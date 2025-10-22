# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""
"""
Train SAM with point prompts (from scribble_points.jsonl)
Author: Adapted for point-based fine-tuning on ACDC
"""
# %% setup environment
import os
import json
import random
import shutil
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import monai
from tqdm import tqdm
from segment_anything import sam_model_registry
from torch.amp import GradScaler, autocast
# ACDC 1842 slices


# ========== 环境配置 ==========
# set seeds
torch.manual_seed(2025)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

# ========== 工具函数 ==========
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(pts_xy, ax, size=20):
    # pts_xy: (K,2) [x,y]
    ax.scatter(pts_xy[:,0], pts_xy[:,1], s=size, marker='o')

# ========== 数据集定义 ==========
# 不再随机挑一个 GT 类别，而是严格用 jsonl 里指定的 class_id 来生成监督的二值 gt2D，与该类的正点一致。
class JsonlPointDataset(Dataset):
    def __init__(self, jsonl_path, max_points_per_sample=32):
        """
        jsonl_path: 你的 scribble_points.jsonl
        max_points_per_sample: 训练时固定采样/重复到的点数K（建议 16/32/64）
        """
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append(obj)

        self.max_points = max_points_per_sample
        print(f"[PointDataset] samples: {len(self.items)}, fixed K={self.max_points}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]

        img = np.load(it["image_path"], "r", allow_pickle=True)  # (1024,1024,3), [0,1]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)    # -> (3,H,W)

        gt_multi = np.load(it["mask_path"], "r", allow_pickle=True)  # 多类
        cls_id = int(it["class_id"])
        gt2D = (gt_multi == cls_id).astype(np.uint8)[None, ...]      # (1,H,W) 二值

        # 点（像素坐标）
        pts = np.asarray(it["point_coords"], dtype=np.float32)  # (N,2), [x,y]
        labs = np.asarray(it["point_labels"], dtype=np.int64)   # (N,), 全1

        # 统一点数为 K
        K = self.max_points
        N = len(pts)
        if N == 0:
            # 保险起见：几乎不会发生（你的jsonl里都有点）
            pts_k = np.zeros((K, 2), dtype=np.float32)
            labs_k = np.zeros((K,), dtype=np.int64)
        elif N >= K:
            sel = np.random.choice(N, size=K, replace=False)
            pts_k = pts[sel]
            labs_k = labs[sel]
        else:
            # 采样有放回凑满 K
            sel = np.random.choice(N, size=K, replace=True)
            pts_k = pts[sel]
            labs_k = labs[sel]

        return (
            torch.from_numpy(img),           # (3,H,W) float32
            torch.from_numpy(gt2D).long(),   # (1,H,W) long(0/1)
            torch.from_numpy(pts_k).float(), # (K,2) float32, 像素坐标
            torch.from_numpy(labs_k).long(), # (K,) int64, 1
            it["slice_id"],                  # 便于可视化
        )

def collate_points(batch):
    # 把列表[(img, gt, pts, labs, name), ...] 堆叠成 batch
    imgs   = torch.stack([b[0] for b in batch], dim=0)   # (B,3,H,W)
    gts    = torch.stack([b[1] for b in batch], dim=0)   # (B,1,H,W)
    points = torch.stack([b[2] for b in batch], dim=0)   # (B,K,2)
    labels = torch.stack([b[3] for b in batch], dim=0)   # (B,K)
    names  = [b[4] for b in batch]
    return imgs, gts, points, labels, names

# ========== 模型定义 ==========
class MedSAM(nn.Module):
    def __init__(self,image_encoder,mask_decoder,prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder and image encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, points, labels):
        """
        image: (B,3,H,W) float32
        points: (B,K,2) float32, 像素坐标 [x,y]，与输入分辨率一致(1024)
        labels: (B,K)   int64, 1 表示正点（你现在全 1）
        """
        image_embedding = self.image_encoder(image)  # (B,256,64,64)
        with torch.no_grad():
            # SAM 的 prompt_encoder 期望 points 为像素坐标；当前图像已是 1024×1024
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(points, labels),
                boxes=None,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

# ========== sanity test ==========
jsonl_path = "/home/gaoqi/HeartSeg/github/MedSAM/data/npy/MR_ACDCtr/scri_points/scribble_points.jsonl"

tr_dataset = JsonlPointDataset(jsonl_path, max_points_per_sample=32)
tr_dataloader = DataLoader(
    tr_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_points,
)

for step, (image, gt, points, labels, names_temp) in enumerate(tr_dataloader):
    # image:  (B,3,H,W)
    # gt:     (B,1,H,W)
    # points: (B,K,2)  像素坐标 [x,y]
    # labels: (B,K)    这里全为 1
    # names_temp: list[str]
    print(image.shape, gt.shape, points.shape, labels.shape)

    _, axs = plt.subplots(1, 2, figsize=(25, 25))

    idx = random.randint(0, image.shape[0]-1)
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[0])                 # (1,H,W) -> OK
    show_points(points[idx].cpu().numpy(), axs[0])           # 画点
    axs[0].axis("off")
    axs[0].set_title(names_temp[idx])

    idx = random.randint(0, image.shape[0]-1)
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[1])
    show_points(points[idx].cpu().numpy(), axs[1])
    axs[1].axis("off")
    axs[1].set_title(names_temp[idx])

    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("./data_sanitycheck_points.png", bbox_inches="tight", dpi=300)
    plt.close()
    break
# ========== 训练主函数 ==========
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-jsonl",
        type=str,
        default="/home/gaoqi/HeartSeg/github/MedSAM/data/npy/MR_ACDCtr/scri_points/scribble_points.jsonl",
        help="path to scribble_points.jsonl",
    )
    parser.add_argument("-task_name", type=str, default="ACDC-SAM-ViT-B")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("-checkpoint", type=str, default="/home/gaoqi/HeartSeg/github/MedSAM/work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument(
        "--load_pretrain", type=bool, default=True, help="load pretrain model"
    )
    parser.add_argument("-pretrain_model_path", type=str, default="")
    parser.add_argument("-work_dir", type=str, default="./work_dir")
    # train
    parser.add_argument("-num_epochs", type=int, default=1000)
    parser.add_argument("-batch_size", type=int, default=2)
    parser.add_argument("-num_workers", type=int, default=0)
    # Optimizer parameters
    parser.add_argument("-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
    parser.add_argument("-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
    )
    parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
    parser.add_argument(
        "--resume", type=str, default="", help="Resuming training from checkpoint"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.use_wandb:
        import wandb
        wandb.login()
        wandb.init(
            project=args.task_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "data_path": args.jsonl,
                "model_type": args.model_type,
            },
        )

    # set up model for training
    device = torch.device(args.device)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = os.path.join(args.work_dir, args.task_name + "-" + run_id)
    os.makedirs(model_save_path, exist_ok=True)
    # 复制脚本快照
    snapshot_path = os.path.join(model_save_path, f"{run_id}_{os.path.basename(__file__) if '__file__' in globals() else 'train.py'}")
    try:
        if '__file__' in globals():
            shutil.copyfile(__file__, snapshot_path)
    except Exception as e:
        print(f"[warn] skip script snapshot: {e}")


    # === 加载 SAM 模型 ===
    args.checkpoint = os.path.abspath(args.checkpoint)
    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252

    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    
    # === 加载数据集 ===
    train_dataset = JsonlPointDataset(
        jsonl_path=args.jsonl,
        max_points_per_sample=32,   # 可调 16/32/64
    ) 
    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_points,  # 关键！
    )

    # === 训练循环 ===
    best_loss = 1e10
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    scaler = GradScaler(enabled=args.use_amp)
    losses = []

    iter_num = 0
    for epoch in range(start_epoch, args.num_epochs):
        medsam_model.train()
        epoch_loss = 0
        for step, (image, gt, points, labels, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            image  = image.to(device)   # (B,3,H,W)
            gt     = gt.to(device)    # (B,1,H,W)
            points = points.to(device)  # (B,K,2)
            labels = labels.to(device)  # (B,K)
            
            with autocast(device_type="cuda", enabled=args.use_amp):
                pred = medsam_model(image, points, labels)
                loss = seg_loss(pred, gt.float()) + ce_loss(pred, gt.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            iter_num += 1

        epoch_loss /= (step + 1)
        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        ## save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_latest.pth"))
        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_best.pth"))

        # plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()

if __name__ == "__main__":
    main()
