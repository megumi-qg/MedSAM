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
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import monai
from tqdm import tqdm
from segment_anything import sam_model_registry
from torch.cuda.amp import GradScaler, autocast
# ACDC 1842 slices


# ========== 环境配置 ==========
# set seeds
torch.manual_seed(2025)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

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
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, points, labels):
        """
        image: (B,3,H,W) float32
        points: (B,K,2) float32, 像素坐标 [x,y]，与输入分辨率一致(1024)
        labels: (B,K)   int64, 1 表示正点（你现在全 1）
        """
        points = points.to(image.device)
        labels = labels.to(image.device)
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
parser = argparse.ArgumentParser()
parser.add_argument(
    "-jsonl",
    type=str,
    default="/home/gaoqi/HeartSeg/github/MedSAM/data/npy/MR_ACDCtr/scri_points/scribble_points.jsonl",
    help="path to scribble_points.jsonl",
)
parser.add_argument("-task_name", type=str, default="ACDC-SAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint",
    type=str,
    default="/home/gaoqi/HeartSeg/github/MedSAM/work_dir/SAM/sam_vit_b_01ec64.pth",
)
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="load pretrain model"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=1000)
parser.add_argument("-batch_size", type=int, default=8)
parser.add_argument("-num_workers", type=int, default=8)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")
parser.add_argument(
    "--max_points_per_sample",
    type=int,
    default=32,
    help="Number of point prompts per sample during training",
)
# Distributed training args
parser.add_argument("--world_size", type=int, help="world size")
parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
parser.add_argument(
    "--bucket_cap_mb",
    type=int,
    default=25,
    help="The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket",
)
parser.add_argument(
    "--grad_acc_steps",
    type=int,
    default=1,
    help="Gradient accumulation steps before syncing gradients for backprop",
)
parser.add_argument("--init_method", type=str, default="env://")

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

run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = os.path.join(args.work_dir, args.task_name + "-" + run_id)

args.checkpoint = os.path.abspath(args.checkpoint)
assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"


def main():
    ngpus_per_node = torch.cuda.device_count()
    if args.world_size is None:
        args.world_size = ngpus_per_node
    print("Spwaning processces")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = args.world_size
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    if is_main_host:
        os.makedirs(model_save_path, exist_ok=True)
        try:
            shutil.copyfile(
                __file__, os.path.join(model_save_path, f"{run_id}_{os.path.basename(__file__)}")
            )
        except Exception as e:
            print(f"[warn] skip script snapshot: {e}")
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(
        backend="nccl", init_method=args.init_method, rank=rank, world_size=world_size
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).cuda()
    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[1] / (
        1024**3
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Total CUDA memory before DDP initialised: {total_cuda_mem} Gb"
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Free CUDA memory before DDP initialised: {free_cuda_mem} Gb"
    )
    if rank % ngpus_per_node == 0:
        print("Before DDP initialization:")
        os.system("nvidia-smi")

    medsam_model = nn.parallel.DistributedDataParallel(
        medsam_model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb,
    )

    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[1] / (
        1024**3
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Total CUDA memory after DDP initialised: {total_cuda_mem} Gb"
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Free CUDA memory after DDP initialised: {free_cuda_mem} Gb"
    )
    if rank % ngpus_per_node == 0:
        print("After DDP initialization:")
        os.system("nvidia-smi")

    medsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )

    img_mask_encdec_params = list(
        medsam_model.module.image_encoder.parameters()
    ) + list(medsam_model.module.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    train_dataset = JsonlPointDataset(
        jsonl_path=args.jsonl, max_points_per_sample=args.max_points_per_sample
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=collate_points,
    )

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(rank, "=> loading checkpoint '{}'".format(args.resume))
            loc = f"cuda:{gpu}"
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                rank,
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                ),
            )
        torch.distributed.barrier()

    if args.use_amp:
        scaler = GradScaler()
        print(f"[RANK {rank}: GPU {gpu}] Using AMP for training")

    losses = []
    best_loss = 1e10
    iter_num = 0

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0
        train_dataloader.sampler.set_epoch(epoch)
        num_steps = 0
        for step, (image, gt, points, labels, _) in enumerate(
            tqdm(train_dataloader, desc=f"[RANK {rank}: GPU {gpu}]")
        ):
            optimizer.zero_grad()
            image = image.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            points = points.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            if args.use_amp:
                with autocast(dtype=torch.float16):
                    pred = medsam_model(image, points, labels)
                    loss = seg_loss(pred, gt.float()) + ce_loss(pred, gt.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                pred = medsam_model(image, points, labels)
                loss = seg_loss(pred, gt.float()) + ce_loss(pred, gt.float())
                if args.grad_acc_steps > 1:
                    loss = loss / args.grad_acc_steps
                    if (step + 1) % args.grad_acc_steps == 0:
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        with medsam_model.no_sync():
                            loss.backward()
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            if step > 10 and step % 100 == 0:
                if is_main_host:
                    checkpoint = {
                        "model": medsam_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(
                        checkpoint,
                        os.path.join(model_save_path, "medsam_model_latest_step.pth"),
                    )

            epoch_loss += loss.item()
            iter_num += 1
            num_steps += 1

        cuda_mem_info = torch.cuda.mem_get_info(gpu)
        free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[
            1
        ] / (1024**3)
        print("\n")
        print(f"[RANK {rank}: GPU {gpu}] Total CUDA memory: {total_cuda_mem} Gb")
        print(f"[RANK {rank}: GPU {gpu}] Free CUDA memory: {free_cuda_mem} Gb")
        print(
            f"[RANK {rank}: GPU {gpu}] Used CUDA memory: {total_cuda_mem - free_cuda_mem} Gb"
        )
        print("\n")

        epoch_loss /= max(1, num_steps)
        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        if is_main_host:
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_latest.pth"))
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_best.pth"))
            plt.plot(losses)
            plt.title("Dice + Cross Entropy Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(os.path.join(model_save_path, args.task_name + "train_loss.png"))
            plt.close()

        torch.distributed.barrier()


if __name__ == "__main__":
    main()
