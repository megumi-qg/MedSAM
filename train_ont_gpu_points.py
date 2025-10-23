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
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import monai
from tqdm import tqdm
tqdm.disable = True
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
        self.logger = logging.getLogger(LOGGER_NAME)
        self.logger.info(
            "[PointDataset] samples: %s, fixed K=%s",
            len(self.items),
            self.max_points,
        )

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

LOGGER_NAME = "medsam_train"


def collate_points(batch):
    # 把列表[(img, gt, pts, labs, name), ...] 堆叠成 batch
    imgs   = torch.stack([b[0] for b in batch], dim=0)   # (B,3,H,W)
    gts    = torch.stack([b[1] for b in batch], dim=0)   # (B,1,H,W)
    points = torch.stack([b[2] for b in batch], dim=0)   # (B,K,2)
    labels = torch.stack([b[3] for b in batch], dim=0)   # (B,K)
    names  = [b[4] for b in batch]
    return imgs, gts, points, labels, names


def compute_batch_dice(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    preds: (B,1,H,W) logits
    targets: (B,1,H,W) binary masks
    """
    probs = torch.sigmoid(preds)
    preds_bin = (probs > 0.5).float()
    targets = targets.float()
    intersection = (preds_bin * targets).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item()


def save_validation_preview(images, gts, preds, names, save_dir, epoch):
    """Save a quick preview of validation predictions for visual monitoring."""
    try:
        idx = 0
        img = images[idx].cpu().permute(1, 2, 0).numpy()
        gt_mask = gts[idx].cpu().numpy()
        pred_mask = torch.sigmoid(preds[idx]).cpu().numpy()
        pred_binary = (pred_mask > 0.5).astype(np.float32)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img)
        axs[0].axis("off")
        axs[0].set_title(f"Image: {names[idx]}")

        axs[1].imshow(img)
        show_mask(gt_mask, axs[1])
        axs[1].axis("off")
        axs[1].set_title("Ground Truth")

        axs[2].imshow(img)
        show_mask(pred_binary[None, ...], axs[2])
        axs[2].axis("off")
        axs[2].set_title("Prediction")

        plt.tight_layout()
        preview_path = os.path.join(save_dir, f"val_preview_epoch_{epoch:04d}.png")
        plt.savefig(preview_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
    except Exception as exc:
        logging.getLogger(LOGGER_NAME).warning(
            "failed to save validation preview: %s", exc
        )


def save_training_curves(save_dir, task_name, train_losses, val_losses, val_dices):
    epochs = range(1, len(train_losses) + 1)
    val_epochs = range(1, len(val_losses) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(epochs, train_losses, label="Train Loss", color="#1f77b4")
    if val_losses:
        axs[0].plot(val_epochs, val_losses, label="Val Loss", color="#ff7f0e")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss Curves")
    axs[0].legend()

    if val_dices:
        axs[1].plot(val_epochs, val_dices, label="Val Dice", color="#2ca02c")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Dice Score")
        axs[1].set_title("Validation Dice")
        axs[1].set_ylim(0, 1)
        axs[1].legend()
    else:
        axs[1].axis("off")
        axs[1].text(
            0.5,
            0.5,
            "No validation data",
            ha="center",
            va="center",
            transform=axs[1].transAxes,
        )

    plt.tight_layout()
    curve_path = os.path.join(save_dir, f"{task_name}_metrics.png")
    plt.savefig(curve_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

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
    parser.add_argument("-batch_size", type=int, default=8)
    parser.add_argument("-num_workers", type=int, default=0)
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="fraction of data reserved for validation (0-1)",
    )
    parser.add_argument(
        "--disable_val_preview",
        action="store_true",
        help="disable saving validation prediction previews",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="training.log",
        help="filename used to store the training log inside work_dir/task",
    )
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

    if not 0.0 <= args.val_split < 1.0:
        raise ValueError("val_split must be in the range [0, 1).")

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

    log_path = os.path.join(model_save_path, args.log_file)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info("Log file: %s", log_path)
    logger.info("Work dir for this run: %s", model_save_path)
    logger.info("Arguments: %s", vars(args))
    # 复制脚本快照
    snapshot_path = os.path.join(
        model_save_path,
        f"{run_id}_{os.path.basename(__file__) if '__file__' in globals() else 'train.py'}",
    )
    try:
        if '__file__' in globals():
            shutil.copyfile(__file__, snapshot_path)
    except Exception as e:
        logger.warning("skip script snapshot: %s", e)


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

    logger.info(
        "Number of total parameters: %s",
        sum(p.numel() for p in medsam_model.parameters()),
    )
    logger.info(
        "Number of trainable parameters: %s",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )

    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    logger.info(
        "Number of image encoder and mask decoder parameters: %s",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    
    # === 加载数据集 ===
    full_dataset = JsonlPointDataset(
        jsonl_path=args.jsonl,
        max_points_per_sample=32,   # 可调 16/32/64
    ) 
    dataset_size = len(full_dataset)
    val_dataset = None
    train_dataset = full_dataset

    if args.val_split > 0 and dataset_size >= 2:
        val_size = max(1, int(dataset_size * args.val_split))
        val_size = min(val_size, dataset_size - 1)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(2025),
        )
        logger.info("Dataset split -> train: %s, val: %s", train_size, val_size)
    else:
        logger.info("Skipping validation split due to configuration or dataset size.")

    logger.info("Number of training samples: %s", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_points,  # 关键！
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_points,
        )
        logger.info("Number of validation samples: %s", len(val_dataset))

    # === 训练循环 ===
    best_metric = float("inf")
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    scaler = GradScaler(enabled=args.use_amp)
    train_losses = []
    val_losses = []
    val_dices = []

    iter_num = 0
    for epoch in range(start_epoch, args.num_epochs):
        medsam_model.train()
        epoch_loss = 0.0
        train_batches = 0
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
            train_batches += 1
            iter_num += 1

        avg_train_loss = epoch_loss / max(1, train_batches)
        train_losses.append(avg_train_loss)

        val_loss = None
        val_dice = None
        if val_dataloader is not None:
            medsam_model.eval()
            val_loss_acc = 0.0
            val_dice_acc = 0.0
            val_batches = 0
            preview_payload = None
            with torch.no_grad():
                for val_step, (image, gt, points, labels, names) in enumerate(val_dataloader):
                    image  = image.to(device)
                    gt     = gt.to(device)
                    points = points.to(device)
                    labels = labels.to(device)

                    with autocast(device_type="cuda", enabled=args.use_amp):
                        pred = medsam_model(image, points, labels)
                        loss = seg_loss(pred, gt.float()) + ce_loss(pred, gt.float())

                    val_loss_acc += loss.item()
                    val_dice_acc += compute_batch_dice(pred, gt)
                    val_batches += 1

                    if preview_payload is None:
                        preview_payload = (
                            image[:1].detach().cpu(),
                            gt[:1].detach().cpu(),
                            pred[:1].detach().cpu(),
                            names[:1],
                        )

            val_loss = val_loss_acc / max(1, val_batches)
            val_dice = val_dice_acc / max(1, val_batches)
            val_losses.append(val_loss)
            val_dices.append(val_dice)
            if preview_payload is not None and not args.disable_val_preview:
                save_validation_preview(*preview_payload, save_dir=model_save_path, epoch=epoch)
            medsam_model.train()

        if args.use_wandb:
            log_payload = {"epoch_loss": avg_train_loss}
            if val_loss is not None:
                log_payload["val_loss"] = val_loss
                log_payload["val_dice"] = val_dice
            wandb.log(log_payload)

        if val_loss is not None:
            logger.info(
                "Epoch %s | TrainLoss %.4f | ValLoss %.4f | ValDice %.4f",
                epoch,
                avg_train_loss,
                val_loss,
                val_dice,
            )
        else:
            logger.info(
                "Epoch %s | Loss %.4f",
                epoch,
                avg_train_loss,
            )
        ## save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_latest.pth"))
        ## save the best model
        metric_to_track = val_loss if val_loss is not None else avg_train_loss
        if metric_to_track < best_metric:
            best_metric = metric_to_track
            torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_best.pth"))
            logger.info("Best model updated at epoch %s (metric %.4f)", epoch, metric_to_track)

        save_training_curves(
            model_save_path,
            args.task_name,
            train_losses,
            val_losses,
            val_dices,
        )

if __name__ == "__main__":
    main()
