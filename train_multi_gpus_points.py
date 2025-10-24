# -*- coding: utf-8 -*-
"""
Multi-GPU (DDP) training for point-prompted SAM fine-tuning.
Converted from single-GPU train_one_gpu_points.py to follow the DDP pattern
used in train_multi_gpus.py (spawn + init_process_group + DistributedSampler,
DDP wrapping, optional AMP and gradient accumulation).

Usage (example on 4 GPUs, single node):
  torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    train_one_gpu_points_ddp.py \
      -jsonl /path/to/scribble_points.jsonl \
      -checkpoint /path/to/sam_vit_b_01ec64.pth \
      -batch_size 4 -num_workers 8 -use_amp

Notes:
- Validation and logging run on rank 0 only to avoid duplication.
- Checkpoints are saved by rank 0; all ranks hit barriers to keep in sync.
- For gradient accumulation, use --grad_acc_steps > 1.
"""

import os
import json
import random
import shutil
from datetime import datetime
import logging
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

import monai
from tqdm import tqdm
from segment_anything import sam_model_registry

# ===================== Utilities =====================
LOGGER_NAME = "medsam_train"

torch.manual_seed(2025)
torch.cuda.empty_cache()

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "6")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# ===================== Dataset =====================
class JsonlPointDataset(Dataset):
    """
    Items are JSON lines with keys: image_path, mask_path, class_id,
    point_coords (Nx2 [x,y]), point_labels (N, usually 1s), slice_id
    """
    def __init__(self, jsonl_path: str, max_points_per_sample: int = 32):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))
        self.max_points = max_points_per_sample

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img = np.load(it["image_path"], "r", allow_pickle=True)  # (1024,1024,3), [0,1]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)      # -> (3,H,W)

        gt_multi = np.load(it["mask_path"], "r", allow_pickle=True)
        cls_id = int(it["class_id"])
        gt2D = (gt_multi == cls_id).astype(np.uint8)[None, ...]    # (1,H,W)

        pts = np.asarray(it["point_coords"], dtype=np.float32)    # (N,2) [x,y]
        labs = np.asarray(it["point_labels"], dtype=np.int64)     # (N,)

        K = self.max_points
        N = len(pts)
        if N == 0:
            pts_k = np.zeros((K, 2), dtype=np.float32)
            labs_k = np.zeros((K,), dtype=np.int64)
        elif N >= K:
            sel = np.random.choice(N, size=K, replace=False)
            pts_k = pts[sel]
            labs_k = labs[sel]
        else:
            sel = np.random.choice(N, size=K, replace=True)
            pts_k = pts[sel]
            labs_k = labs[sel]

        return (
            torch.from_numpy(img),           # (3,H,W)
            torch.from_numpy(gt2D).long(),   # (1,H,W)
            torch.from_numpy(pts_k).float(), # (K,2)
            torch.from_numpy(labs_k).long(), # (K,)
            it.get("slice_id", os.path.basename(it["image_path"]))
        )

def collate_points(batch):
    imgs   = torch.stack([b[0] for b in batch], dim=0)
    gts    = torch.stack([b[1] for b in batch], dim=0)
    points = torch.stack([b[2] for b in batch], dim=0)
    labels = torch.stack([b[3] for b in batch], dim=0)
    names  = [b[4] for b in batch]
    return imgs, gts, points, labels, names

# ===================== Model =====================
class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder and image encoder (as in single-GPU version)
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.image_encoder.parameters():
            p.requires_grad = False

    def forward(self, image, points, labels):
        image_embedding = self.image_encoder(image)  # (B,256,64,64)
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(points, labels), boxes=None, masks=None
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

# ===================== Helpers =====================

def compute_batch_dice(preds: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(preds)
    preds_bin = (probs > 0.5).float()
    targets = targets.float()
    intersection = (preds_bin * targets).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item()


def save_validation_preview(images, gts, preds, names, save_dir, epoch):
    try:
        idx = 0
        img = images[idx].cpu().permute(1, 2, 0).numpy()
        gt_mask = gts[idx].cpu().numpy()
        pred_mask = torch.sigmoid(preds[idx]).cpu().numpy()
        pred_binary = (pred_mask > 0.5).astype(np.float32)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img); axs[0].axis("off"); axs[0].set_title(f"Image: {names[idx]}")
        axs[1].imshow(img); show_mask(gt_mask, axs[1]); axs[1].axis("off"); axs[1].set_title("Ground Truth")
        axs[2].imshow(img); show_mask(pred_binary[None, ...], axs[2]); axs[2].axis("off"); axs[2].set_title("Prediction")
        plt.tight_layout()
        preview_path = os.path.join(save_dir, f"val_preview_epoch_{epoch:04d}.png")
        plt.savefig(preview_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
    except Exception as exc:
        logging.getLogger(LOGGER_NAME).warning("failed to save validation preview: %s", exc)


def save_training_curves(save_dir, task_name, train_losses, val_losses, val_dices):
    epochs = range(1, len(train_losses) + 1)
    val_epochs = range(1, len(val_losses) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(list(epochs), train_losses, label="Train Loss")
    if val_losses:
        axs[0].plot(list(val_epochs), val_losses, label="Val Loss")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss"); axs[0].set_title("Loss Curves"); axs[0].legend()

    if val_dices:
        axs[1].plot(list(val_epochs), val_dices, label="Val Dice")
        axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Dice Score"); axs[1].set_ylim(0, 1); axs[1].legend(); axs[1].set_title("Validation Dice")
    else:
        axs[1].axis("off"); axs[1].text(0.5, 0.5, "No validation data", ha="center", va="center", transform=axs[1].transAxes)

    plt.tight_layout()
    curve_path = os.path.join(save_dir, f"{task_name}_metrics.png")
    plt.savefig(curve_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

# ===================== Main (DDP) =====================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-jsonl", type=str, required=True, help="path to scribble_points.jsonl")
    parser.add_argument("-task_name", type=str, default="ACDC-SAM-ViT-B")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("-checkpoint", type=str, required=True)
    parser.add_argument("--load_pretrain", type=bool, default=True)
    parser.add_argument("-pretrain_model_path", type=str, default="")
    parser.add_argument("-work_dir", type=str, default="./work_dir")
    # train
    parser.add_argument("-num_epochs", type=int, default=1000)
    parser.add_argument("-batch_size", type=int, default=8)
    parser.add_argument("-num_workers", type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--disable_val_preview", action="store_true")
    parser.add_argument("--log_file", type=str, default="training.log")
    # Optimizer
    parser.add_argument("-weight_decay", type=float, default=0.01)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-use_wandb", type=bool, default=False)
    parser.add_argument("-use_amp", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default="")
    # DDP related
    parser.add_argument("--world_size", type=int, help="Total processes (GPUs) across all nodes")
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--init_method", type=str, default="env://")
    parser.add_argument("--bucket_cap_mb", type=int, default=25)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    # Derive world size if not set explicitly
    if args.world_size is None:
        args.world_size = torch.cuda.device_count()

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    global_save_dir = os.path.join(args.work_dir, f"{args.task_name}-{run_id}")
    # Create work_dir only on rank 0 inside workers (after rank known). Here we pre-create for convenience when single-process debugging.
    os.makedirs(global_save_dir, exist_ok=True)

    ngpus_per_node = torch.cuda.device_count()
    print(f"Spawning {ngpus_per_node} processes (GPUs)")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, global_save_dir, run_id))


def build_logger(log_path: str):
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(formatter)
    sh = logging.StreamHandler(); sh.setFormatter(formatter)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def main_worker(gpu: int, ngpus_per_node: int, args, save_dir_root: str, run_id: str):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    is_main_host = rank == 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl", init_method=args.init_method, rank=rank, world_size=args.world_size)

    # Create per-run dir and snapshot on rank 0
    if is_main_host:
        os.makedirs(save_dir_root, exist_ok=True)
        try:
            if '__file__' in globals():
                shutil.copyfile(__file__, os.path.join(save_dir_root, f"{run_id}_" + os.path.basename(__file__)))
        except Exception as e:
            print("[Rank 0] Skip script snapshot:", e)

    log_path = os.path.join(save_dir_root, args.log_file)
    logger = build_logger(log_path) if is_main_host else logging.getLogger(LOGGER_NAME)

    if is_main_host:
        logger.info("Work dir: %s", save_dir_root)
        logger.info("Args: %s", vars(args))

    # ===== Model =====
    checkpoint_abs = os.path.abspath(args.checkpoint)
    assert os.path.exists(checkpoint_abs), f"Checkpoint not found: {checkpoint_abs}"

    sam_model = sam_model_registry[args.model_type](checkpoint=checkpoint_abs)
    model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).cuda()

    # Wrap DDP
    if is_main_host:
        print(f"[Rank {rank}] CUDA mem before DDP: {torch.cuda.mem_get_info(gpu)[0]/(1024**3):.2f} GB free")
    model = DDP(
        model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb,
    )

    # Optimizer / loss
    img_mask_params = list(model.module.image_encoder.parameters()) + list(model.module.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(img_mask_params, lr=args.lr, weight_decay=args.weight_decay)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss  = nn.BCEWithLogitsLoss(reduction="mean")

    # ===== Data =====
    full_dataset = JsonlPointDataset(jsonl_path=args.jsonl, max_points_per_sample=32)
    dataset_size = len(full_dataset)

    if args.val_split > 0 and dataset_size >= 2:
        val_size = max(1, int(dataset_size * args.val_split))
        val_size = min(val_size, dataset_size - 1)
        train_size = dataset_size - val_size
        # Ensure identical split on all ranks
        g = torch.Generator().manual_seed(2025)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)
    else:
        train_dataset, val_dataset = full_dataset, None

    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True, drop_last=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # sampler does shuffling
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=collate_points,
    )

    # For simplicity, run validation only on rank 0
    if val_dataset is not None and is_main_host:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_points,
        )
    else:
        val_loader = None

    # ===== Resume =====
    start_epoch = 0
    if args.resume:
        loc = f"cuda:{gpu}"
        if os.path.isfile(args.resume):
            ckpt = torch.load(args.resume, map_location=loc)
            start_epoch = ckpt.get("epoch", -1) + 1
            model.load_state_dict(ckpt["model"])  # DDP-safe
            optimizer.load_state_dict(ckpt["optimizer"])
            if is_main_host:
                logger.info("Resumed from %s (epoch %s)", args.resume, start_epoch)
        dist.barrier()

    scaler = GradScaler(enabled=args.use_amp)

    train_losses, val_losses, val_dices = [], [], []
    best_metric = float("inf")

    # ===== Train loop =====
    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for step, (image, gt, points, labels, _) in enumerate(tqdm(train_loader, disable=not is_main_host)):
            optimizer.zero_grad(set_to_none=True)
            image  = image.cuda(non_blocking=True)
            gt     = gt.cuda(non_blocking=True).float()
            points = points.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            with autocast(device_type="cuda", enabled=args.use_amp):
                pred = model(image, points, labels)
                loss = seg_loss(pred, gt) + ce_loss(pred, gt)

            if args.grad_acc_steps > 1:
                loss = loss / args.grad_acc_steps
                if (step + 1) % args.grad_acc_steps == 0:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    with model.no_sync():
                        scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        # Gather average loss (each rank has its own view); report rank 0 only
        avg_train_loss = epoch_loss / max(1, num_batches)
        train_losses.append(avg_train_loss)

        # ===== Validation on rank 0 =====
        if val_loader is not None and is_main_host:
            model.eval()
            val_loss_acc = 0.0
            val_dice_acc = 0.0
            val_batches = 0
            preview_payload = None
            with torch.no_grad():
                for vstep, (image, gt, points, labels, names) in enumerate(val_loader):
                    image  = image.cuda(non_blocking=True)
                    gt     = gt.cuda(non_blocking=True).float()
                    points = points.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                    with autocast(device_type="cuda", enabled=args.use_amp):
                        pred = model.module(image, points, labels)  # use .module in eval on rank 0
                        vloss = seg_loss(pred, gt) + ce_loss(pred, gt)
                    val_loss_acc += vloss.item()
                    val_dice_acc += compute_batch_dice(pred, gt)
                    val_batches += 1
                    if preview_payload is None:
                        preview_payload = (image[:1].cpu(), gt[:1].cpu(), pred[:1].cpu(), names[:1])

            v_loss = val_loss_acc / max(1, val_batches)
            v_dice = val_dice_acc / max(1, val_batches)
            val_losses.append(v_loss)
            val_dices.append(v_dice)
            if preview_payload is not None and not args.disable_val_preview:
                save_validation_preview(*preview_payload, save_dir=save_dir_root, epoch=epoch)
            logger.info("Epoch %d | TrainLoss %.4f | ValLoss %.4f | ValDice %.4f", epoch, avg_train_loss, v_loss, v_dice)
        elif is_main_host:
            logger.info("Epoch %d | Loss %.4f", epoch, avg_train_loss)

        # ===== Save checkpoints (rank 0) =====
        if is_main_host:
            ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
            torch.save(ckpt, os.path.join(save_dir_root, "medsam_model_latest.pth"))
            metric = val_losses[-1] if (val_losses and val_loader is not None) else avg_train_loss
            if metric < best_metric:
                best_metric = metric
                torch.save(ckpt, os.path.join(save_dir_root, "medsam_model_best.pth"))
                logger.info("Best model updated at epoch %d (metric %.4f)", epoch, metric)
            save_training_curves(save_dir_root, args.task_name, train_losses, val_losses, val_dices)

        # Keep ranks in sync
        dist.barrier()

    if is_main_host:
        logger.info("Training completed.")


if __name__ == "__main__":
    main()
