import os
import torch
import logging
import shutil
from datetime import datetime
import monai
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import argparse

from segment_anything import sam_model_registry
from train.dataset_points import JsonlPointDataset, collate_points
from train.model_medsam import MedSAM
from train.utils_visual import save_validation_preview
from train.utils_train import compute_batch_dice, save_training_curves

LOGGER_NAME = "medsam_train"

# ========== 环境配置 ==========
# set seeds
torch.manual_seed(2025)
torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


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
    parser.add_argument("-work_dir", type=str, default="/home/gaoqi/HeartSeg/github/MedSAM/work_dir")
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
    parser.add_argument("-use_wandb", type=bool, default=False, help="use wandb to monitor training")
    parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
    parser.add_argument("--resume", type=str, default="/home/gaoqi/HeartSeg/github/MedSAM/work_dir/ACDC-SAM-ViT-B-20251023-1527/medsam_model_latest.pth", help="Resuming training from checkpoint")
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

    # >>> 新增：根据是否 resume 决定保存目录
    if args.resume and os.path.isfile(args.resume):
        # 继续在同一目录训练
        model_save_path = os.path.dirname(os.path.abspath(args.resume))
        resume_mode = True
    else:
        model_save_path = os.path.join(args.work_dir, args.task_name + "-" + run_id)
        resume_mode = False
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
    scaler = GradScaler(enabled=args.use_amp)
    train_losses, val_losses, val_dices = [], [], []

    if args.resume and os.path.isfile(args.resume):
        logger.info("=> Resuming from checkpoint: %s", args.resume)
        checkpoint = torch.load(args.resume, map_location=device)

        # 兼容键名
        model_state = checkpoint.get("model", checkpoint.get("state_dict"))
        if model_state is None:
            raise RuntimeError("Checkpoint missing 'model' state_dict")

        medsam_model.load_state_dict(model_state, strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if args.use_amp and "scaler" in checkpoint:
            try:
                scaler.load_state_dict(checkpoint["scaler"])
            except Exception as e:
                logger.warning("AMP scaler state load failed, continue fresh scaler: %s", e)

        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_metric = float(checkpoint.get("best_metric", float("inf")))

        # 恢复历史曲线（若存在）
        train_losses = list(checkpoint.get("train_losses", []))
        val_losses   = list(checkpoint.get("val_losses", []))
        val_dices    = list(checkpoint.get("val_dices", []))

        logger.info(
            "=> Resume OK | start_epoch=%d | best_metric=%.6f | history: train=%d, val=%d",
            start_epoch, best_metric, len(train_losses), len(val_losses)
        )
    else:
        logger.info("=> Fresh training run in: %s", model_save_path)

    iter_num = 0
    for epoch in range(start_epoch, args.num_epochs):
        medsam_model.train()
        epoch_loss = 0.0
        train_batches = 0
        #for step, (image, gt, points, labels, _) in enumerate(tqdm(train_dataloader)):
        for step, (image, gt, points, labels, _) in enumerate(train_dataloader):
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
            "best_metric": best_metric,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_dices": val_dices,
        }
        if args.use_amp:
            checkpoint["scaler"] = scaler.state_dict()
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
