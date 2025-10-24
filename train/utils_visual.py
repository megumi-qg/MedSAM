import numpy as np
import matplotlib.pyplot as plt
import os, logging
import torch

LOGGER_NAME = "medsam_train"

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