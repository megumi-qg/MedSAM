import torch, os, matplotlib.pyplot as plt

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