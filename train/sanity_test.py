import random
from matplotlib import pyplot as plt
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from train.dataset_points import JsonlPointDataset, collate_points
from train.utils_visual import show_mask, show_points


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
