import torch
from torch.utils.data import Dataset
import numpy as np
import json, logging

LOGGER_NAME = "medsam_train"

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


