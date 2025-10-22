# 我想要微调SAM模型，我现在已经有了针对数据的scribble，我想把scribble按像素转换为point进行微调。 我使用的数据集是ACDC。我现在已经将图像数据按照切片转换为1024*1024
# 的npy文件了，例如MR_ACDCtr_patient001_frame01-000.npy。相应的ground truth也转换为了npy文件，例如MR_ACDCtr_patient001_frame01-000.npy。并不是每一张切片都转换为
# 了npy,只有那些包含前景的切片转换为了npy. 但是我的scirbble文件依然是.nii.gz格式的，并且是3D的，例如patient001_frame01.nii.gz。在scribble中，背景为0，右心室为1，
# 心肌为2，左心室为3，0123都是scribble，而这些scribble之外的内容则全是4. 我要微调SAM的话，就只有正样本，例如想要分割右心室，就取值为1的所有点，要分割左心室的话，
# 就取值为3的所有点。 我要怎么转换这个数据使得它可以微调SAM模型呢？

import os
import re
import json
import glob
import math
import random
from pathlib import Path

import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom, binary_dilation
import cv2
from skimage.measure import label as cc_label

# ----------------------------
# 路径与配置（按需修改）
# ----------------------------
# 你的2D图像/GT npy 所在目录（已经是 1024x1024）
IMG_DIR = "/home/gaoqi/HeartSeg/github/MedSAM/data/npy/MR_ACDCtr/imgs"  # e.g. MR_ACDCtr_patient001_frame01-000.npy
GT_DIR  = "/home/gaoqi/HeartSeg/github/MedSAM/data/npy/MR_ACDCtr/gts"   # e.g. MR_ACDCtr_patient001_frame01-000.npy

# 3D scribble 的 nii.gz 所在目录（每个case类似 patient001_frame01.nii.gz）
SCRIBBLE_DIR = "/home/gaoqi/HeartSeg/data/nnU-Net/raw/Dataset001_ACDC/SlabelsTr"

# 输出
OUT_JSONL = "/home/gaoqi/HeartSeg/github/MedSAM/data/npy/MR_ACDCtr/scri_points/scribble_points.jsonl"

# 类别映射（与你的约定一致）
CLASS_ID_TO_NAME = {1: "RV", 2: "MYO", 3: "LV"}
TARGET_CLASS_IDS = [1, 2, 3]  # 只做正点

# 采样上限
MAX_POINTS_PER_CLASS_PER_SLICE = 64  # 可调，32/64/128
BOUNDARY_RATIO = 0.5  # 边界点与内部点比例各占50%

# 当 scribble 非 1024×1024 时如何缩放到 1024（与图像/GT对齐）
RESIZE_TO = (1024, 1024)

# 随机种子（复现）
random.seed(2025)
np.random.seed(2025)


def natural_sort_key(s):
    # 把 ...-000.npy, ...-001.npy 自然排序
    _nsre = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text for text in _nsre.split(s)]

def load_nifti(path):
    nii = nib.load(path)
    data = nii.get_fdata()
    # 转成 int，确保值域是 0/1/2/3/4
    data = data.astype(np.int16)
    data = np.transpose(data, (1, 0, 2))  # 转为 (Y,X,Z)
    return data

def resize_nn(mask2d, target_hw=(1024, 1024)):
    """最近邻缩放 2D mask 到 target_hw=(H,W)"""
    h, w = mask2d.shape
    th, tw = target_hw
    # 用 OpenCV 的最近邻，注意宽高顺序
    return cv2.resize(mask2d, (tw, th), interpolation=cv2.INTER_NEAREST)

def find_boundary(mask_bin):
    """找前景边界（形态学膨胀-原图）"""
    dil = binary_dilation(mask_bin, iterations=1)
    boundary = np.logical_and(dil, np.logical_not(mask_bin))
    return boundary

def sample_points_from_coords(coords, k):
    """从 coords(N,2) 中随机采样 k 个（不足则全保留）"""
    if len(coords) <= k:
        return coords
    idx = np.random.choice(len(coords), size=k, replace=False)
    return coords[idx]

def coords_from_mask(mask_bin):
    """从二值mask中取 (y,x) 像素坐标 -> 再转为 (x,y)"""
    ys, xs = np.where(mask_bin)
    coords = np.stack([xs, ys], axis=1)  # (N,2), [x,y]
    return coords

def collect_2d_npy_pairs(img_dir, gt_dir):
    """收集 2D 图像/GT 对应文件列表，按自然排序一一对应"""
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.npy")), key=natural_sort_key)
    gt_files  = sorted(glob.glob(os.path.join(gt_dir,  "*.npy")), key=natural_sort_key)
    assert len(img_files) == len(gt_files), "图像和GT数量不一致！"
    # 简单的配对校验（文件名前缀应一致）
    pairs = []
    for img_path, gt_path in zip(img_files, gt_files):
        img_stem = Path(img_path).stem
        gt_stem  = Path(gt_path).stem
        assert img_stem == gt_stem, f"文件名不匹配: {img_stem} vs {gt_stem}"
        pairs.append((img_path, gt_path))
    return pairs

def infer_case_key_from_filename(fname):
    """
    从 2D 文件名推断它属于哪个 3D case（patientXXX_frameYY）。
    例如：MR_ACDCtr_patient001_frame01-000.npy -> patient001_frame01
    你若有其它命名规则，请在这里自定义解析。
    """
    base = Path(fname).stem  # MR_ACDCtr_patient001_frame01-000
    # 找到 patientXXX_frameYY
    m = re.search(r"(patient\d+_frame\d+)", base)
    if m:
        return m.group(1)
    # 兜底：从 scribble 列表里找能包含的那一个
    return None

def group_pairs_by_case(pairs):
    """
    把 2D (img,gt) 按 case 分组，并记录在该 case 下的“切片顺序”
    假设相同 case 的切片在 2D 导出时就是按 z 递增排列（-000, -001, ...）
    """
    groups = {}
    for img_path, gt_path in pairs:
        key = infer_case_key_from_filename(img_path)
        if key is None:
            # 若解析不到，放到一个特殊组
            key = "__UNKNOWN__"
        groups.setdefault(key, []).append((img_path, gt_path))
    # 组内再按自然顺序
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: natural_sort_key(x[0]))
    return groups

def load_scribble_3d_for_case(case_key):
    """
    根据 case_key 找到对应的 3D scribble nii.gz
    例如 case_key=patient001_frame01 -> 匹配 SCRIBBLE_DIR/patient001_frame01.nii.gz
    你如有其它前缀，可在这里补上。
    """
    cand = os.path.join(SCRIBBLE_DIR, f"{case_key}.nii.gz")
    if os.path.exists(cand):
        return load_nifti(cand)
    # 容错：尝试在目录下模糊匹配
    hits = glob.glob(os.path.join(SCRIBBLE_DIR, f"*{case_key}*.nii.gz"))
    if len(hits) == 1:
        return load_nifti(hits[0])
    elif len(hits) > 1:
        # 选最短路径名
        hits.sort(key=lambda p: len(Path(p).name))
        return load_nifti(hits[0])
    else:
        return None

def get_nonempty_slice_indices(scrib3d):
    """
    返回包含 {1,2,3} 任一的z切片索引（按升序）
    假设 scrib3d 形状为 (H,W,Z) 或 (Z,H,W)？——多数 ACDC 是 (H,W,Depth)
    这里假设 (H,W,Z)。如你的维度不同，改 axis 即可。
    """
    assert scrib3d.ndim == 3, "Scribble 不是3D"
    H, W, Z = scrib3d.shape
    idxs = []
    for z in range(Z):
        sl = scrib3d[:, :, z]
        if np.any((sl == 1) | (sl == 2) | (sl == 3)):
            idxs.append(z)
    return idxs

def sample_points_for_class(mask2d_for_class, max_points=64, boundary_ratio=0.5):
    """
    从某一类的 2D scribble 掩码中采点。
    - 先求连通域
    - 为每个连通域抽边界点 & 内部点
    """
    if not np.any(mask2d_for_class):
        return np.zeros((0, 2), dtype=np.int32)

    # 找连通域
    lab = cc_label(mask2d_for_class.astype(np.uint8), connectivity=1)
    coords_all = []

    num_cc = lab.max()
    # 总配额（简单按连通域均分）
    alloc_per_cc = max(1, max_points // max(1, num_cc))

    for cc in range(1, num_cc + 1):
        comp = (lab == cc)
        boundary = find_boundary(comp)
        interior = np.logical_and(comp, np.logical_not(boundary))

        bcoords = coords_from_mask(boundary)
        icoords = coords_from_mask(interior)
        if len(bcoords) == 0 and len(icoords) == 0:
            continue

        k_cc = alloc_per_cc
        k_b = int(round(k_cc * boundary_ratio))
        k_i = k_cc - k_b

        sb = sample_points_from_coords(bcoords, k_b) if len(bcoords) else np.zeros((0,2), dtype=np.int32)
        si = sample_points_from_coords(icoords, k_i) if len(icoords) else np.zeros((0,2), dtype=np.int32)

        coords_all.append(sb)
        coords_all.append(si)

    if len(coords_all) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    coords_all = np.vstack(coords_all)

    # 若总数超上限，再全局裁一遍
    if len(coords_all) > max_points:
        coords_all = sample_points_from_coords(coords_all, max_points)

    return coords_all.astype(np.int32)

def main():
    pairs = collect_2d_npy_pairs(IMG_DIR, GT_DIR)   # [(img, gt), ...]
    groups = group_pairs_by_case(pairs)              # case_key -> [(img, gt), ...]
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    n_written = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as fw:
        for case_key, slice_pairs in tqdm(groups.items(), desc="cases"):
            if case_key == "__UNKNOWN__":
                print("[警告] 有些 2D 文件无法解析出 patientXX_frameYY；将跳过scribble对齐。")
                continue

            scrib3d = load_scribble_3d_for_case(case_key)
            if scrib3d is None:
                print(f"[跳过] 找不到 scribble: {case_key}")
                continue

            # 假设 scrib3d: (H,W,Z)
            H, W, Z = scrib3d.shape

            # 把 non-empty 切片索引拿出来（这应与 2D 导出规则一致）
            nonempty_z = get_nonempty_slice_indices(scrib3d)

            # 断言与 2D 同数（若不同，给出提示但继续）
            if len(nonempty_z) != len(slice_pairs):
                print(f"[提示] case {case_key}: nonempty_scrib_slices={len(nonempty_z)}, 2D_pairs={len(slice_pairs)}。"
                      f"若不一致，建议检查2D导出规则是否一致。")

            # 逐切片对齐：按序匹配 nonempty_z 与 2D 对（假设导出顺序相同）
            for i, (img_path, gt_path) in enumerate(slice_pairs):
                if i >= len(nonempty_z):
                    # 2D 切片多于 scribble 非空切片，这种情况通常是导出规则不一致
                    break
                z = nonempty_z[i]
                scrib2d = scrib3d[:, :, z].astype(np.int16)

                # 缩放到 1024×1024（最近邻）
                if scrib2d.shape != RESIZE_TO:
                    scrib2d = resize_nn(scrib2d, RESIZE_TO)

                # 针对每个目标类别，各写一条样本行
                for cls_id in TARGET_CLASS_IDS:
                    mask_cls = (scrib2d == cls_id)
                    if not np.any(mask_cls):
                        continue

                    pts = sample_points_for_class(
                        mask2d_for_class=mask_cls,
                        max_points=MAX_POINTS_PER_CLASS_PER_SLICE,
                        boundary_ratio=BOUNDARY_RATIO
                    )
                    if len(pts) == 0:
                        continue

                    sample = {
                        "image_path": img_path,
                        "mask_path": gt_path,           # 监督用
                        "slice_id": Path(img_path).stem,
                        "case_key": case_key,
                        "class_id": int(cls_id),
                        "class_name": CLASS_ID_TO_NAME.get(cls_id, str(cls_id)),
                        "point_coords": pts.tolist(),   # [[x,y],...]
                        "point_labels": [1] * len(pts), # 全正样本
                        "scribble_values": [0,1,2,3,4], # 元数据
                        "image_shape": list(RESIZE_TO[::-1][::-1])  # 保留占位；可改为[H,W]=[1024,1024]
                    }
                    fw.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    n_written += 1

    print(f"完成！写入 {n_written} 条样本到 {OUT_JSONL}")

if __name__ == "__main__":
    main()
