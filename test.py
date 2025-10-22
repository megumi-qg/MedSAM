# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from skimage import transform
from segment_anything import sam_model_registry
import torch.nn.functional as F
import json
from tqdm import tqdm

# 指标计算
def dice_score(pred, gt):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    intersection = np.logical_and(pred, gt).sum()
    return 2. * intersection / (pred.sum() + gt.sum() + 1e-8)

def iou_score(pred, gt):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)

def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None, boxes=box_torch, masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(H, W), mode="bilinear", align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().detach().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

def main():
    # 路径设置
    test_root = "data/npy/MR_heart_test"
    img_dir = os.path.join(test_root, "imgs")
    gt_dir = os.path.join(test_root, "gts")
    box_dir = os.path.join(test_root, "boxes")
    model_type = "vit_b"
    # 需要修改
    checkpoint = "work_dir/SAM/sam_vit_b_01ec64.pth"
    device = "cuda:0"

    # 加载模型
    # 暂且先加载官方权重
    medsam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    # 加载保存的checkpoint
    # 需要修改
    checkpoint = torch.load("work_dir/MedSAM-ViT-B-ACDC-20250723-2019/medsam_model_best.pth")
    medsam_model.load_state_dict(checkpoint['model'])

    # 统计指标
    all_dice = []
    all_iou = []
    class_dice = {}  # {label: [dice1, dice2, ...]}
    class_iou = {}

    img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
    for img_name in tqdm(img_names[:10]):
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name)
        box_path = os.path.join(box_dir, img_name.replace('.npy', '.json'))

        # 读取图片和标签
        img_1024 = np.load(img_path)  # (1024,1024,3), [0,1]
        gt = np.load(gt_path)         # (1024,1024), 多类标签
        H, W = gt.shape

        # 读取检测框
        with open(box_path, 'r') as f:
            box_info = json.load(f)
        bboxes = box_info['bboxes']

        # 预处理图片
        img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3,1024,1024)
        img_1024_tensor = torch.tensor(img_1024).float().unsqueeze(0).to(device)

        # 提取特征
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)

        # 针对每个类别分别推理
        for bbox_item in bboxes:
            label = bbox_item['label']
            box_1024 = np.array([bbox_item['bbox']])  # shape (1,4)

            # 推理
            pred_mask = medsam_inference(medsam_model, image_embedding, box_1024, H, W)

            # 取该类别的gt
            mapped_label = label_map(label, gt)
            if mapped_label is None:
                continue    
            gt_mask = (gt == label_map(label, gt)).astype(np.uint8)  # label_map见下

            # 注释：保存预测mask
            save_path = test_root+"/pred_masks/sam/"+img_name.split(".")[0]+"_"+str(mapped_label)+".npy"
            np.save(save_path, pred_mask.astype(np.uint8))

            # 计算指标
            dice = dice_score(pred_mask, gt_mask)
            iou = iou_score(pred_mask, gt_mask)
            all_dice.append(dice)
            all_iou.append(iou)
            class_dice.setdefault(label, []).append(dice)
            class_iou.setdefault(label, []).append(iou)

    # 输出结果
    print("整体Dice: {:.4f}, IoU: {:.4f}".format(np.mean(all_dice), np.mean(all_iou)))
    for label in class_dice:
        print("类别 {}: Dice={:.4f}, IoU={:.4f}".format(
            label, np.mean(class_dice[label]), np.mean(class_iou[label])
        ))

# label_map函数：将label字符串映射为gt中的数字标签
def label_map(label_str, gt):
    label_dict = {
        "RV": 1,
        "Myo": 2,
        "LV": 3,
    }
    unique_labels = sorted([x for x in np.unique(gt) if x != 0])

    if set(label_dict.values()) == set(unique_labels):
        return label_dict[label_str]
    else:
        label_list = ["RV", "Myo", "LV"]
        idx = label_list.index(label_str)
        if idx < len(unique_labels):
            return unique_labels[idx]
        else:
            return None  # 当前gt中没有该类别


if __name__ == "__main__":
    main()