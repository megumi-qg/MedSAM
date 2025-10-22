import os
import numpy as np
import nibabel as nib
import cv2
import json
from tqdm import tqdm

def extract_bboxes_from_mask(mask_2d, class_id):
    binary_mask = (mask_2d == class_id).astype(np.uint8)
    if np.sum(binary_mask) == 0:
        return []
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append([int(x), int(y), int(x + w), int(y + h)])
    return bboxes

def process_one_nii_file(npy_path, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    # seg_nii = nib.load(nii_path)
    # seg_data = seg_nii.get_fdata().astype(np.uint8)  # shape: (H, W, T)
    seg_data = np.load(npy_path) # shape:(H,W)

    mask_2d = seg_data
    bboxes_all = []

    for class_id, label_name in zip([1, 2, 3], ["RV", "Myo", "LV"]):
        bboxes = extract_bboxes_from_mask(mask_2d, class_id)
        for box in bboxes:
            bboxes_all.append({
                "bbox": box,
                "label": label_name
            })

    if len(bboxes_all) == 0:
        return

    output = {
        "slice_name": f"{prefix}",
        "bboxes": bboxes_all
    }

    json_path = os.path.join(output_dir, f"{prefix}.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved: {json_path}")

def process_all_gt_masks(gt_dir, output_dir):
    npy_files = [f for f in os.listdir(gt_dir) if f.endswith('.npy')]
    for npy_file in tqdm(npy_files):
        prefix = npy_file.replace('.npy', '')  # e.g. MR_heart_test_patient101_frame01-000
        npy_path = os.path.join(gt_dir, npy_file)
        process_one_nii_file(npy_path, output_dir, prefix)

# 调用入口
gt_mask_dir = 'data/npy/MR_heart_test/gts'  # 所有 _gt.nii.gz 文件的文件夹
json_output_dir = 'data/npy/MR_heart_test/boxes'
process_all_gt_masks(gt_mask_dir, json_output_dir)
