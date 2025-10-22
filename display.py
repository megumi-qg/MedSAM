import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize
import os

# 设置路径
nii_path = "data/ACDC/test/images/patient101_frame01.nii.gz"
# 注释：gt路径
# gt_path = "data/ACDC/test/labels/patient101_frame01_gt.nii.gz"
# 注释：mask路径
mask_paths = [
    "data/npy/MR_heart_test/pred_masks/sam/MR_heart_test_patient101_frame01-001_1.npy",
    "data/npy/MR_heart_test/pred_masks/sam/MR_heart_test_patient101_frame01-001_2.npy",
    "data/npy/MR_heart_test/pred_masks/sam/MR_heart_test_patient101_frame01-001_3.npy"
]
slice_index = 1  # 你要查看的切片索引

# 加载nii.gz图像
nii_img = nib.load(nii_path)
nii_data = nii_img.get_fdata()  # shape: (H, W, D)
slice_img = nii_data[:, :, slice_index]  # shape: (H, W)
# 注释：加载gt的nii.gz图像
# gt_nii_img = nib.load(gt_path)
# gt_nii_data = gt_nii_img.get_fdata()
# gt_slice_img = gt_nii_data[:, :, slice_index]

# 获取原始尺寸
H, W = slice_img.shape

# 注释：加载并resize mask
resized_masks = []
for path in mask_paths:
    mask = np.load(path)
    mask = np.transpose(mask)
    mask_resized = resize(mask, (H, W), order=0, preserve_range=True, anti_aliasing=False)
    mask_resized = (mask_resized > 0.5).astype(np.uint8)
    resized_masks.append(mask_resized)

# 可视化叠加
plt.figure(figsize=(8, 8))
plt.imshow(slice_img, cmap='gray')


# 叠加预测掩膜，使用与 GT 相同的 colormap 和透明度
pred_colors = {
    1: ('Reds', 0.4),
    2: ('Greens', 0.4),
    3: ('Blues', 0.4),
}

for label_idx, mask in enumerate(resized_masks, start=1):  # label_idx 从 1 开始
    cmap_name, alpha_val = pred_colors[label_idx]
    masked = np.ma.masked_where(mask == 0, mask)
    plt.imshow(masked, cmap=cmap_name, alpha=alpha_val, vmin=0, vmax=1)


# 注释：叠加gt可视化
# colors = {
#     1: ('Reds', 0.4),
#     2: ('Greens', 0.4),
#     3: ('Blues', 0.4),
# }
# for label_val, (cmap_name, alpha_val) in colors.items():
#     mask = (gt_slice_img == label_val).astype(np.uint8)
#     if np.any(mask):
#         masked = np.ma.masked_where(mask == 0, mask)
#         plt.imshow(masked, cmap=cmap_name, alpha=alpha_val, vmin=0, vmax=1)


plt.axis('off')
plt.title(f"Slice {slice_index} with 3 Masks")
plt.tight_layout()

output_path = f"slice_{slice_index}_with_masks.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")

