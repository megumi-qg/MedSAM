import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 模型定义 ==========
class MedSAM(nn.Module):
    def __init__(self,image_encoder,mask_decoder,prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder and image encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, points, labels):
        """
        image: (B,3,H,W) float32
        points: (B,K,2) float32, 像素坐标 [x,y]，与输入分辨率一致(1024)
        labels: (B,K)   int64, 1 表示正点（你现在全 1）
        """
        image_embedding = self.image_encoder(image)  # (B,256,64,64)
        with torch.no_grad():
            # SAM 的 prompt_encoder 期望 points 为像素坐标；当前图像已是 1024×1024
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(points, labels),
                boxes=None,
                masks=None,
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
