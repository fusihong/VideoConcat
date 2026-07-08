import torch
import torch.nn.functional as F

class SimpleVideoConcat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video1": ("IMAGE",),
                "video2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video",)
    FUNCTION = "concat"
    CATEGORY = "Video/Utils"

    def concat(self, video1, video2):
        # ComfyUI 中的视频（图像批次）形状通常为 (B, H, W, C)
        # B 是帧数，H 是高度，W 是宽度，C 是通道数
        
        # 检查尺寸是否一致，如果不一致则将 video2 缩放到 video1 的尺寸
        if video1.shape[1:3] != video2.shape[1:3]:
            print(f"Video sizes do not match. Resizing video2 {video2.shape} to match video1 {video1.shape}")
            # 调整维度顺序为 (B, C, H, W) 以便使用 interpolate
            v2_permuted = video2.permute(0, 3, 1, 2)
            # 进行双线性插值缩放
            v2_resized = F.interpolate(v2_permuted, size=(video1.shape[1], video1.shape[2]), mode='bilinear')
            # 恢复维度顺序为 (B, H, W, C)
            video2 = v2_resized.permute(0, 2, 3, 1)
        
        # 直接在第 0 维度（帧数/批次维度）进行拼接
        out = torch.cat((video1, video2), dim=0)
        return (out,)
