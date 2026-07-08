import torch
import torch.nn.functional as F

# 这是一个 ComfyUI 的小技巧，用来创建一个“万能类型”，可以连接任何输出节点（无论是 IMAGE 还是 VIDEO 类型）
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

class SimpleVideoConcat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video1": (any_type,),
                "video2": (any_type,),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("video",)
    FUNCTION = "concat"
    CATEGORY = "Video/Utils"

    def concat(self, video1, video2):
        try:
            # 辅助函数：提取 Tensor 并记录原始数据格式
            def extract_tensor(v):
                if isinstance(v, torch.Tensor):
                    return v, None
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    return torch.cat(v, dim=0), None
                elif isinstance(v, dict):
                    # 尝试从字典中提取常见名称的 Tensor
                    for k in ["samples", "video", "image", "images", "frames"]:
                        if k in v and isinstance(v[k], torch.Tensor):
                            return v[k], k
                    # 实在找不到，取第一个 Tensor
                    for k, val in v.items():
                        if isinstance(val, torch.Tensor):
                            return val, k
                return None, None

            v1, k1 = extract_tensor(video1)
            v2, k2 = extract_tensor(video2)

            # 如果传进来的不是 Tensor，打印报错信息原路返回
            if v1 is None or v2 is None:
                print(f"VideoConcat Error: Could not extract tensor. Type v1: {type(video1)}, Type v2: {type(video2)}")
                return (video1,)

            # 确保处于同一个设备和数据类型（例如都放在 GPU/CPU）
            v2 = v2.to(v1.device, dtype=v1.dtype)

            # 补充缺失的 Batch 维度（例如有些节点传过来的是 [H, W, C]）
            if v1.ndim == 3:
                v1 = v1.unsqueeze(0)
            if v2.ndim == 3:
                v2 = v2.unsqueeze(0)

            # ComfyUI 中的视频形状通常为 (B, H, W, C)
            # 检查尺寸是否一致，如果不一致则将 v2 缩放到 v1 的尺寸
            if v1.shape[1:3] != v2.shape[1:3]:
                print(f"VideoConcat: Resizing v2 {v2.shape} to match v1 {v1.shape}")
                v2_permuted = v2.permute(0, 3, 1, 2)
                v2_resized = F.interpolate(v2_permuted, size=(v1.shape[1], v1.shape[2]), mode='bilinear')
                v2 = v2_resized.permute(0, 2, 3, 1)
            
            # 直接在第 0 维度（帧数维度）进行拼接
            out_tensor = torch.cat((v1, v2), dim=0)

            # 还原输出格式（如果之前输入的是字典，原样包回去，以适配下游 BA 节点）
            if isinstance(video1, dict) and k1 is not None:
                out_dict = video1.copy()
                out_dict[k1] = out_tensor
                return (out_dict,)

            return (out_tensor,)
        except Exception as e:
            # 防止流程彻底崩溃，如果拼接失败打印日志，返回原视频1
            print(f"VideoConcat Exception: {e}")
            import traceback
            traceback.print_exc()
            return (video1,)
