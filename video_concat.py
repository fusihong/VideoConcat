import os
import torch
import torch.nn.functional as F
import traceback

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
        # 建立一个本地日志文件，方便在云房间中排查任何疑难杂症
        log_path = os.path.join(os.path.dirname(__file__), "debug.log")
        def log_debug(msg):
            print(msg)
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except:
                pass

        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("--- VideoConcat Run ---\n")
            
            log_debug(f"Input type v1: {type(video1)}")
            log_debug(f"Input type v2: {type(video2)}")

            # 辅助函数：提取 Tensor 并记录原始数据格式
            def extract_tensor(v):
                if isinstance(v, torch.Tensor):
                    return v, None
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    return torch.cat(v, dim=0), None
                elif isinstance(v, tuple) and len(v) > 0:
                    return extract_tensor(v[0])
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
                log_debug(f"Failed to extract tensor. v1 type: {type(v1)}, v2 type: {type(v2)}")
                return (video1,)

            log_debug(f"v1 shape: {v1.shape}, dtype: {v1.dtype}, device: {v1.device}")
            log_debug(f"v2 shape: {v2.shape}, dtype: {v2.dtype}, device: {v2.device}")

            # 确保处于同一个设备和数据类型（例如都放在 GPU/CPU）
            v2 = v2.to(v1.device, dtype=v1.dtype)

            # 自动推断应该在哪个维度进行拼接 (Time 维度) 以及 高宽 (H, W) 的维度
            concat_dim = 0
            h_dim, w_dim = 1, 2
            
            if v1.ndim == 5:
                if v1.shape[1] in [1, 3, 4]: # [B, C, T, H, W]
                    concat_dim = 2
                    h_dim, w_dim = 3, 4
                elif v1.shape[2] in [1, 3, 4]: # [B, T, C, H, W]
                    concat_dim = 1
                    h_dim, w_dim = 3, 4
                elif v1.shape[4] in [1, 3, 4]: # [B, T, H, W, C]
                    concat_dim = 1
                    h_dim, w_dim = 2, 3
                else:
                    concat_dim = 1 # 默认猜测 1 是时间维度
                    h_dim, w_dim = 3, 4
            elif v1.ndim == 4:
                if v1.shape[3] in [1, 3, 4]: # [T, H, W, C] - 这是 ComfyUI 最标准的图像批次/视频格式
                    concat_dim = 0
                    h_dim, w_dim = 1, 2
                elif v1.shape[1] in [1, 3, 4]: # [B, C, H, W] - PyTorch 原生格式
                    concat_dim = 0
                    h_dim, w_dim = 2, 3
                else:
                    concat_dim = 0
                    h_dim, w_dim = 1, 2

            log_debug(f"Determined concat_dim: {concat_dim}, h_dim: {h_dim}, w_dim: {w_dim}")

            # 检查尺寸是否一致，如果不一致则将 v2 缩放到 v1 的尺寸
            if v1.shape[h_dim] != v2.shape[h_dim] or v1.shape[w_dim] != v2.shape[w_dim]:
                log_debug(f"Resizing v2 from {v2.shape} to match v1 {v1.shape}")
                target_h, target_w = v1.shape[h_dim], v1.shape[w_dim]
                
                if v2.ndim == 4 and v2.shape[3] in [1, 3, 4]: 
                    v2_p = v2.permute(0, 3, 1, 2) 
                    v2_p = F.interpolate(v2_p, size=(target_h, target_w), mode='bilinear')
                    v2 = v2_p.permute(0, 2, 3, 1) 
                elif v2.ndim == 5 and v2.shape[1] in [1, 3, 4]: 
                    b, c, t, h, w = v2.shape
                    v2_p = v2.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
                    v2_p = F.interpolate(v2_p, size=(target_h, target_w), mode='bilinear')
                    v2 = v2_p.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4)
                elif v2.ndim == 5 and v2.shape[2] in [1, 3, 4]: 
                    b, t, c, h, w = v2.shape
                    v2_p = v2.reshape(b*t, c, h, w)
                    v2_p = F.interpolate(v2_p, size=(target_h, target_w), mode='bilinear')
                    v2 = v2_p.reshape(b, t, c, target_h, target_w)
                elif v2.ndim == 5 and v2.shape[4] in [1, 3, 4]: 
                    b, t, h, w, c = v2.shape
                    v2_p = v2.view(b*t, h, w, c).permute(0, 3, 1, 2) 
                    v2_p = F.interpolate(v2_p, size=(target_h, target_w), mode='bilinear')
                    v2 = v2_p.permute(0, 2, 3, 1).view(b, t, target_h, target_w, c)
                elif v2.ndim == 4 and v2.shape[1] in [1, 3, 4]: 
                    v2 = F.interpolate(v2, size=(target_h, target_w), mode='bilinear')
                else:
                    log_debug("Unsupported shape for resizing, skipping resize")

            # 真正进行视频拼接！
            log_debug(f"Concatenating on dim {concat_dim}...")
            out_tensor = torch.cat((v1, v2), dim=concat_dim)
            log_debug(f"Output tensor shape: {out_tensor.shape}")

            # 还原输出格式（如果之前输入的是字典，原样包回去，以适配下游 BA 节点）
            if isinstance(video1, dict) and k1 is not None:
                out_dict = video1.copy()
                out_dict[k1] = out_tensor
                log_debug("Returning as dict.")
                return (out_dict,)

            log_debug("Returning as tensor.")
            return (out_tensor,)

        except Exception as e:
            # 防止流程彻底崩溃，如果拼接失败打印日志，返回原视频1
            log_debug(f"Exception occurred: {e}")
            log_debug(traceback.format_exc())
            return (video1,)