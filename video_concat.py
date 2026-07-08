import os
import torch
import torch.nn.functional as F

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
            # 1. 递归提取 Tensor，并生成“还原函数”（确保下游节点收到的是它认识的格式：比如 Tuple 或 Dict）
            def extract(v):
                if isinstance(v, torch.Tensor):
                    return v, lambda x: x
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    return torch.cat(v, dim=0), lambda x: x 
                elif isinstance(v, tuple) and len(v) > 0:
                    t, _ = extract(v[0])
                    if t is not None:
                        def pack_tuple(x):
                            l = list(v)
                            l[0] = x
                            return tuple(l)
                        return t, pack_tuple
                elif isinstance(v, dict):
                    for k in ["samples", "video", "image", "images", "frames"]:
                        if k in v and isinstance(v[k], torch.Tensor):
                            def pack_dict(x):
                                d = v.copy()
                                d[k] = x
                                if "audio" in d: del d["audio"]
                                if "frame_count" in d: d["frame_count"] = x.shape[0] if x.ndim==4 else x.shape[1]
                                return d
                            return v[k], pack_dict
                    for k, val in v.items():
                        if isinstance(val, torch.Tensor):
                            def pack_dict_any(x):
                                d = v.copy()
                                d[k] = x
                                return d
                            return val, pack_dict_any
                return None, None

            v1, pack_v1 = extract(video1)
            v2, pack_v2 = extract(video2)

            if v1 is None or v2 is None:
                raise ValueError(f"无法从输入中提取视频 Tensor。v1类型:{type(video1)}, v2类型:{type(video2)}")

            # 2. 对齐设备和数据类型
            v2 = v2.to(v1.device, dtype=v1.dtype)

            # 3. 维度对齐与推断
            if v1.ndim == 3: v1 = v1.unsqueeze(0)
            if v2.ndim == 3: v2 = v2.unsqueeze(0)

            concat_dim = 0
            h_dim, w_dim = 1, 2
            
            if v1.ndim == 4:
                if v1.shape[3] in [1, 3, 4]: # [T, H, W, C]
                    concat_dim = 0; h_dim, w_dim = 1, 2
                elif v1.shape[1] in [1, 3, 4]: # [B, C, H, W]
                    concat_dim = 0; h_dim, w_dim = 2, 3
            elif v1.ndim == 5: # [B, T, C, H, W] 等
                concat_dim = 1; h_dim, w_dim = 3, 4

            # 4. 高宽缩放 (转换到 float32 以防 float16/uint8 在 interpolate 时报错)
            if v1.shape[h_dim] != v2.shape[h_dim] or v1.shape[w_dim] != v2.shape[w_dim]:
                target_h, target_w = v1.shape[h_dim], v1.shape[w_dim]
                v2_float = v2.to(torch.float32)
                
                if v2_float.ndim == 4 and v2_float.shape[3] in [1, 3, 4]: 
                    v2_p = v2_float.permute(0, 3, 1, 2)
                    v2_p = F.interpolate(v2_p, size=(target_h, target_w), mode='bilinear')
                    v2_float = v2_p.permute(0, 2, 3, 1)
                elif v2_float.ndim == 4 and v2_float.shape[1] in [1, 3, 4]:
                    v2_float = F.interpolate(v2_float, size=(target_h, target_w), mode='bilinear')
                
                v2 = v2_float.to(v1.dtype)

            # 5. 通道数对齐 (防止一个带 Alpha 通道，一个不带导致 cat 报错)
            if v1.shape[-1] != v2.shape[-1] and v1.ndim == 4 and v1.shape[3] in [1, 3, 4]:
                min_c = min(v1.shape[-1], v2.shape[-1])
                v1 = v1[..., :min_c]
                v2 = v2[..., :min_c]

            # 6. 拼接
            out_tensor = torch.cat((v1, v2), dim=concat_dim)

            # 7. 完美还原包装
            out_final = pack_v1(out_tensor)
            return (out_final,)

        except Exception as e:
            # 【取消静默报错】如果出错了，直接抛出异常让节点变紫爆红！
            import traceback
            error_msg = f"拼接失败 (VideoConcat Error):\n{str(e)}\n\n"
            error_msg += traceback.format_exc()
            raise RuntimeError(error_msg)
