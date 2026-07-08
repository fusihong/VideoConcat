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
            # === 新增：文件级硬核拼接 (专为 VideoFromFile 等包含物理文件路径的对象设计) ===
            file1 = getattr(video1, "_VideoFromFile__file", None)
            file2 = getattr(video2, "_VideoFromFile__file", None)
            
            if file1 and file2 and os.path.exists(file1) and os.path.exists(file2):
                import subprocess
                import tempfile
                import uuid
                import copy
                
                # 在临时目录生成合并后的视频文件
                out_file = os.path.join(tempfile.gettempdir(), f"concat_{uuid.uuid4().hex}.mp4")
                
                # 使用 FFmpeg 强制拼接，并自动处理分辨率不一致的问题
                cmd = [
                    "ffmpeg", "-y",
                    "-i", file1,
                    "-i", file2,
                    "-filter_complex",
                    "[1:v][0:v]scale2ref[v1][v0]; [v0]setsar=1[v0_sar]; [v1]setsar=1[v1_sar]; [v0_sar][v1_sar]concat=n=2:v=1:a=0[outv]",
                    "-map", "[outv]",
                    out_file
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    # 克隆原来的对象，并将文件路径指向新生成的 10s 视频文件
                    new_video = copy.copy(video1)
                    setattr(new_video, "_VideoFromFile__file", out_file)
                    
                    # 修正其他可能限制长度的属性
                    if hasattr(new_video, "get_duration"):
                        try:
                            d1 = video1.get_duration()
                            d2 = video2.get_duration()
                            if hasattr(new_video, "duration"):
                                new_video.duration = d1 + d2
                        except:
                            pass
                            
                    return (new_video,)
                except Exception as ffmpeg_err:
                    print(f"FFmpeg fallback failed: {ffmpeg_err}")
            # =========================================================================

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
                            def pack_dict(x, k=k):
                                d = v.copy()
                                d[k] = x
                                if "audio" in d: del d["audio"]
                                if "frame_count" in d: d["frame_count"] = x.shape[0] if x.ndim==4 else x.shape[1]
                                return d
                            return v[k], pack_dict
                    for k, val in v.items():
                        if isinstance(val, torch.Tensor):
                            def pack_dict_any(x, k=k):
                                d = v.copy()
                                d[k] = x
                                return d
                            return val, pack_dict_any
                
                # 处理未知的自定义对象 (例如 BA Load Video 返回的 VideoFromFile)
                # 1. 尝试直接从它的属性 (vars) 中寻找 Tensor
                if hasattr(v, '__dict__'):
                    for k, val in vars(v).items():
                        if isinstance(val, torch.Tensor):
                            def pack_obj(x, key=k):
                                import copy
                                try:
                                    new_v = copy.copy(v)
                                except:
                                    new_v = v
                                setattr(new_v, key, x)
                                if hasattr(new_v, "audio"): setattr(new_v, "audio", None)
                                if hasattr(new_v, "frame_count"): setattr(new_v, "frame_count", x.shape[0] if x.ndim==4 else x.shape[1])
                                return new_v
                            return val, pack_obj
                
                # 2. 尝试调用可能的方法获取 Tensor
                for method in ["get_video", "get_tensor", "to_tensor", "get_frames"]:
                    if hasattr(v, method) and callable(getattr(v, method)):
                        try:
                            val = getattr(v, method)()
                            if isinstance(val, torch.Tensor):
                                # 如果是通过方法获取的，直接返回拼接后的 Tensor，看下游节点是否兼容
                                return val, lambda x: x
                        except:
                            pass

                return None, None

            v1, pack_v1 = extract(video1)
            v2, pack_v2 = extract(video2)

            if v1 is None or v2 is None:
                def get_debug_info(v):
                    try:
                        return f"\nvars: {vars(v)}\n\ndir: {dir(v)}"
                    except:
                        try:
                            return f"\ndir: {dir(v)}"
                        except:
                            return "无法获取属性"
                
                debug_v1 = get_debug_info(video1)
                raise ValueError(f"无法从输入中提取视频 Tensor。\n\nv1类型: {type(video1)}\nv1内部结构: {debug_v1}")

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
