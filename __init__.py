from .video_concat import SimpleVideoConcat

# 节点类映射字典
NODE_CLASS_MAPPINGS = {
    "SimpleVideoConcat": SimpleVideoConcat
}

# 节点显示名称映射字典
NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleVideoConcat": "Video Concat (Direct)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
