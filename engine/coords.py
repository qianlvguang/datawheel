"""相对坐标转换工具模块。
统一规范：
- 以图像左上角作为坐标原点 (0, 0)
- 横坐标 (x) 与纵坐标 (y) 归一化映射至 [0, 1000] 整数相对区间
- 格式统一为 [ymin, xmin, ymax, xmax]，与主流 VLM (如 Qwen-VL) 视觉定位协议完全对齐。
"""
from typing import List, Tuple, Union


def to_rel_coord_1000(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    """将绝对像素坐标 (x, y) 转换为 [0, 1000] 相对坐标"""
    rx = max(0, min(1000, round((x / width) * 1000)))
    ry = max(0, min(1000, round((y / height) * 1000)))
    return rx, ry


def to_rel_bbox_1000(bbox_pixels: Union[List[int], Tuple[int, int, int, int]],
                     width: int, height: int) -> List[int]:
    """将绝对像素边界框 [y1, x1, y2, x2] 转换为 [0, 1000] 相对边界框 [ymin, xmin, ymax, xmax]"""
    y1, x1, y2, x2 = bbox_pixels
    ymin = max(0, min(1000, round((min(y1, y2) / height) * 1000)))
    xmin = max(0, min(1000, round((min(x1, x2) / width) * 1000)))
    ymax = max(0, min(1000, round((max(y1, y2) / height) * 1000)))
    xmax = max(0, min(1000, round((max(x1, x2) / width) * 1000)))
    return [ymin, xmin, ymax, xmax]


def mask_to_rel_bbox_1000(mask_np, width: int = None, height: int = None, threshold: int = 10):
    """从变换后的 2D 布尔/Alpha Mask 中提取紧凑的外接边界框 [ymin, xmin, ymax, xmax] (0~1000)。
    避免直接使用初始粘贴矩形经变换后的理论坐标，保证与最终实际可见像素严格对应。
    """
    import numpy as np
    if not isinstance(mask_np, np.ndarray):
        mask_np = np.array(mask_np)

    if mask_np.ndim == 3:
        if mask_np.shape[2] == 4:
            mask_np = mask_np[:, :, 3]  # Alpha 通道
        else:
            mask_np = np.mean(mask_np, axis=2)

    h, w = mask_np.shape[:2]
    total_w = width if width is not None else w
    total_h = height if height is not None else h

    active_indices = np.where(mask_np > threshold)
    if len(active_indices[0]) == 0:
        return None

    y_min_px = int(np.min(active_indices[0]))
    y_max_px = int(np.max(active_indices[0]))
    x_min_px = int(np.min(active_indices[1]))
    x_max_px = int(np.max(active_indices[1]))

    return to_rel_bbox_1000([y_min_px, x_min_px, y_max_px, x_max_px], total_w, total_h)


def bbox_iou_1000(box1: List[int], box2: List[int]) -> float:
    """计算两个 [ymin, xmin, ymax, xmax] 相对坐标框的 IoU。"""
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    inter_ymin = max(y1_min, y2_min)
    inter_xmin = max(x1_min, x2_min)
    inter_ymax = min(y1_max, y2_max)
    inter_xmax = min(x1_max, x2_max)

    inter_h = max(0, inter_ymax - inter_ymin)
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_area = inter_h * inter_w

    area1 = max(0, y1_max - y1_min) * max(0, x1_max - x1_min)
    area2 = max(0, y2_max - y2_min) * max(0, x2_max - x2_min)

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)
