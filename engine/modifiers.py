"""复合与对抗扰动修改器。
实现：
1. 挖空与空间完整性识别（Hollow / Cutout / Negative Space）
2. 跨旗帜拼接与来源溯源（Splicing / Hybrid Composition）
"""
import random
from typing import Dict, Any, Tuple
from PIL import Image, ImageDraw

from .coords import to_rel_bbox_1000


def apply_cutout_or_hollowing(img: Image.Image, cutout_type: str = "circular") -> Tuple[Image.Image, Dict[str, Any]]:
    """在旗帜图案上施加挖空/破损/缺角扰动，测试模型对空间完整性的判断。"""
    w, h = img.size
    result_img = img.copy()
    draw = ImageDraw.Draw(result_img)
    void_color = (40, 40, 40, 255)  # 灰色镂空缺口底色

    meta = {
        "is_complete": False,
        "completeness_status": "不完整（存在挖空或缺口）",
        "cutout_type": cutout_type
    }

    if cutout_type == "circular_cutout":
        # 中心或随机偏位圆形挖空
        r = random.randint(int(h * 0.15), int(h * 0.25))
        cx = random.randint(int(w * 0.35), int(w * 0.65))
        cy = random.randint(int(h * 0.35), int(h * 0.65))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=void_color)
        meta["cutout_desc"] = f"在相对区域存在一个半径约为{round((r/h)*1000)}的圆形镂空孔洞"
        meta["cutout_bbox"] = to_rel_bbox_1000([cy - r, cx - r, cy + r, cx + r], w, h)

    elif cutout_type == "rectangular_cutout":
        # 矩形区域缺失
        rw = random.randint(int(w * 0.2), int(w * 0.35))
        rh = random.randint(int(h * 0.2), int(h * 0.35))
        x1 = random.randint(int(w * 0.2), int(w * 0.6))
        y1 = random.randint(int(h * 0.2), int(h * 0.6))
        draw.rectangle([x1, y1, x1 + rw, y1 + rh], fill=void_color)
        meta["cutout_desc"] = "在画面局部存在明显的矩形挖空缺失"
        meta["cutout_bbox"] = to_rel_bbox_1000([y1, x1, y1 + rh, x1 + rw], w, h)

    elif cutout_type == "missing_corner":
        # 缺角（如右上角或右下角三角缺失）
        corner = random.choice(["top_right", "bottom_right"])
        cut_sz = random.randint(int(h * 0.3), int(h * 0.45))
        if corner == "top_right":
            draw.polygon([(w - cut_sz, 0), (w, 0), (w, cut_sz)], fill=void_color)
            meta["cutout_desc"] = "右上角存在斜向三角形残缺撕裂"
            meta["cutout_bbox"] = to_rel_bbox_1000([0, w - cut_sz, cut_sz, w], w, h)
        else:
            draw.polygon([(w - cut_sz, h), (w, h), (w, h - cut_sz)], fill=void_color)
            meta["cutout_desc"] = "右下角存在斜向三角形残缺撕裂"
            meta["cutout_bbox"] = to_rel_bbox_1000([h - cut_sz, w - cut_sz, h, w], w, h)

    return result_img, meta


def splice_two_flags(img_a: Image.Image, meta_a: Dict[str, Any],
                     img_b: Image.Image, meta_b: Dict[str, Any],
                     split_mode: str = "vertical") -> Tuple[Image.Image, Dict[str, Any]]:
    """将两面不同的旗帜/图案拼接，测试模型对混合拼接来源的溯源推理。"""
    w, h = img_a.size
    composite = Image.new("RGBA", (w, h))

    if split_mode == "vertical":
        # 左半边 A，右半边 B
        half_w = w // 2
        crop_a = img_a.crop((0, 0, half_w, h))
        crop_b = img_b.crop((half_w, 0, w, h))
        composite.paste(crop_a, (0, 0))
        composite.paste(crop_b, (half_w, 0))
        # 绘制明显的中缝分割微线
        draw = ImageDraw.Draw(composite)
        draw.line([(half_w, 0), (half_w, h)], fill=(200, 200, 200), width=2)
        splice_desc = (f"该图像是左右垂直拼接图：左半边来源于[{meta_a.get('layout_desc', '图案A')}]，"
                       f"右半边来源于[{meta_b.get('layout_desc', '图案B')}]。")

    else:
        # 上半边 A，下半边 B
        half_h = h // 2
        crop_a = img_a.crop((0, 0, w, half_h))
        crop_b = img_b.crop((0, half_h, w, h))
        composite.paste(crop_a, (0, 0))
        composite.paste(crop_b, (0, half_h))
        draw = ImageDraw.Draw(composite)
        draw.line([(0, half_h), (w, half_h)], fill=(200, 200, 200), width=2)
        splice_desc = (f"该图像是上下水平拼接图：上半部分来源于[{meta_a.get('layout_desc', '图案A')}]，"
                       f"下半部分来源于[{meta_b.get('layout_desc', '图案B')}]。")

    splice_meta = {
        "is_spliced": True,
        "splice_mode": split_mode,
        "splice_desc": splice_desc,
        "source_a": meta_a,
        "source_b": meta_b
    }
    return composite, splice_meta
