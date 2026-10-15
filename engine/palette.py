"""标准旗帜色彩与纹章学调色板。
提供严格的 RGB 映射与自然语言色彩标签，防止大模型色彩幻觉。
"""

COLORS = {
    "red": {"name": "红色", "rgb": (204, 0, 0), "hex": "#CC0000"},
    "dark_red": {"name": "深红色", "rgb": (153, 0, 0), "hex": "#990000"},
    "crimson": {"name": "绯红色", "rgb": (220, 20, 60), "hex": "#DC143C"},
    "yellow": {"name": "黄色", "rgb": (255, 204, 0), "hex": "#FFCC00"},
    "gold": {"name": "金色", "rgb": (218, 165, 32), "hex": "#DAA520"},
    "blue": {"name": "蓝色", "rgb": (0, 51, 153), "hex": "#003399"},
    "sky_blue": {"name": "天蓝色", "rgb": (0, 153, 204), "hex": "#0099CC"},
    "navy_blue": {"name": "藏青色", "rgb": (0, 24, 72), "hex": "#001848"},
    "white": {"name": "白色", "rgb": (255, 255, 255), "hex": "#FFFFFF"},
    "black": {"name": "黑色", "rgb": (20, 20, 20), "hex": "#141414"},
    "green": {"name": "绿色", "rgb": (0, 136, 68), "hex": "#008844"},
    "dark_green": {"name": "深绿色", "rgb": (0, 85, 34), "hex": "#005522"},
    "orange": {"name": "橙色", "rgb": (255, 102, 0), "hex": "#FF6600"},
}

def get_color(key: str) -> dict:
    return COLORS.get(key, COLORS["white"])
