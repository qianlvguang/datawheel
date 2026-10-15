"""多态参考素材库治理与管理模块 (Reference Bank)。
核心职责：
1. 维护标准参考素材 (Canonical Assets)、官方矢量图与标准渲染图。
2. 维护细粒度部件层语义分解 (Hierarchical Components)：区分“区分性特征”与“通用非区分性色块/几何形”。
3. 建立细粒度混淆干扰对 (Confusable Hard-Negative Pairs)，用于生成高质量困难负样本。
4. 区分“色相不变性”与“严格色相拓扑约束”。
"""
import io
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import resvg_py

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "official_flags"


REFERENCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "cn": {
        "id": "cn",
        "name_zh": "中华人民共和国国旗 (五星红旗)",
        "name_en": "Flag of China",
        "category": "flag",
        "color_topology_strict": True,
        "canonical_colors": ["red", "yellow"],
        "confusable_distractors": ["vn", "so"],
        "components": [
            {
                "name": "主大五角星 (Major 5-Point Star)",
                "relative_pos": "canton_major",
                "is_discriminative": True,
                "notes": "位于左上方，尖角向上"
            },
            {
                "name": "四颗拱卫小五角星 (Four Minor Stars)",
                "relative_pos": "canton_minor_arc",
                "is_discriminative": True,
                "notes": "围绕大星右侧呈半环形排列，各有一个角尖正对大星中心"
            },
            {
                "name": "红色底板 (Red Field)",
                "relative_pos": "background",
                "is_discriminative": False,
                "notes": "纯红底色，为通用背景，单独出现不足以确认身份"
            }
        ],
        "distractor_reason": {
            "vn": "越南国旗同为纯红底色配黄色五角星，但仅有正中央单颗大星，缺少四星拱卫结构。",
            "so": "索马里国旗同为单星，但底色为浅蓝，星为白色。"
        }
    },
    "vn": {
        "id": "vn",
        "name_zh": "越南社会主义共和国国旗 (金星红旗)",
        "name_en": "Flag of Vietnam",
        "category": "flag",
        "color_topology_strict": True,
        "canonical_colors": ["red", "yellow"],
        "confusable_distractors": ["cn", "so"],
        "components": [
            {
                "name": "中央金黄色五角星 (Central Yellow Star)",
                "relative_pos": "center",
                "is_discriminative": True,
                "notes": "位于旗帜正几何中心"
            },
            {
                "name": "红色底板 (Red Field)",
                "relative_pos": "background",
                "is_discriminative": False,
                "notes": "通用红色底板"
            }
        ],
        "distractor_reason": {
            "cn": "中国国旗五角星位于左上角且为一主四辅组合，越南仅正中一颗。"
        }
    },
    "fr": {
        "id": "fr",
        "name_zh": "法兰西共和国国旗 (蓝白红三色旗)",
        "name_en": "Flag of France",
        "category": "flag",
        "color_topology_strict": True,
        "canonical_colors": ["blue", "white", "red"],
        "confusable_distractors": ["it", "de", "pl"],
        "components": [
            {
                "name": "垂直三色条纹拓扑 (Vertical Tricolor Bands)",
                "relative_pos": "vertical_stripes_left_to_right",
                "is_discriminative": True,
                "notes": "自左向右严格为：蓝、白、红"
            }
        ],
        "distractor_reason": {
            "it": "意大利国旗三条纹布局完全一致，但左侧第一条为纯绿，而非蓝色。",
            "pl": "波兰为水平双色 (白红)，法兰西为垂直三色。"
        }
    },
    "it": {
        "id": "it",
        "name_zh": "意大利共和国国旗 (绿白红三色旗)",
        "name_en": "Flag of Italy",
        "category": "flag",
        "color_topology_strict": True,
        "canonical_colors": ["green", "white", "red"],
        "confusable_distractors": ["fr"],
        "components": [
            {
                "name": "垂直三色条纹拓扑 (Vertical Tricolor Bands)",
                "relative_pos": "vertical_stripes_left_to_right",
                "is_discriminative": True,
                "notes": "自左向右严格为：绿、白、红"
            }
        ],
        "distractor_reason": {
            "fr": "法国国旗第一条为蓝，意大利为绿。仅凭几何轮廓无法区分，必须依靠色相拓扑。"
        }
    },
    "de": {
        "id": "de",
        "name_zh": "德意志联邦共和国国旗 (黑红金三色旗)",
        "name_en": "Flag of Germany",
        "category": "flag",
        "color_topology_strict": True,
        "canonical_colors": ["black", "red", "gold"],
        "confusable_distractors": ["pl", "ua"],
        "components": [
            {
                "name": "水平三色条纹拓扑 (Horizontal Tricolor Bands)",
                "relative_pos": "horizontal_stripes_top_to_bottom",
                "is_discriminative": True,
                "notes": "自上向下严格为：黑、红、金"
            }
        ],
        "distractor_reason": {
            "pl": "波兰为水平白红双色，德国为黑红金三色。",
            "ua": "乌克兰为水平蓝黄双色。"
        }
    },
    "es": {
        "id": "es",
        "name_zh": "西班牙王国国旗 (带国徽官方版)",
        "name_en": "Flag of Spain",
        "category": "flag",
        "color_topology_strict": True,
        "canonical_colors": ["red", "yellow"],
        "confusable_distractors": ["mx", "de"],
        "components": [
            {
                "name": "水平红黄红三条纹 (Horizontal Red-Yellow-Red)",
                "relative_pos": "horizontal_stripes",
                "is_discriminative": False,
                "notes": "黄色中条宽度为上下红条的两倍"
            },
            {
                "name": "西班牙官方国徽 (Coat of Arms of Spain)",
                "relative_pos": "left_center_yellow_band",
                "is_discriminative": True,
                "notes": "包含四分盾徽、石榴花、波旁百合盾心、大力神海格力斯双柱与王冠"
            },
            {
                "name": "PLVS VLTRA 铭文飘带 (Plus Ultra Inscription)",
                "relative_pos": "hercules_pillars_ribbon",
                "is_discriminative": True,
                "notes": "海格力斯双柱两侧环绕的拉丁文箴言"
            }
        ],
        "distractor_reason": {
            "mx": "墨西哥国旗亦带复杂国徽(鹰蛇仙人掌)，但底条纹为绿白红竖条。",
            "civil_spain": "若缺少国徽，则为西班牙民用旗/基础条纹，不足以确认官方带徽规格。"
        }
    },
    "br": {
        "id": "br",
        "name_zh": "巴西联邦共和国国旗",
        "name_en": "Flag of Brazil",
        "category": "flag",
        "color_topology_strict": True,
        "canonical_colors": ["green", "yellow", "blue", "white"],
        "confusable_distractors": ["es"],
        "components": [
            {
                "name": "绿底黄色菱形 (Yellow Rhombus on Green Field)",
                "relative_pos": "center_rhombus",
                "is_discriminative": True,
                "notes": "大面积绿色底版内嵌黄色正菱形"
            },
            {
                "name": "天球仪与南十字星座 (Celestial Globe & Constellations)",
                "relative_pos": "center_circle",
                "is_discriminative": True,
                "notes": "菱形中央的深蓝天球仪，包含27颗白色五角星"
            },
            {
                "name": "ORDEM E PROGRESSO 铭文弧带 (Inscription Ribbon)",
                "relative_pos": "globe_white_band",
                "is_discriminative": True,
                "notes": "贯穿天球仪的白色弧形飘带，绿字铭文"
            }
        ],
        "distractor_reason": {
            "es": "巴西为绿底黄菱形天球仪，构图独一无二。"
        }
    },
    "eu": {
        "id": "eu",
        "name_zh": "欧洲联盟盟旗 (十二星环旗)",
        "name_en": "Flag of the European Union",
        "category": "flag",
        "color_topology_strict": True,
        "canonical_colors": ["blue", "yellow"],
        "confusable_distractors": ["so", "cn"],
        "components": [
            {
                "name": "十二颗五角金星圆环 (Circle of 12 Golden Stars)",
                "relative_pos": "center_circle",
                "is_discriminative": True,
                "notes": "12颗正立五角金星均布成正圆，象征完美与统一"
            },
            {
                "name": "深蓝色底板 (Azure Blue Field)",
                "relative_pos": "background",
                "is_discriminative": False,
                "notes": "通用蓝色底板"
            }
        ],
        "distractor_reason": {
            "so": "索马里同为蓝底五角星，但仅有中央单颗白星；欧盟为十二星环形金星。"
        }
    },
    "so": {
        "id": "so",
        "name_zh": "索马里联邦共和国国旗 (单星天蓝旗)",
        "name_en": "Flag of Somalia",
        "category": "flag",
        "color_topology_strict": True,
        "canonical_colors": ["blue", "white"],
        "confusable_distractors": ["vn", "eu"],
        "components": [
            {
                "name": "中央白色五角星 (Central White Star)",
                "relative_pos": "center",
                "is_discriminative": True,
                "notes": "正中正立白色大五角星"
            },
            {
                "name": "天蓝色底板 (Azure Blue Field)",
                "relative_pos": "background",
                "is_discriminative": False,
                "notes": "纯浅蓝色底色"
            }
        ],
        "distractor_reason": {
            "vn": "越南为红底金星，索马里为天蓝底白星。",
            "eu": "欧盟为十二颗金星环，索马里仅中央单颗白星。"
        }
    },
    "pl": {
        "id": "pl",
        "name_zh": "波兰共和国国旗 (白红双色旗)",
        "name_en": "Flag of Poland",
        "category": "flag",
        "color_topology_strict": True,
        "canonical_colors": ["white", "red"],
        "confusable_distractors": ["ua", "de"],
        "components": [
            {
                "name": "水平白红双色条带 (Horizontal White-Red Bands)",
                "relative_pos": "horizontal_stripes",
                "is_discriminative": True,
                "notes": "上白下红横向平分"
            }
        ],
        "distractor_reason": {
            "ua": "乌克兰为上蓝下黄，波兰为上白下红。"
        }
    },
    "ua": {
        "id": "ua",
        "name_zh": "乌克兰国旗 (蓝黄双色旗)",
        "name_en": "Flag of Ukraine",
        "category": "flag",
        "color_topology_strict": True,
        "canonical_colors": ["blue", "yellow"],
        "confusable_distractors": ["pl", "se"],
        "components": [
            {
                "name": "水平蓝黄双色条带 (Horizontal Blue-Yellow Bands)",
                "relative_pos": "horizontal_stripes",
                "is_discriminative": True,
                "notes": "上蓝下黄横向平分"
            }
        ],
        "distractor_reason": {
            "pl": "波兰为上白下红，乌克兰为上蓝下黄。"
        }
    }
}


class ReferenceBank:
    """参考素材库总控类"""

    @classmethod
    def list_supported_ids(cls) -> List[str]:
        return list(REFERENCE_REGISTRY.keys())

    @classmethod
    def get_meta(cls, ref_id: str) -> Dict[str, Any]:
        ref_id = ref_id.lower()
        if ref_id not in REFERENCE_REGISTRY:
            raise KeyError(f"未注册的参考目标: {ref_id}")
        return REFERENCE_REGISTRY[ref_id]

    @classmethod
    def get_confusable_distractor_id(cls, ref_id: str) -> Optional[str]:
        meta = cls.get_meta(ref_id)
        distractors = meta.get("confusable_distractors", [])
        return distractors[0] if distractors else None

    @classmethod
    def load_canonical_image(cls, ref_id: str, width: int = 512, height: int = 384) -> Image.Image:
        """读取官方标准矢量资产并渲染为高质量 RGBA PIL 图像（作为标准参考图）。"""
        ref_id = ref_id.lower()
        svg_file = ASSETS_DIR / f"{ref_id}.svg"
        if not svg_file.exists():
            raise FileNotFoundError(f"未找到官方矢量文件: {svg_file}")
        svg_code = svg_file.read_text(encoding="utf-8")
        png_bytes = resvg_py.svg_to_bytes(svg_code, width=width, height=height)
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    @classmethod
    def render_reference_card(cls, ref_id: str, card_size: Tuple[int, int] = (400, 300)) -> Image.Image:
        """生成供模型输入使用的独立标准参考图卡片 (Clean Reference Image)。
        带有标准留白与边界线，方便作为多模态输入 <ref_image> 传入。
        """
        ref_img = cls.load_canonical_image(ref_id, width=card_size[0] - 40, height=card_size[1] - 40)
        card = Image.new("RGBA", card_size, (248, 249, 250, 255))
        # 居中贴附
        paste_x = (card_size[0] - ref_img.width) // 2
        paste_y = (card_size[1] - ref_img.height) // 2
        card.alpha_composite(ref_img, (paste_x, paste_y))
        return card
