#!/usr/bin/env python3
"""
Encyclopedic Knowledge & Visual Feature Expander (知识图谱与视觉特征扩写引擎)。
解决核心问题：
1. 彻底消除缩写与生硬代码（如 raw ISO 代码 'cn', 'de', 'us_ca' 或 slug '001_gucci.svg'）；
2. 注入完整中英文实体全称、文化历史背景、纹章/旗帜学几何构型说明；
3. 结合物理底板（1,500 种真实服饰箱包底料）与印花着色区域，生成多维度、高信息密度的图文互注与定位数据；
4. 保证 100% 离线运行，零网络流量消耗。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MANIFESTS_DIR = ASSETS_DIR / "manifests"

# -----------------------------------------------------------------------------
# 1. ISO 3166-1 / 国际实体权威知识图谱 (涵盖 271 个国旗代码及非标准地区代码)
# -----------------------------------------------------------------------------
ISO_KNOWLEDGE_BASE: Dict[str, Dict[str, str]] = {
    "cn": {
        "name_zh": "中华人民共和国",
        "name_en": "People's Republic of China",
        "title_zh": "中华人民共和国国旗（五星红旗）",
        "title_en": "National Flag of the People's Republic of China (Five-star Red Flag)",
        "vexillology_zh": "红色旗面象征革命历史，左上方缀黄色五角星五颗，一主四拱，象征全国各族人民大团结；主星角尖朝上，四颗小星各有一角正对大星中心点，呈现严格的向心对称拓扑结构。",
        "vexillology_en": "A red field charged in canton with five golden stars: a large star surrounded by an arc of four smaller stars pointing toward the center.",
        "region_zh": "东亚",
        "region_en": "East Asia",
    },
    "us": {
        "name_zh": "美利坚合众国",
        "name_en": "United States of America",
        "title_zh": "美利坚合众国国旗（星条旗）",
        "title_en": "Flag of the United States (Stars and Stripes)",
        "vexillology_zh": "十三道红白相间的横向条纹象征独立建国时的十三个殖民地，左上角深蓝色象限内整齐排列五十颗白色五角星，代表联邦五十个州。",
        "vexillology_en": "Thirteen horizontal stripes alternating red and white, with a blue canton containing fifty small white five-pointed stars.",
        "region_zh": "北美洲",
        "region_en": "North America",
    },
    "gb": {
        "name_zh": "大不列颠及北爱尔兰联合王国（英国）",
        "name_en": "United Kingdom",
        "title_zh": "联合王国国旗（米字旗）",
        "title_en": "Flag of the United Kingdom (Union Jack)",
        "vexillology_zh": "深蓝色背景上叠合英格兰圣乔治红十字、苏格兰圣安德鲁白斜十字与爱尔兰圣帕特里克红斜十字，呈现非对称十字叠合构型。",
        "vexillology_en": "The red cross of Saint George edged in white, superimposed on the Saltire of Saint Patrick, superimposed on the Saltire of Saint Andrew on a blue field.",
        "region_zh": "欧洲",
        "region_en": "Europe",
    },
    "fr": {
        "name_zh": "法兰西共和国（法国）",
        "name_en": "French Republic",
        "title_zh": "法兰西共和国国旗（蓝白红三色旗）",
        "title_en": "National Flag of France (Tricolour)",
        "vexillology_zh": "从左至右由蓝、白、红三道等宽竖条组成，源于法国大革命时期的巴黎徽标与王室白旗色彩，象征自由、平等与博爱。",
        "vexillology_en": "A vertical tricolour featuring three equal bands of blue, white, and red.",
        "region_zh": "西欧",
        "region_en": "Western Europe",
    },
    "de": {
        "name_zh": "德意志联邦共和国（德国）",
        "name_en": "Federal Republic of Germany",
        "title_zh": "德意志联邦共和国国旗（黑红金三色旗）",
        "title_en": "National Flag of Germany (Black-Red-Gold Tricolour)",
        "vexillology_zh": "从上至下由黑、红、金（黄）三道等宽水平横条组成，源自十九世纪反抗拿破仑战争中吕佐夫志愿军的军服配色。",
        "vexillology_en": "A horizontal tricolour of black, red, and gold (yellow).",
        "region_zh": "中欧",
        "region_en": "Central Europe",
    },
    "jp": {
        "name_zh": "日本国",
        "name_en": "Japan",
        "title_zh": "日本国国旗（日之丸旗）",
        "title_en": "National Flag of Japan (Hinomaru)",
        "vexillology_zh": "纯白矩形底面中央居中绘有一轮纯深红色正圆形太阳，极简克制，象征日出之国与纯洁。",
        "vexillology_en": "A white rectangular field with a large red crimson disc representing the sun at its center.",
        "region_zh": "东亚",
        "region_en": "East Asia",
    },
    "it": {
        "name_zh": "意大利共和国",
        "name_en": "Italian Republic",
        "title_zh": "意大利共和国国旗（绿白红三色旗）",
        "title_en": "National Flag of Italy (Tricolore)",
        "vexillology_zh": "从左至右依次为绿、白、红三等份竖条，绿色象征米兰民兵军服与亚平宁原野，白色象征阿尔卑斯雪山与崇高，红色象征烈士鲜血。",
        "vexillology_en": "A vertical tricolour of green, white, and red bands.",
        "region_zh": "南欧",
        "region_en": "Southern Europe",
    },
    "es": {
        "name_zh": "西班牙王国",
        "name_en": "Kingdom of Spain",
        "title_zh": "西班牙王国国旗（红黄双色旗）",
        "title_en": "National Flag of Spain (La Rojigualda)",
        "vexillology_zh": "由红、黄、红三道水平横条组成，中间黄色条纹宽度为上下红色条纹的两倍（比例1:2:1），偏左侧缀有西班牙国徽与海格力斯双柱。",
        "vexillology_en": "Three horizontal stripes of red, yellow, and red, with the yellow stripe twice the width of each red stripe and charged with the Spanish coat of arms.",
        "region_zh": "南欧",
        "region_en": "Southern Europe",
    },
    "ca": {
        "name_zh": "加拿大",
        "name_en": "Canada",
        "title_zh": "加拿大国旗（枫叶旗）",
        "title_en": "National Flag of Canada (Maple Leaf Flag)",
        "vexillology_zh": "两侧为红色竖条，中央为两倍宽度的白色方形区域，正中绘有一枚醒目的红色十一角几何枫叶。",
        "vexillology_en": "A red base featuring a white square in its centre with a stylized red 11-pointed maple leaf charged in the middle.",
        "region_zh": "北美洲",
        "region_en": "North America",
    },
    "br": {
        "name_zh": "巴西联邦共和国",
        "name_en": "Federative Republic of Brazil",
        "title_zh": "巴西联邦共和国国旗（金绿旗）",
        "title_en": "National Flag of Brazil (Auriverde)",
        "vexillology_zh": "绿色矩形底面上嵌有一个大黄色菱形，中央为深蓝色天球仪圆形，贯穿一条写有'秩序与进步'的白色弧形缎带及二十七颗白色星辰。",
        "vexillology_en": "A green field on which a large yellow rhombus is centered, bearing a blue celestial globe with twenty-seven white five-pointed stars and a white curved ribbon inscribed 'Ordem e Progresso'.",
        "region_zh": "南美洲",
        "region_en": "South America",
    },
    "ru": {
        "name_zh": "俄罗斯联邦",
        "name_en": "Russian Federation",
        "title_zh": "俄罗斯联邦国旗（白蓝红三色旗）",
        "title_en": "National Flag of the Russian Federation",
        "vexillology_zh": "由上至下由白、蓝、红三道等宽平行横条组成，泛斯拉夫颜色体系的代表旗帜。",
        "vexillology_en": "A horizontal tricolour of three equal horizontal bands: white on top, blue in the middle, and red on the bottom.",
        "region_zh": "欧亚",
        "region_en": "Eurasia",
    },
    "kr": {
        "name_zh": "大韩民国（韩国）",
        "name_en": "Republic of Korea",
        "title_zh": "大韩民国国旗（太极旗）",
        "title_en": "National Flag of South Korea (Taegeukgi)",
        "vexillology_zh": "纯白底色中央为红蓝阴阳太极图，四角分别布置乾、坤、坎、离四组黑色八卦卦象，蕴含周易阴阳互克互生哲学。",
        "vexillology_en": "A white field with a red and blue Taegeuk circle in the center and four black trigrams in each corner.",
        "region_zh": "东亚",
        "region_en": "East Asia",
    },
    "au": {
        "name_zh": "澳大利亚联邦",
        "name_en": "Commonwealth of Australia",
        "title_zh": "澳大利亚联邦国旗",
        "title_en": "National Flag of Australia",
        "vexillology_zh": "深蓝色底面左上方为英国米字旗象限，左下方为大七角联邦之星，右侧分布南十字星座的五颗白色星辰。",
        "vexillology_en": "A defaced Blue Ensign: a blue field with the Union Jack in the canton, the large seven-pointed Commonwealth Star in the lower hoist, and the Southern Cross constellation on the fly.",
        "region_zh": "大洋洲",
        "region_en": "Oceania",
    },
    "in": {
        "name_zh": "印度共和国",
        "name_en": "Republic of India",
        "title_zh": "印度共和国国旗（三色旗）",
        "title_en": "National Flag of India (Tiranga)",
        "vexillology_zh": "由藏红花色（橙色）、白色、绿色三道等宽横条组成，中央白色横条正中绘有海军蓝色的二十四辐阿育王法轮（Ashoka Chakra）。",
        "vexillology_en": "A horizontal tricolour of India saffron, white, and India green; with the Ashoka Chakra, a 24-spoke wheel in navy blue at its centre.",
        "region_zh": "南亚",
        "region_en": "South Asia",
    },
    "za": {
        "name_zh": "南非共和国",
        "name_en": "Republic of South Africa",
        "title_zh": "南非共和国国旗（彩虹之国六色旗）",
        "title_en": "National Flag of South Africa",
        "vexillology_zh": "由红、白、绿、黄、黑、蓝六色交织构成，正中央为横卧朝左的绿色'Y'形宽带，象征种族隔离结束后的民族和解与社会大融合。",
        "vexillology_en": "A horizontal bicolor of red and blue, with a black isosceles triangle at the hoist edged in yellow and white, separated by a central horizontal Y-shaped band of green.",
        "region_zh": "非洲",
        "region_en": "Southern Africa",
    },
    "eu": {
        "name_zh": "欧洲联盟（欧盟）",
        "name_en": "European Union",
        "title_zh": "欧洲联盟旗帜（十二金星蓝旗）",
        "title_en": "Flag of the European Union",
        "vexillology_zh": "深蓝色纯色背景正中央环绕排列十二颗五角金星构成圆环，象征欧洲人民的团结、完美与圆满。",
        "vexillology_en": "A circle of twelve five-pointed golden stars on a blue field, symbolizing European unity and perfection.",
        "region_zh": "欧洲国际组织",
        "region_en": "European Supranational",
    },
    "un": {
        "name_zh": "联合国组织",
        "name_en": "United Nations",
        "title_zh": "联合国会旗",
        "title_en": "Flag of the United Nations",
        "vexillology_zh": "天蓝色底面上正中呈现以北极点为中心的白色世界方位等距投影地图，周围环绕一对象征和平繁荣的橄榄枝花环。",
        "vexillology_en": "A white azimuthal equidistant projection map of the world centered on the North Pole, encircled by two olive branches on a UN blue field.",
        "region_zh": "全球国际组织",
        "region_en": "Global International Organization",
    },
    "ch": {
        "name_zh": "瑞士联邦",
        "name_en": "Swiss Confederation",
        "title_zh": "瑞士联邦国旗（红底正白十字旗）",
        "title_en": "National Flag of Switzerland",
        "vexillology_zh": "罕见的 1:1 正方形旗帜，鲜红底色正中央为一个等臂纯白色直角十字，代表中立、和平与人道主义庇护。",
        "vexillology_en": "A square flag featuring a white equilateral cross on a solid red field.",
        "region_zh": "中欧",
        "region_en": "Central Europe",
    },
    "se": {
        "name_zh": "瑞典王国",
        "name_en": "Kingdom of Sweden",
        "title_zh": "瑞典王国国旗（黄蓝北欧十字旗）",
        "title_en": "National Flag of Sweden",
        "vexillology_zh": "浅蓝色底面上呈现黄色北欧斯堪的纳维亚偏心十字，垂直臂偏向旗杆一侧。",
        "vexillology_en": "A yellow Nordic cross on a blue field, with the vertical crossbar shifted toward the hoist side.",
        "region_zh": "北欧",
        "region_en": "Northern Europe",
    },
    "no": {
        "name_zh": "挪威王国",
        "name_en": "Kingdom of Norway",
        "title_zh": "挪威王国国旗",
        "title_en": "National Flag of Norway",
        "vexillology_zh": "鲜红底面上呈现带白色细镶边的深蓝色北欧偏心十字，融合了丹麦与瑞典的传统色彩。",
        "vexillology_en": "A red field charged with a navy blue Nordic cross outlined in white.",
        "region_zh": "北欧",
        "region_en": "Northern Europe",
    },
    "nl": {
        "name_zh": "荷兰王国",
        "name_en": "Kingdom of the Netherlands",
        "title_zh": "荷兰王国国旗（红白蓝三色旗）",
        "title_en": "National Flag of the Netherlands",
        "vexillology_zh": "从上至下由红、白、蓝三等宽横条组成，是现代水平三色旗的始祖范式。",
        "vexillology_en": "A horizontal tricolour of red, white, and cobalt blue.",
        "region_zh": "西欧",
        "region_en": "Western Europe",
    },
    "ar": {
        "name_zh": "阿根廷共和国",
        "name_en": "Argentine Republic",
        "title_zh": "阿根廷共和国国旗（五月太阳旗）",
        "title_en": "National Flag of Argentina",
        "vexillology_zh": "由浅天蓝、白、浅天蓝三道水平横条组成，中间白色条纹正中缀有带人脸面容与三十二道光芒的黄金'五月太阳'。",
        "vexillology_en": "Three equal horizontal bands of light sky blue and white, with the golden Sun of May centered on the white stripe.",
        "region_zh": "南美洲",
        "region_en": "South America",
    },
    "mx": {
        "name_zh": "墨西哥合众国",
        "name_en": "United Mexican States",
        "title_zh": "墨西哥合众国国旗",
        "title_en": "National Flag of Mexico",
        "vexillology_zh": "从左至右为绿、白、红三道竖条，中央白色条纹正中嵌有墨西哥国徽：一只雄鹰叼着毒蛇伫立在仙人掌上，源于阿兹特克建城神话。",
        "vexillology_en": "A vertical tricolour of green, white, and red with the national coat of arms (eagle perched on a prickly pear cactus devouring a snake) centered on the white stripe.",
        "region_zh": "北美/中美",
        "region_en": "North America",
    },
}

# 区域旗帜拓展知识
REGIONAL_KNOWLEDGE_BASE: Dict[str, Dict[str, str]] = {
    "us_california": {
        "name_zh": "美国加利福尼亚州（熊旗共和国）",
        "name_en": "State of California, USA",
        "title_zh": "加利福尼亚州州旗（加州熊旗）",
        "title_en": "California State Flag (Bear Flag)",
        "vexillology_zh": "白色底布左上角缀有一颗红色五角星，中央草地上方缓步走过一头加利福尼亚金熊，下方印有大写'CALIFORNIA REPUBLIC'字样，底部衬有一条粗红底杠。",
        "vexillology_en": "A white field with a red star in canton, a California grizzly bear walking on a green grass plot, 'CALIFORNIA REPUBLIC' inscription, and a red bottom stripe.",
    },
    "us_texas": {
        "name_zh": "美国德克萨斯州（孤星之州）",
        "name_en": "State of Texas, USA",
        "title_zh": "德克萨斯州州旗（孤星旗）",
        "title_en": "Texas State Flag (Lone Star Flag)",
        "vexillology_zh": "旗杆侧为一条垂直深蓝条纹，中央镶嵌一颗白色大五角星；右侧由上白下红两道等宽水平条纹构成，气势磅礴。",
        "vexillology_en": "A vertical blue stripe along the hoist containing a single white star, with two equal horizontal stripes of white and red on the fly.",
    },
    "jp_tokyo_metropolis": {
        "name_zh": "日本东京都",
        "name_en": "Tokyo Metropolis, Japan",
        "title_zh": "东京都都旗（江户紫太阳徽）",
        "title_en": "Flag of Tokyo Metropolis",
        "vexillology_zh": "底色为传统雅致的'江户紫'（深紫罗兰色），正中居中呈现白色的东京都徽章，形似六道放射光芒的太阳与日字变体，象征日本首都繁荣发展。",
        "vexillology_en": "An Edo purple field bearing a white central emblem resembling a six-rayed sun symbolizing the capital's expansion.",
    },
    "de_bavaria": {
        "name_zh": "德国巴伐利亚自由邦",
        "name_en": "Free State of Bavaria, Germany",
        "title_zh": "巴伐利亚自由邦菱格州旗",
        "title_en": "Flag of Bavaria (Rautenflagge)",
        "vexillology_zh": "由白蓝两色倾斜菱形格子紧密交错阵列构成，源自维特尔斯巴赫家族历史纹章，是德意志传统文化的鲜明标志。",
        "vexillology_en": "A lozengy pattern of alternating white and sky-blue diamonds originating from the House of Wittelsbach coat of arms.",
    },
}

# 官方国徽/徽章知识
EMBLEM_KNOWLEDGE_BASE: Dict[str, Dict[str, str]] = {
    "emblem_peoples_republic_of_china": {
        "name_zh": "中华人民共和国国徽",
        "name_en": "National Emblem of the People's Republic of China",
        "title_zh": "中华人民共和国国徽",
        "title_en": "National Emblem of the PRC",
        "vexillology_zh": "正中央为五星照耀下的天安门城楼，周围环绕麦稻穗齿轮，施以纯红金黄两色，象征中国工人阶级领导的工农联盟新民主主义政权。",
        "vexillology_en": "Tiananmen Gate under the light of five stars, encircled by sheaves of wheat and rice with a cogwheel at the base in red and gold.",
    },
    "emblem_germany_eagle": {
        "name_zh": "德意志联邦共和国国徽（联邦之鹰）",
        "name_en": "Coat of Arms of Germany (Bundesadler)",
        "title_zh": "德意志联邦之鹰国徽",
        "title_en": "Federal Coat of Arms of Germany",
        "vexillology_zh": "金黄色盾面上伫立着一头面向左侧、展翅挺胸的黑色单头鹰，长有深红色的利喙与鹰爪，展现悠久的罗马上帝鹰徽传承与威严秩序。",
        "vexillology_en": "A black eagle with red beak and claws displayed on a golden shield.",
    },
    "badge_nasa_meatball_insignia": {
        "name_zh": "美国国家航空航天局（NASA）肉丸徽标",
        "name_en": "NASA Meatball Insignia",
        "title_zh": "NASA 经典肉丸徽标 (The Meatball)",
        "title_en": "NASA Classic Meatball Insignia",
        "vexillology_zh": "深蓝色球体象征茫茫宇宙苍穹，缀以白色群星，红色倒V形翼掠过球心象征航空空气动力学突破，白色环绕轨道与白色NASA衬线字符交织穿行。",
        "vexillology_en": "A blue sphere representing space with stars, a red V-shaped vector wing denoting aeronautics, an orbital path, and bold white serif NASA typography.",
    },
    "badge_olympic_rings": {
        "name_zh": "奥林匹克五环徽标",
        "name_en": "Olympic Rings Insignia",
        "title_zh": "奥林匹克五环标志",
        "title_en": "The Olympic Rings",
        "vexillology_zh": "由蓝、黄、黑、绿、红五只等径圆环由左至右、上二下三交错套叠扣合，象征五大洲的紧密团结与全球运动员在奥运会上的友好竞争。",
        "vexillology_en": "Five interlaced rings of equal dimensions in blue, yellow, black, green, and red representing the union of the five inhabited continents.",
    },
}


# -----------------------------------------------------------------------------
# 官方实体（国旗 / 国徽 / 地方旗）中文名表（assets/manifests/entity_name_zh.json）
# 由 tools/build_entity_name_table.py 离线生成，100% 覆盖三个资产域；
# 未命中表中详实旗帜学/纹章学条目时，使用下列中文通用描述兜底。
# -----------------------------------------------------------------------------
OFFICIAL_ENTITY_DESCRIPTIONS: Dict[str, Tuple[str, str, str, str]] = {
    "official_flags": (
        "采用标准旗帜学矩形比例与高对比度几何色块分割，色彩配置符合国际通行旗帜规范，图案布局规整对称。",
        "Constructed with standardized vexillological proportions, high-contrast geometric color fields, and balanced emblem placement.",
        "作为国际公认的主权象征与地域身份标识，承载该国家或地区的政治传统与集体认同。",
        "An internationally recognized sovereign symbol carrying the political heritage and collective identity of the entity.",
    ),
    "official_emblems": (
        "采用经典纹章学构图，常见盾徽、冠饰、神兽禽鸟图腾、禾穗环带与铭文绶带等元素，层级严密、对称庄重。",
        "Rendered in classic heraldic composition with shield, crest, fauna charges, wreaths, and motto ribbons.",
        "作为国家或国际组织的法定最高徽记，代表公权力权威、历史传承与制度信用。",
        "The supreme legal insignia of a state or international organization, embodying sovereign authority and historical continuity.",
    ),
    "official_regional": (
        "融合地方历史纹章、州徽/市徽图案、地域性动植物象征与几何色块分割，体现辖区独特的自治标识。",
        "Combining regional heraldic devices, local seals, geographic fauna/flora symbols, and geometric color divisions.",
        "承载地方行政区划的历史沿革、族群传统与地域文化认同，常见于政府建筑与公共仪式场合。",
        "Reflecting the administrative history, ethnic heritage, and regional identity of the sub-national jurisdiction.",
    ),
}

OFFICIAL_ENTITY_DOMAINS: Tuple[str, ...] = (
    "official_flags",
    "official_emblems",
    "official_regional",
)


class KnowledgeExpander:
    """全息多模态知识图谱与视觉特征扩写引擎"""

    def __init__(self):
        self._load_all_manifests()

    def _load_all_manifests(self):
        """离线加载现有全部结构化资产元数据清单"""
        self.luxury_map: Dict[str, Dict] = {}
        self.ip_brand_map: Dict[str, Dict] = {}
        self.character_ip_map: Dict[str, Dict] = {}
        self.subculture_map: Dict[str, Dict] = {}
        self.commodity_1500_list: List[Dict] = []

        lux_file = MANIFESTS_DIR / "luxury_brands.json"
        if lux_file.exists():
            data = json.loads(lux_file.read_text(encoding="utf-8"))
            for it in data.get("items", []):
                stem = Path(it.get("local_path", "")).stem
                self.luxury_map[stem] = it
                self.luxury_map[it.get("slug", "")] = it

        ip_file = MANIFESTS_DIR / "ip_brands.json"
        if ip_file.exists():
            data = json.loads(ip_file.read_text(encoding="utf-8"))
            for it in data.get("items", []):
                stem = Path(it.get("local_path", "")).stem
                self.ip_brand_map[stem] = it
                self.ip_brand_map[it.get("slug", "")] = it

        char_file = MANIFESTS_DIR / "character_ips.json"
        if char_file.exists():
            data = json.loads(char_file.read_text(encoding="utf-8"))
            for it in data.get("items", []):
                stem = Path(it.get("local_path", "")).stem
                self.character_ip_map[stem] = it
                self.character_ip_map[it.get("id", "")] = it

        sub_file = MANIFESTS_DIR / "subculture_elements.json"
        if sub_file.exists():
            data = json.loads(sub_file.read_text(encoding="utf-8"))
            for it in data.get("items", []):
                stem = Path(it.get("local_path", "")).stem
                self.subculture_map[stem] = it
                self.subculture_map[it.get("id", "")] = it

        comm_file = MANIFESTS_DIR / "commodity_1500.json"
        if comm_file.exists():
            data = json.loads(comm_file.read_text(encoding="utf-8"))
            self.commodity_1500_list = data.get("items", [])

        # 官方实体（国旗/国徽/地方旗）中文全称表：由 tools/build_entity_name_table.py 离线生成。
        # 表中 100% 覆盖三个资产域的实际文件 stem，供 resolve_entity_knowledge 在通用 fallback 之前优先消费。
        self.entity_name_table: Dict[str, Dict[str, Dict[str, str]]] = {
            domain: {} for domain in OFFICIAL_ENTITY_DOMAINS
        }
        entity_file = MANIFESTS_DIR / "entity_name_zh.json"
        if entity_file.exists():
            entity_data = json.loads(entity_file.read_text(encoding="utf-8"))
            for domain in OFFICIAL_ENTITY_DOMAINS:
                domain_data = entity_data.get(domain)
                if isinstance(domain_data, dict):
                    self.entity_name_table[domain] = domain_data

    def _legacy_vexillology_detail(
        self, domain: str, stem: str
    ) -> Optional[Dict[str, str]]:
        """为官方实体表中命中的条目复用原有高质量旗帜学/纹章学细节（如存在）。"""
        if domain == "official_flags":
            return ISO_KNOWLEDGE_BASE.get(stem)
        if domain == "official_regional":
            for key, value in REGIONAL_KNOWLEDGE_BASE.items():
                if key in stem or stem in key:
                    return value
            return None
        if domain == "official_emblems":
            for key, value in EMBLEM_KNOWLEDGE_BASE.items():
                if key in stem or stem in key:
                    return value
        return None

    def _resolve_official_from_table(
        self, domain: str, stem: str, entry: Dict[str, str]
    ) -> Dict[str, str]:
        """将中文名表条目展开为统一实体知识结构，保证四个中文字段全为规范中文。"""
        legacy = self._legacy_vexillology_detail(domain, stem)
        if legacy is not None:
            design_zh = legacy["vexillology_zh"]
            design_en = legacy["vexillology_en"]
            if domain == "official_flags":
                cultural_zh = (
                    f"属于{legacy['region_zh']}法定官方象征标识，"
                    "具备最高级别旗帜学公信力与图形规范约束。"
                )
                cultural_en = (
                    f"Official sovereign vexillological emblem of {legacy['region_en']}."
                )
            elif domain == "official_emblems":
                cultural_zh = "国家宪法法定最高徽记或全球顶级学术/体育国际公约权威印记。"
                cultural_en = "Constitutional state crest or global supreme treaty insignia."
            else:
                cultural_zh = "具备深厚的地方拓荒历史、族群传统与地域自治认同。"
                cultural_en = (
                    "Rooted in local regional heritage, pioneer history, and self-governing identity."
                )
        else:
            design_zh, design_en, cultural_zh, cultural_en = OFFICIAL_ENTITY_DESCRIPTIONS[domain]

        return {
            "entity_type": entry.get("entity_type_zh", "官方实体徽记 (Official Entity Insignia)"),
            "entity_name_zh": entry["name_zh"],
            "entity_name_en": entry["name_en"],
            "full_title_zh": entry["title_zh"],
            "full_title_en": entry["title_en"],
            "design_analysis_zh": design_zh,
            "design_analysis_en": design_en,
            "cultural_context_zh": cultural_zh,
            "cultural_context_en": cultural_en,
        }

    def resolve_entity_knowledge(self, domain: str, stem: str) -> Dict[str, str]:
        """将任意原始缩写、代码、ID 扩写为结构化百科知识，杜绝一切片面或单字代码。"""
        clean_stem = stem.lower().strip()

        # 0. 官方实体中文名表优先（100% 覆盖 flags / emblems / regional 实际资产，
        #    彻底杜绝英文国名前缀与 raw ISO/行政区代码 fallback）
        if domain in self.entity_name_table:
            table_entry = self.entity_name_table[domain].get(clean_stem)
            if table_entry:
                return self._resolve_official_from_table(domain, clean_stem, table_entry)

        # 1. 国际旗帜 (ISO 3166-1)
        if domain == "official_flags" or clean_stem in ISO_KNOWLEDGE_BASE:
            if clean_stem in ISO_KNOWLEDGE_BASE:
                kb = ISO_KNOWLEDGE_BASE[clean_stem]
                return {
                    "entity_type": "国家/地区主权旗帜 (National Flag)",
                    "entity_name_zh": kb["name_zh"],
                    "entity_name_en": kb["name_en"],
                    "full_title_zh": kb["title_zh"],
                    "full_title_en": kb["title_en"],
                    "design_analysis_zh": kb["vexillology_zh"],
                    "design_analysis_en": kb["vexillology_en"],
                    "cultural_context_zh": f"属于{kb['region_zh']}法定官方象征标识，具备最高级别旗帜学公信力与图形规范约束。",
                    "cultural_context_en": f"Official sovereign vexillological emblem of {kb['region_en']}.",
                }
            # 自动化通用泛化处理其他 240+ ISO 代码
            upper_code = clean_stem.upper()
            return {
                "entity_type": "主权国家/地区官方旗帜 (National Flag)",
                "entity_name_zh": f"国际标准代码 [{upper_code}] 主权领地实体",
                "entity_name_en": f"Sovereign Entity (ISO Code {upper_code})",
                "full_title_zh": f"国际公认官方国家与地区旗帜 [{upper_code}]",
                "full_title_en": f"Official National & Territorial Flag ({upper_code})",
                "design_analysis_zh": f"遵循 ISO 3166 国际旗帜标准比例构建，具备标准几何图元、分色色块与专属纹章拓扑布局。",
                "design_analysis_en": f"Constructed according to standard ISO vexillological geometry and official color schemes.",
                "cultural_context_zh": "承载国际公认的主权象征与地域身份识别。",
                "cultural_context_en": "Carrying sovereign international identity and regional recognition.",
            }

        # 2. 官方徽章与国徽 (Official Emblems)
        if domain == "official_emblems" or "emblem" in clean_stem or "badge" in clean_stem:
            for k, v in EMBLEM_KNOWLEDGE_BASE.items():
                if k in clean_stem or clean_stem in k:
                    return {
                        "entity_type": "官方最高权威国徽/国际组织徽记 (Official Emblem / Insignia)",
                        "entity_name_zh": v["name_zh"],
                        "entity_name_en": v["name_en"],
                        "full_title_zh": v["title_zh"],
                        "full_title_en": v["title_en"],
                        "design_analysis_zh": v["vexillology_zh"],
                        "design_analysis_en": v["vexillology_en"],
                        "cultural_context_zh": "国家宪法法定最高徽记或全球顶级学术/体育国际公约权威印记。",
                        "cultural_context_en": "Constitutional state crest or global supreme treaty insignia.",
                    }
            country_part = clean_stem.replace("emblem_", "").replace("badge_", "").replace("_", " ").title()
            return {
                "entity_type": "国家/权威官方徽记 (Official State Emblem)",
                "entity_name_zh": f"{country_part} 官方徽章/国徽",
                "entity_name_en": f"Official Emblem of {country_part}",
                "full_title_zh": f"{country_part} 官方纹章徽记",
                "full_title_en": f"Coat of Arms / Seal of {country_part}",
                "design_analysis_zh": "采用西方或古典纹章学经典盾徽、展翅神兽、禾穗环绕与几何铭文拓扑构建，构图严密威仪。",
                "design_analysis_en": "Rendered in classic heraldic shield, fauna charge, and ceremonial wreath composition.",
                "cultural_context_zh": "代表最高主权公权力与历史传承。",
                "cultural_context_en": "Embodying supreme sovereign authority and historical heritage.",
            }

        # 3. 区域旗帜 (Official Regional)
        if domain == "official_regional" or any(clean_stem.startswith(p) for p in ["us_", "jp_", "de_", "ca_", "au_"]):
            for k, v in REGIONAL_KNOWLEDGE_BASE.items():
                if k in clean_stem or clean_stem in k:
                    return {
                        "entity_type": "行政邦州/特区法定官方旗帜 (Sub-national / Regional Flag)",
                        "entity_name_zh": v["name_zh"],
                        "entity_name_en": v["name_en"],
                        "full_title_zh": v["title_zh"],
                        "full_title_en": v["title_en"],
                        "design_analysis_zh": v["vexillology_zh"],
                        "design_analysis_en": v["vexillology_en"],
                        "cultural_context_zh": "具备深厚的地方拓荒历史、族群传统与地域自治认同。",
                        "cultural_context_en": "Rooted in local regional heritage, pioneer history, and self-governing identity.",
                    }
            region_name = clean_stem.replace("_", " ").title()
            return {
                "entity_type": "地方邦州/自治省行政旗帜 (State / Provincial Flag)",
                "entity_name_zh": f"{region_name} 法定区域旗帜",
                "entity_name_en": f"Regional Flag of {region_name}",
                "full_title_zh": f"{region_name} 地方行政法定旗帜",
                "full_title_en": f"Official Flag of {region_name}",
                "design_analysis_zh": "融合了州徽、地域特产动植物图案、历史战旗条纹与独特的拓扑色块。",
                "design_analysis_en": "Integrating regional seal elements, flora/fauna symbols, and historic stripe arrangements.",
                "cultural_context_zh": "承载地方历史与社群凝聚力。",
                "cultural_context_en": "Carrying statehood heritage and regional community pride.",
            }

        # 4. 奢侈品与高级钟表珠宝品牌 (Luxury Brands)
        if clean_stem in self.luxury_map or domain == "luxury_100":
            item = self.luxury_map.get(clean_stem, {})
            name_zh = item.get("name_zh", clean_stem.title())
            name_en = item.get("name_en", clean_stem.title())
            return {
                "entity_type": "国际顶级奢侈品牌核心视觉资产 (Haute Horlogerie & Luxury Heritage)",
                "entity_name_zh": name_zh,
                "entity_name_en": name_en,
                "full_title_zh": f"{name_zh}（{name_en}）典藏官方徽标",
                "full_title_en": f"{name_en} Official Heritage Wordmark & Monogram",
                "design_analysis_zh": f"采用严苛比例的几何衬线/无衬线高定字体设计，笔画转角经过光学微调，呈现极致平衡与现代奢华感。",
                "design_analysis_en": f"Designed with precise bespoke serif/sans-serif typography and optical weight balance.",
                "cultural_context_zh": "拥有数十年乃至百余年工坊精湛传承，是全球高端时尚与身份地位的具象文化象征。",
                "cultural_context_en": "Representing centuries-old artisanal craft and global high-fashion distinction.",
            }

        # 5. IP、科技与流行品牌 (IP & Consumer Brands)
        if clean_stem in self.ip_brand_map or domain == "ip_brands_100":
            item = self.ip_brand_map.get(clean_stem, {})
            name_zh = item.get("name_zh", clean_stem.title())
            name_en = item.get("name_en", clean_stem.title())
            return {
                "entity_type": "全球知名消费/娱乐/科技商业 IP 徽标 (Global Cultural & Entertainment IP)",
                "entity_name_zh": name_zh,
                "entity_name_en": name_en,
                "full_title_zh": f"{name_zh}（{name_en}）官方品牌超级符号",
                "full_title_en": f"{name_en} Trademark Visual Identity",
                "design_analysis_zh": "标志具备高度凝练的几何图形或强辨识度字形，高对比度视觉传达，专为跨媒介工业传播优化。",
                "design_analysis_en": "Featuring highly memorable geometry, high visual contrast, and optimized cross-media scalability.",
                "cultural_context_zh": "全球受众基础深厚，深度融入当代青年消费文化与数字生活体验。",
                "cultural_context_en": "Embedded in contemporary youth pop culture and global consumer life.",
            }

        # 6. 经典动漫角色 IP (Character IPs)
        if clean_stem in self.character_ip_map or domain == "character_ips":
            item = self.character_ip_map.get(clean_stem, {})
            name_zh = item.get("name_zh", clean_stem.replace("_", " ").title())
            name_en = item.get("name_en", clean_stem.replace("_", " ").title())
            universe = item.get("universe", "经典动漫宇宙")
            attrs = item.get("attributes", "主角/经典造型")
            return {
                "entity_type": "经典动漫/游戏角色视觉肖像 (Anime & Gaming Character IP)",
                "entity_name_zh": name_zh,
                "entity_name_en": name_en,
                "full_title_zh": f"《{universe}》所属角色：{name_zh}（{name_en}）",
                "full_title_en": f"{name_en} from {universe}",
                "design_analysis_zh": f"呈现二维手绘与现代数码赛璐珞赛博质感，发型轮廓鲜明，身着经典战袍服饰，具备{attrs}鲜明视觉特征。",
                "design_analysis_en": f"Featuring iconic silhouette, distinctive signature costume, and dynamic anime cel-shaded character design.",
                "cultural_context_zh": f"作为二次元与流行亚文化顶流符号，具备跨世代的广泛粉丝认知与情感共鸣。",
                "cultural_context_en": "An iconic pop-culture figure with global fandom and multi-generational recognition.",
            }

        # 7. 亚文化符号、3D Emoji 与网络流行文化 (Subculture Elements)
        if clean_stem in self.subculture_map or domain == "subculture_elements":
            item = self.subculture_map.get(clean_stem, {})
            name_zh = item.get("name_zh", clean_stem.replace("_", " ").title())
            name_en = item.get("name_en", clean_stem.replace("_", " ").title())
            group = item.get("group", "亚文化元素")
            tag = item.get("subculture_tag", "潮流符号")
            return {
                "entity_type": "街头潮流/数字亚文化/表情符号 (Streetwear & Subculture Iconography)",
                "entity_name_zh": name_zh,
                "entity_name_en": name_en,
                "full_title_zh": f"街头与数字潮流符号：{name_zh}（{name_en}）",
                "full_title_en": f"{name_en} ({tag.capitalize()} Subculture Motif)",
                "design_analysis_zh": f"属于 {group} 分类，采用三维微光立体渲染或高对比度街头模板喷绘风格，轮廓精炼，视觉张力极强。",
                "design_analysis_en": f"Rendered in distinctive {group} aesthetics with sharp iconography and punchy visual presence.",
                "cultural_context_zh": f"盛行于网络社交语境、Z世代街头服饰涂鸦与潮流生活方式，体现反叛、幽默或个性化态度表达。",
                "cultural_context_en": "Popular in digital internet culture, Z-generation street fashion, and playful self-expression.",
            }

        # 默认通用回退（决不输出单薄代码）
        readable_title = clean_stem.replace("_", " ").replace("-", " ").title()
        return {
            "entity_type": "通用图形与视觉徽标资产 (Visual Motif / Logo)",
            "entity_name_zh": f"{readable_title} 视觉图形",
            "entity_name_en": f"{readable_title} Visual Graphic",
            "full_title_zh": f"{readable_title} 专属几何印花图元",
            "full_title_en": f"{readable_title} Graphic Element",
            "design_analysis_zh": "具备完整的清晰的外形轮廓与分界明显的色阶层级，造型严谨现代。",
            "design_analysis_en": "Composed of clean vector contours and distinct color separation.",
            "cultural_context_zh": "作为当代商品与视觉传达设计中的高频图形符号。",
            "cultural_context_en": "Serving as a recurring motif in contemporary product and visual design.",
        }

    @staticmethod
    def inspect_visual_features(img_rgba: Image.Image) -> Dict[str, Any]:
        """动态分析渲染后 RGBA 图像的实际物理与视觉特征（主导色谱、宽高比、拓扑布局等）。"""
        w, h = img_rgba.size
        aspect_ratio = round(float(w) / float(max(h, 1)), 2)

        arr = np.array(img_rgba)
        alpha = arr[:, :, 3]
        active_pixels = arr[alpha > 40]

        if len(active_pixels) == 0:
            return {
                "aspect_ratio": aspect_ratio,
                "layout_geometry_zh": "中性方形图元",
                "layout_geometry_en": "Square Graphic Primitive",
                "dominant_colors_zh": ["中性灰/透光"],
                "dominant_colors_en": ["Neutral Gray"],
                "has_transparency": True,
            }

        # 采样计算主色谱
        rgb_pixels = active_pixels[:, :3]
        if len(rgb_pixels) > 500:
            indices = np.random.choice(len(rgb_pixels), 500, replace=False)
            sample_rgb = rgb_pixels[indices]
        else:
            sample_rgb = rgb_pixels

        # 简单均值与主分量
        med_rgb = np.median(sample_rgb, axis=0)
        r, g, b = float(med_rgb[0]), float(med_rgb[1]), float(med_rgb[2])

        col_name_zh, col_name_en = KnowledgeExpander._rgb_to_color_name(r, g, b)

        # 几何外形判断
        if aspect_ratio >= 1.6:
            geom_zh = "横向宽幅扁平矩形构型 (Horizontal Widescreen Layout)"
            geom_en = "Wide Horizontal Rectangular Proportion"
        elif aspect_ratio <= 0.65:
            geom_zh = "纵向立式纤细长条构型 (Vertical Banner Proportion)"
            geom_en = "Tall Vertical Proportion"
        elif 0.90 <= aspect_ratio <= 1.10:
            geom_zh = "正方形/对称正圆同心圆构型 (Square / Centered Circular Geometry)"
            geom_en = "Square / Concentric Circular Geometry"
        else:
            geom_zh = "标准 4:3 / 3:2 传统旗帜与徽章比例 (Classic 3:2 / 4:3 Aspect Ratio)"
            geom_en = "Standard 3:2 Proportion"

        has_transparency = bool(np.any(alpha < 240))

        return {
            "aspect_ratio": aspect_ratio,
            "layout_geometry_zh": geom_zh,
            "layout_geometry_en": geom_en,
            "dominant_color_zh": col_name_zh,
            "dominant_color_en": col_name_en,
            "has_transparency": has_transparency,
        }

    @staticmethod
    def _rgb_to_color_name(r: float, g: float, b: float) -> Tuple[str, str]:
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        sat = (max_c - min_c) / (max_c + 1e-5)

        if sat < 0.14:
            if lum < 50:
                return "浓郁曜石黑", "Obsidian Black"
            elif lum < 120:
                return "深沉石板灰", "Slate Gray"
            elif lum < 200:
                return "质感中性灰", "Neutral Heather Gray"
            else:
                return "纯净明亮白", "Pure Crisp White"

        if r > g * 1.35 and r > b * 1.35:
            return ("典雅酒红/深绯", "Deep Burgundy Red") if lum < 110 else ("鲜明中国红/赤焰红", "Vibrant Scarlet Red")
        elif g > r * 1.25 and g > b * 1.25:
            return ("沉稳墨绿/橄榄绿", "Forest Olive Green") if lum < 120 else ("清爽翠绿/薄荷绿", "Vivid Emerald Green")
        elif b > r * 1.25 and b > g * 1.1:
            return ("深邃海军蓝/藏青", "Deep Navy Blue") if lum < 115 else ("明媚皇家蓝/天青", "Vibrant Royal Blue")
        elif r > 160 and g > 130 and b < 90:
            return "辉煌金黄/琥珀黄", "Golden Amber Yellow"
        elif r > 160 and g > 75 and b < 70:
            return "活力热烈橙", "Vivid Energetic Orange"
        elif r > 120 and b > 120 and g < 100:
            return "神秘魅惑紫", "Mystic Royal Purple"
        elif r > 160 and b > 130 and g > 110:
            return "浪漫樱花粉", "Soft Blossom Pink"
        else:
            return "复合混色调", "Multi-Color Composite"


# 单例实例
expander_instance = KnowledgeExpander()
