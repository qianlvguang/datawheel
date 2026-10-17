#!/usr/bin/env python3
"""
官方实体中文名表离线生成器 (Entity Name Table Builder)。

目标：彻底消除 knowledge_expander 对 271 款国旗、121 款国徽/徽章、190 款地方旗
的英文名/原始代码 fallback，生成 `assets/manifests/entity_name_zh.json`。

数据来源（100% 离线，禁止联网）：
1. 国旗：调用本机 Node.js 的 `Intl.DisplayNames(['zh-CN'] / ['en'], {type:'region'})`
   遍历 a-z 两字母组合批量导出 ISO 3166-1 alpha-2 中英文名；
   非 ISO 代码（arab / asean / cefta / eac / pc / eu / un / es-ct / gb-eng /
   sh-ac / xk / xx 等）与地区特例（tw / hk / mo / ic / dg / cp ...）手工补录。
2. 国徽：以 `assets/official_emblems` 实际目录 stem 为准，剥 emblem_/badge_ 前缀后，
   用规范化 token 做「国家英文名 token ⊆ stem token」匹配（自带 the/of/and 等噪声
   token 容忍，以及 espa→spain、per→peru、liechtensteins→liechtenstein 等词形
   别名）；国际组织/特殊徽章（united_nations、nasa、olympic、sede_vacante、
   somaliland、cn_stars 等）手工补全。
3. 地方旗：以 `assets/official_regional` 实际目录 stem 为准，人工映射 190 条
   （美/加/德/西/日/英/法/意/澳全部现有资产），无法从 ISO 名称表推导。

输出结构：
{
  "meta": {"generated_at", "node_bin", "counts", "sources"},
  "official_flags":    {stem: {name_zh, name_en, title_zh, title_en, entity_type_zh, entity_type_en}},
  "official_emblems":  {...},
  "official_regional": {...}
}

用法：
    .venv/bin/python tools/build_entity_name_table.py

脚本会强制校验三个域目录的 100% 覆盖与中文质量（含 CJK、去括号后无 ASCII 字母、
不含劣质 fallback 模式），校验失败直接非零退出，绝不产出半成品表。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MANIFESTS_DIR = ASSETS_DIR / "manifests"
OUTPUT_PATH = MANIFESTS_DIR / "entity_name_zh.json"

DOMAINS = ("official_flags", "official_emblems", "official_regional")

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
PAREN_RE = re.compile(r"[（(][^（）()]*[）)]")
INNER_PAREN_RE = re.compile(r"[（(]([^（）()]*)[）)]")
LATIN_RUN_RE = re.compile(r"[A-Za-z][A-Za-z.&'’\- ]*")

# 劣质 fallback 模式（必须彻底消失）
FORBIDDEN_ZH_FRAGMENTS = (
    "国际标准代码",
    "官方徽章/国徽",
    "法定区域旗帜",
    "主权领地实体",
    "视觉图形",
    "专属几何印花图元",
)

# 官方实体中文名中允许出现在括号内的拉丁缩写白名单（非国家名）
ALLOWED_BRACKET_LATIN = {"NASA"}

# -----------------------------------------------------------------------------
# 常见实体类型（中英）
# -----------------------------------------------------------------------------
FLAG_TYPES: Dict[str, Tuple[str, str]] = {
    "sovereign": ("国家/地区主权旗帜 (National Flag)", "Sovereign National Flag"),
    "territory": ("地区/属地官方旗帜 (Territorial / Dependency Flag)", "Territorial / Dependency Flag"),
    "organization": ("国际组织官方旗帜 (International Organization Flag)", "International Organization Flag"),
    "special": ("特殊/非官方象征旗帜 (Special / Unofficial Flag)", "Special / Unofficial Flag"),
}

REGIONAL_TYPES: Dict[str, Tuple[str, str]] = {
    "state": ("行政州/省级法定官方旗帜 (State / Provincial Flag)", "State / Provincial Flag"),
    "territory": ("地区/海外领地法定官方旗帜 (Territory / Dependency Flag)", "Territory / Dependency Flag"),
    "historic": ("历史政区/民族官方旗帜 (Historical Administrative / Ethnic Flag)", "Historical Administrative / Ethnic Flag"),
    "national_variant": ("国家旗帜特殊版式 (National Flag Variant)", "National Flag Variant"),
    "constituent": ("构成国官方旗帜 (Constituent Country Flag)", "Constituent Country Flag"),
    "special": ("联邦特区/特殊行政区旗帜 (Federal District / Special Administrative Flag)", "Federal District / Special Administrative Flag"),
}

EMBLEM_TYPE = ("国家/权威官方徽记 (Official State Emblem)", "Official State Emblem")
EMBLEM_ORG_TYPE = ("国际组织/航天机构官方徽记 (International Organization Insignia)", "International Organization / Space Agency Insignia")


def _record(
    name_zh: str,
    name_en: str,
    title_zh: str,
    title_en: str,
    entity_type_zh: str,
    entity_type_en: str,
) -> Dict[str, str]:
    return {
        "name_zh": name_zh.strip(),
        "name_en": name_en.strip(),
        "title_zh": title_zh.strip(),
        "title_en": title_en.strip(),
        "entity_type_zh": entity_type_zh,
        "entity_type_en": entity_type_en,
    }


def _flag(name_zh: str, name_en: str, title_zh: str, title_en: str, kind: str) -> Dict[str, str]:
    type_zh, type_en = FLAG_TYPES[kind]
    return _record(name_zh, name_en, title_zh, title_en, type_zh, type_en)


def _regional(name_zh: str, name_en: str, title_zh: str, title_en: str, kind: str) -> Dict[str, str]:
    type_zh, type_en = REGIONAL_TYPES[kind]
    return _record(name_zh, name_en, title_zh, title_en, type_zh, type_en)


def _emblem(name_zh: str, name_en: str, title_zh: str, title_en: str, org: bool = False) -> Dict[str, str]:
    type_zh, type_en = EMBLEM_ORG_TYPE if org else EMBLEM_TYPE
    return _record(name_zh, name_en, title_zh, title_en, type_zh, type_en)


# -----------------------------------------------------------------------------
# 1. Node.js Intl.DisplayNames 离线导出 ISO 3166-1 中英文名
# -----------------------------------------------------------------------------
NODE_BIN_CANDIDATES: List[str] = [
    "/opt/homebrew/bin/node",
    "/usr/local/bin/node",
    "/usr/bin/node",
]

NODE_ISO_SCRIPT = r"""
const zh = new Intl.DisplayNames(["zh-CN"], {type: "region"});
const en = new Intl.DisplayNames(["en"], {type: "region"});
const out = {};
for (let i = 97; i <= 122; i++) {
  for (let j = 97; j <= 122; j++) {
    const code = String.fromCharCode(i) + String.fromCharCode(j);
    const upper = code.toUpperCase();
    try {
      const z = zh.of(upper);
      const e = en.of(upper);
      if (!z || !e) continue;
      if (z === upper || e === upper) continue;
      if (z === "未知地区" || e === "Unknown Region") continue;
      out[code] = {name_zh: z, name_en: e};
    } catch (err) {
      /* 未分配代码会抛 RangeError，跳过 */
    }
  }
}
process.stdout.write(JSON.stringify(out));
"""


def _resolve_node_bin() -> str:
    candidates = list(NODE_BIN_CANDIDATES)
    which_node = shutil.which("node")
    if which_node:
        candidates.insert(0, which_node)
    for cand in candidates:
        if cand and Path(cand).exists():
            return cand
    raise SystemExit(
        "[FATAL] 未找到 Node.js 可执行文件（尝试过 /opt/homebrew/bin/node 等）。"
        "本脚本依赖系统 node 的 Intl.DisplayNames 生成 ISO 3166-1 中英文名。"
    )


def load_iso_display_names() -> Tuple[Dict[str, Dict[str, str]], str]:
    """遍历 a-z 两字母组合，导出 ISO 3166-1 alpha-2 中英文名（离线）。"""
    node_bin = _resolve_node_bin()
    try:
        proc = subprocess.run(
            [node_bin, "-e", NODE_ISO_SCRIPT],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - 环境异常
        raise SystemExit(f"[FATAL] 调用 Node.js 失败：{exc}") from exc
    if proc.returncode != 0:
        raise SystemExit(f"[FATAL] Node.js 执行失败（{node_bin}）：\n{proc.stderr}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise SystemExit(f"[FATAL] 无法解析 Node.js 输出：{exc}") from exc
    if not isinstance(data, dict) or len(data) < 200:
        raise SystemExit(f"[FATAL] Node.js 导出的 ISO 名称过少（{len(data)} 条），请检查 ICU 支持。")
    return data, node_bin


# -----------------------------------------------------------------------------
# 2. 国旗：ISO 自动导出 + 非 ISO / 特例手工补录
# -----------------------------------------------------------------------------
# 2.1 高质量条目（与 engine/knowledge_expander.py 的 ISO_KNOWLEDGE_BASE 保持一致）
FLAG_OVERRIDES: Dict[str, Dict[str, str]] = {
    "cn": _flag("中华人民共和国", "People's Republic of China", "中华人民共和国国旗（五星红旗）", "National Flag of the People's Republic of China (Five-star Red Flag)", "sovereign"),
    "us": _flag("美利坚合众国", "United States of America", "美利坚合众国国旗（星条旗）", "Flag of the United States (Stars and Stripes)", "sovereign"),
    "gb": _flag("大不列颠及北爱尔兰联合王国（英国）", "United Kingdom", "联合王国国旗（米字旗）", "Flag of the United Kingdom (Union Jack)", "sovereign"),
    "fr": _flag("法兰西共和国（法国）", "French Republic", "法兰西共和国国旗（蓝白红三色旗）", "National Flag of France (Tricolour)", "sovereign"),
    "de": _flag("德意志联邦共和国（德国）", "Federal Republic of Germany", "德意志联邦共和国国旗（黑红金三色旗）", "National Flag of Germany (Black-Red-Gold Tricolour)", "sovereign"),
    "jp": _flag("日本国", "Japan", "日本国国旗（日之丸旗）", "National Flag of Japan (Hinomaru)", "sovereign"),
    "it": _flag("意大利共和国", "Italian Republic", "意大利共和国国旗（绿白红三色旗）", "National Flag of Italy (Tricolore)", "sovereign"),
    "es": _flag("西班牙王国", "Kingdom of Spain", "西班牙王国国旗（红黄双色旗）", "National Flag of Spain (La Rojigualda)", "sovereign"),
    "ca": _flag("加拿大", "Canada", "加拿大国旗（枫叶旗）", "National Flag of Canada (Maple Leaf Flag)", "sovereign"),
    "br": _flag("巴西联邦共和国", "Federative Republic of Brazil", "巴西联邦共和国国旗（金绿旗）", "National Flag of Brazil (Auriverde)", "sovereign"),
    "ru": _flag("俄罗斯联邦", "Russian Federation", "俄罗斯联邦国旗（白蓝红三色旗）", "National Flag of the Russian Federation", "sovereign"),
    "kr": _flag("大韩民国（韩国）", "Republic of Korea", "大韩民国国旗（太极旗）", "National Flag of South Korea (Taegeukgi)", "sovereign"),
    "au": _flag("澳大利亚联邦", "Commonwealth of Australia", "澳大利亚联邦国旗", "National Flag of Australia", "sovereign"),
    "in": _flag("印度共和国", "Republic of India", "印度共和国国旗（三色旗）", "National Flag of India (Tiranga)", "sovereign"),
    "za": _flag("南非共和国", "Republic of South Africa", "南非共和国国旗（彩虹之国六色旗）", "National Flag of South Africa", "sovereign"),
    "eu": _flag("欧洲联盟（欧盟）", "European Union", "欧洲联盟旗帜（十二金星蓝旗）", "Flag of the European Union", "organization"),
    "un": _flag("联合国组织", "United Nations", "联合国会旗", "Flag of the United Nations", "organization"),
    "ch": _flag("瑞士联邦", "Swiss Confederation", "瑞士联邦国旗（红底正白十字旗）", "National Flag of Switzerland", "sovereign"),
    "se": _flag("瑞典王国", "Kingdom of Sweden", "瑞典王国国旗（黄蓝北欧十字旗）", "National Flag of Sweden", "sovereign"),
    "no": _flag("挪威王国", "Kingdom of Norway", "挪威王国国旗", "National Flag of Norway", "sovereign"),
    "nl": _flag("荷兰王国", "Kingdom of the Netherlands", "荷兰王国国旗（红白蓝三色旗）", "National Flag of the Netherlands", "sovereign"),
    "ar": _flag("阿根廷共和国", "Argentine Republic", "阿根廷共和国国旗（五月太阳旗）", "National Flag of Argentina", "sovereign"),
    "mx": _flag("墨西哥合众国", "United Mexican States", "墨西哥合众国国旗", "National Flag of Mexico", "sovereign"),
    # 2.2 非 ISO 3166-1 alpha-2 代码（以 flag-icons 实际资产为准）
    "arab": _flag("阿拉伯国家联盟（阿盟）", "League of Arab States (Arab League)", "阿拉伯国家联盟旗帜", "Flag of the Arab League", "organization"),
    "asean": _flag("东南亚国家联盟（东盟）", "Association of Southeast Asian Nations (ASEAN)", "东南亚国家联盟会旗", "Flag of ASEAN", "organization"),
    "cefta": _flag("中欧自由贸易协定", "Central European Free Trade Agreement (CEFTA)", "中欧自由贸易协定旗帜", "Flag of CEFTA", "organization"),
    "eac": _flag("东非共同体", "East African Community (EAC)", "东非共同体旗帜", "Flag of the East African Community", "organization"),
    "pc": _flag("太平洋共同体", "Pacific Community (SPC)", "太平洋共同体旗帜", "Flag of the Pacific Community", "organization"),
    "es-ct": _flag("西班牙加泰罗尼亚自治区", "Catalonia, Spain", "加泰罗尼亚自治区旗帜（塞尼耶拉旗）", "Flag of Catalonia (Senyera)", "territory"),
    "es-ga": _flag("西班牙加利西亚自治区", "Galicia, Spain", "加利西亚自治区旗帜", "Flag of Galicia", "territory"),
    "es-pv": _flag("西班牙巴斯克自治区", "Basque Country, Spain", "巴斯克自治区旗帜（伊库里尼亚旗）", "Flag of the Basque Country (Ikurriña)", "territory"),
    "gb-eng": _flag("英格兰", "England", "英格兰旗帜（圣乔治十字旗）", "Flag of England (St George's Cross)", "special"),
    "gb-sct": _flag("苏格兰", "Scotland", "苏格兰旗帜（圣安德鲁十字旗）", "Flag of Scotland (Saltire)", "special"),
    "gb-wls": _flag("威尔士", "Wales", "威尔士旗帜（红龙旗）", "Flag of Wales (Red Dragon)", "special"),
    "gb-nir": _flag("北爱尔兰", "Northern Ireland", "北爱尔兰旗帜（阿尔斯特旗）", "Flag of Northern Ireland (Ulster Banner)", "special"),
    "sh-ac": _flag("阿森松岛（英国海外领地）", "Ascension Island, United Kingdom", "阿森松岛旗帜", "Flag of Ascension Island", "territory"),
    "sh-hl": _flag("圣赫勒拿（英国海外领地）", "Saint Helena, United Kingdom", "圣赫勒拿旗帜", "Flag of Saint Helena", "territory"),
    "sh-ta": _flag("特里斯坦-达库尼亚群岛（英国海外领地）", "Tristan da Cunha, United Kingdom", "特里斯坦-达库尼亚群岛旗帜", "Flag of Tristan da Cunha", "territory"),
    "xx": _flag("未知地区", "Unknown Region", "未知地区占位旗帜", "Placeholder Flag for an Unknown Region", "special"),
    # 2.3 ISO 代码的规范译名/政治表述精修
    "tw": _flag("中国台湾地区", "Taiwan Region of China", "中国台湾地区旗帜", "Flag of the Taiwan Region of China", "territory"),
    "hk": _flag("中国香港特别行政区", "Hong Kong SAR, China", "中国香港特别行政区区旗", "Regional Flag of the Hong Kong SAR of China", "territory"),
    "mo": _flag("中国澳门特别行政区", "Macao SAR, China", "中国澳门特别行政区区旗", "Regional Flag of the Macao SAR of China", "territory"),
    "ps": _flag("巴勒斯坦国", "State of Palestine", "巴勒斯坦国国旗", "National Flag of the State of Palestine", "sovereign"),
    "cd": _flag("刚果民主共和国", "Democratic Republic of the Congo", "刚果民主共和国国旗", "National Flag of the Democratic Republic of the Congo", "sovereign"),
    "cg": _flag("刚果共和国", "Republic of the Congo", "刚果共和国国旗", "National Flag of the Republic of the Congo", "sovereign"),
    "fm": _flag("密克罗尼西亚联邦", "Federated States of Micronesia", "密克罗尼西亚联邦国旗", "National Flag of the Federated States of Micronesia", "sovereign"),
    "va": _flag("梵蒂冈城国", "Vatican City State", "梵蒂冈城国国旗", "National Flag of Vatican City State", "sovereign"),
    "eh": _flag("西撒哈拉（阿拉伯撒哈拉民主共和国）", "Western Sahara (Sahrawi Arab Democratic Republic)", "西撒哈拉旗帜", "Flag of Western Sahara", "special"),
    "aq": _flag("南极洲", "Antarctica", "南极洲非官方旗帜", "Unofficial Flag of Antarctica", "special"),
    "xk": _flag("科索沃地区", "Kosovo", "科索沃地区旗帜", "Flag of Kosovo", "territory"),
    "ic": _flag("西班牙加那利群岛自治区", "Canary Islands, Spain", "加那利群岛自治区旗帜", "Flag of the Canary Islands", "territory"),
    "dg": _flag("迪戈加西亚岛（英属印度洋领地）", "Diego Garcia, British Indian Ocean Territory", "迪戈加西亚岛旗帜", "Flag of Diego Garcia", "territory"),
    "cp": _flag("克利珀顿岛（法属）", "Clipperton Island, France", "克利珀顿岛旗帜", "Flag of Clipperton Island", "territory"),
    "um": _flag("美国本土外小岛屿", "U.S. Outlying Islands", "美国本土外小岛屿旗帜", "Flag of the U.S. Outlying Islands", "territory"),
}

# 未列入 overrides 的 ISO 代码：主权国家 / 属地分类
NON_SOVEREIGN_FLAG_CODES = {
    "ai", "as", "aw", "ax", "bl", "bm", "bq", "bv", "cc", "ck", "cw", "cx",
    "fk", "fo", "gf", "gg", "gi", "gl", "gp", "gs", "gu", "hm", "im", "io",
    "je", "ky", "mf", "mp", "mq", "ms", "nc", "nf", "nu", "pf", "pm", "pn",
    "pr", "re", "sh", "sj", "sx", "tc", "tf", "tk", "vg", "vi", "wf", "yt",
}


def build_official_flags(iso: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    flag_dir = ASSETS_DIR / "official_flags"
    stems = sorted(p.stem for p in flag_dir.glob("*.svg"))
    if not stems:
        raise SystemExit(f"[FATAL] 未在 {flag_dir} 找到任何国旗 SVG。")

    table: Dict[str, Dict[str, str]] = {}
    for stem in stems:
        if stem in FLAG_OVERRIDES:
            table[stem] = dict(FLAG_OVERRIDES[stem])
            continue
        meta = iso.get(stem)
        if meta is None:
            raise SystemExit(
                f"[FATAL] 国旗 stem '{stem}' 既不在手工补录表，也不是可识别的 ISO 3166-1 代码。"
            )
        name_zh = meta["name_zh"]
        name_en = meta["name_en"]
        if stem in NON_SOVEREIGN_FLAG_CODES:
            table[stem] = _flag(name_zh, name_en, f"{name_zh}旗帜", f"Flag of {name_en}", "territory")
        else:
            table[stem] = _flag(name_zh, name_en, f"{name_zh}国旗", f"National Flag of {name_en}", "sovereign")

    # 覆盖校验：目录里每一个文件都必须有中文条目
    missing = [s for s in stems if s not in table]
    if missing:
        raise SystemExit(f"[FATAL] 国旗覆盖缺失 {len(missing)} 条：{missing}")
    return table


# -----------------------------------------------------------------------------
# 3. 国徽：按国家英文名 token 匹配 + 特殊徽章手工补录
# -----------------------------------------------------------------------------
EMBLEM_OVERRIDES: Dict[str, Dict[str, str]] = {
    # 国际组织 / 航天机构 / 特殊徽记
    "badge_nasa_meatball_insignia": _emblem("美国国家航空航天局（NASA）", "National Aeronautics and Space Administration (NASA)", "美国国家航空航天局（NASA）肉丸徽标", "NASA Meatball Insignia", org=True),
    "badge_olympic_rings": _emblem("国际奥林匹克委员会", "International Olympic Committee", "奥林匹克五环标志", "The Olympic Rings", org=True),
    "badge_united_nations": _emblem("联合国", "United Nations", "联合国徽记（橄榄枝世界地图）", "Emblem of the United Nations", org=True),
    "emblem_sede_vacante": _emblem("梵蒂冈城国（圣座）", "Vatican City (Holy See)", "宗座空缺期教廷徽记", "Sede Vacante Coat of Arms of the Holy See", org=True),
    "emblem_somaliland": _emblem("索马里兰", "Somaliland", "索马里兰地区徽记", "Emblem of Somaliland"),
    "emblem_the_republic_of_china": _emblem("中国台湾地区", "Taiwan Region of China", "中国台湾地区徽记", "Emblem of the Taiwan Region of China"),
    # 特殊图案（国旗局部 / 徽章局部）
    "cn_stars": _emblem("中华人民共和国", "People's Republic of China", "中华人民共和国五星红旗（五星图案）", "Five-Star Pattern of the National Flag of the People's Republic of China"),
    "es_shield": _emblem("西班牙王国", "Kingdom of Spain", "西班牙王国国徽盾徽图案", "Coat of Arms Shield of the Kingdom of Spain"),
    "eu_stars": _emblem("欧洲联盟（欧盟）", "European Union", "欧洲联盟盟徽（十二金星环）", "Emblem of the European Union (Circle of Twelve Stars)", org=True),
    "fr_diplomatic": _emblem("法兰西共和国（法国）", "French Republic", "法兰西共和国外交徽记", "Diplomatic Emblem of France"),
    "mx_eagle": _emblem("墨西哥合众国", "United Mexican States", "墨西哥合众国国徽雄鹰图案", "Eagle Motif of the Coat of Arms of Mexico"),
    "ru_double_eagle": _emblem("俄罗斯联邦", "Russian Federation", "俄罗斯联邦国徽（双头鹰）", "Coat of Arms of Russia (Double-headed Eagle)"),
    "us_great_seal": _emblem("美利坚合众国（美国）", "United States of America", "美国国徽（大印章正面）", "Great Seal of the United States (Obverse)"),
    # 同国多资产：用细分标题区分
    "emblem_brazil": _emblem("巴西联邦共和国", "Federative Republic of Brazil", "巴西联邦共和国国徽", "Coat of Arms of Brazil"),
    "emblem_national_seal_of_brazil__color_": _emblem("巴西联邦共和国", "Federative Republic of Brazil", "巴西国家印章（彩色版）", "National Seal of Brazil (Color Version)"),
    "emblem_bangladesh": _emblem("孟加拉人民共和国", "People's Republic of Bangladesh", "孟加拉国国徽", "National Emblem of Bangladesh"),
    "emblem_government_seal_of_bangladesh": _emblem("孟加拉人民共和国", "People's Republic of Bangladesh", "孟加拉国政府印章（国徽）", "Government Seal of Bangladesh (National Emblem)"),
    "emblem_mexico": _emblem("墨西哥合众国", "United Mexican States", "墨西哥合众国国徽（雄鹰叼蛇）", "Coat of Arms of Mexico"),
    "emblem_the_united_states__reverse_": _emblem("美利坚合众国（美国）", "United States of America", "美国国徽（大印章背面）", "Great Seal of the United States (Reverse)"),
    "emblem_the_united_kingdom__2022__variant_2_": _emblem("大不列颠及北爱尔兰联合王国（英国）", "United Kingdom", "英国国徽（2022 年版）", "Royal Coat of Arms of the United Kingdom (2022 Version)"),
    # 语义化精修标题
    "emblem_peoples_republic_of_china": _emblem("中华人民共和国", "People's Republic of China", "中华人民共和国国徽", "National Emblem of the People's Republic of China"),
    "emblem_germany": _emblem("德意志联邦共和国（德国）", "Federal Republic of Germany", "德意志联邦共和国国徽（联邦之鹰）", "Coat of Arms of Germany (Bundesadler)"),
    "emblem_imperial_seal_of_japan": _emblem("日本国", "Japan", "日本皇室菊花纹章（十六瓣八重表菊纹）", "Imperial Seal of Japan (Chrysanthemum Seal)"),
    "emblem_indonesia_garuda_pancasila": _emblem("印度尼西亚共和国", "Republic of Indonesia", "印度尼西亚国徽（金翅鸟潘查希拉）", "National Emblem of Indonesia (Garuda Pancasila)"),
    "emblem_qatar-2022": _emblem("卡塔尔国", "State of Qatar", "卡塔尔国国徽（2022 年版）", "Emblem of Qatar (2022 Version)"),
    "emblem_palestine": _emblem("巴勒斯坦国", "State of Palestine", "巴勒斯坦国国徽", "Coat of Arms of Palestine"),
    "emblem_egypt__official_": _emblem("阿拉伯埃及共和国", "Arab Republic of Egypt", "阿拉伯埃及共和国国徽（萨拉丁之鹰）", "Coat of Arms of Egypt"),
    # 语言干扰词 / 词形变体（法语、西语、德语、意大利语）
    "emblem_armoiries_du_togo": _emblem("多哥共和国", "Togolese Republic", "多哥共和国国徽", "Coat of Arms of Togo"),
    "emblem_austria_bundesadler": _emblem("奥地利共和国", "Republic of Austria", "奥地利共和国国徽（联邦之鹰）", "Coat of Arms of Austria (Bundesadler)"),
    "emblem_escudo_de_espa_a__mazonado_": _emblem("西班牙王国", "Kingdom of Spain", "西班牙王国国徽（盾徽版）", "Coat of Arms of Spain (Escudo de España)"),
    "emblem_escudo_nacional_del_per_": _emblem("秘鲁共和国", "Republic of Peru", "秘鲁共和国国徽（国家盾徽）", "National Coat of Arms of Peru"),
    "emblem_staatswappen-liechtensteins": _emblem("列支敦士登公国", "Principality of Liechtenstein", "列支敦士登公国国徽", "Coat of Arms of Liechtenstein"),
    "emblem_east_timor": _emblem("东帝汶民主共和国", "Democratic Republic of Timor-Leste", "东帝汶民主共和国国徽", "National Emblem of Timor-Leste"),
}

# 英文名 token 词形别名（用于西班牙语/法语/德语资产名中的变体）
EMBLEM_TOKEN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "espa": ("spain",),
    "espana": ("spain",),
    "espagne": ("spain",),
    "spanien": ("spain",),
    "per": ("peru",),
    "liechtensteins": ("liechtenstein",),
}

EMBLEM_NOISE_TOKENS = {
    "the", "of", "and", "or", "a", "an", "du", "de", "la", "le", "les", "des",
    "del", "della", "di", "da", "das", "der", "die", "dos", "do",
}


def _tokens(text: str) -> List[str]:
    """小写 + 去变音符号 + 按非字母数字切分（Réunion→reunion, Côte→cote, Türkiye→turkiye）。"""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return [tok for tok in re.split(r"[^a-z0-9]+", ascii_text.lower()) if tok]


def _country_tokens(name_en: str) -> set:
    """国家英文名 token 集合；先剥离括号别名（如 'Myanmar (Burma)'、'Congo - Kinshasa' 后缀说明）。"""
    primary = re.sub(r"[（(][^）)]*[）)]", " ", name_en)
    return {tok for tok in _tokens(primary) if tok not in EMBLEM_NOISE_TOKENS and len(tok) >= 2}


def build_official_emblems(
    iso: Dict[str, Dict[str, str]],
    allowed_codes: Iterable[str],
) -> Dict[str, Dict[str, str]]:
    emblem_dir = ASSETS_DIR / "official_emblems"
    stems = sorted(p.stem for p in emblem_dir.glob("*.svg"))
    if not stems:
        raise SystemExit(f"[FATAL] 未在 {emblem_dir} 找到任何国徽 SVG。")

    # 候选国家索引：只使用国旗目录实际存在的现行 ISO 3166-1 两字母代码，
    # 避免 CLDR 中的历史代码（hv=上沃尔特、cs=塞尔维亚和黑山、vd=北越、yd=南也门等）造成歧义。
    current_codes = {code for code in allowed_codes if len(code) == 2 and code.isalpha()}
    country_index: List[Tuple[str, str, str, set]] = []
    for code, meta in iso.items():
        if code not in current_codes:
            continue
        tokens = _country_tokens(meta["name_en"])
        if tokens:
            country_index.append((code, meta["name_zh"], meta["name_en"], tokens))

    table: Dict[str, Dict[str, str]] = {}
    unmatched: List[str] = []
    ambiguous: List[str] = []

    for stem in stems:
        if stem in EMBLEM_OVERRIDES:
            table[stem] = dict(EMBLEM_OVERRIDES[stem])
            continue
        body = stem
        for prefix in ("emblem_", "badge_"):
            if body.startswith(prefix):
                body = body[len(prefix):]
                break
        stem_tokens = set(_tokens(body))
        expanded = set(stem_tokens)
        for tok in stem_tokens:
            expanded.update(EMBLEM_TOKEN_ALIASES.get(tok, ()))
        candidates = [
            (len(tokens), code, name_zh, name_en)
            for code, name_zh, name_en, tokens in country_index
            if tokens <= expanded
        ]
        if not candidates:
            unmatched.append(stem)
            continue
        candidates.sort(key=lambda item: (-item[0], item[1]))
        top_len = candidates[0][0]
        top_codes = {item[1] for item in candidates if item[0] == top_len}
        if len(top_codes) > 1:
            ambiguous.append(f"{stem} -> {sorted(top_codes)}")
            continue
        _, _code, name_zh, name_en = candidates[0]
        table[stem] = _emblem(name_zh, name_en, f"{name_zh}国徽", f"National Emblem of {name_en}")

    if unmatched or ambiguous:
        raise SystemExit(
            "[FATAL] 国徽自动匹配失败，请补充手工条目：\n"
            f"  未匹配: {unmatched}\n  有歧义: {ambiguous}"
        )
    missing = [s for s in stems if s not in table]
    if missing:
        raise SystemExit(f"[FATAL] 国徽覆盖缺失 {len(missing)} 条：{missing}")
    return table


# -----------------------------------------------------------------------------
# 4. 地方旗：190 条人工映射（以 assets/official_regional 实际 stem 为准）
# -----------------------------------------------------------------------------
REGIONAL_GROUPS: Dict[str, Dict[str, Any]] = {
    "au": {
        "default_type": "state",
        "items": [
            ("australian_capital_territory", "澳大利亚首都领地", "Australian Capital Territory, Australia", "澳大利亚首都领地旗帜", "Flag of the Australian Capital Territory"),
            ("new_south_wales", "澳大利亚新南威尔士州", "New South Wales, Australia", "新南威尔士州州旗", "Flag of New South Wales"),
            ("norfolk_island", "澳大利亚诺福克岛", "Norfolk Island, Australia", "诺福克岛旗帜", "Flag of Norfolk Island", "territory"),
            ("northern_territory", "澳大利亚北领地", "Northern Territory, Australia", "北领地旗帜", "Flag of the Northern Territory", "territory"),
            ("queensland", "澳大利亚昆士兰州", "Queensland, Australia", "昆士兰州州旗", "Flag of Queensland"),
            ("south_australia", "澳大利亚南澳大利亚州", "South Australia, Australia", "南澳大利亚州州旗", "Flag of South Australia"),
            ("tasmania", "澳大利亚塔斯马尼亚州", "Tasmania, Australia", "塔斯马尼亚州州旗", "Flag of Tasmania"),
            ("victoria__australia_", "澳大利亚维多利亚州", "Victoria, Australia", "维多利亚州州旗", "Flag of Victoria"),
            ("western_australia", "澳大利亚西澳大利亚州", "Western Australia, Australia", "西澳大利亚州州旗", "Flag of Western Australia"),
        ],
    },
    "ca": {
        "default_type": "state",
        "items": [
            ("alberta", "加拿大阿尔伯塔省", "Alberta, Canada", "阿尔伯塔省省旗", "Flag of Alberta"),
            ("british_columbia", "加拿大不列颠哥伦比亚省", "British Columbia, Canada", "不列颠哥伦比亚省省旗", "Flag of British Columbia"),
            ("manitoba", "加拿大曼尼托巴省", "Manitoba, Canada", "曼尼托巴省省旗", "Flag of Manitoba"),
            ("new_brunswick", "加拿大新不伦瑞克省", "New Brunswick, Canada", "新不伦瑞克省省旗", "Flag of New Brunswick"),
            ("newfoundland_and_labrador", "加拿大纽芬兰与拉布拉多省", "Newfoundland and Labrador, Canada", "纽芬兰与拉布拉多省省旗", "Flag of Newfoundland and Labrador"),
            ("northwest_territories", "加拿大西北地区", "Northwest Territories, Canada", "西北地区区旗", "Flag of the Northwest Territories", "territory"),
            ("nova_scotia", "加拿大新斯科舍省", "Nova Scotia, Canada", "新斯科舍省省旗", "Flag of Nova Scotia"),
            ("nunavut", "加拿大努纳武特地区", "Nunavut, Canada", "努纳武特地区区旗", "Flag of Nunavut", "territory"),
            ("ontario", "加拿大安大略省", "Ontario, Canada", "安大略省省旗", "Flag of Ontario"),
            ("prince_edward_island", "加拿大爱德华王子岛省", "Prince Edward Island, Canada", "爱德华王子岛省省旗", "Flag of Prince Edward Island"),
            ("quebec", "加拿大魁北克省", "Quebec, Canada", "魁北克省省旗", "Flag of Quebec"),
            ("saskatchewan", "加拿大萨斯喀彻温省", "Saskatchewan, Canada", "萨斯喀彻温省省旗", "Flag of Saskatchewan"),
        ],
    },
    "de": {
        "default_type": "state",
        "items": [
            ("bavaria__lozengy_", "德国巴伐利亚自由邦", "Free State of Bavaria, Germany", "巴伐利亚自由邦菱格州旗", "Flag of Bavaria (Lozengy Variant)"),
            ("bavaria__striped_", "德国巴伐利亚自由邦", "Free State of Bavaria, Germany", "巴伐利亚自由邦条纹州旗", "Flag of Bavaria (Striped Variant)"),
            ("berlin", "德国柏林州", "Berlin, Germany", "柏林州州旗", "Flag of Berlin"),
            ("brandenburg", "德国勃兰登堡州", "Brandenburg, Germany", "勃兰登堡州州旗", "Flag of Brandenburg"),
            ("bremen", "德国不来梅州", "Bremen, Germany", "不来梅州民用旗（红白条纹旗）", "Civil Flag of Bremen"),
            ("civil_flag_of_prussia_1701-1935", "普鲁士王国（历史政区）", "Kingdom of Prussia (Historical)", "普鲁士王国历史民用旗（1701–1935）", "Historical Civil Flag of Prussia (1701–1935)", "historic"),
            ("germany__3-2_", "德意志联邦共和国（德国）", "Federal Republic of Germany", "德国国旗（3:2 比例版）", "National Flag of Germany (3:2 Variant)", "national_variant"),
            ("hamburg", "德国汉堡州", "Hamburg, Germany", "汉堡州州旗", "Flag of Hamburg"),
            ("hesse", "德国黑森州", "Hesse, Germany", "黑森州民用旗（红白双色旗）", "Civil Flag of Hesse"),
            ("hesse__state_", "德国黑森州", "Hesse, Germany", "黑森州州旗（带州徽版）", "State Service Flag of Hesse"),
            ("lower_saxony", "德国下萨克森州", "Lower Saxony, Germany", "下萨克森州州旗", "Flag of Lower Saxony"),
            ("mecklenburg-western_pomerania__state_", "德国梅克伦堡-前波美拉尼亚州", "Mecklenburg-Western Pomerania, Germany", "梅克伦堡-前波美拉尼亚州州旗", "Flag of Mecklenburg-Western Pomerania"),
            ("north_rhine-westphalia", "德国北莱茵-威斯特法伦州", "North Rhine-Westphalia, Germany", "北莱茵-威斯特法伦州民用旗", "Civil Flag of North Rhine-Westphalia"),
            ("north_rhine-westphalia__state_", "德国北莱茵-威斯特法伦州", "North Rhine-Westphalia, Germany", "北莱茵-威斯特法伦州州旗（带州徽版）", "State Service Flag of North Rhine-Westphalia"),
            ("prussia__1892-1918_", "普鲁士王国（历史政区）", "Kingdom of Prussia (Historical)", "普鲁士王国国旗（1892–1918）", "Flag of the Kingdom of Prussia (1892–1918)", "historic"),
            ("prussia__1918_1933_", "普鲁士自由邦（历史政区）", "Free State of Prussia (Historical)", "普鲁士自由邦旗帜（1918–1933）", "Flag of the Free State of Prussia (1918–1933)", "historic"),
            ("rhineland-palatinate", "德国莱茵兰-普法尔茨州", "Rhineland-Palatinate, Germany", "莱茵兰-普法尔茨州州旗", "Flag of Rhineland-Palatinate"),
            ("saxony-anhalt__state_", "德国萨克森-安哈尔特州", "Saxony-Anhalt, Germany", "萨克森-安哈尔特州州旗", "Flag of Saxony-Anhalt"),
            ("saxony__state_", "德国萨克森州", "Saxony, Germany", "萨克森州州旗", "Flag of Saxony"),
            ("schleswig-holstein", "德国石勒苏益格-荷尔斯泰因州", "Schleswig-Holstein, Germany", "石勒苏益格-荷尔斯泰因州民用旗", "Civil Flag of Schleswig-Holstein"),
            ("schleswig-holstein__state_", "德国石勒苏益格-荷尔斯泰因州", "Schleswig-Holstein, Germany", "石勒苏益格-荷尔斯泰因州州旗（带州徽版）", "State Service Flag of Schleswig-Holstein"),
            ("sorbs", "德国索布族（文德族）", "Sorbs (Wends), Germany", "索布族民族旗帜", "Flag of the Sorbs", "historic"),
            ("state_flag_of_bremen", "德国不来梅州", "Bremen, Germany", "不来梅州州旗（带市徽钥匙版）", "State Flag of Bremen"),
            ("the_free_city_of_l_beck", "吕贝克自由市（历史汉萨城市）", "Free City of Lübeck (Historical)", "吕贝克自由市旗帜", "Flag of the Free City of Lübeck", "historic"),
            ("thuringia", "德国图林根州", "Thuringia, Germany", "图林根州民用旗（红白双色旗）", "Civil Flag of Thuringia"),
            ("thuringia__state_", "德国图林根州", "Thuringia, Germany", "图林根州州旗（带州徽版）", "State Service Flag of Thuringia"),
        ],
    },
    "es": {
        "default_type": "state",
        "items": [
            ("andaluc_a", "西班牙安达卢西亚自治区", "Andalusia, Spain", "安达卢西亚自治区区旗", "Flag of Andalusia"),
            ("aragon", "西班牙阿拉贡自治区", "Aragon, Spain", "阿拉贡自治区区旗", "Flag of Aragon"),
            ("asturias", "西班牙阿斯图里亚斯自治区", "Asturias, Spain", "阿斯图里亚斯自治区区旗", "Flag of Asturias"),
            ("balearic_islands", "西班牙巴利阿里群岛自治区", "Balearic Islands, Spain", "巴利阿里群岛自治区区旗", "Flag of the Balearic Islands"),
            ("basque_country", "西班牙巴斯克自治区", "Basque Country, Spain", "巴斯克自治区区旗（伊库里尼亚旗）", "Flag of the Basque Country (Ikurriña)"),
            ("canary_islands", "西班牙加那利群岛自治区", "Canary Islands, Spain", "加那利群岛自治区区旗", "Flag of the Canary Islands"),
            ("catalonia", "西班牙加泰罗尼亚自治区", "Catalonia, Spain", "加泰罗尼亚自治区区旗（塞尼耶拉旗）", "Flag of Catalonia (Senyera)"),
            ("ceuta", "西班牙休达自治市", "Ceuta, Spain", "休达自治市市旗", "Flag of Ceuta", "special"),
            ("community_of_madrid", "西班牙马德里自治区", "Community of Madrid, Spain", "马德里自治区区旗", "Flag of the Community of Madrid"),
            ("galicia", "西班牙加利西亚自治区", "Galicia, Spain", "加利西亚自治区区旗", "Flag of Galicia"),
            ("region_of_murcia", "西班牙穆尔西亚自治区", "Region of Murcia, Spain", "穆尔西亚自治区区旗", "Flag of the Region of Murcia"),
            ("valencian_community__2x3_", "西班牙巴伦西亚自治区", "Valencian Community, Spain", "巴伦西亚自治区区旗", "Flag of the Valencian Community"),
        ],
    },
    "fr": {
        "default_type": "state",
        "items": [
            ("brittany__gwenn_ha_du_", "法国布列塔尼大区", "Brittany, France", "布列塔尼大区区旗（格温哈杜旗）", "Flag of Brittany (Gwenn-ha-du)"),
            ("centre-val_de_loire", "法国中央-卢瓦尔河谷大区", "Centre-Val de Loire, France", "中央-卢瓦尔河谷大区区旗", "Flag of Centre-Val de Loire"),
            ("corsica", "法国科西嘉大区", "Corsica, France", "科西嘉大区区旗（摩尔人头像旗）", "Flag of Corsica (Moor's Head)"),
            ("hauts-de-france", "法国上法兰西大区", "Hauts-de-France, France", "上法兰西大区区旗", "Flag of Hauts-de-France"),
            ("region_auvergne-rh_ne-alpes", "法国奥弗涅-罗讷-阿尔卑斯大区", "Auvergne-Rhône-Alpes, France", "奥弗涅-罗讷-阿尔卑斯大区区旗", "Flag of Auvergne-Rhône-Alpes"),
        ],
    },
    "gb": {
        "default_type": "constituent",
        "items": [
            ("bermuda", "英国百慕大群岛（海外领地）", "Bermuda, United Kingdom", "百慕大群岛旗帜", "Flag of Bermuda", "territory"),
            ("cayman_islands", "英国开曼群岛（海外领地）", "Cayman Islands, United Kingdom", "开曼群岛旗帜", "Flag of the Cayman Islands", "territory"),
            ("falkland_islands", "英国福克兰群岛（海外领地）", "Falkland Islands, United Kingdom", "福克兰群岛旗帜", "Flag of the Falkland Islands", "territory"),
            ("gibraltar", "英国直布罗陀（海外领地）", "Gibraltar, United Kingdom", "直布罗陀旗帜", "Flag of Gibraltar", "territory"),
            ("guernsey", "英国根西岛（王室属地）", "Guernsey, United Kingdom", "根西岛旗帜", "Flag of Guernsey", "territory"),
            ("isle_of_mann", "英国马恩岛（王室属地）", "Isle of Man, United Kingdom", "马恩岛旗帜", "Flag of the Isle of Man", "territory"),
            ("jersey", "英国泽西岛（王室属地）", "Jersey, United Kingdom", "泽西岛旗帜", "Flag of Jersey", "territory"),
            ("northern_ireland__1953_1972_", "英国北爱尔兰", "Northern Ireland, United Kingdom", "北爱尔兰阿尔斯特旗（1953–1972）", "Ulster Banner of Northern Ireland (1953–1972)", "constituent"),
            ("scotland", "英国苏格兰", "Scotland, United Kingdom", "苏格兰旗帜（圣安德鲁十字旗）", "Flag of Scotland (Saltire)", "constituent"),
            ("wales", "英国威尔士", "Wales, United Kingdom", "威尔士旗帜（红龙旗）", "Flag of Wales (Red Dragon)", "constituent"),
        ],
    },
    "it": {
        "default_type": "state",
        "items": [
            ("abruzzo", "意大利阿布鲁佐大区", "Abruzzo, Italy", "阿布鲁佐大区区旗", "Flag of Abruzzo"),
            ("apulia", "意大利普利亚大区", "Apulia, Italy", "普利亚大区区旗", "Flag of Apulia"),
            ("basilicata", "意大利巴西利卡塔大区", "Basilicata, Italy", "巴西利卡塔大区区旗", "Flag of Basilicata"),
            ("campania", "意大利坎帕尼亚大区", "Campania, Italy", "坎帕尼亚大区区旗", "Flag of Campania"),
            ("emilia-romagna__de_facto_", "意大利艾米利亚-罗马涅大区", "Emilia-Romagna, Italy", "艾米利亚-罗马涅大区区旗（事实使用版）", "Flag of Emilia-Romagna (De Facto)"),
            ("friuli-venezia_giulia", "意大利弗留利-威尼斯朱利亚大区", "Friuli-Venezia Giulia, Italy", "弗留利-威尼斯朱利亚大区区旗", "Flag of Friuli-Venezia Giulia"),
            ("lombardy", "意大利伦巴第大区", "Lombardy, Italy", "伦巴第大区区旗", "Flag of Lombardy"),
            ("marche", "意大利马尔凯大区", "Marche, Italy", "马尔凯大区区旗", "Flag of Marche"),
            ("molise", "意大利莫利塞大区", "Molise, Italy", "莫利塞大区区旗", "Flag of Molise"),
            ("piedmont", "意大利皮埃蒙特大区", "Piedmont, Italy", "皮埃蒙特大区区旗", "Flag of Piedmont"),
            ("sardinia__italy", "意大利撒丁尼亚大区", "Sardinia, Italy", "撒丁尼亚大区区旗", "Flag of Sardinia"),
            ("sicily", "意大利西西里大区", "Sicily, Italy", "西西里大区区旗", "Flag of Sicily"),
            ("tuscany", "意大利托斯卡纳大区", "Tuscany, Italy", "托斯卡纳大区区旗", "Flag of Tuscany"),
            ("umbria", "意大利翁布里亚大区", "Umbria, Italy", "翁布里亚大区区旗", "Flag of Umbria"),
        ],
    },
    "jp": {
        "default_type": "state",
        "items": [
            ("aichi_prefecture", "日本爱知县", "Aichi Prefecture, Japan", "爱知县县旗", "Flag of Aichi Prefecture"),
            ("akita_prefecture", "日本秋田县", "Akita Prefecture, Japan", "秋田县县旗", "Flag of Akita Prefecture"),
            ("aomori_prefecture", "日本青森县", "Aomori Prefecture, Japan", "青森县县旗", "Flag of Aomori Prefecture"),
            ("chiba_prefecture", "日本千叶县", "Chiba Prefecture, Japan", "千叶县县旗", "Flag of Chiba Prefecture"),
            ("ehime_prefecture", "日本爱媛县", "Ehime Prefecture, Japan", "爱媛县县旗", "Flag of Ehime Prefecture"),
            ("fukui_prefecture", "日本福井县", "Fukui Prefecture, Japan", "福井县县旗", "Flag of Fukui Prefecture"),
            ("fukuoka_prefecture", "日本福冈县", "Fukuoka Prefecture, Japan", "福冈县县旗", "Flag of Fukuoka Prefecture"),
            ("fukushima_prefecture", "日本福岛县", "Fukushima Prefecture, Japan", "福岛县县旗", "Flag of Fukushima Prefecture"),
            ("gunma_prefecture", "日本群马县", "Gunma Prefecture, Japan", "群马县县旗", "Flag of Gunma Prefecture"),
            ("hiroshima_prefecture", "日本广岛县", "Hiroshima Prefecture, Japan", "广岛县县旗", "Flag of Hiroshima Prefecture"),
            ("hokkaido_prefecture", "日本北海道", "Hokkaido, Japan", "北海道道旗", "Flag of Hokkaido"),
            ("hyogo_prefecture", "日本兵库县", "Hyogo Prefecture, Japan", "兵库县县旗", "Flag of Hyogo Prefecture"),
            ("ibaraki_prefecture", "日本茨城县", "Ibaraki Prefecture, Japan", "茨城县县旗", "Flag of Ibaraki Prefecture"),
            ("ishikawa_prefecture", "日本石川县", "Ishikawa Prefecture, Japan", "石川县县旗", "Flag of Ishikawa Prefecture"),
            ("iwate_prefecture", "日本岩手县", "Iwate Prefecture, Japan", "岩手县县旗", "Flag of Iwate Prefecture"),
            ("kagawa_prefecture", "日本香川县", "Kagawa Prefecture, Japan", "香川县县旗", "Flag of Kagawa Prefecture"),
            ("kagoshima_prefecture", "日本鹿儿岛县", "Kagoshima Prefecture, Japan", "鹿儿岛县县旗", "Flag of Kagoshima Prefecture"),
            ("kanagawa_prefecture", "日本神奈川县", "Kanagawa Prefecture, Japan", "神奈川县县旗", "Flag of Kanagawa Prefecture"),
            ("kochi_prefecture", "日本高知县", "Kochi Prefecture, Japan", "高知县县旗", "Flag of Kochi Prefecture"),
            ("kumamoto_prefecture", "日本熊本县", "Kumamoto Prefecture, Japan", "熊本县县旗", "Flag of Kumamoto Prefecture"),
            ("kyoto_prefecture", "日本京都府", "Kyoto Prefecture, Japan", "京都府府旗", "Flag of Kyoto Prefecture"),
            ("mie_prefecture", "日本三重县", "Mie Prefecture, Japan", "三重县县旗", "Flag of Mie Prefecture"),
            ("miyagi_prefecture", "日本宫城县", "Miyagi Prefecture, Japan", "宫城县县旗", "Flag of Miyagi Prefecture"),
            ("miyazaki_prefecture", "日本宫崎县", "Miyazaki Prefecture, Japan", "宫崎县县旗", "Flag of Miyazaki Prefecture"),
            ("nagano_prefecture", "日本长野县", "Nagano Prefecture, Japan", "长野县县旗", "Flag of Nagano Prefecture"),
            ("nagasaki_prefecture", "日本长崎县", "Nagasaki Prefecture, Japan", "长崎县县旗", "Flag of Nagasaki Prefecture"),
            ("nara_prefecture", "日本奈良县", "Nara Prefecture, Japan", "奈良县县旗", "Flag of Nara Prefecture"),
            ("niigata_prefecture", "日本新潟县", "Niigata Prefecture, Japan", "新潟县县旗", "Flag of Niigata Prefecture"),
            ("oita_prefecture", "日本大分县", "Oita Prefecture, Japan", "大分县县旗", "Flag of Oita Prefecture"),
            ("okayama_prefecture", "日本冈山县", "Okayama Prefecture, Japan", "冈山县县旗", "Flag of Okayama Prefecture"),
            ("okinawa_prefecture", "日本冲绳县", "Okinawa Prefecture, Japan", "冲绳县县旗", "Flag of Okinawa Prefecture"),
            ("osaka_prefecture", "日本大阪府", "Osaka Prefecture, Japan", "大阪府府旗", "Flag of Osaka Prefecture"),
            ("saga_prefecture", "日本佐贺县", "Saga Prefecture, Japan", "佐贺县县旗", "Flag of Saga Prefecture"),
            ("saitama_prefecture", "日本埼玉县", "Saitama Prefecture, Japan", "埼玉县县旗", "Flag of Saitama Prefecture"),
            ("shiga_prefecture", "日本滋贺县", "Shiga Prefecture, Japan", "滋贺县县旗", "Flag of Shiga Prefecture"),
            ("shimane_prefecture", "日本岛根县", "Shimane Prefecture, Japan", "岛根县县旗", "Flag of Shimane Prefecture"),
            ("shizuoka_prefecture", "日本静冈县", "Shizuoka Prefecture, Japan", "静冈县县旗", "Flag of Shizuoka Prefecture"),
            ("tochigi_prefecture", "日本栃木县", "Tochigi Prefecture, Japan", "栃木县县旗", "Flag of Tochigi Prefecture"),
            ("tokushima_prefecture", "日本德岛县", "Tokushima Prefecture, Japan", "德岛县县旗", "Flag of Tokushima Prefecture"),
            ("tokyo_metropolis", "日本东京都", "Tokyo Metropolis, Japan", "东京都都旗（江户紫太阳徽）", "Flag of Tokyo Metropolis"),
            ("tokyo_metropolis__symbol_", "日本东京都", "Tokyo Metropolis, Japan", "东京都都徽旗（白底紫徽版）", "Symbol Flag of Tokyo Metropolis"),
            ("toyama_prefecture", "日本富山县", "Toyama Prefecture, Japan", "富山县县旗", "Flag of Toyama Prefecture"),
            ("wakayama_prefecture", "日本和歌山县", "Wakayama Prefecture, Japan", "和歌山县县旗", "Flag of Wakayama Prefecture"),
            ("yamagata_prefecture", "日本山形县", "Yamagata Prefecture, Japan", "山形县县旗", "Flag of Yamagata Prefecture"),
            ("yamaguchi_prefecture", "日本山口县", "Yamaguchi Prefecture, Japan", "山口县县旗", "Flag of Yamaguchi Prefecture"),
            ("yamanashi_prefecture", "日本山梨县", "Yamanashi Prefecture, Japan", "山梨县县旗", "Flag of Yamanashi Prefecture"),
        ],
    },
    "us": {
        "default_type": "state",
        "items": [
            ("alabama", "美国亚拉巴马州", "Alabama, United States", "亚拉巴马州州旗", "Flag of Alabama"),
            ("alaska", "美国阿拉斯加州", "Alaska, United States", "阿拉斯加州州旗", "Flag of Alaska"),
            ("american_samoa", "美国美属萨摩亚", "American Samoa, United States", "美属萨摩亚旗帜", "Flag of American Samoa", "territory"),
            ("arizona", "美国亚利桑那州", "Arizona, United States", "亚利桑那州州旗", "Flag of Arizona"),
            ("arkansas", "美国阿肯色州", "Arkansas, United States", "阿肯色州州旗", "Flag of Arkansas"),
            ("california", "美国加利福尼亚州（熊旗共和国）", "State of California, United States", "加利福尼亚州州旗（加州熊旗）", "California State Flag (Bear Flag)"),
            ("colorado_designed_by_andrew_carlisle_carson", "美国科罗拉多州", "Colorado, United States", "科罗拉多州州旗", "Flag of Colorado"),
            ("connecticut", "美国康涅狄格州", "Connecticut, United States", "康涅狄格州州旗", "Flag of Connecticut"),
            ("delaware", "美国特拉华州", "Delaware, United States", "特拉华州州旗", "Flag of Delaware"),
            ("florida", "美国佛罗里达州", "Florida, United States", "佛罗里达州州旗", "Flag of Florida"),
            ("georgia_", "美国佐治亚州", "Georgia, United States", "佐治亚州州旗", "Flag of Georgia"),
            ("guam", "美国关岛", "Guam, United States", "关岛旗帜", "Flag of Guam", "territory"),
            ("hawaii", "美国夏威夷州", "Hawaii, United States", "夏威夷州州旗", "Flag of Hawaii"),
            ("idaho", "美国爱达荷州", "Idaho, United States", "爱达荷州州旗", "Flag of Idaho"),
            ("illinois", "美国伊利诺伊州", "Illinois, United States", "伊利诺伊州州旗", "Flag of Illinois"),
            ("indiana", "美国印第安纳州", "Indiana, United States", "印第安纳州州旗", "Flag of Indiana"),
            ("iowa", "美国艾奥瓦州", "Iowa, United States", "艾奥瓦州州旗", "Flag of Iowa"),
            ("kansas", "美国堪萨斯州", "Kansas, United States", "堪萨斯州州旗", "Flag of Kansas"),
            ("kentucky", "美国肯塔基州", "Kentucky, United States", "肯塔基州州旗", "Flag of Kentucky"),
            ("louisiana", "美国路易斯安那州", "Louisiana, United States", "路易斯安那州州旗", "Flag of Louisiana"),
            ("maine", "美国缅因州", "Maine, United States", "缅因州州旗", "Flag of Maine"),
            ("maryland", "美国马里兰州", "Maryland, United States", "马里兰州州旗", "Flag of Maryland"),
            ("massachusetts", "美国马萨诸塞州", "Massachusetts, United States", "马萨诸塞州州旗", "Flag of Massachusetts"),
            ("michigan", "美国密歇根州", "Michigan, United States", "密歇根州州旗", "Flag of Michigan"),
            ("minnesota", "美国明尼苏达州", "Minnesota, United States", "明尼苏达州州旗", "Flag of Minnesota"),
            ("mississippi", "美国密西西比州", "Mississippi, United States", "密西西比州州旗", "Flag of Mississippi"),
            ("missouri", "美国密苏里州", "Missouri, United States", "密苏里州州旗", "Flag of Missouri"),
            ("montana", "美国蒙大拿州", "Montana, United States", "蒙大拿州州旗", "Flag of Montana"),
            ("nebraska", "美国内布拉斯加州", "Nebraska, United States", "内布拉斯加州州旗", "Flag of Nebraska"),
            ("nevada", "美国内华达州", "Nevada, United States", "内华达州州旗", "Flag of Nevada"),
            ("new_hampshire", "美国新罕布什尔州", "New Hampshire, United States", "新罕布什尔州州旗", "Flag of New Hampshire"),
            ("new_jersey", "美国新泽西州", "New Jersey, United States", "新泽西州州旗", "Flag of New Jersey"),
            ("new_mexico", "美国新墨西哥州", "New Mexico, United States", "新墨西哥州州旗", "Flag of New Mexico"),
            ("new_york", "美国纽约州", "New York, United States", "纽约州州旗", "Flag of New York"),
            ("north_carolina", "美国北卡罗来纳州", "North Carolina, United States", "北卡罗来纳州州旗", "Flag of North Carolina"),
            ("north_dakota", "美国北达科他州", "North Dakota, United States", "北达科他州州旗", "Flag of North Dakota"),
            ("ohio", "美国俄亥俄州", "Ohio, United States", "俄亥俄州州旗", "Flag of Ohio"),
            ("oklahoma", "美国俄克拉何马州", "Oklahoma, United States", "俄克拉何马州州旗", "Flag of Oklahoma"),
            ("oregon", "美国俄勒冈州", "Oregon, United States", "俄勒冈州州旗", "Flag of Oregon"),
            ("pennsylvania", "美国宾夕法尼亚州", "Pennsylvania, United States", "宾夕法尼亚州州旗", "Flag of Pennsylvania"),
            ("puerto_rico", "美国波多黎各", "Puerto Rico, United States", "波多黎各旗帜", "Flag of Puerto Rico", "territory"),
            ("rhode_island", "美国罗得岛州", "Rhode Island, United States", "罗得岛州州旗", "Flag of Rhode Island"),
            ("south_carolina", "美国南卡罗来纳州", "South Carolina, United States", "南卡罗来纳州州旗", "Flag of South Carolina"),
            ("south_dakota", "美国南达科他州", "South Dakota, United States", "南达科他州州旗", "Flag of South Dakota"),
            ("tennessee", "美国田纳西州", "Tennessee, United States", "田纳西州州旗", "Flag of Tennessee"),
            ("texas", "美国德克萨斯州（孤星之州）", "State of Texas, United States", "德克萨斯州州旗（孤星旗）", "Texas State Flag (Lone Star Flag)"),
            ("the_district_of_columbia", "美国哥伦比亚特区", "District of Columbia, United States", "哥伦比亚特区区旗", "Flag of the District of Columbia", "special"),
            ("the_northern_mariana_islands", "美国北马里亚纳群岛", "Northern Mariana Islands, United States", "北马里亚纳群岛旗帜", "Flag of the Northern Mariana Islands", "territory"),
            ("the_united_states_virgin_islands", "美国美属维尔京群岛", "United States Virgin Islands", "美属维尔京群岛旗帜", "Flag of the United States Virgin Islands", "territory"),
            ("utah", "美国犹他州", "Utah, United States", "犹他州州旗", "Flag of Utah"),
            ("vermont", "美国佛蒙特州", "Vermont, United States", "佛蒙特州州旗", "Flag of Vermont"),
            ("virginia", "美国弗吉尼亚州", "Virginia, United States", "弗吉尼亚州州旗", "Flag of Virginia"),
            ("washington", "美国华盛顿州", "Washington, United States", "华盛顿州州旗", "Flag of Washington"),
            ("west_virginia", "美国西弗吉尼亚州", "West Virginia, United States", "西弗吉尼亚州州旗", "Flag of West Virginia"),
            ("wisconsin", "美国威斯康星州", "Wisconsin, United States", "威斯康星州州旗", "Flag of Wisconsin"),
            ("wyoming", "美国怀俄明州", "Wyoming, United States", "怀俄明州州旗", "Flag of Wyoming"),
        ],
    },
}


def build_official_regional() -> Dict[str, Dict[str, str]]:
    regional_dir = ASSETS_DIR / "official_regional"
    stems = sorted(p.stem for p in regional_dir.glob("*.svg"))
    if not stems:
        raise SystemExit(f"[FATAL] 未在 {regional_dir} 找到任何地方旗 SVG。")

    table: Dict[str, Dict[str, str]] = {}
    for prefix, group in REGIONAL_GROUPS.items():
        default_type = group["default_type"]
        for item in group["items"]:
            suffix, name_zh, name_en, title_zh, title_en = item[:5]
            kind = item[5] if len(item) > 5 else default_type
            stem = f"{prefix}_{suffix}"
            if stem in table:
                raise SystemExit(f"[FATAL] 地方旗人工映射重复 stem：{stem}")
            table[stem] = _regional(name_zh, name_en, title_zh, title_en, kind)

    missing = [s for s in stems if s not in table]
    extra = [s for s in table if s not in stems]
    if missing or extra:
        raise SystemExit(
            "[FATAL] 地方旗人工映射与实际目录不一致：\n"
            f"  目录有但未映射 ({len(missing)}): {missing}\n"
            f"  已映射但目录不存在 ({len(extra)}): {extra}"
        )
    return table


# -----------------------------------------------------------------------------
# 5. 质量校验与落盘
# -----------------------------------------------------------------------------
def _check_zh_text(value: str, field: str, stem: str, domain: str) -> None:
    """校验单个中文字段：必须含 CJK、括号外不得出现拉丁字母、不得含劣质模式。"""
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"[FATAL] {domain}/{stem} 的 {field} 为空。")
    if not CJK_RE.search(value):
        raise SystemExit(f"[FATAL] {domain}/{stem} 的 {field} 不含中文：{value!r}")
    for bad in FORBIDDEN_ZH_FRAGMENTS:
        if bad in value:
            raise SystemExit(f"[FATAL] {domain}/{stem} 的 {field} 含劣质模式 '{bad}'：{value!r}")
    outside = PAREN_RE.sub("", value)
    latin_outside = LATIN_RUN_RE.findall(outside)
    if latin_outside:
        raise SystemExit(
            f"[FATAL] {domain}/{stem} 的 {field} 括号外出现拉丁字母 {latin_outside}：{value!r}"
        )
    for inner in INNER_PAREN_RE.findall(value):
        for word in LATIN_RUN_RE.findall(inner):
            normalized = word.strip(" .'’-")
            if normalized and normalized not in ALLOWED_BRACKET_LATIN:
                raise SystemExit(
                    f"[FATAL] {domain}/{stem} 的 {field} 括号内出现未授权拉丁词 '{normalized}'：{value!r}"
                )


def _check_entry(domain: str, stem: str, entry: Dict[str, str]) -> None:
    for field in ("name_zh", "title_zh"):
        _check_zh_text(entry.get(field, ""), field, stem, domain)
    for field in ("name_en", "title_en", "entity_type_zh", "entity_type_en"):
        value = entry.get(field, "")
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"[FATAL] {domain}/{stem} 的 {field} 为空。")
    for field in ("name_en", "title_en"):
        if not re.search(r"[A-Za-z]", entry[field]):
            raise SystemExit(f"[FATAL] {domain}/{stem} 的 {field} 不是英文：{entry[field]!r}")
    if not CJK_RE.search(entry["entity_type_zh"]):
        raise SystemExit(f"[FATAL] {domain}/{stem} 的 entity_type_zh 不含中文：{entry['entity_type_zh']!r}")


def validate_table(table: Dict[str, Dict[str, Dict[str, str]]]) -> None:
    for domain in DOMAINS:
        domain_map = table[domain]
        if not domain_map:
            raise SystemExit(f"[FATAL] 域 {domain} 为空。")
        for stem, entry in domain_map.items():
            _check_entry(domain, stem, entry)
    # 附加：国旗中文名不得与英文国家名同形（英文已被禁止，此处兜底检查 raw code）
    for domain in DOMAINS:
        for stem, entry in table[domain].items():
            lowered = entry["name_zh"].lower()
            raw_token = stem.replace("-", "_")
            if raw_token and raw_token in lowered:
                raise SystemExit(f"[FATAL] {domain}/{stem} 的中文名疑似含 raw code：{entry['name_zh']!r}")


def main() -> int:
    print("=" * 78)
    print("Datawheel 官方实体中文名表生成器（离线）")
    print("=" * 78)

    iso, node_bin = load_iso_display_names()
    print(f"[1/4] Node.js ({node_bin}) Intl.DisplayNames 导出 ISO 3166-1 中英文名：{len(iso)} 条")

    flags = build_official_flags(iso)
    print(f"[2/4] 国旗 official_flags：{len(flags)} 条（ISO 自动 + 非 ISO 手工补录）")

    emblems = build_official_emblems(iso, allowed_codes=flags.keys())
    print(f"[3/4] 国徽 official_emblems：{len(emblems)} 条（名称 token 匹配 + 特殊徽章手工补录）")

    regional = build_official_regional()
    print(f"[4/4] 地方旗 official_regional：{len(regional)} 条（人工映射全覆盖）")

    table = {
        "official_flags": flags,
        "official_emblems": emblems,
        "official_regional": regional,
    }
    validate_table(table)
    print("[OK] 中文质量校验通过：全部条目含 CJK、括号外无拉丁字母、无 raw code / 劣质 fallback 模式。")

    payload = {
        "meta": {
            "description": "Datawheel 官方实体（国旗/国徽/地方旗）中文全称表",
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "generator": "tools/build_entity_name_table.py",
            "node_bin": node_bin,
            "sources": [
                "Node.js Intl.DisplayNames(['zh-CN']/['en'], {type:'region'}) for ISO 3166-1 alpha-2",
                "手工补录：非 ISO 代码（arab/asean/cefta/eac/pc/eu/un/es-ct/gb-eng/sh-ac/xk/xx 等）",
                "手工补录：国际组织/特殊徽章（united_nations/nasa/olympic/sede_vacante/somaliland 等）",
                "人工映射：assets/official_regional 190 条地方行政区划旗帜",
            ],
            "counts": {domain: len(table[domain]) for domain in DOMAINS},
            "total": sum(len(v) for v in table.values()),
        },
        **table,
    }

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("-" * 78)
    print(f"[DONE] 已写入 {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"       覆盖统计：flags={len(flags)}  emblems={len(emblems)}  regional={len(regional)}  total={payload['meta']['total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
