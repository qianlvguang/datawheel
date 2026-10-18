#!/usr/bin/env python3
"""Datawheel v2 多模态数据飞轮生成器 (DW_V2)。

相比 v1（10,000 条 caption + 250 张预览图）的关键迭代：
1. 全量渲染 10,000 张图像（640x640；参考匹配阶段 640x840），图文一一对应；
2. 五阶段课程：A 几何色块 / B 真实商品单图案 / C 多图案空间关系 /
   D 参考图匹配（含 hard negative、no-match、候选框）/ E 负样本与证据不足；
3. 实体级 + 底板级 split 隔离，支持 unseen-entity / unseen-substrate 泛化评测；
4. 正确区位词表：按商品类别生成「胸前/下摆/包盖/鞋帮/帽面」等真实工艺位置，
   不再出现「鞋子上胸前印花」这类语义错误；
5. 图案按宽高比等比投放、深色/繁复面料使用不透明塑胶浆模式，并在记录中
   写入打印可见度、bbox 面积、底板可印性等质量指标；
6. 每条 QA 带 answer_type / answer_bbox / evidence，支持 grounding 与
   yes/no / uncertain 监督；实体名称全部走知识扩写（无 raw code）。

用法：
  .venv/bin/python tools/generate_datawheel_v2.py --start 1 --end 10000 \\
      --out-dir output/datawheel_v2 --images-dir output/datawheel_v2/images \\
      --records-file output/datawheel_v2/_generation/shard_records/full.jsonl
  .venv/bin/python tools/generate_datawheel_v2.py --merge --out-dir output/datawheel_v2
  .venv/bin/python tools/generate_datawheel_v2.py --preview --out-dir /tmp/v2_preview
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import resvg_py
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.compositor import CommodityCompositor  # noqa: E402
from engine.coords import mask_to_rel_bbox_1000, to_rel_bbox_1000  # noqa: E402
from engine.knowledge_expander import expander_instance  # noqa: E402

ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"

CANVAS = 640
D_BAND = 200
D_HEIGHT = 840
SEED = 20261017
DATASET_VERSION = "v2.0"

DOMAINS = [
    "official_flags", "official_emblems", "official_regional",
    "luxury_brands", "ip_brands", "character_ips", "subculture_elements",
]
DOMAIN_GLOBS = {
    "official_flags": ["official_flags/*.svg"],
    "official_emblems": ["official_emblems/*.svg"],
    "official_regional": ["official_regional/*.svg"],
    "luxury_brands": ["brand_logos/luxury_100/*.svg"],
    "ip_brands": ["brand_logos/ip_brands_100/*.svg"],
    "character_ips": ["character_ips/**/*.png"],
    "subculture_elements": ["subculture_elements/**/*.png", "subculture_elements/**/*.svg"],
}
DOMAIN_WEIGHTS = {
    "official_flags": 0.17,
    "official_emblems": 0.12,
    "official_regional": 0.13,
    "luxury_brands": 0.14,
    "ip_brands": 0.14,
    "character_ips": 0.15,
    "subculture_elements": 0.15,
}

STAGE_RANGES = {
    "stage_a_geometry": (1, 1000),
    "stage_b_single_element": (1001, 5000),
    "stage_c_multi_element": (5001, 7500),
    "stage_d_reference_match": (7501, 9000),
    "stage_e_negative_evidence": (9001, 10000),
}
STAGE_TITLES = {
    "stage_a_geometry": "阶段A：基础色块与颜色区域",
    "stage_b_single_element": "阶段B：真实商品单图案展示",
    "stage_c_multi_element": "阶段C：真实商品多图案组合",
    "stage_d_reference_match": "阶段D：参考图匹配与实物对照",
    "stage_e_negative_evidence": "阶段E：负样本与证据不足",
}
SPLIT_QUOTAS = {
    "stage_a_geometry": {"train": 800, "val": 80, "test_unseen_entity": 60,
                         "test_unseen_substrate": 30, "test_unseen_both": 30},
    "stage_b_single_element": {"train": 3200, "val": 320, "test_unseen_entity": 240,
                               "test_unseen_substrate": 140, "test_unseen_both": 100},
    "stage_c_multi_element": {"train": 2000, "val": 200, "test_unseen_entity": 150,
                              "test_unseen_substrate": 90, "test_unseen_both": 60},
    "stage_d_reference_match": {"train": 1200, "val": 120, "test_unseen_entity": 90,
                                "test_unseen_substrate": 60, "test_unseen_both": 30},
    "stage_e_negative_evidence": {"train": 800, "val": 80, "test_unseen_entity": 60,
                                  "test_unseen_substrate": 40, "test_unseen_both": 20},
}
SPLITS = ["train", "val", "test_unseen_entity", "test_unseen_substrate", "test_unseen_both"]

COLOR_PRESETS = [
    ("#ee1c25", "中国红/绯红", "Vibrant Scarlet Red"),
    ("#00209f", "法兰西深蓝", "Deep French Blue"),
    ("#ffffff", "纯白", "Pure White"),
    ("#ffcc00", "明金黄", "Bright Golden Yellow"),
    ("#000000", "深黑", "Deep Black"),
    ("#007a3d", "森林绿", "Forest Green"),
    ("#ff6600", "活力橙", "Vibrant Orange"),
    ("#75225b", "皇家紫", "Royal Purple"),
    ("#708090", "石板灰", "Slate Gray"),
    ("#87ceeb", "天青蓝", "Sky Blue"),
]

# 各类商品的自然印花/标识位置（相对底板可印花区的比例：y0, x0, y1, x1）
ZONE_SPECS: Dict[str, List[Dict[str, Any]]] = {
    "Topwear": [
        {"id": "center_chest_graphic", "rect": (0.12, 0.18, 0.82, 0.82),
         "zh": "前胸中央主视觉印花区", "en": "center chest hero graphic area", "kind": "screenprint"},
        {"id": "left_chest_badge", "rect": (0.05, 0.04, 0.32, 0.32),
         "zh": "左胸上方小型徽标区", "en": "left chest badge area", "kind": "screenprint"},
        {"id": "lower_hem_patch", "rect": (0.72, 0.30, 0.94, 0.74),
         "zh": "下摆边缘织唛/小标区", "en": "lower hem label patch area", "kind": "woven_label"},
    ],
    "Bags": [
        {"id": "bag_front_flap", "rect": (0.12, 0.12, 0.58, 0.88),
         "zh": "包盖/正面外袋装饰徽标区", "en": "bag front flap emblem area", "kind": "heat_transfer"},
        {"id": "bag_front_panel", "rect": (0.30, 0.16, 0.82, 0.86),
         "zh": "包身正面主视觉印制区", "en": "bag front panel graphic area", "kind": "heat_transfer"},
    ],
    "Shoes": [
        {"id": "shoe_lateral_quarter", "rect": (0.18, 0.12, 0.68, 0.88),
         "zh": "鞋面/鞋帮主标识区", "en": "shoe upper emblem area", "kind": "heat_transfer"},
        {"id": "shoe_tongue_panel", "rect": (0.06, 0.26, 0.40, 0.74),
         "zh": "鞋舌/鞋面小标区", "en": "shoe tongue small-label area", "kind": "woven_label"},
    ],
    "Headwear": [
        {"id": "hat_front_panel", "rect": (0.20, 0.18, 0.72, 0.82),
         "zh": "帽冠正面主视觉区", "en": "hat front panel area", "kind": "embroidery"},
    ],
    "Bottomwear": [
        {"id": "thigh_side_graphic", "rect": (0.22, 0.16, 0.74, 0.84),
         "zh": "裤腿正面/侧面印花区", "en": "pants leg graphic area", "kind": "screenprint"},
        {"id": "pants_hem_patch", "rect": (0.74, 0.18, 0.94, 0.80),
         "zh": "裤脚口织唛/小标区", "en": "pants hem label area", "kind": "woven_label"},
    ],
    "Dress": [
        {"id": "bodice_center_graphic", "rect": (0.16, 0.16, 0.62, 0.84),
         "zh": "裙身上半部主视觉印花区", "en": "bodice center graphic area", "kind": "screenprint"},
        {"id": "skirt_hem_patch", "rect": (0.72, 0.22, 0.94, 0.78),
         "zh": "裙摆边缘下摆小标区", "en": "skirt hem label area", "kind": "woven_label"},
    ],
}
DEFAULT_ZONE_SPEC = ZONE_SPECS["Topwear"]
CATEGORY_ALIAS = {
    "Top": "Topwear", "Shirt": "Topwear", "Jacket": "Topwear", "Coat": "Topwear",
    "Cardigan": "Topwear", "Sweater": "Topwear", "Vest": "Topwear",
    "Pants": "Bottomwear", "Shorts": "Bottomwear", "Skirt": "Bottomwear",
    "Topwear": "Topwear", "Bottomwear": "Bottomwear", "Dress": "Dress",
    "Bags": "Bags", "Shoes": "Shoes", "Headwear": "Headwear",
}
TECHNIQUE_ZH = {"screenprint": "高精度工业丝网胶印", "woven_label": "织唛/织标",
                "heat_transfer": "高精度热转印", "embroidery": "精细电脑刺绣"}
TECHNIQUE_EN = {"screenprint": "screenprint", "woven_label": "woven label",
                "heat_transfer": "heat transfer", "embroidery": "embroidery"}

# 多图案布局：在底板可印花区内取比例矩形，保证互不重叠
LAYOUTS = [
    {"id": "main_center_plus_left_badge", "rects": [(0.10, 0.20, 0.86, 0.84), (0.03, 0.03, 0.30, 0.30)],
     "zh": "中央主视觉 + 左上角小徽标", "en": "center main graphic plus upper-left badge"},
    {"id": "vertical_stack", "rects": [(0.06, 0.24, 0.50, 0.80), (0.58, 0.24, 0.94, 0.80)],
     "zh": "上下双排垂直排布", "en": "vertical stacked arrangement"},
    {"id": "diagonal_composition", "rects": [(0.04, 0.14, 0.54, 0.58), (0.58, 0.56, 0.96, 0.96)],
     "zh": "对角式复合排布", "en": "diagonal composition"},
    {"id": "chest_plus_hem", "rects": [(0.04, 0.18, 0.54, 0.82), (0.70, 0.24, 0.95, 0.78)],
     "zh": "胸前主图 + 下摆小标", "en": "chest graphic plus hem label"},
]
SMALL_LAYOUTS = [
    {"id": "compact_side_by_side", "rects": [(0.02, 0.02, 0.54, 0.98), (0.58, 0.24, 0.96, 0.76)],
     "zh": "紧凑左右双图案排布", "en": "compact side-by-side dual arrangement"},
    {"id": "compact_top_bottom", "rects": [(0.02, 0.06, 0.56, 0.94), (0.60, 0.22, 0.96, 0.78)],
     "zh": "紧凑上下双图案排布", "en": "compact stacked dual arrangement"},
]
TRIPLE_LAYOUT = {"id": "triple_triangle", "rects": [(0.04, 0.22, 0.46, 0.72), (0.56, 0.18, 0.80, 0.46),
                                                   (0.56, 0.56, 0.80, 0.84)],
                 "zh": "三图案三角式组合", "en": "three-element triangular arrangement"}

QUESTION_TEMPLATES = {
    "identify": ["图中的印花图案是什么？", "请识别商品上印制/贴合的主要视觉图案。",
                 "这件商品上的图案内容是什么？"],
    "grounding": ["请给出该图案的边界框 [ymin,xmin,ymax,xmax]。",
                  "定位该图案，输出 [ymin,xmin,ymax,xmax] 相对坐标。",
                  "该图案在整图中的边界框坐标是多少？"],
    "color": ["该图案的主导颜色是什么？", "请描述该图案的主色调。",
              "印花的代表色是？"],
    "location": ["该图案位于商品的哪个工艺位置？", "印花在商品上的具体位置是？"],
    "geometry": ["该图案的几何构型属于哪一类？", "该图案的整体外形比例如何？"],
    "count": ["图中共有几处独立印花？", "商品上共有几个主要视觉图案？"],
    "relation": ["{a}与{b}的相对位置关系是什么？", "{b}位于{a}的什么方位？"],
    "match_which": ["顶部候选参考图（{letters}）中，哪一张与下方商品上的印花一致？",
                    "比对候选参考图，指出与商品印花匹配的那一张。"],
    "match_exists": ["顶部候选参考图中是否有与下方商品印花一致的吗？",
                     "候选参考图中存在与商品图案匹配的图案吗？"],
    "candidate_bbox": ["候选{letter}参考图在整图中的位置是？", "候选{letter}的边界框坐标？"],
    "present": ["图中商品上是否存在【{name}】图案？", "商品表面能看到【{name}】的印花吗？"],
    "distractor": ["图中商品上的图案是什么？", "请识别商品上实际出现的图案。"],
}

REQUIRED_FIELDS = [
    "sample_id", "dataset_version", "task_stage", "stage_title_zh", "task_family",
    "difficulty", "split", "domain", "entity_ids_json", "entity_names_zh", "entity_names_en",
    "num_entities", "substrate_id", "substrate_category_zh", "substrate_category_en",
    "substrate_source", "scene_style_zh", "placement_zone", "print_technique_zh",
    "bboxes_norm_1000", "entities_json", "ref_candidates_json", "match_label",
    "caption_zh", "caption_en", "visual_analysis", "knowledge_expansion", "qa_pairs_json",
    "image_path", "image_sha256", "image_width", "image_height",
    "render_meta_json", "quality_json", "content_hash",
]


def _seed_for(*parts: Any) -> int:
    joined = "|".join(str(p) for p in parts)
    return int(hashlib.md5(joined.encode("utf-8")).hexdigest()[:12], 16)


def _md5_bucket(key: str) -> int:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 100


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _largest_remainder(weights: Dict[str, float], total: int) -> Dict[str, int]:
    s = sum(weights.values())
    raw = {k: total * v / s for k, v in weights.items()}
    base = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = total - sum(base.values())
    order = sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)
    for k in order[:remainder]:
        base[k] += 1
    return base


class DatawheelV2Generator:
    """Datawheel v2 五阶段数据集生成器（可复现、可分片）。"""

    def __init__(self, seed: int = SEED):
        self.seed = seed
        self.compositor = CommodityCompositor(CANVAS, CANVAS)
        self.expander = expander_instance
        self._substrate_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._knowledge_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        comm_path = ASSETS_DIR / "manifests" / "commodity_1500.json"
        self.commodity_meta = {it["substrate_id"]: it for it in
                               json.loads(comm_path.read_text(encoding="utf-8")).get("items", [])} if comm_path.exists() else {}
        self._load_assets()
        self._load_substrates()
        self._build_plan()

    # ------------------------------------------------------------------
    # 初始化：资产 / 底板 / 计划
    # ------------------------------------------------------------------
    def _load_assets(self) -> None:
        index_path = ASSETS_DIR / "manifests" / "asset_visual_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"missing {index_path}; run tools/build_asset_index.py first")
        self.asset_index = json.loads(index_path.read_text(encoding="utf-8"))
        self.asset_domains: Dict[str, Dict[str, Dict[str, Any]]] = self.asset_index["domains"]
        self.neighbors: Dict[str, List[Dict[str, Any]]] = self.asset_index["neighbors"]
        self.asset_paths: Dict[str, Dict[str, Path]] = {}
        for domain, patterns in DOMAIN_GLOBS.items():
            paths: Dict[str, Path] = {}
            for pattern in patterns:
                for p in ASSETS_DIR.glob(pattern):
                    if p.is_file() and not p.name.startswith(".") and p.stem not in paths:
                        paths[p.stem] = p
            self.asset_paths[domain] = paths
        for domain in DOMAINS:
            n_index = len(self.asset_domains.get(domain, {}))
            n_files = len(self.asset_paths.get(domain, {}))
            print(f"[v2] domain {domain}: indexed={n_index} files={n_files}")

    def _load_substrates(self) -> None:
        q_path = ASSETS_DIR / "manifests" / "substrate_quality.json"
        if not q_path.exists():
            raise FileNotFoundError(f"missing {q_path}; run tools/build_substrate_quality.py first")
        payload = json.loads(q_path.read_text(encoding="utf-8"))
        self.substrates = [it for it in payload["items"] if it.get("print_eligible")
                           and not (it.get("source") == "fashionpedia_editorial"
                                    and it.get("sub_category") in ("Bottomwear", "Pants", "Shorts", "Skirt"))]
        if not self.substrates:
            raise RuntimeError("no print-eligible substrates found")
        self.substrates_by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for it in self.substrates:
            self.substrates_by_cat.setdefault(it["sub_category"], []).append(it)
        print(f"[v2] eligible substrates: {len(self.substrates)} "
              f"({len(self.substrates_by_cat)} categories), difficulty="
              f"{ {d: sum(1 for s in self.substrates if s['difficulty'] == d) for d in ('easy','medium','hard')} }")

    def _entity_pool_filter(self, split: str):
        if split in ("train", "test_unseen_substrate"):
            lo, hi = 0, 80
        elif split == "val":
            lo, hi = 80, 90
        else:  # test_unseen_entity / test_unseen_both
            lo, hi = 90, 100
        return lambda domain, stem: lo <= _md5_bucket(f"{domain}/{stem}") < hi

    def _substrate_split_filter(self, split: str):
        heldout = split in ("test_unseen_substrate", "test_unseen_both")
        if heldout:
            return lambda s: 80 <= _md5_bucket(s["substrate_id"]) < 100
        return lambda s: _md5_bucket(s["substrate_id"]) < 80

    def _get_entity_perm(self, stage: str, split: str, domain: str) -> List[str]:
        key = (stage, split, domain)
        if key not in self._entity_perm:
            filt = self._entity_pool_filter(split)
            pool = [stem for stem in self.asset_domains[domain] if filt(domain, stem) and stem in self.asset_paths[domain]]
            rng = random.Random(_seed_for(self.seed, "entperm", stage, split, domain))
            rng.shuffle(pool)
            self._entity_perm[key] = pool
        return self._entity_perm[key]

    def _substrate_perm_list(self, stage: str, split: str) -> List[Dict[str, Any]]:
        key = (stage, split)
        if key not in self._substrate_perm:
            filt = self._substrate_split_filter(split)
            pool = [s for s in self.substrates if filt(s)]
            if not pool:
                pool = list(self.substrates)
            rng = random.Random(_seed_for(self.seed, "subperm", stage, split))
            rng.shuffle(pool)
            if stage in ("stage_b_single_element", "stage_c_multi_element"):
                # 平铺/无人像底板语义更可靠，加权 2x；真人上身图保留以增加真实感
                pool = pool + [s for s in pool if not s.get("person_worn")]
                rng.shuffle(pool)
            self._substrate_perm[key] = pool
        return self._substrate_perm[key]

    def _build_plan(self) -> None:
        self.assign: Dict[int, Dict[str, Any]] = {}
        self._entity_perm: Dict[Tuple[str, str, str], List[str]] = {}
        self._substrate_perm: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self._domain_seq: Dict[Tuple[str, str], List[str]] = {}
        for stage, (start, end) in STAGE_RANGES.items():
            ids = list(range(start, end + 1))
            shuffled = ids[:]
            random.Random(_seed_for(self.seed, "stageids", stage)).shuffle(shuffled)
            pos = 0
            for split in SPLITS:
                count = SPLIT_QUOTAS[stage][split]
                for idx in shuffled[pos:pos + count]:
                    self.assign[idx] = {"stage": stage, "split": split}
                pos += count
        # ordinal + 领域序列
        counters: Dict[Tuple[str, str], int] = {}
        for idx in sorted(self.assign):
            plan = self.assign[idx]
            key = (plan["stage"], plan["split"])
            plan["ordinal"] = counters.get(key, 0)
            counters[key] = plan["ordinal"] + 1
        for stage, (start, end) in STAGE_RANGES.items():
            for split in SPLITS:
                count = SPLIT_QUOTAS[stage][split]
                filt = self._entity_pool_filter(split)
                weights = {}
                for domain in DOMAINS:
                    avail = sum(1 for stem in self.asset_domains.get(domain, {})
                                if filt(domain, stem) and stem in self.asset_paths[domain])
                    if avail > 0:
                        weights[domain] = DOMAIN_WEIGHTS[domain]
                counts = _largest_remainder(weights, count)
                seq: List[str] = []
                for domain, c in counts.items():
                    seq.extend([domain] * c)
                random.Random(_seed_for(self.seed, "domseq", stage, split)).shuffle(seq)
                self._domain_seq[(stage, split)] = seq
                for domain in set(seq):
                    self._get_entity_perm(stage, split, domain)
                self._substrate_perm_list(stage, split)
        # 为每条样本分配主领域 / 次领域（仅 C 用）
        occ_main: Dict[Tuple[str, str, str], int] = {}
        occ_second: Dict[Tuple[str, str, str], int] = {}
        for idx in sorted(self.assign):
            plan = self.assign[idx]
            stage, split = plan["stage"], plan["split"]
            seq = self._domain_seq[(stage, split)]
            domain = seq[plan["ordinal"] % len(seq)]
            plan["domain"] = domain
            key = (stage, split, domain)
            plan["domain_occ"] = occ_main.get(key, 0)
            occ_main[key] = plan["domain_occ"] + 1
            if stage == "stage_c_multi_element":
                rng = random.Random(_seed_for(self.seed, "dom2", idx))
                candidates = [d for d in DOMAINS if d != domain and self._get_entity_perm(stage, split, d)]
                weights2 = {d: DOMAIN_WEIGHTS[d] for d in candidates}
                tot = sum(weights2.values())
                r = rng.random() * tot
                acc = 0.0
                domain2 = candidates[-1]
                for d, w in weights2.items():
                    acc += w
                    if r <= acc:
                        domain2 = d
                        break
                plan["domain2"] = domain2
                key2 = (stage, split, domain2)
                plan["domain2_occ"] = occ_second.get(key2, 0)
                occ_second[key2] = plan["domain2_occ"] + 1

    # ------------------------------------------------------------------
    # 资源读取
    # ------------------------------------------------------------------
    def _entity(self, domain: str, stem: str) -> Dict[str, Any]:
        key = (domain, stem)
        if key not in self._knowledge_cache:
            info = dict(self.expander.resolve_entity_knowledge(domain, stem))
            feat = self.asset_domains[domain][stem]
            info.update({k: feat[k] for k in ("dominant_color_zh", "dominant_color_en", "geometry_zh",
                                              "geometry_en", "aspect_ratio")})
            info["domain"] = domain
            info["stem"] = stem
            self._knowledge_cache[key] = info
        return self._knowledge_cache[key]

    def _pick_entity(self, stage: str, split: str, domain: str, occ: int) -> Dict[str, Any]:
        perm = self._get_entity_perm(stage, split, domain)
        return self._entity(domain, perm[occ % len(perm)])

    def _pick_substrate(self, stage: str, split: str, ordinal: int) -> Dict[str, Any]:
        perm = self._substrate_perm_list(stage, split)
        return perm[ordinal % len(perm)]

    def _substrate_bg(self, sub: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        sid = sub["substrate_id"]
        if sid not in self._substrate_cache:
            path = PROJECT_ROOT / sub["file_path"]
            rgb, lighting, _meta = self.compositor.load_real_packshot_letterboxed(str(path))
            if len(self._substrate_cache) > 24:
                self._substrate_cache.pop(next(iter(self._substrate_cache)))
            self._substrate_cache[sid] = (rgb, lighting)
        rgb, lighting = self._substrate_cache[sid]
        return rgb.copy(), lighting

    def _load_foreground(self, domain: str, stem: str, target_w: int = 420) -> Optional[Image.Image]:
        path = self.asset_paths[domain].get(stem)
        if path is None:
            return None
        try:
            if path.suffix.lower() == ".svg":
                svg = path.read_text(encoding="utf-8")
                png = resvg_py.svg_to_bytes(svg, width=target_w, dpi=96.0)
                return Image.open(__import__("io").BytesIO(png)).convert("RGBA")
            img = Image.open(path).convert("RGBA")
            if img.width > target_w:
                h = max(1, int(img.height * target_w / img.width))
                img = img.resize((target_w, h), Image.Resampling.LANCZOS)
            return img
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 几何与渲染工具
    # ------------------------------------------------------------------
    def _category(self, sub: Dict[str, Any]) -> str:
        return CATEGORY_ALIAS.get(sub["sub_category"], "Topwear")

    @staticmethod
    def _zone_desc(sub: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[str, str]:
        """返回该样本可用的区位中文/英文描述。

        平铺商品图可精确使用「前胸/下摆/包盖」等工艺位置；
        真人上身图无法从无分割标注保证服装部位，改用画面相对位置描述（事实可验证）。
        """
        if not sub.get("person_worn"):
            return spec["zh"], spec["en"]
        y0, x0, y1, x1 = sub["zone_canvas_norm"]
        cy = (y0 + y1) / 2
        cx = (x0 + x1) / 2
        vert = "上部" if cy < 0.38 else ("中部" if cy < 0.64 else "下部")
        horiz = "左侧" if cx < 0.38 else ("居中" if cx < 0.64 else "右侧")
        return (f"商品正面可视面的{vert}{horiz}区域（约在画面 {cy:.0%} 高度、{cx:.0%} 宽度处）",
                f"{vert} {horiz} area of the product frontal view (about {cy:.0%} height, {cx:.0%} width)")

    def _zone_rect(self, sub: Dict[str, Any], spec: Dict[str, Any],
                   rng: random.Random) -> Tuple[int, int, int, int]:
        y0, x0, y1, x1 = sub["zone_canvas_norm"]
        ry0, rx0, ry1, rx1 = spec["rect"]
        px0 = int((x0 + rx0 * (x1 - x0)) * CANVAS)
        px1 = int((x0 + rx1 * (x1 - x0)) * CANVAS)
        py0 = int((y0 + ry0 * (y1 - y0)) * CANVAS)
        py1 = int((y0 + ry1 * (y1 - y0)) * CANVAS)
        # 轻微抖动，避免固定死板框
        jx = max(0, int((px1 - px0) * 0.04))
        jy = max(0, int((py1 - py0) * 0.04))
        if jx:
            d = rng.randint(-jx, jx)
            px0, px1 = px0 + d, px1 + d
        if jy:
            d = rng.randint(-jy, jy)
            py0, py1 = py0 + d, py1 + d
        px0, py0 = max(2, px0), max(2, py0)
        px1, py1 = min(CANVAS - 3, px1), min(CANVAS - 3, py1)
        return px0, py0, px1, py1

    def _fit_quad(self, element: Image.Image, rect: Tuple[int, int, int, int],
                  rng: random.Random, fill: float = 0.82) -> Tuple[Image.Image, np.ndarray]:
        x0, y0, x1, y1 = rect
        zw, zh = max(1, x1 - x0), max(1, y1 - y0)
        ew, eh = element.size
        scale = min(zw * fill / ew, zh * fill / eh)
        tw = max(8, int(round(ew * scale)))
        th = max(8, int(round(eh * scale)))
        if tw > zw:
            th = max(8, int(th * zw / tw)); tw = zw
        if th > zh:
            tw = max(8, int(tw * zh / th)); th = zh
        elem = element.resize((tw, th), Image.Resampling.LANCZOS)
        span_x = max(0, zw - tw)
        span_y = max(0, zh - th)
        cx = x0 + tw / 2 + rng.uniform(-0.42, 0.42) * span_x
        cy = y0 + th / 2 + rng.uniform(-0.42, 0.42) * span_y
        cx = min(max(cx, x0 + tw / 2), x1 - tw / 2)
        cy = min(max(cy, y0 + th / 2), y1 - th / 2)
        angle = math.radians(rng.uniform(-2.5, 2.5))
        hw, hh = tw / 2, th / 2
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        quad = np.float32([
            [cx - hw * cos_a + hh * sin_a, cy - hw * sin_a - hh * cos_a],
            [cx + hw * cos_a + hh * sin_a, cy + hw * sin_a - hh * cos_a],
            [cx + hw * cos_a - hh * sin_a, cy + hw * sin_a + hh * cos_a],
            [cx - hw * cos_a - hh * sin_a, cy - hw * sin_a + hh * cos_a],
        ])
        return elem, quad

    def _visibility(self, bg: np.ndarray, blended: np.ndarray, alpha: np.ndarray) -> Tuple[float, Optional[List[int]]]:
        mask = alpha > 40
        if not mask.any():
            return 0.0, None
        diff = np.abs(blended.astype(np.int16) - bg.astype(np.int16)).mean(axis=2)
        vis = float(diff[mask].mean() / 255.0)
        box = mask_to_rel_bbox_1000(alpha, CANVAS, CANVAS, threshold=25)
        return vis, box

    def _render_element(self, bg: np.ndarray, lighting: np.ndarray, element: Image.Image,
                        rect: Tuple[int, int, int, int], rng: random.Random,
                        sub: Dict[str, Any], fill: float = 0.82) -> Dict[str, Any]:
        """把单个图案等比投放到指定矩形并物理融合，返回 bbox/可见度等。"""
        opaque = bool(sub.get("dark") or sub.get("difficulty") == "hard")
        mode = "opaque_screenprint" if opaque else "real_garment_screenprint"
        best = None
        for attempt in range(5):
            cur_fill = min(0.98, fill + attempt * 0.06)
            elem, quad = self._fit_quad(element, rect, rng, cur_fill)
            warped, _ = self.compositor.apply_homography_warp(elem, quad)
            displaced, alpha = self.compositor.apply_fold_displacement(
                warped, bg, displacement_scale=rng.uniform(3.0, 5.0))
            blended = self.compositor.blend_onto_substrate(
                bg, lighting, displaced, blend_mode=mode,
                ink_opacity=0.97 if opaque else 0.94,
                weave_strength=0.35 if opaque else 0.65)
            vis, box = self._visibility(bg, blended, alpha)
            if box is not None:
                bbox_area = ((box[2] - box[0]) * (box[3] - box[1])) / 1_000_000.0
            else:
                bbox_area = 0.0
            if best is None or vis > best["visibility"]:
                best = {"rgb": blended, "alpha": alpha, "bbox": box, "visibility": vis,
                        "area": bbox_area, "mode": mode, "fill": cur_fill, "attempt": attempt}
            if vis >= 0.06 and bbox_area >= 0.0015:
                break
        # 极小实例兜底：以 1.3x 扩张矩形再渲染一次，尽量满足 ≥0.1% 画布面积
        if best is not None and best["area"] < 0.0012:
            x0, y0, x1, y1 = rect
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hw, hh = (x1 - x0) * 0.65, (y1 - y0) * 0.65
            ex0, ey0 = max(1, int(cx - hw)), max(1, int(cy - hh))
            ex1, ey1 = min(CANVAS - 2, int(cx + hw)), min(CANVAS - 2, int(cy + hh))
            elem2, quad2 = self._fit_quad(element, (ex0, ey0, ex1, ey1), rng, 0.98)
            warped2, _ = self.compositor.apply_homography_warp(elem2, quad2)
            displaced2, alpha2 = self.compositor.apply_fold_displacement(
                warped2, bg, displacement_scale=rng.uniform(3.0, 5.0))
            blended2 = self.compositor.blend_onto_substrate(
                bg, lighting, displaced2, blend_mode=mode,
                ink_opacity=0.97 if opaque else 0.94,
                weave_strength=0.35 if opaque else 0.65)
            vis2, box2 = self._visibility(bg, blended2, alpha2)
            if box2 is not None:
                area2 = ((box2[2] - box2[0]) * (box2[3] - box2[1])) / 1_000_000.0
                if area2 > best["area"]:
                    best = {"rgb": blended2, "alpha": alpha2, "bbox": box2, "visibility": vis2,
                            "area": area2, "mode": mode, "fill": 0.98, "attempt": 5}
        return best

    # ------------------------------------------------------------------
    # 记录组装
    # ------------------------------------------------------------------
    def _record(self, idx: int, stage: str, split: str, *, task_family: str, difficulty: str,
                domain: str, entity_ids: List[Dict[str, Any]], entity_names_zh: str,
                entity_names_en: str, num_entities: int, substrate: Optional[Dict[str, Any]],
                placement_zone: str, technique: str, bboxes: List[List[int]],
                entities: List[Dict[str, Any]], ref_candidates: List[Dict[str, Any]],
                match_label: str, caption_zh: str, caption_en: str, visual_analysis: str,
                knowledge_expansion: str, qa_pairs: List[Dict[str, Any]],
                render_meta: Dict[str, Any], quality: Dict[str, Any]) -> Dict[str, Any]:
        sub = substrate or {}
        return {
            "sample_id": f"DW_V2_{idx:06d}",
            "dataset_version": DATASET_VERSION,
            "task_stage": stage,
            "stage_title_zh": STAGE_TITLES[stage],
            "task_family": task_family,
            "difficulty": difficulty,
            "split": split,
            "domain": domain,
            "entity_ids_json": _json(entity_ids),
            "entity_names_zh": entity_names_zh,
            "entity_names_en": entity_names_en,
            "num_entities": int(num_entities),
            "substrate_id": sub.get("substrate_id", "procedural_canvas"),
            "substrate_category_zh": sub.get("category_zh", "纯色平面画布"),
            "substrate_category_en": sub.get("category_en", "Procedural Geometry Canvas"),
            "substrate_source": sub.get("source", "procedural"),
            "scene_style_zh": sub.get("scene_style_zh", "纯色平面背景"),
            "placement_zone": placement_zone,
            "print_technique_zh": TECHNIQUE_ZH.get(technique, "高精度工业丝网胶印"),
            "bboxes_norm_1000": _json(bboxes),
            "entities_json": _json(entities),
            "ref_candidates_json": _json(ref_candidates),
            "match_label": match_label,
            "caption_zh": caption_zh,
            "caption_en": caption_en,
            "visual_analysis": visual_analysis,
            "knowledge_expansion": knowledge_expansion,
            "qa_pairs_json": _json(qa_pairs),
            "image_path": "",
            "image_sha256": "",
            "image_width": 0,
            "image_height": 0,
            "render_meta_json": _json(render_meta),
            "quality_json": _json(quality),
            "content_hash": "",
        }

    def _qa(self, question: str, answer: str, answer_type: str,
            answer_bbox: Optional[List[int]] = None, evidence: str = "") -> Dict[str, Any]:
        return {"question_zh": question, "answer_zh": answer, "answer_type": answer_type,
                "answer_bbox_1000": answer_bbox, "evidence_zh": evidence}

    @staticmethod
    def _relation_label(box_a: List[int], box_b: List[int]) -> str:
        """b 相对 a 的方位（基于 bbox 中心）。"""
        ay, ax = (box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2
        by, bx = (box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2
        dx, dy = bx - ax, by - ay
        horiz = "右侧" if dx > 8 else ("左侧" if dx < -8 else "")
        vert = "下方" if dy > 8 else ("上方" if dy < -8 else "")
        if horiz and vert:
            return f"{vert}{horiz}"
        if horiz:
            return horiz
        if vert:
            return vert
        return "重叠/邻近"

    # ------------------------------------------------------------------
    # 阶段 A：几何色块
    # ------------------------------------------------------------------
    def _gen_stage_a(self, idx: int, rng: random.Random, plan: Dict[str, Any]):
        modes = ["tricolor_vertical", "tricolor_horizontal", "quadrant_grid", "centered_disc"]
        mode = modes[(idx - 1) % 4]
        combo_k = (idx - 1) // 4
        import itertools
        if mode in ("tricolor_vertical", "tricolor_horizontal"):
            perms = list(itertools.permutations(range(len(COLOR_PRESETS)), 3))
            c1, c2, c3 = (COLOR_PRESETS[perms[combo_k % len(perms)][0]],
                          COLOR_PRESETS[perms[combo_k % len(perms)][1]],
                          COLOR_PRESETS[perms[combo_k % len(perms)][2]])
        elif mode == "quadrant_grid":
            perms4 = list(itertools.permutations(range(len(COLOR_PRESETS)), 4))
            idxs = perms4[combo_k % len(perms4)]
            c1, c2, c3 = COLOR_PRESETS[idxs[0]], COLOR_PRESETS[idxs[1]], COLOR_PRESETS[idxs[2]]
            c4 = COLOR_PRESETS[idxs[3]]
        else:
            pairs = list(itertools.permutations(range(len(COLOR_PRESETS)), 2))
            disc_radius = (0.24, 0.28, 0.30)[(combo_k // len(pairs)) % 3]
            pi, pj = pairs[combo_k % len(pairs)]
            c1, c2 = COLOR_PRESETS[pi], COLOR_PRESETS[pj]
            c3 = COLOR_PRESETS[(pj + 1) % len(COLOR_PRESETS)]
        img = Image.new("RGB", (CANVAS, CANVAS), (245, 245, 245))
        draw = ImageDraw.Draw(img)
        regions: List[Dict[str, Any]] = []
        if mode == "tricolor_vertical":
            w = CANVAS // 3
            for i, (c, name) in enumerate(zip((c1, c2, c3), ("左侧", "中间", "右侧"))):
                draw.rectangle([i * w, 0, (i + 1) * w if i < 2 else CANVAS, CANVAS], fill=c[0])
                regions.append({"name_zh": f"{name}{c[1]}竖向色块", "name_en": f"{name} {c[2]} vertical panel",
                                "color_zh": c[1], "color_en": c[2],
                                "bbox": [0, int(i * 1000 / 3), 1000, int((i + 1) * 1000 / 3)]})
        elif mode == "tricolor_horizontal":
            h = CANVAS // 3
            for i, (c, name) in enumerate(zip((c1, c2, c3), ("顶层", "中层", "底层"))):
                draw.rectangle([0, i * h, CANVAS, (i + 1) * h if i < 2 else CANVAS], fill=c[0])
                regions.append({"name_zh": f"{name}{c[1]}水平条带", "name_en": f"{name} {c[2]} horizontal stripe",
                                "color_zh": c[1], "color_en": c[2],
                                "bbox": [int(i * 1000 / 3), 0, int((i + 1) * 1000 / 3), 1000]})
        elif mode == "quadrant_grid":
            colors = [c1, c2, c3, c4]
            names = ["左上", "右上", "左下", "右下"]
            for i, (c, name) in enumerate(zip(colors, names)):
                x0 = 0 if i % 2 == 0 else CANVAS // 2
                y0 = 0 if i < 2 else CANVAS // 2
                draw.rectangle([x0, y0, x0 + CANVAS // 2, y0 + CANVAS // 2], fill=c[0])
                regions.append({"name_zh": f"{name}{c[1]}方形分区", "name_en": f"{name} {c[2]} quadrant panel",
                                "color_zh": c[1], "color_en": c[2],
                                "bbox": [int(y0 * 1000 / CANVAS), int(x0 * 1000 / CANVAS),
                                         int((y0 + CANVAS // 2) * 1000 / CANVAS),
                                         int((x0 + CANVAS // 2) * 1000 / CANVAS)]})
        else:  # centered_disc
            draw.rectangle([0, 0, CANVAS, CANVAS], fill=c1[0])
            r0 = int(CANVAS * (0.5 - disc_radius / 2))
            r1 = int(CANVAS * (0.5 + disc_radius / 2))
            draw.ellipse([r0, r0, r1, r1], fill=c2[0])
            regions = [
                {"name_zh": f"全幅{c1[1]}底板", "name_en": f"full-field {c1[2]} background",
                 "color_zh": c1[1], "color_en": c1[2], "bbox": [0, 0, 1000, 1000]},
                {"name_zh": f"正中{c2[1]}圆形徽记", "name_en": f"centered {c2[2]} disc",
                 "color_zh": c2[1], "color_en": c2[2],
                 "bbox": [int(r0 * 1000 / CANVAS), int(r0 * 1000 / CANVAS),
                          int(r1 * 1000 / CANVAS), int(r1 * 1000 / CANVAS)]},
            ]
        bboxes = [r["bbox"] for r in regions]
        if mode == "centered_disc":
            main = regions[1]
            other = regions[0]
            relation_answer = "圆形徽记位于整幅画布的正中央（水平与垂直均居中）。"
        else:
            main = regions[0]
            other = regions[-1]
            relation_answer = self._relation_label(main["bbox"], other["bbox"])
        count_templates = ["图中共有几个独立的颜色分区？", "画面被分割为几个纯色区域？",
                           "这张几何图中共包含几个颜色区块？"]
        rel_templates = ["「{b}」位于「{a}」的什么方位？", "相对「{a}」，「{b}」处于什么位置？"]
        qa = [
            self._qa(count_templates[combo_k % len(count_templates)],
                     f"共有 {len(regions)} 个独立分区。", "count", None,
                     "按色块边界线统计封闭区域数量。"),
            self._qa(f"请给出「{main['name_zh']}」的边界框 [ymin,xmin,ymax,xmax]。",
                     str(main["bbox"]), "bbox", main["bbox"], "直接读取该色块的四条边界。"),
            self._qa(f"「{main['name_zh']}」呈现什么颜色？", main["color_zh"], "color", None,
                     "该分区为单一纯色填充。"),
            self._qa(rel_templates[combo_k % len(rel_templates)].format(a=main["name_zh"], b=other["name_zh"]),
                     relation_answer, "relation", None,
                     "比较两个分区的边界框中心点坐标。"),
        ]
        entities = [{"entity_type": "几何色彩分区", "entity_name_zh": r["name_zh"],
                     "entity_name_en": r["name_en"], "bbox_1000": r["bbox"],
                     "placement_zone": "整幅画面", "visibility": "full"} for r in regions]
        entity_ids: List[Dict[str, Any]] = []  # 几何样本无实体泄漏风险
        caption_zh = (f"画面由 {len(regions)} 个纯色区域组成。"
                      f"其中「{main['name_zh']}」的颜色为{main['color_zh']}，边界框为 {main['bbox']}；"
                      f"「{other['name_zh']}」位于其{('中央' if mode == 'centered_disc' else self._relation_label(main['bbox'], other['bbox']))}。"
                      f"各颜色区域边界清晰，整体是一幅平面图案。")
        caption_en = (f"A flat graphic composed of {len(regions)} solid-color regions. "
                      f"The {main['name_en']} is {main['color_en']} and spans [ymin,xmin,ymax,xmax]={main['bbox']}; "
                      f"the {other['name_en']} lies to its {self._relation_label(main['bbox'], other['bbox'])}.")
        rgb = np.asarray(img)
        record = self._record(idx, "stage_a_geometry", plan["split"], task_family="geometry_count_color_grounding",
                              difficulty="easy", domain="geometric_primitives", entity_ids=entity_ids,
                              entity_names_zh=" / ".join(r["name_zh"] for r in regions),
                              entity_names_en=" / ".join(r["name_en"] for r in regions),
                              num_entities=len(regions), substrate=None, placement_zone=mode,
                              technique="screenprint", bboxes=bboxes, entities=entities,
                              ref_candidates=[], match_label="not_applicable",
                              caption_zh=caption_zh, caption_en=caption_en,
                              visual_analysis=f"画面由 {len(regions)} 个颜色区域组成，色块边界清晰。",
                              knowledge_expansion="画面由纯色区域组成，可作为颜色识别与区域定位的练习。",
                              qa_pairs=qa, render_meta={"seed": self.seed, "mode": mode, "canvas": [CANVAS, CANVAS]},
                              quality={"print_visibility": None, "bbox_area_ratio_min": round(min(
                                  (b[2] - b[0]) * (b[3] - b[1]) / 1_000_000 for b in bboxes), 4),
                                  "substrate_print_eligible": False, "checks_passed": True})
        return record, rgb

    # ------------------------------------------------------------------
    # 阶段 B：单图案真实商品贴合
    # ------------------------------------------------------------------
    def _gen_stage_b(self, idx: int, rng: random.Random, plan: Dict[str, Any]):
        stage, split = plan["stage"], plan["split"]
        domain = plan["domain"]
        entity = self._pick_entity(stage, split, domain, plan["domain_occ"])
        sub = self._pick_substrate(stage, split, plan["ordinal"])
        cat = self._category(sub)
        bg, lighting = self._substrate_bg(sub)
        elem = self._load_foreground(domain, entity["stem"])
        if elem is None:
            elem = Image.new("RGBA", (240, 240), (220, 40, 40, 255))
        best_choice = None
        zone_specs = list(ZONE_SPECS[cat])
        rng.shuffle(zone_specs)
        for zone_spec in zone_specs[:3]:
            rect = self._zone_rect(sub, zone_spec, rng)
            if min(rect[2] - rect[0], rect[3] - rect[1]) < 34:
                continue
            render_try = self._render_element(bg, lighting, elem, rect, rng, sub, fill=rng.uniform(0.78, 0.94))
            if render_try["bbox"] is None:
                continue
            choice = (render_try["area"], render_try["visibility"], zone_spec, rect, render_try)
            if best_choice is None or choice[0] > best_choice[0]:
                best_choice = choice
            if render_try["area"] >= 0.0015 and render_try["visibility"] >= 0.06:
                break
        if best_choice is None:
            zone_spec = zone_specs[0]
            rect = self._zone_rect(sub, zone_spec, rng)
            render = self._render_element(bg, lighting, elem, rect, rng, sub, fill=0.9)
        else:
            _, _, zone_spec, rect, render = best_choice
        if render["area"] < 0.0012:
            zy0, zx0, zy1, zx1 = sub["zone_canvas_norm"]
            full_rect = (max(2, int(zx0 * CANVAS)), max(2, int(zy0 * CANVAS)),
                         min(CANVAS - 3, int(zx1 * CANVAS)), min(CANVAS - 3, int(zy1 * CANVAS)))
            if min(full_rect[2] - full_rect[0], full_rect[3] - full_rect[1]) >= 34:
                render_full = self._render_element(bg, lighting, elem, full_rect, rng, sub, fill=0.97)
                if render_full["bbox"] is not None and render_full["area"] > render["area"]:
                    render = render_full
                    rect = full_rect
        zone_zh, zone_en = self._zone_desc(sub, zone_spec)
        bbox = render["bbox"] or [int(rect[1] * 1000 / CANVAS), int(rect[0] * 1000 / CANVAS),
                                  int(rect[3] * 1000 / CANVAS), int(rect[2] * 1000 / CANVAS)]
        tech = zone_spec["kind"]
        technique_zh = TECHNIQUE_ZH[tech]
        qa = [
            self._qa(rng.choice(QUESTION_TEMPLATES["identify"]), entity["full_title_zh"], "name", None,
                     "依据图案的轮廓、配色与内部文字/纹样判断。"),
            self._qa(rng.choice(QUESTION_TEMPLATES["grounding"]), str(bbox), "bbox", bbox,
                     "边界框由图中图案的实际可见区域确定。"),
            self._qa(rng.choice(QUESTION_TEMPLATES["color"]), entity["dominant_color_zh"], "color", None,
                     "取图案主体区域的主要颜色。"),
            self._qa(rng.choice(QUESTION_TEMPLATES["location"]), zone_zh, "location", None,
                     f"依据该图案在画面中的相对位置判断：{zone_zh}。"),
        ]
        if rng.random() < 0.55:
            qa.append(self._qa(rng.choice(QUESTION_TEMPLATES["geometry"]), entity["geometry_zh"], "name",
                               None, "由图案外接宽高比判断。"))
        entities = [{"entity_type": entity["entity_type"], "entity_name_zh": entity["entity_name_zh"],
                     "entity_name_en": entity["entity_name_en"], "full_title_zh": entity["full_title_zh"],
                     "bbox_1000": bbox, "placement_zone": zone_spec["id"], "visibility": "full",
                     "technique": technique_zh}]
        entity_ids = [{"domain": domain, "stem": entity["stem"], "role": "primary",
                       "name_zh": entity["entity_name_zh"], "name_en": entity["entity_name_en"]}]
        surface_zh = sub["surface_zh"]
        caption_zh = (f"图像为一件{surface_zh}的{sub['category_zh']}实拍商品（{sub.get('scene_style_zh', '')}），"
                      f"在{zone_zh}以{technique_zh}工艺贴附有【{entity['full_title_zh']}】。"
                      f"图案特征：{entity['design_analysis_zh']}"
                      f"印花主色为{entity['dominant_color_zh']}，外形呈{entity['geometry_zh']}；"
                      f"边缘随面料褶皱起伏，与底布形成自然的丝印/热转印质感。"
                      f"该图案在整图中的边界框为 {bbox}。")
        caption_en = (f"A real product photo of a {sub['category_en']} with {sub['surface_en']}. "
                      f"Printed at the {zone_en} via {TECHNIQUE_EN[tech]} is the "
                      f"{entity['full_title_en']}. Design analysis: {entity['design_analysis_en']} "
                      f"The print is {entity['dominant_color_en']} with {entity['geometry_en']}; "
                      f"it conforms to fabric folds. Ground-truth box: [ymin,xmin,ymax,xmax]={bbox}.")
        vis = round(render["visibility"], 4)
        record = self._record(idx, stage, split, task_family="single_element_grounding_attributes",
                              difficulty=sub["difficulty"], domain=domain, entity_ids=entity_ids,
                              entity_names_zh=entity["entity_name_zh"], entity_names_en=entity["entity_name_en"],
                              num_entities=1, substrate=sub, placement_zone=zone_spec["id"], technique=tech,
                              bboxes=[bbox], entities=entities, ref_candidates=[], match_label="not_applicable",
                              caption_zh=caption_zh, caption_en=caption_en,
                              visual_analysis=f"面料: {surface_zh}; 工艺: {technique_zh}; 印花可见度: {vis}。",
                              knowledge_expansion=f"{entity['full_title_zh']} - {entity['cultural_context_zh']}",
                              qa_pairs=qa,
                              render_meta={"seed": self.seed, "canvas": [CANVAS, CANVAS], "stage": stage,
                                           "blend_mode": render["mode"], "fill": round(render["fill"], 3),
                                           "attempt": render["attempt"], "zone_rect_px": list(rect)},
                              quality={"print_visibility": vis, "bbox_area_ratio_min": round(render["area"], 4),
                                       "substrate_print_eligible": bool(sub["print_eligible"]),
                                       "substrate_difficulty": sub["difficulty"], "checks_passed": vis >= 0.05})
        return record, render["rgb"]

    # ------------------------------------------------------------------
    # 阶段 C：多图案空间排布
    # ------------------------------------------------------------------
    def _gen_stage_c(self, idx: int, rng: random.Random, plan: Dict[str, Any]):
        stage, split = plan["stage"], plan["split"]
        domain = plan["domain"]
        entity1 = self._pick_entity(stage, split, domain, plan["domain_occ"])
        domain2 = plan["domain2"]
        entity2 = self._pick_entity(stage, split, domain2, plan["domain2_occ"])
        sub = self._pick_substrate(stage, split, plan["ordinal"])
        cat = self._category(sub)
        bg, lighting = self._substrate_bg(sub)
        y0, x0, y1, x1 = sub["zone_canvas_norm"]
        zone_px_w = int((x1 - x0) * CANVAS)
        zone_px_h = int((y1 - y0) * CANVAS)
        candidate_layouts = list(LAYOUTS) + list(SMALL_LAYOUTS)
        use_triple = (rng.random() < 0.12 and cat in ("Topwear", "Bags", "Dress")
                      and min(zone_px_w, zone_px_h) >= 260)
        if use_triple:
            candidate_layouts = [TRIPLE_LAYOUT]
        scored = []
        for lay in candidate_layouts:
            min_dim = min(min((fx1 - fx0) * zone_px_w, (fy1 - fy0) * zone_px_h)
                          for fy0, fx0, fy1, fx1 in lay["rects"])
            scored.append((min_dim, lay))
        best_dim = max(sc for sc, _ in scored)
        feasible = [lay for sc, lay in scored if sc >= max(30, best_dim - 8)]
        if not feasible:
            feasible = [max(scored, key=lambda item: item[0])[1]]
        layout = rng.choice(feasible)
        if use_triple:
            domains3 = [d for d in DOMAINS if d not in (domain, domain2)]
            domain3 = rng.choice(domains3)
            entity3 = self._entity(domain3, rng.choice(self._get_entity_perm(stage, split, domain3)))
            entities_src = [entity1, entity2, entity3]
        else:
            entities_src = [entity1, entity2]
        rgb = bg
        placed: List[Dict[str, Any]] = []
        render_meta_items = []
        for i, ((fy0, fx0, fy1, fx1), ent) in enumerate(zip(layout["rects"], entities_src)):
            px0 = int((x0 + fx0 * (x1 - x0)) * CANVAS)
            px1 = int((x0 + fx1 * (x1 - x0)) * CANVAS)
            py0 = int((y0 + fy0 * (y1 - y0)) * CANVAS)
            py1 = int((y0 + fy1 * (y1 - y0)) * CANVAS)
            if px1 - px0 < 28 or py1 - py0 < 28:
                continue
            elem = self._load_foreground(ent["domain"], ent["stem"])
            if elem is None:
                continue
            render = self._render_element(rgb, lighting, elem, (px0, py0, px1, py1), rng, sub,
                                          fill=0.86 if i == 0 else 0.78)
            rgb = render["rgb"]
            bbox = render["bbox"]
            if bbox is None:
                continue
            placed.append({"entity": ent, "bbox": bbox, "visibility": render["visibility"],
                           "mode": render["mode"], "area": render["area"]})
            render_meta_items.append({"role": "main" if i == 0 else f"secondary_{i}",
                                      "blend_mode": render["mode"], "fill": round(render["fill"], 3),
                                      "visibility": round(render["visibility"], 4)})
        placed = [p for p in placed if p["area"] >= 0.001]
        if len(placed) < 2:
            # 极小可印区的兜底：仍保留 stage_c 记录（单图案），避免阶段配额漂移
            rec, rgb_b = self._gen_stage_b(idx, rng, plan)
            rec["task_stage"] = stage
            rec["stage_title_zh"] = STAGE_TITLES[stage]
            rec["task_family"] = "multi_element_compact_fallback"
            return rec, rgb_b
        bboxes = [p["bbox"] for p in placed]
        names_zh = " / ".join(p["entity"]["entity_name_zh"] for p in placed)
        names_en = " / ".join(p["entity"]["entity_name_en"] for p in placed)
        main = placed[0]
        relations = [self._relation_label(main["bbox"], p["bbox"]) for p in placed[1:]]
        qa = [
            self._qa(rng.choice(QUESTION_TEMPLATES["count"]), f"共有 {len(placed)} 处独立印花/视觉元素。",
                     "count", None, "按互不重叠的图案实例逐一计数。"),
            self._qa(f"主视觉图案是什么？", main["entity"]["full_title_zh"], "name", None,
                     "主图案面积最大、位于画面中心区域。"),
            self._qa(rng.choice(QUESTION_TEMPLATES["grounding"]).replace("该图案", "主视觉图案"),
                     str(main["bbox"]), "bbox", main["bbox"], "读取主图案的实际可见边界。"),
            self._qa(f"次要图案「{placed[1]['entity']['entity_name_zh']}」的边界框坐标是多少？",
                     str(placed[1]["bbox"]), "bbox", placed[1]["bbox"], "读取次要图案的外接框。"),
            self._qa(QUESTION_TEMPLATES["relation"][0].format(a="主视觉图案", b="次要图案"),
                     relations[0], "relation", None, "比较两图案边界框中心点的相对方位。"),
            self._qa(f"次要图案「{placed[1]['entity']['entity_name_zh']}」的主体颜色是什么？",
                     placed[1]["entity"]["dominant_color_zh"], "color", None, "取次要图案有效像素中位色。"),
        ]
        # 负样本问题：随机询问一个未出现在图中的图案
        absent_pool = [s for s in self.asset_domains[domain] if s not in
                       {p["entity"]["stem"] for p in placed}]
        if absent_pool and rng.random() < 0.6:
            absent_stem = rng.choice(absent_pool)
            absent = self._entity(domain, absent_stem)
            qa.append(self._qa(rng.choice(QUESTION_TEMPLATES["present"]).format(name=absent["entity_name_zh"]),
                               f"否，图中不存在【{absent['entity_name_zh']}】。", "yes_no", None,
                               "逐项比对画面中的实际印花，未发现该实体。"))
        entities_json = [{"entity_type": p["entity"]["entity_type"],
                          "entity_name_zh": p["entity"]["entity_name_zh"],
                          "entity_name_en": p["entity"]["entity_name_en"],
                          "full_title_zh": p["entity"]["full_title_zh"],
                          "bbox_1000": p["bbox"], "placement_zone": layout["id"],
                          "visibility": "full"} for p in placed]
        entity_ids = [{"domain": p["entity"]["domain"], "stem": p["entity"]["stem"],
                       "role": "primary" if i == 0 else f"secondary_{i}",
                       "name_zh": p["entity"]["entity_name_zh"], "name_en": p["entity"]["entity_name_en"]}
                      for i, p in enumerate(placed)]
        caption_zh = (f"该{sub['category_zh']}表面采用「{layout['zh']}」的多图案组合，"
                      f"共包含 {len(placed)} 个独立印花。主视觉图案为【{main['entity']['full_title_zh']}】"
                      f"（边界框 {main['bbox']}）；其余图案依次为 "
                      + "；".join(f"【{p['entity']['full_title_zh']}】（边界框 {p['bbox']}，位于主图的{relations[i]}）"
                                 for i, p in enumerate(placed[1:]))
                      + "。各图案随面料褶皱自然起伏，空间上互不重叠。")
        caption_en = (f"Multi-element composition on a {sub['category_en']} using the {layout['en']} layout "
                      f"with {len(placed)} independent prints. Primary: {main['entity']['full_title_en']} "
                      f"[ymin,xmin,ymax,xmax]={main['bbox']}. Others: "
                      + "; ".join(f"{p['entity']['full_title_en']} {p['bbox']} ({relations[i]})"
                                  for i, p in enumerate(placed[1:])) + ".")
        record = self._record(idx, stage, split, task_family="multi_element_count_relation_grounding",
                              difficulty=sub["difficulty"], domain=domain + "+" + domain2 if not use_triple else
                              domain + "+" + domain2 + "+" + entity3["domain"],
                              entity_ids=entity_ids, entity_names_zh=names_zh, entity_names_en=names_en,
                              num_entities=len(placed), substrate=sub, placement_zone=layout["id"],
                              technique="screenprint", bboxes=bboxes, entities=entities_json,
                              ref_candidates=[], match_label="not_applicable",
                              caption_zh=caption_zh, caption_en=caption_en,
                              visual_analysis=f"布局: {layout['zh']}; 图案数: {len(placed)}; 空间关系: {'、'.join(relations)}。",
                              knowledge_expansion="主图案: " + main["entity"]["full_title_zh"] + "; 其余: " +
                                                  "、".join(p["entity"]["full_title_zh"] for p in placed[1:]),
                              qa_pairs=qa,
                              render_meta={"seed": self.seed, "canvas": [CANVAS, CANVAS], "stage": stage,
                                           "layout": layout["id"], "items": render_meta_items},
                              quality={"print_visibility": round(min(p["visibility"] for p in placed), 4),
                                       "bbox_area_ratio_min": round(min(p["area"] for p in placed), 4),
                                       "substrate_print_eligible": bool(sub["print_eligible"]),
                                       "substrate_difficulty": sub["difficulty"],
                                       "checks_passed": all(p["visibility"] >= 0.045 for p in placed)})
        return record, rgb

    # ------------------------------------------------------------------
    # 阶段 D：参考图匹配
    # ------------------------------------------------------------------
    def _gen_stage_d(self, idx: int, rng: random.Random, plan: Dict[str, Any]):
        stage, split = plan["stage"], plan["split"]
        domain = plan["domain"]
        target = self._pick_entity(stage, split, domain, plan["domain_occ"])
        sub = self._pick_substrate(stage, split, plan["ordinal"])
        cat = self._category(sub)
        bg, lighting = self._substrate_bg(sub)

        n_candidates = rng.choices([2, 3, 4], weights=[0.3, 0.4, 0.3])[0]
        case_match = rng.random() < 0.72
        if case_match:
            printed = target
            neg_count = n_candidates - 1
            negatives = self._hard_negatives(domain, target["stem"], neg_count, rng)
            candidate_entities = [target] + [self._entity(domain, s) for s in negatives]
            match_label = "match"
        else:
            pool = self._get_entity_perm(stage, split, domain)
            printed_stem = rng.choice([s for s in pool if s != target["stem"]])
            printed = self._entity(domain, printed_stem)
            negatives = self._hard_negatives(domain, printed["stem"], n_candidates, rng)
            candidate_entities = [self._entity(domain, s) for s in negatives]
            match_label = "no_match"
        rng.shuffle(candidate_entities)
        letters = [chr(ord("A") + i) for i in range(len(candidate_entities))]
        positive_idx = next((i for i, e in enumerate(candidate_entities) if e["stem"] == printed["stem"]), None)

        # 商品图（640x640）渲染：最多尝试 3 个区位，确保实例 bbox 不过小
        elem = self._load_foreground(printed["domain"], printed["stem"]) or Image.new("RGBA", (240, 240), (220, 40, 40, 255))
        zone_specs = list(ZONE_SPECS[cat]); rng.shuffle(zone_specs)
        best_zone = None
        for zs in zone_specs[:3]:
            r_try = self._zone_rect(sub, zs, rng)
            if min(r_try[2] - r_try[0], r_try[3] - r_try[1]) < 34:
                continue
            render_try = self._render_element(bg, lighting, elem, r_try, rng, sub, fill=rng.uniform(0.84, 0.98))
            if render_try["bbox"] is None:
                continue
            if best_zone is None or render_try["area"] > best_zone[0]:
                best_zone = (render_try["area"], zs, r_try, render_try)
            if render_try["area"] >= 0.0015 and render_try["visibility"] >= 0.06:
                break
        if best_zone is None:
            zs = zone_specs[0]; r_try = self._zone_rect(sub, zs, rng)
            render_try = self._render_element(bg, lighting, elem, r_try, rng, sub, fill=0.95)
            best_zone = (render_try.get("area", 0), zs, r_try, render_try)
        _, zone_spec, rect, render = best_zone
        zone_zh, zone_en = self._zone_desc(sub, zone_spec)
        product_rgb = render["rgb"]
        bbox640 = render["bbox"] or [int(rect[1] * 1000 / CANVAS), int(rect[0] * 1000 / CANVAS),
                                     int(rect[3] * 1000 / CANVAS), int(rect[2] * 1000 / CANVAS)]
        product_box = to_rel_bbox_1000([D_BAND + int(bbox640[0] * CANVAS / 1000), int(bbox640[1] * CANVAS / 1000),
                                        D_BAND + int(bbox640[2] * CANVAS / 1000), int(bbox640[3] * CANVAS / 1000)],
                                      CANVAS, D_HEIGHT)

        # 拼装 640x840 参考图 + 商品图
        full = Image.new("RGB", (CANVAS, D_HEIGHT), (250, 250, 250))
        full.paste(Image.fromarray(product_rgb), (0, D_BAND))
        pane_w = CANVAS / len(candidate_entities)
        ref_candidates: List[Dict[str, Any]] = []
        draw = ImageDraw.Draw(full)
        for i, ent in enumerate(candidate_entities):
            x_off = int(i * pane_w)
            tile = Image.new("RGB", (int(pane_w), D_BAND - 12), (255, 255, 255))
            ref = self._load_foreground(ent["domain"], ent["stem"], target_w=300)
            if ref is not None:
                max_w = int(pane_w) - 22
                max_h = D_BAND - 42
                ref.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                rx = (tile.width - ref.width) // 2
                ry = (tile.height - ref.height) // 2 + 10
                white = Image.new("RGBA", ref.size, (255, 255, 255, 255))
                white.alpha_composite(ref)
                tile.paste(white.convert("RGB"), (rx, ry))
                box = to_rel_bbox_1000([ry + 6, x_off + rx, ry + 6 + ref.height, x_off + rx + ref.width],
                                       CANVAS, D_HEIGHT)
            else:
                box = to_rel_bbox_1000([20, x_off + 20, 180, x_off + int(pane_w) - 20], CANVAS, D_HEIGHT)
            full.paste(tile, (x_off, 6))
            draw.rectangle([x_off + 1, 7, x_off + int(pane_w) - 2, D_BAND - 7], outline=(170, 170, 175), width=1)
            draw.text((x_off + 8, 10), letters[i], fill=(30, 30, 30))
            ref_candidates.append({"candidate_id": letters[i], "pane_index": i,
                                   "entity_name_zh": ent["entity_name_zh"], "entity_name_en": ent["entity_name_en"],
                                   "bbox_1000": box, "role": "positive" if ent["stem"] == printed["stem"] else "negative"})
        draw.line([(0, D_BAND - 2), (CANVAS, D_BAND - 2)], fill=(160, 160, 165), width=2)
        rgb = np.asarray(full)

        if case_match and positive_idx is not None:
            pos = ref_candidates[positive_idx]
            qa = [
                self._qa(QUESTION_TEMPLATES["match_which"][0].format(letters="/".join(letters)),
                         f"候选{pos['candidate_id']}（{printed['full_title_zh']}）与商品印花一致。", "name", None,
                         "参考图与商品印花的轮廓、配色和内部结构逐项对应。"),
                self._qa("商品上该图案的边界框是多少？", str(product_box), "bbox", product_box,
                         "由商品图中的实际图案区域确定并映射到整图坐标。"),
                self._qa(QUESTION_TEMPLATES["candidate_bbox"][0].format(letter=pos["candidate_id"]),
                         str(pos["bbox_1000"]), "bbox", pos["bbox_1000"], "读取候选参考图在整图中的位置。"),
            ]
        else:
            qa = [
                self._qa(QUESTION_TEMPLATES["match_exists"][0],
                         f"没有。候选参考图均与商品印花不匹配；商品上的图案为【{printed['full_title_zh']}】。",
                         "yes_no", None, "逐张比对候选参考图，均不存在与商品印花一致的图案。"),
                self._qa("商品上实际出现的图案是什么？", printed["full_title_zh"], "name", None,
                         "直接识别商品图上的印花内容。"),
                self._qa("商品上该图案的边界框是多少？", str(product_box), "bbox", product_box,
                         "读取商品图区域中该图案的外接框。"),
            ]
        caption_zh = (f"参考图与商品对照：图像顶部为 {len(candidate_entities)} 张候选参考图"
                      f"（{'/'.join(letters)}），下方为一件{sub['category_zh']}实拍商品。"
                      f"商品在{zone_zh}印有【{printed['full_title_zh']}】。"
                      + (f"经比对，候选{letters[positive_idx]}与其一致（候选框 {ref_candidates[positive_idx]['bbox_1000']}，"
                         f"商品实例框 {product_box}）。" if case_match and positive_idx is not None else
                         f"经比对，所有候选参考图均不匹配；商品图案为【{printed['full_title_zh']}】（实例框 {product_box}）。")
                      + "可用于核对同一图案在参考图与实物商品上的一致性。")
        caption_en = (f"In-context reference matching sample with {len(candidate_entities)} candidate references "
                      f"({'/'.join(letters)}) on top and a real product photo below. "
                      f"The product shows {printed['full_title_en']} at the {zone_en}. "
                      + (f"Candidate {letters[positive_idx]} matches the product instance." if case_match and positive_idx is not None
                         else f"None of the candidates matches; the printed entity is {printed['full_title_en']}.")
                      + " This sample checks whether the same pattern appears on both the reference images and the physical product.")
        entities_json = [{"entity_type": "商品印花实例", "entity_name_zh": printed["entity_name_zh"],
                          "entity_name_en": printed["entity_name_en"], "full_title_zh": printed["full_title_zh"],
                          "bbox_1000": product_box, "placement_zone": zone_spec["id"], "visibility": "full"}]
        entity_ids = [{"domain": domain, "stem": printed["stem"], "role": "printed_instance",
                       "name_zh": printed["entity_name_zh"], "name_en": printed["entity_name_en"]}]
        vis = round(render["visibility"], 4)
        record = self._record(idx, stage, split, task_family="reference_matching_grounding",
                              difficulty="medium", domain=domain, entity_ids=entity_ids,
                              entity_names_zh=printed["entity_name_zh"], entity_names_en=printed["entity_name_en"],
                              num_entities=1, substrate=sub, placement_zone=zone_spec["id"],
                              technique=zone_spec["kind"], bboxes=[product_box], entities=entities_json,
                              ref_candidates=ref_candidates, match_label=match_label,
                              caption_zh=caption_zh, caption_en=caption_en,
                              visual_analysis=f"候选数: {len(candidate_entities)}; 匹配标签: {match_label}; 印花可见度: {vis}。",
                              knowledge_expansion=f"{printed['full_title_zh']} - {printed['cultural_context_zh']}",
                              qa_pairs=qa,
                              render_meta={"seed": self.seed, "canvas": [CANVAS, D_HEIGHT], "stage": stage,
                                           "candidate_count": len(candidate_entities), "match_label": match_label,
                                           "blend_mode": render["mode"], "product_offset_y": D_BAND},
                              quality={"print_visibility": vis, "bbox_area_ratio_min": round(render["area"], 4),
                                       "substrate_print_eligible": bool(sub["print_eligible"]),
                                       "substrate_difficulty": sub["difficulty"],
                                       "checks_passed": vis >= 0.045})
        return record, rgb

    # ------------------------------------------------------------------
    # 阶段 E：负样本与证据不足
    # ------------------------------------------------------------------
    def _substrate_color(self, sub: Dict[str, Any]) -> Tuple[str, str]:
        meta = self.commodity_meta.get(sub["substrate_id"], {})
        return meta.get("dominant_color_zh", "自然色"), meta.get("dominant_color_en", "natural tone")

    def _gen_stage_e(self, idx: int, rng: random.Random, plan: Dict[str, Any]):
        stage, split = plan["stage"], plan["split"]
        domain = plan["domain"]
        target = self._pick_entity(stage, split, domain, plan["domain_occ"])
        case = rng.choices(["absent_clean", "occluded", "hard_distractor"], weights=[0.4, 0.3, 0.3])[0]
        pool = self._substrate_perm_list(stage, split)
        if case == "hard_distractor":
            allowed = pool
        else:
            allowed = [s for s in pool if s["difficulty"] in ("easy", "medium")]
            if not allowed:
                allowed = pool
        sub = allowed[plan["ordinal"] % len(allowed)]
        sub_color_zh, sub_color_en = self._substrate_color(sub)
        cat = self._category(sub)
        bg, lighting = self._substrate_bg(sub)
        zone_specs = list(ZONE_SPECS[cat]); rng.shuffle(zone_specs)
        zone_spec = zone_specs[0]
        rect = self._zone_rect(sub, zone_spec, rng)
        zone_zh, zone_en = self._zone_desc(sub, zone_spec)
        render = None
        bboxes: List[List[int]] = []
        entities_json: List[Dict[str, Any]] = []
        ref_candidates: List[Dict[str, Any]] = []
        match_label = "no_match"

        if case == "occluded":
            elem = self._load_foreground(domain, target["stem"]) or Image.new("RGBA", (240, 240), (200, 30, 30, 255))
            choices = []
            for zs in zone_specs[:3]:
                r_try = self._zone_rect(sub, zs, rng)
                if min(r_try[2] - r_try[0], r_try[3] - r_try[1]) < 34:
                    continue
                rr = self._render_element(bg, lighting, elem, r_try, rng, sub, fill=rng.uniform(0.84, 0.98))
                if rr["bbox"] is None:
                    continue
                choices.append((rr["area"], zs, r_try, rr))
            if choices:
                _, zone_spec, rect, render = max(choices, key=lambda item: item[0])
                zone_zh, zone_en = self._zone_desc(sub, zone_spec)
            rgb = render["rgb"]
            bbox = render["bbox"] or [int(rect[1] * 1000 / CANVAS), int(rect[0] * 1000 / CANVAS),
                                      int(rect[3] * 1000 / CANVAS), int(rect[2] * 1000 / CANVAS)]
            rgb = self._cover_bbox_with_fabric(rgb, bbox, rng, sub)
            match_label = "insufficient_evidence"
            qa = [
                self._qa(f"图中能否确认商品上存在【{target['full_title_zh']}】图案？",
                         f"无法确认：{zone_zh}被同色面料/遮挡物覆盖，未暴露可辨识的图案。",
                         "uncertain", None, "目标区域被完全遮挡，无法观察有效证据。"),
                self._qa("当前可观察到该图案吗？", "不可见。", "yes_no", None, "遮挡区域无图案像素。"),
            ]
            caption_zh = (f"该{sub['category_zh']}的{zone_zh}被同色面料/遮挡物覆盖，"
                          f"无法确认是否存在【{target['full_title_zh']}】。"
                          f"该区域没有可见图案，无法进一步确认。")
            caption_en = (f"The {zone_en} of this {sub['category_en']} is fully occluded by a matching "
                          f"fabric patch; whether {target['full_title_en']} is present is unverifiable. "
                          f"The region shows no visible pattern, so the answer cannot be confirmed.")
            quality = {"print_visibility": None, "bbox_area_ratio_min": 0.0,
                       "substrate_print_eligible": bool(sub["print_eligible"]), "occluded": True,
                       "checks_passed": True}
        elif case == "hard_distractor":
            neg_stem = self._hard_negatives(domain, target["stem"], 1, rng)[0]
            distractor = self._entity(domain, neg_stem)
            elem = self._load_foreground(domain, distractor["stem"]) or Image.new("RGBA", (240, 240), (30, 120, 220, 255))
            choices = []
            for zs in zone_specs[:3]:
                r_try = self._zone_rect(sub, zs, rng)
                if min(r_try[2] - r_try[0], r_try[3] - r_try[1]) < 34:
                    continue
                rr = self._render_element(bg, lighting, elem, r_try, rng, sub, fill=rng.uniform(0.84, 0.98))
                if rr["bbox"] is None:
                    continue
                choices.append((rr["area"], zs, r_try, rr))
            if choices:
                _, zone_spec, rect, render = max(choices, key=lambda item: item[0])
                zone_zh, zone_en = self._zone_desc(sub, zone_spec)
            rgb = render["rgb"]
            bbox = render["bbox"] or [int(rect[1] * 1000 / CANVAS), int(rect[0] * 1000 / CANVAS),
                                      int(rect[3] * 1000 / CANVAS), int(rect[2] * 1000 / CANVAS)]
            bboxes = [bbox]
            entities_json = [{"entity_type": "高相似度干扰图案", "entity_name_zh": distractor["entity_name_zh"],
                              "entity_name_en": distractor["entity_name_en"], "full_title_zh": distractor["full_title_zh"],
                              "bbox_1000": bbox, "placement_zone": zone_spec["id"], "visibility": "full",
                              "role": "distractor"}]
            qa = [
                self._qa(QUESTION_TEMPLATES["present"][0].format(name=target["entity_name_zh"]),
                         f"否，图中不存在【{target['entity_name_zh']}】。", "yes_no", None,
                         "图中实际图案与目标实体在轮廓/细节上存在差异。"),
                self._qa(QUESTION_TEMPLATES["distractor"][0],
                         f"图中实际出现的图案为【{distractor['full_title_zh']}】（边界框 {bbox}）。",
                         "name", None, "识别画面中真实存在的图案。"),
                self._qa("图中实际图案的边界框是多少？", str(bbox), "bbox", bbox,
                         "读取干扰图案的外接框。"),
            ]
            caption_zh = (f"图中商品在{zone_spec['zh']}印有与【{target['full_title_zh']}】高度相似的"
                          f"【{distractor['full_title_zh']}】，但并非目标实体。"
                          f"两处图案外形相近，需要仔细区分。")
            caption_en = (f"The product shows {distractor['full_title_en']} at the {zone_spec['en']}, which is "
                          f"visually similar to but distinct from {target['full_title_en']}. "
                          f"The two patterns look similar and require careful discrimination.")
            entities_json[0]["is_target"] = False
            quality = {"print_visibility": round(render["visibility"], 4), "bbox_area_ratio_min": round(render["area"], 4),
                       "substrate_print_eligible": bool(sub["print_eligible"]), "distractor": True,
                       "checks_passed": render["visibility"] >= 0.045}
        else:  # absent_clean
            rgb = bg.copy()
            present_qs = ["图中商品上是否存在【{name}】图案？", "商品表面能看到【{name}】的印花吗？",
                          "这张商品图中是否出现了【{name}】？"]
            color_qs = ["该商品的主要底色是什么？", "商品底布是什么颜色？", "画面中商品的面料底色是？"]
            part_qs = ["图中任何位置能看到【{name}】的任何部分吗？", "画面中有【{name}】的局部或整体吗？",
                       "商品上是否有与【{name}】相符的图案？"]
            qa = [
                self._qa(rng.choice(present_qs).format(name=target["entity_name_zh"]),
                         f"否，图中商品表面没有【{target['entity_name_zh']}】图案。", "yes_no", None,
                         "画面中未出现该实体的任何印花像素。"),
                self._qa(rng.choice(color_qs), sub_color_zh, "color", None,
                         "取商品可见区域的中位色。"),
                self._qa(rng.choice(part_qs).format(name=target["entity_name_zh"]),
                         "没有，画面中未出现该实体的任何部分。", "yes_no", None,
                         "检查商品各区域均未发现该实体的可见像素。"),
            ]
            caption_zh = rng.choice([
                f"该{sub_color_zh}{sub['category_zh']}（{sub['surface_zh']}，{sub.get('scene_style_zh', '')}）上未见【{target['full_title_zh']}】图案。"
                f"画面中没有出现该图案。",
                f"这张{sub_color_zh}{sub['category_zh']}实拍画面中并不存在【{target['full_title_zh']}】，"
                f"商品表面为{sub['surface_zh']}，没有出现该图案。",
                f"画面展示一件{sub_color_zh}{sub['category_zh']}（{sub['surface_zh']}），"
                f"其上没有出现【{target['full_title_zh']}】。",
            ])
            caption_en = (f"The {sub_color_en} {sub['category_en']} ({sub['surface_en']}) shows no "
                          f"{target['full_title_en']}. This negative sample trains correct "
                          f"rejection and reduces false positives.")
            quality = {"print_visibility": None, "bbox_area_ratio_min": 0.0,
                       "substrate_print_eligible": bool(sub["print_eligible"]), "absent": True,
                       "checks_passed": True}
        entity_ids = [{"domain": domain, "stem": target["stem"], "role": "queried_target",
                       "name_zh": target["entity_name_zh"], "name_en": target["entity_name_en"]}]
        record = self._record(idx, stage, split, task_family=f"negative_{case}",
                              difficulty=sub["difficulty"], domain=domain, entity_ids=entity_ids,
                              entity_names_zh=target["entity_name_zh"], entity_names_en=target["entity_name_en"],
                              num_entities=len(bboxes), substrate=sub, placement_zone=zone_spec["id"],
                              technique=zone_spec["kind"], bboxes=bboxes, entities=entities_json,
                              ref_candidates=ref_candidates, match_label=match_label,
                              caption_zh=caption_zh, caption_en=caption_en,
                              visual_analysis=f"样本类型: {case}; 目标可见性: {'不可见' if case != 'hard_distractor' else '相似干扰'}。",
                              knowledge_expansion=f"{target['full_title_zh']} - {target['cultural_context_zh']}",
                              qa_pairs=qa,
                              render_meta={"seed": self.seed, "canvas": [CANVAS, CANVAS], "stage": stage,
                                           "case": case, "zone_rect_px": list(rect)},
                              quality=quality)
        return record, rgb

    # ------------------------------------------------------------------
    # 辅助：硬负样本 / 遮挡
    # ------------------------------------------------------------------
    def _hard_negatives(self, domain: str, stem: str, k: int, rng: random.Random) -> List[str]:
        nb = self.neighbors.get(f"{domain}/{stem}", [])
        distinct = [n["stem"] for n in nb if n.get("distance", 0) >= 6
                    and n["stem"] != stem and n["stem"] in self.asset_domains.get(domain, {})]
        near_dup = [n["stem"] for n in nb if n["stem"] != stem
                    and n["stem"] in self.asset_domains.get(domain, {}) and n["stem"] not in distinct]
        cands = distinct + near_dup
        if len(cands) < k:
            others = [s for s in self.asset_domains[domain] if s != stem and s not in cands]
            rng.shuffle(others)
            cands.extend(others[: k - len(cands)])
        return rng.sample(cands, min(k, len(cands)))

    def _cover_bbox_with_fabric(self, rgb: np.ndarray, bbox_1000: List[int],
                                rng: random.Random, sub: Dict[str, Any]) -> np.ndarray:
        """用与周围面料一致的纹理块完全遮挡目标区域（模拟同色布料/遮挡物）。

        从可印区内选取与 bbox 外圈面料颜色最接近的纹理块，避免把白底背景误当遮挡物。
        """
        y0, x0, y1, x1 = [int(v * CANVAS / 1000) for v in bbox_1000]
        margin = 7
        py0, py1 = max(2, y0 - margin), min(CANVAS - 3, y1 + margin)
        px0, px1 = max(2, x0 - margin), min(CANVAS - 3, x1 + margin)
        bh, bw = py1 - py0, px1 - px0
        if bh <= 2 or bw <= 2:
            return rgb
        zy0, zx0, zy1, zx1 = sub["zone_canvas_norm"]
        zx0, zy0 = max(0, int(zx0 * CANVAS)), max(0, int(zy0 * CANVAS))
        zx1, zy1 = min(CANVAS, int(zx1 * CANVAS)), min(CANVAS, int(zy1 * CANVAS))
        # 外圈面料中位色（用于挑选最接近的遮挡纹理块）
        ry0, ry1 = max(0, py0 - 14), min(CANVAS, py1 + 14)
        rx0, rx1 = max(0, px0 - 14), min(CANVAS, px1 + 14)
        ring = rgb[ry0:ry1, rx0:rx1].reshape(-1, 3)
        mask = np.ones((ry1 - ry0, rx1 - rx0), dtype=bool)
        mask[py0 - ry0:py1 - ry0, px0 - rx0:px1 - rx0] = False
        ring = rgb[ry0:ry1, rx0:rx1][mask]
        ring_med = np.median(ring, axis=0) if len(ring) else np.array([120, 120, 120])
        candidates = []
        for step in (1.15, 1.6, 2.2):
            for dx, dy in ((bw * step, 0), (-bw * step, 0), (0, bh * step), (0, -bh * step),
                           (bw * step, bh * step), (-bw * step, -bh * step)):
                sx, sy = int(px0 + dx), int(py0 + dy)
                if sx < zx0 or sy < zy0 or sx + bw > zx1 or sy + bh > zy1:
                    continue
                patch = rgb[sy:sy + bh, sx:sx + bw]
                dist = float(np.linalg.norm(np.median(patch.reshape(-1, 3), axis=0) - ring_med))
                candidates.append((dist, patch.copy()))
        out = rgb.copy()
        if candidates:
            _, patch = min(candidates, key=lambda item: item[0])
            patch = cv2.GaussianBlur(patch, (3, 3), sigmaX=0.8)
        else:
            base = np.clip(ring_med[None, None, :] + np.random.default_rng(7).normal(0, 2.5, (bh, bw, 3)), 0, 255)
            patch = cv2.GaussianBlur(base.astype(np.uint8), (5, 5), sigmaX=1.2)
        out[py0:py1, px0:px1] = patch
        return out

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def _dispatch(self, idx: int, rng: random.Random, plan: Dict[str, Any]):
        stage = plan["stage"]
        if stage == "stage_a_geometry":
            return self._gen_stage_a(idx, rng, plan)
        if stage == "stage_b_single_element":
            return self._gen_stage_b(idx, rng, plan)
        if stage == "stage_c_multi_element":
            return self._gen_stage_c(idx, rng, plan)
        if stage == "stage_d_reference_match":
            return self._gen_stage_d(idx, rng, plan)
        return self._gen_stage_e(idx, rng, plan)

    def generate_sample(self, idx: int, images_dir: Path, debug_dir: Optional[Path] = None):
        plan = self.assign[idx]
        record, rgb, best_min_area = None, None, 0.0
        for attempt in range(3):
            rng = random.Random(_seed_for(self.seed, "sample", idx, attempt) if attempt else
                                _seed_for(self.seed, "sample", idx))
            rec_try, rgb_try = self._dispatch(idx, rng, plan)
            boxes = json.loads(rec_try["bboxes_norm_1000"])
            min_area = min(((b[2] - b[0]) * (b[3] - b[1]) / 1_000_000 for b in boxes), default=1.0)
            if record is None or min_area > best_min_area:
                record, rgb, best_min_area = rec_try, rgb_try, min_area
            if min_area >= 0.001:
                break
        quality = json.loads(record["quality_json"])
        quality["min_bbox_area"] = round(best_min_area, 6)
        record["quality_json"] = json.dumps(quality, ensure_ascii=False)
        pil = self.compositor.finalize_image_symmetric(rgb, quality=90, seed=self.seed * 7 + idx)
        filename = f"{record['sample_id']}.jpg"
        path = images_dir / filename
        pil.save(path, format="JPEG", quality=92)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            rel = str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            rel = str(path)
        record["image_path"] = rel
        record["image_sha256"] = sha
        record["image_width"], record["image_height"] = pil.size
        hash_src = json.dumps({k: record[k] for k in ("caption_zh", "qa_pairs_json", "entities_json",
                                                      "bboxes_norm_1000", "match_label",
                                                      "substrate_id", "placement_zone")}, sort_keys=True)
        record["content_hash"] = hashlib.sha1(hash_src.encode("utf-8")).hexdigest()[:16]
        if debug_dir is not None:
            self._save_debug(record, pil, debug_dir)
        return record

    def _save_debug(self, record: Dict[str, Any], pil: Image.Image, debug_dir: Path) -> None:
        from engine.coords import draw_debug_bboxes
        boxes = json.loads(record["bboxes_norm_1000"])
        dbg = draw_debug_bboxes(pil, boxes, label=record["sample_id"]) if boxes else pil.copy()
        dbg.save(debug_dir / f"{record['sample_id']}_debug.jpg", format="JPEG", quality=85)

    # ------------------------------------------------------------------
    # 写盘 / 合并 / 报告
    # ------------------------------------------------------------------
    def run_range(self, start: int, end: int, images_dir: Path, records_file: Path,
                  debug_dir: Optional[Path] = None, limit: Optional[int] = None) -> int:
        images_dir.mkdir(parents=True, exist_ok=True)
        records_file.parent.mkdir(parents=True, exist_ok=True)
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        count = 0
        with open(records_file, "w", encoding="utf-8") as f:
            for idx in range(start, end + 1):
                if limit is not None and count >= limit:
                    break
                record = self.generate_sample(idx, images_dir, debug_dir)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if count % 200 == 0 or idx == end:
                    elapsed = time.time() - t0
                    print(f"  [{start}-{end}] {count} samples, {count / max(elapsed, 1e-6):.1f} samples/s", flush=True)
        print(f"[v2] wrote {count} records -> {records_file} in {time.time() - t0:.1f}s")
        return count


# ----------------------------------------------------------------------
# 合并 shard / 生成报告
# ----------------------------------------------------------------------
def _arrow_schema() -> pa.Schema:
    return pa.schema([
        ("sample_id", pa.string()), ("dataset_version", pa.string()), ("task_stage", pa.string()),
        ("stage_title_zh", pa.string()), ("task_family", pa.string()), ("difficulty", pa.string()),
        ("split", pa.string()), ("domain", pa.string()), ("entity_ids_json", pa.string()),
        ("entity_names_zh", pa.string()), ("entity_names_en", pa.string()), ("num_entities", pa.int32()),
        ("substrate_id", pa.string()), ("substrate_category_zh", pa.string()),
        ("substrate_category_en", pa.string()), ("substrate_source", pa.string()),
        ("scene_style_zh", pa.string()), ("placement_zone", pa.string()), ("print_technique_zh", pa.string()),
        ("bboxes_norm_1000", pa.string()), ("entities_json", pa.string()),
        ("ref_candidates_json", pa.string()), ("match_label", pa.string()),
        ("caption_zh", pa.string()), ("caption_en", pa.string()),
        ("visual_analysis", pa.string()), ("knowledge_expansion", pa.string()),
        ("qa_pairs_json", pa.string()), ("image_path", pa.string()), ("image_sha256", pa.string()),
        ("image_width", pa.int32()), ("image_height", pa.int32()),
        ("render_meta_json", pa.string()), ("quality_json", pa.string()), ("content_hash", pa.string()),
    ])


def merge_shards(out_dir: Path) -> None:
    from collections import Counter
    shard_dir = out_dir / "shard_records"
    if not shard_dir.exists():
        shard_dir = out_dir / "_generation" / "shard_records"
    records: List[Dict[str, Any]] = []
    for f in sorted(shard_dir.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"no shard records under {shard_dir}")
    records.sort(key=lambda r: r["sample_id"])
    for r in records:
        hash_src = json.dumps({k: r.get(k, "") for k in ("caption_zh", "qa_pairs_json", "entities_json",
                                                         "bboxes_norm_1000", "match_label",
                                                         "substrate_id", "placement_zone")}, sort_keys=True)
        r["content_hash"] = hashlib.sha1(hash_src.encode("utf-8")).hexdigest()[:16]
    print(f"[merge] {len(records)} records from {len(list(shard_dir.glob('*.jsonl')))} shards")
    table = pa.Table.from_pylist(records, schema=_arrow_schema())
    pq.write_table(table, out_dir / "dataset.parquet", compression="zstd")
    print(f"[merge] wrote {out_dir / 'dataset.parquet'} "
          f"({(out_dir / 'dataset.parquet').stat().st_size / 1024 / 1024:.2f} MB)")

    # 报告
    stage_counts = Counter(r["task_stage"] for r in records)
    split_counts = Counter(r["split"] for r in records)
    diff_counts = Counter(r["difficulty"] for r in records)
    domain_counts = Counter(r["domain"] for r in records)
    family_counts = Counter(r["task_family"] for r in records)
    match_counts = Counter(r["match_label"] for r in records)
    answer_types = Counter()
    qa_per_sample = Counter()
    for r in records:
        qa = json.loads(r["qa_pairs_json"])
        qa_per_sample[len(qa)] += 1
        for q in qa:
            answer_types[q["answer_type"]] += 1
    vis = [json.loads(r["quality_json"]).get("print_visibility") for r in records]
    vis = [v for v in vis if isinstance(v, (int, float))]
    lines = [
        "# Datawheel v2 数据集报告 (10,000 条全量渲染图文数据)",
        "",
        f"- 版本: {DATASET_VERSION}",
        f"- 样本数: {len(records):,}",
        f"- 图像: 全量渲染，{len(records):,} 张（640x640；参考匹配阶段 640x840），位于 `output/datawheel_v2/images/`",
        f"- 数据集: `output/datawheel_v2/dataset.parquet` (ZSTD)",
        f"- 生成方式: 100% 离线离线合成，0 网络流量",
        f"- 可印性底板: {len({r['substrate_id'] for r in records if r['substrate_id'] != 'procedural_canvas'})} 张通过质检的真实商品底板",
        "",
        "## 1. 阶段与能力分布",
        "| 阶段 | 样本数 | 说明 |",
        "|---|---|---|",
    ]
    for st in STAGE_RANGES:
        lines.append(f"| `{st}` | {stage_counts[st]:,} | {STAGE_TITLES[st]} |")
    lines += ["", "## 2. Split 分布（实体/底板级隔离）", "| Split | 样本数 | 说明 |", "|---|---|---|"]
    split_desc = {"train": "训练集（实体/底板均见）", "val": "验证集（未见实体）",
                  "test_unseen_entity": "测试（未见实体/见过底板）",
                  "test_unseen_substrate": "测试（见过实体/未见底板）",
                  "test_unseen_both": "测试（实体与底板均未见）"}
    for sp in SPLITS:
        lines.append(f"| `{sp}` | {split_counts[sp]:,} | {split_desc[sp]} |")
    lines += ["", "## 3. 难度 / 任务族 / 匹配标签", "",
              "难度: " + json.dumps(dict(diff_counts), ensure_ascii=False),
              "", "任务族: ```" + json.dumps(dict(family_counts), ensure_ascii=False, indent=1) + "```",
              "", "匹配标签: " + json.dumps(dict(match_counts), ensure_ascii=False),
              "", "## 4. QA 答案类型分布", "| answer_type | 数量 |", "|---|---|"]
    for at, c in answer_types.most_common():
        lines.append(f"| `{at}` | {c:,} |")
    lines += ["", f"每样本 QA 数分布: {dict(qa_per_sample)}",
              f"印花可见度: 均值 {np.mean(vis):.3f} / 中位 {np.median(vis):.3f} / P10 {np.percentile(vis, 10):.3f}",
              "", "## 5. 领域分布", "| domain | 样本数 |", "|---|---|"]
    for d, c in domain_counts.most_common(25):
        lines.append(f"| `{d}` | {c:,} |")
    lines += ["", "## 6. 代表性样本", "```json",
              json.dumps(records[min(1600, len(records) - 1)], ensure_ascii=False, indent=1)[:4000],
              "```", ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[merge] wrote {out_dir / 'report.md'}")
    make_contact_sheet(out_dir, records)


def make_contact_sheet(out_dir: Path, records: List[Dict[str, Any]]) -> None:
    from engine.coords import draw_debug_bboxes
    picks: List[Dict[str, Any]] = []
    for stage in STAGE_RANGES:
        stage_records = [r for r in records if r["task_stage"] == stage]
        if not stage_records:
            continue
        step = max(1, len(stage_records) // 4)
        picks.extend(stage_records[::step][:4])
    tiles = []
    for r in picks[:20]:
        img = Image.open(PROJECT_ROOT / r["image_path"]).convert("RGB")
        boxes = json.loads(r["bboxes_norm_1000"])
        if boxes:
            img = draw_debug_bboxes(img, boxes, label=r["sample_id"])
        img.thumbnail((320, 320), Image.Resampling.LANCZOS)
        tiles.append((img, r["sample_id"]))
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 320, rows * 320), (240, 240, 240))
    for i, (tile, sid) in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * 320, (i // cols) * 320))
    sheet.save(out_dir / "contact_sheet.jpg", quality=90)
    print(f"[merge] wrote {out_dir / 'contact_sheet.jpg'} ({len(tiles)} tiles)")


def preview(seed: int, out_dir: Path, n_per_stage: int = 2) -> None:
    gen = DatawheelV2Generator(seed)
    images_dir = out_dir / "images"
    debug_dir = out_dir / "debug"
    images_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for stage, (start, end) in STAGE_RANGES.items():
        for k in range(n_per_stage):
            idx = start + int((end - start) * (k + 1) / (n_per_stage + 1))
            records.append(gen.generate_sample(idx, images_dir, debug_dir))
    (out_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    make_contact_sheet(out_dir, records)
    print(f"[preview] {len(records)} samples -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Datawheel v2 数据集生成器")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10000)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "datawheel_v2")
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--records-file", type=Path, default=None)
    parser.add_argument("--debug-images", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--shards", type=int, default=1, help="并行分片数（>1 时自动启动子进程并合并）")
    args = parser.parse_args()

    if args.merge:
        merge_shards(args.out_dir)
        return
    if args.shards and args.shards > 1:
        import math as _math
        import subprocess as _subprocess
        out_dir = args.out_dir
        (out_dir / "_generation" / "logs").mkdir(parents=True, exist_ok=True)
        total = args.end - args.start + 1
        per = _math.ceil(total / args.shards)
        procs = []
        for i in range(args.shards):
            s_idx = args.start + i * per
            e_idx = min(args.end, s_idx + per - 1)
            if s_idx > e_idx:
                break
            log = out_dir / "_generation" / "logs" / f"shard_{i + 1}.log"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--start", str(s_idx), "--end", str(e_idx),
                   "--out-dir", str(out_dir), "--seed", str(args.seed)]
            procs.append((_subprocess.Popen(cmd, stdout=open(log, "w"), stderr=_subprocess.STDOUT), s_idx, e_idx, log))
        rc = 0
        for proc, s_idx, e_idx, log in procs:
            proc.wait()
            rc = rc or proc.returncode
            print(f"[shard {s_idx}-{e_idx}] exit={proc.returncode} log={log}")
        if rc:
            raise SystemExit(rc)
        merge_shards(out_dir)
        return
    if args.preview:
        preview(args.seed, args.out_dir)
        return
    images_dir = args.images_dir or (args.out_dir / "images")
    records_file = args.records_file or (args.out_dir / "_generation" / "shard_records" / f"shard_{args.start:05d}_{args.end:05d}.jsonl")
    gen = DatawheelV2Generator(args.seed)
    gen.run_range(args.start, args.end, images_dir, records_file,
                  debug_dir=(args.out_dir / "debug") if args.debug_images else None,
                  limit=args.limit)


if __name__ == "__main__":
    main()
