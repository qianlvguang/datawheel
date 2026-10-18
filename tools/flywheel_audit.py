#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flywheel_audit.py —— Datawheel 多模态数据集独立质检审计工具

支持两类数据集:
  * v2（Datawheel v2，output/datawheel_v2/dataset.parquet）:
      完整 35 列契约校验（schema / bbox / 图片 / 重复 / split 泄漏 / 分布配额 / 质量启发式）
  * v1 legacy（output/legacy_v1/datawheel_10k_captions.parquet，18 列）:
      自动识别并降级检查（缺列自动跳过，保留 v1 可验证的 bbox / 图片 / 重复 / 分布检查）

用法:
  .venv/bin/python tools/flywheel_audit.py \\
      --dataset output/datawheel_v2/dataset.parquet \\
      --images-root . \\
      --out output/datawheel_v2/audit/

产物（--out 目录下）:
  audit_report.md    中文审计报告（表格 + 结论 / 风险 / 建议）
  audit_metrics.json 机器可读指标（total_samples / 各分布 dict / critical、warning 计数 ...）
  issues.jsonl       逐条问题，每行 {"severity","code","sample_id","detail"}

退出码:
  0  无 critical 问题（可含 warning / info）
  1  存在 critical 问题
  2  运行错误（数据集不存在 / 无法读取 / 参数错误 ...）

约束: 只读数据集与图片；不联网；依赖 pyarrow + PIL + 标准库。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TOOL_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# 常量与契约
# ---------------------------------------------------------------------------

# v2 契约列（35 列）
V2_REQUIRED_COLUMNS: List[str] = [
    "sample_id", "dataset_version", "task_stage", "stage_title_zh",
    "task_family", "difficulty", "split", "domain",
    "entity_ids_json", "entity_names_zh", "entity_names_en", "num_entities",
    "substrate_id", "substrate_category_zh", "substrate_category_en",
    "substrate_source", "scene_style_zh", "placement_zone", "print_technique_zh",
    "bboxes_norm_1000", "entities_json", "ref_candidates_json", "match_label",
    "caption_zh", "caption_en", "visual_analysis", "knowledge_expansion",
    "qa_pairs_json", "image_path", "image_sha256", "image_width", "image_height",
    "render_meta_json", "quality_json", "content_hash",
]

# v1 legacy 必需列（18 列中核心可校验的 10 列）
V1_REQUIRED_COLUMNS: List[str] = [
    "sample_id", "task_stage", "domain", "substrate_id", "num_entities",
    "entities_json", "bboxes_norm_1000", "caption_zh", "qa_pairs_json",
    "image_path",
]

# 只要出现这些 v2 专有列，就按 v2 处理（即使列不全 → 报 schema critical）
V2_MARKERS = {
    "dataset_version", "split", "difficulty", "task_family", "content_hash",
    "image_sha256", "task_stage", "stage_title_zh",
}
# 注意: task_stage / stage_title_zh 是两版共有的，不能作为 marker
V2_MARKERS = {
    "dataset_version", "split", "difficulty", "task_family", "content_hash",
    "image_sha256", "match_label", "entity_ids_json", "ref_candidates_json",
    "render_meta_json", "quality_json", "substrate_source", "scene_style_zh",
    "print_technique_zh", "substrate_category_en", "image_width",
}

# v2 阶段枚举 → 设计配额
V2_STAGES: Dict[str, Dict[str, Any]] = {
    "stage_a_geometry":             {"code": "A", "quota": 1000, "title": "阶段A：几何构图"},
    "stage_b_single_element":       {"code": "B", "quota": 4000, "title": "阶段B：单图元"},
    "stage_c_multi_element":        {"code": "C", "quota": 2500, "title": "阶段C：多图元"},
    "stage_d_reference_match":      {"code": "D", "quota": 1500, "title": "阶段D：参考匹配"},
    "stage_e_negative_evidence":    {"code": "E", "quota": 1000, "title": "阶段E：负证据"},
}
STAGE_ORDER = list(V2_STAGES.keys())

# v1 legacy 阶段枚举（已知）
LEGACY_STAGES = {
    "stage_a_color_blocks", "stage_b_single_element",
    "stage_c_multi_elements", "stage_d_in_context_pairs",
}

DIFFICULTIES = {"easy", "medium", "hard"}
SPLITS = ["train", "val", "test_unseen_entity", "test_unseen_substrate", "test_unseen_both"]
MATCH_LABELS = {"match", "no_match", "insufficient_evidence", "not_applicable"}
ANSWER_TYPES = {"bbox", "name", "color", "count", "yes_no", "relation", "location", "uncertain"}

JSON_COLUMNS = [
    "entity_ids_json", "entities_json", "ref_candidates_json", "bboxes_norm_1000",
    "qa_pairs_json", "render_meta_json", "quality_json",
]
JSON_TO_SAMPLE_KEY = {
    "entity_ids_json": "entity_ids",
    "entities_json": "entities",
    "ref_candidates_json": "ref_candidates",
    "bboxes_norm_1000": "bboxes",
    "qa_pairs_json": "qa",
    "render_meta_json": "render_meta",
    "quality_json": "quality",
}
INT_COLUMNS = {"num_entities", "image_width", "image_height"}

# 审计阈值（在报告中透明展示）
MIN_BBOX_AREA = 2000.0          # 0.2% × (1000×1000) 画布面积
EXPECTED_IMAGE_SIZE = (640, 640)
# 程序化生成物：不对应真实资产池，跨 split 共享属设计预期，不计入泄漏 critical
PROCEDURAL_DOMAINS = {"geometric_primitives"}
PROCEDURAL_SUBSTRATES = {"procedural_canvas"}

# v2 分阶段图片尺寸（Lead 契约澄清：stage_d 顶部 200px 参考图带 → 640x840，其余 640x640）
STAGE_IMAGE_SIZES = {
    "stage_a_geometry": (640, 640),
    "stage_b_single_element": (640, 640),
    "stage_c_multi_element": (640, 640),
    "stage_d_reference_match": (640, 840),
    "stage_e_negative_evidence": (640, 640),
}
# 泄漏判定只纳入监督目标实体；候选参考图/干扰图元不参与
LEAK_EXCLUDED_ROLE_PREFIXES = (
    "candidate", "distractor", "negative", "reference", "ref_", "in_context",
)
BBOX_ASPECT_EXTREME = 20.0      # bbox 宽高比超过该值视为异常（细长条）
CAPTION_DUP_WARN_RATE = 0.05    # caption_zh 完全重复率 warning 阈值
CAPTION_NEAR_DUP_WARN_RATE = 0.10
QA_PAIR_DUP_WARN_RATE = 0.15    # 跨样本 QA 对完全重复（模板重复）warning 阈值（v2 目标 <15%；v1 基线 42.34%）
QA_ANSWER_CONCENTRATION_WARN = 0.50   # Top3 答案占比 warning 阈值
EVIDENCE_EMPTY_WARN_RATE = 0.10
QUOTA_TOLERANCE = 0.02          # 配额偏差 ±2% 视为通过

# 聚合 issue 阈值：同类逐样本问题超过该数量时合并为一条（detail 附示例）
AGGREGATE_THRESHOLD = 50

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# 任意实体/嵌套结构里可能承载 bbox 的 key（统一小写比较）
BBOX_KEYS = {
    "bbox_1000", "bboxes_1000", "bbox_norm_1000", "bboxes_norm_1000",
    "bbox", "bboxes", "bbox_1000_list", "bbox_px",
}
BBOX_COORD_KEYS = [("ymin", "xmin", "ymax", "xmax"), ("y_min", "x_min", "y_max", "x_max")]

COORD_RE = re.compile(
    r"\[\s*(-?\d{1,4}(?:\.\d+)?)\s*,\s*(-?\d{1,4}(?:\.\d+)?)\s*,"
    r"\s*(-?\d{1,4}(?:\.\d+)?)\s*,\s*(-?\d{1,4}(?:\.\d+)?)\s*\]"
)


class AuditError(Exception):
    """运行级错误（退出码 2）。"""


# ---------------------------------------------------------------------------
# 通用工具函数
# ---------------------------------------------------------------------------

def now_str() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def coerce_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        try:
            f = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(f):
        return None
    return f


def parse_json_value(value: Any) -> Tuple[Any, Optional[str]]:
    """把 parquet 中的 JSON 列解析为 Python 对象。

    返回 (obj, error)。json 列既可能是字符串，也可能已是 list/dict（arrow struct）。
    空值返回 (None, None)，由调用方决定是否算问题。
    """
    if value is None:
        return None, None
    if isinstance(value, (list, dict)):
        return value, None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None, None
        try:
            return json.loads(s), None
        except Exception as exc:  # noqa: BLE001
            return None, f"JSON 解析失败: {exc}"
    # 其他标量（int/bool 等）只可能是损坏数据
    return value, None


def coerce_bbox(value: Any) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[str]]:
    """把各种 bbox 表示统一为 (ymin, xmin, ymax, xmax) 浮点四元组。

    支持: list/tuple 四元组、JSON 字符串、{"ymin":...} 字典。
    返回 (bbox, error)。
    """
    if value is None:
        return None, "bbox 为空"
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None, "bbox 为空"
        try:
            value = json.loads(s)
        except Exception as exc:  # noqa: BLE001
            return None, f"bbox 字符串不是合法 JSON: {exc}"
    if isinstance(value, dict):
        low = {str(k).lower(): v for k, v in value.items()}
        for key_seq in BBOX_COORD_KEYS:
            if all(k in low for k in key_seq):
                value = [low[k] for k in key_seq]
                break
        else:
            return None, "bbox 字典缺少 ymin/xmin/ymax/xmax"
    if not isinstance(value, (list, tuple)):
        return None, f"bbox 类型非法: {type(value).__name__}"
    if len(value) != 4:
        return None, f"bbox 长度应为 4，实际 {len(value)}"
    nums: List[float] = []
    for item in value:
        if isinstance(item, bool):
            return None, "bbox 含布尔值"
        f = coerce_number(item)
        if f is None:
            return None, f"bbox 含非数值: {item!r}"
        nums.append(f)
    return (nums[0], nums[1], nums[2], nums[3]), None


def iter_entity_bboxes(node: Any, path: str = "$") -> Iterable[Tuple[str, Any]]:
    """递归挖掘 entities_json 中的 bbox 字段，产出 (路径, bbox 原始值)。"""
    if isinstance(node, dict):
        for key, value in node.items():
            key_low = str(key).lower()
            if key_low in BBOX_KEYS or key_low.endswith("bbox_1000"):
                yield f"{path}.{key}", value
            else:
                yield from iter_entity_bboxes(value, f"{path}.{key}")
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            yield from iter_entity_bboxes(item, f"{path}[{idx}]")


def normalize_qa(item: Any) -> Dict[str, Any]:
    """兼容 v2 (question_zh/answer_zh/...) 与 v1 (q_zh/a_zh) 的 QA 归一化。"""
    if not isinstance(item, dict):
        return {
            "question": None, "answer": None, "answer_type": None,
            "answer_bbox": None, "evidence": None, "raw": item,
        }

    def pick(*keys: str) -> Any:
        for key in keys:
            if key in item:
                return item[key]
        return None

    return {
        "question": pick("question_zh", "q_zh", "question"),
        "answer": pick("answer_zh", "a_zh", "answer"),
        "answer_type": pick("answer_type", "type"),
        "answer_bbox": pick("answer_bbox_1000", "answer_bbox", "a_bbox_1000"),
        "evidence": pick("evidence_zh", "evidence", "evidence_en"),
        "raw": item,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_caption(text: str) -> str:
    """去数字/标点/空白的 caption 归一化，用于近重复检测。"""
    s = unicodedata.normalize("NFKC", str(text)).lower()
    s = re.sub(r"\d+", "", s)
    return "".join(ch for ch in s if ch.isalnum())


def quantile_stats(values: Sequence[float]) -> Dict[str, Any]:
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return {"count": 0}

    def q(p: float) -> float:
        if n == 1:
            return vals[0]
        pos = (n - 1) * p
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)

    return {
        "count": n,
        "min": round(vals[0], 6),
        "p10": round(q(0.10), 6),
        "p25": round(q(0.25), 6),
        "p50": round(q(0.50), 6),
        "p75": round(q(0.75), 6),
        "p90": round(q(0.90), 6),
        "max": round(vals[-1], 6),
        "mean": round(sum(vals) / n, 6),
    }


def arrow_kind(dtype: Any) -> str:
    s = str(dtype)
    if s in ("string", "large_string", "utf8"):
        return "text"
    if s.startswith(("int", "uint")):
        return "int"
    if s.startswith(("float", "double")):
        return "float"
    if s == "bool":
        return "bool"
    if s.startswith(("list", "large_list", "fixed_size_list", "struct", "map")):
        return "json"
    if s.startswith(("binary", "large_binary")):
        return "binary"
    if s.startswith(("timestamp", "date", "time")):
        return "time"
    return "other"


def find_key_values(obj: Any, key_substr: str, out: List[Any], depth: int = 0) -> None:
    """递归查找键名包含 key_substr 的原始值（用于 print_visibility 等）。"""
    if depth > 8:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key_substr in str(key).lower():
                out.append(value)
            find_key_values(value, key_substr, out, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            find_key_values(item, key_substr, out, depth + 1)


def pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def md_escape(value: Any) -> str:
    text = "—" if value is None else str(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(md_escape(h) for h in headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(md_escape(c) for c in row) + " |")
    return "\n".join(lines)


def counter_to_dist(counter: Counter, total: int, sort_by_count: bool = True) -> Dict[str, Dict[str, Any]]:
    items = list(counter.items())
    if sort_by_count:
        items.sort(key=lambda kv: (-kv[1], str(kv[0])))
    else:
        items.sort(key=lambda kv: str(kv[0]))
    return {
        str(k): {"count": int(v), "ratio": round(v / total, 6) if total else 0.0}
        for k, v in items
    }


# ---------------------------------------------------------------------------
# 审计器
# ---------------------------------------------------------------------------

class Auditor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cwd = Path.cwd()
        self.script_path = Path(__file__).resolve()
        self.project_root = self.script_path.parent.parent
        self.dataset_path = Path(args.dataset).expanduser()
        if not self.dataset_path.is_absolute():
            self.dataset_path = self.cwd / self.dataset_path
        self.dataset_path = self.dataset_path.resolve()
        self.dataset_dir = self.dataset_path.parent

        images_root = Path(args.images_root).expanduser()
        if not images_root.is_absolute():
            images_root = self.cwd / images_root
        self.images_root = images_root.resolve()

        out = Path(args.out).expanduser() if args.out else self.dataset_dir / "audit"
        if not out.is_absolute():
            out = self.cwd / out
        self.out_dir = out.resolve()

        self.min_bbox_area = float(args.min_bbox_area) * 1_000_000.0       # critical 阈值
        self.warn_bbox_area = float(args.warn_bbox_area) * 1_000_000.0     # warning 阈值
        self.mode: str = "legacy"
        self.cols: List[str] = []
        self.raw: Dict[str, List[Any]] = {}
        self.total_rows: int = 0
        self.samples: List[Dict[str, Any]] = []
        self.issues: List[Dict[str, Any]] = []
        self.m: Dict[str, Any] = {
            "meta": {}, "counts": {}, "schema": {}, "bbox": {}, "images": {},
            "duplicates": {}, "split": {}, "distributions": {}, "quality": {},
        }

    # -- 问题收集 ---------------------------------------------------------
    def add_issue(self, code: str, severity: str, sample_id: Optional[str], detail: str) -> None:
        detail = str(detail).replace("\r", " ").replace("\n", " ").strip()
        self.issues.append({
            "severity": severity,
            "code": code,
            "sample_id": sample_id,
            "detail": detail,
        })

    def emit_offenders(
        self,
        code: str,
        severity: str,
        offenders: Sequence[Tuple[str, str]],
        summary: str,
        threshold: int = AGGREGATE_THRESHOLD,
    ) -> None:
        """逐样本问题：数量 ≤ threshold 时逐条记录，否则合并为一条并附示例。"""
        if not offenders:
            return
        if len(offenders) <= threshold:
            for sample_id, detail in offenders:
                self.add_issue(code, severity, sample_id, detail)
        else:
            examples = "; ".join(f"{sid}: {det}" for sid, det in offenders[:10])
            self.add_issue(
                code, severity, None,
                f"{summary}：共 {len(offenders)} 个样本。示例: {examples}",
            )

    # -- 加载与预处理 -----------------------------------------------------
    def load(self) -> None:
        if not self.dataset_path.is_file():
            raise AuditError(
                f"数据集文件不存在: {self.dataset_path}\n"
                f"（若 v2 尚未生成，请等待 output/datawheel_v2/dataset.parquet 产出后重跑）"
            )
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise AuditError(f"缺少 pyarrow 依赖，请使用项目 .venv 运行: {exc}") from exc

        try:
            table = pq.read_table(str(self.dataset_path))
        except Exception as exc:  # noqa: BLE001
            raise AuditError(f"无法读取 parquet 文件 {self.dataset_path}: {exc}") from exc

        self.total_rows = table.num_rows
        if self.total_rows == 0:
            raise AuditError(f"数据集为空（0 行）: {self.dataset_path}")
        self.cols = list(table.column_names)
        try:
            self.raw = table.to_pydict()
        except Exception as exc:  # noqa: BLE001
            raise AuditError(f"parquet 列无法转换为 Python 对象: {exc}") from exc
        self.table_schema = table.schema

    def detect_mode(self) -> None:
        legacy_opt = self.args.legacy
        if legacy_opt == "on":
            self.mode = "legacy"
            return
        if legacy_opt == "off":
            self.mode = "v2"
            return
        present = set(self.cols)
        self.mode = "v2" if (V2_MARKERS & present) else "legacy"

    def _audit_indices(self) -> List[int]:
        total = self.total_rows
        limit = self.args.sample_limit
        if not limit or limit >= total:
            return list(range(total))
        # 均匀抽样，保证覆盖各阶段，而不是只取前 N 行
        step = total / float(limit)
        picked = sorted({min(total - 1, int(round(k * step))) for k in range(limit)})
        return picked

    def build_samples(self) -> None:
        self.audit_indices = self._audit_indices()
        self.samples = []
        for i in self.audit_indices:
            row = {col: self.raw[col][i] for col in self.cols}
            sid = row.get("sample_id")
            sid = str(sid).strip() if not is_empty(sid) else None
            sample: Dict[str, Any] = {
                "i": i,
                "sample_id": sid or f"row#{i}",
                "row": row,
                "parse_errors": {},
            }
            for col, key in JSON_TO_SAMPLE_KEY.items():
                if col in self.cols:
                    obj, err = parse_json_value(row.get(col))
                    sample[key] = obj
                    if err:
                        sample["parse_errors"][col] = err
                else:
                    sample[key] = None
            self.samples.append(sample)

    # -- 1. Schema --------------------------------------------------------
    def check_schema(self) -> None:
        present = set(self.cols)
        required = V1_REQUIRED_COLUMNS if self.mode == "legacy" else V2_REQUIRED_COLUMNS
        missing = [c for c in required if c not in present]
        extra = [c for c in self.cols if c not in required]

        for col in missing:
            self.add_issue(
                "schema_missing_column", "critical", None,
                f"缺少必需列 {col!r}（{self.mode} 契约）",
            )

        # 物理类型检查
        column_types: Dict[str, str] = {}
        type_mismatches: List[Dict[str, str]] = []
        schema = self.table_schema
        for col in self.cols:
            kind = arrow_kind(schema.field(col).type)
            column_types[col] = f"{schema.field(col).type} ({kind})"
            if col in JSON_COLUMNS:
                ok = kind in ("text", "json", "binary")
                expected = "JSON字符串/嵌套结构"
            elif col in INT_COLUMNS:
                ok = kind in ("int", "float")
                expected = "整数"
            else:
                ok = kind in ("text", "binary")
                expected = "字符串"
            if not ok:
                type_mismatches.append({
                    "column": col, "expected": expected,
                    "actual": str(schema.field(col).type), "kind": kind,
                })
                self.add_issue(
                    "schema_type_mismatch", "critical", None,
                    f"列 {col} 类型为 {schema.field(col).type}（{kind}），契约要求 {expected}",
                )
            elif col in INT_COLUMNS and kind == "float":
                self.add_issue(
                    "schema_int_as_float", "warning", None,
                    f"列 {col} 以浮点存储（{schema.field(col).type}），建议改为整数",
                )

        # 必需字段空值检查
        if self.mode == "legacy":
            critical_nonempty = {
                "sample_id", "task_stage", "domain", "substrate_id", "caption_zh",
            }
        else:
            critical_nonempty = {
                "sample_id", "dataset_version", "task_stage", "task_family",
                "difficulty", "split", "domain", "substrate_id", "caption_zh",
                "content_hash", "match_label", "image_sha256",
            }
        for col in sorted(critical_nonempty & present):
            offenders = [
                (s["sample_id"], f"{col} 为空")
                for s in self.samples if is_empty(s["row"].get(col))
            ]
            self.emit_offenders(f"empty_value_{col}", "critical", offenders, f"必需字段 {col} 为空")

        # JSON 列空值：v2 的负证据样本（absent/occluded）允许 num_entities=0 且 bboxes/entities 为空
        def _num_entities_zero(sample: Dict[str, Any]) -> bool:
            value = coerce_number(sample["row"].get("num_entities"))
            return value is not None and value == 0

        allowed_empty_json: Dict[str, int] = {}
        for col in ("bboxes_norm_1000", "entities_json", "qa_pairs_json"):
            if col not in present:
                continue
            sample_key = JSON_TO_SAMPLE_KEY[col]
            offenders = []
            allowed = 0
            for s in self.samples:
                if not is_empty(s.get(sample_key)):
                    continue
                if self.mode == "v2" and col in ("bboxes_norm_1000", "entities_json") and _num_entities_zero(s):
                    allowed += 1  # 负证据 absent/occluded 样本允许无可见 bbox / 实体
                    continue
                offenders.append((s["sample_id"], f"{col} 为空或 null"))
            if allowed:
                allowed_empty_json[col] = allowed
            self.emit_offenders(f"empty_json_{col}", "critical", offenders, f"必需 JSON 列 {col} 为空")

        # entity_ids 为空：stage_a_geometry 合法（纯程序化色块，无实体泄漏风险）；
        # 其他阶段仅 info（该样本不参与实体泄漏判定）
        if "entity_ids_json" in present:
            empty_entity_ids = 0
            offenders = []
            for s in self.samples:
                if not is_empty(s.get("entity_ids")):
                    continue
                empty_entity_ids += 1
                stage = str(s["row"].get("task_stage") or "")
                if self.mode == "v2" and stage == "stage_a_geometry":
                    continue
                offenders.append((s["sample_id"], "entity_ids_json 为空（泄漏检查不完整）"))
            if empty_entity_ids:
                allowed_empty_json["entity_ids_json"] = empty_entity_ids
            self.emit_offenders("empty_json_entity_ids_json", "info", offenders,
                                "entity_ids_json 为空")
        if "ref_candidates_json" in present:
            empty_ref_count = sum(1 for s in self.samples if is_empty(s.get("ref_candidates")))
            if empty_ref_count:
                allowed_empty_json["ref_candidates_json"] = empty_ref_count

        # JSON 解析错误
        json_errors: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for s in self.samples:
            for col, err in s["parse_errors"].items():
                json_errors[col].append((s["sample_id"], f"{col} {err}"))
        for col, offenders in sorted(json_errors.items()):
            severity = "warning" if col in ("render_meta_json", "quality_json") else "critical"
            self.emit_offenders(f"json_parse_error_{col}", severity, offenders, f"{col} 存在解析错误")

        # JSON 顶层结构校验（v2 契约：list / dict / 实体条目字段）
        expected_list_cols = ["bboxes_norm_1000", "entities_json", "qa_pairs_json"]
        if self.mode == "v2":
            expected_list_cols += ["entity_ids_json", "ref_candidates_json"]
        for col in expected_list_cols:
            if col not in present:
                continue
            offenders = []
            sample_key = JSON_TO_SAMPLE_KEY[col]
            for s in self.samples:
                obj = s.get(sample_key)
                if obj is None:
                    continue  # 空值已由 empty_json_* 检查处理
                if not isinstance(obj, list):
                    offenders.append(
                        (s["sample_id"], f"{col} 顶层应为 JSON 数组，实际 {type(obj).__name__}")
                    )
            self.emit_offenders(f"json_structure_{col}", "critical", offenders,
                                f"{col} 顶层结构非法")
        for col in ("render_meta_json", "quality_json"):
            if col not in present:
                continue
            offenders = []
            sample_key = JSON_TO_SAMPLE_KEY[col]
            for s in self.samples:
                obj = s.get(sample_key)
                if obj is None:
                    continue
                if not isinstance(obj, dict):
                    offenders.append(
                        (s["sample_id"], f"{col} 顶层应为 JSON 对象，实际 {type(obj).__name__}")
                    )
            self.emit_offenders(f"json_structure_{col}", "warning", offenders,
                                f"{col} 顶层结构建议为对象")
        if self.mode == "v2" and "entity_ids_json" in present:
            offenders = []
            for s in self.samples:
                obj = s.get("entity_ids")
                if not isinstance(obj, list):
                    continue
                bad = []
                for idx, item in enumerate(obj):
                    if not isinstance(item, dict):
                        bad.append(idx)
                    elif is_empty(item.get("domain")) or is_empty(item.get("stem")):
                        bad.append(idx)
                if bad:
                    offenders.append(
                        (s["sample_id"],
                         f"entity_ids_json 第 {bad[:5]} 项缺少 domain/stem 或不是对象")
                    )
            self.emit_offenders("json_structure_entity_ids_items", "critical", offenders,
                                "entity_ids_json 条目缺少 domain/stem")

        # 枚举检查
        enum_invalid: Dict[str, Dict[str, int]] = {}
        if "task_stage" in present:
            if self.mode == "v2":
                counts = self.check_enum(
                    "task_stage", set(V2_STAGES), "enum_invalid_task_stage", "critical",
                )
            else:
                counts = self.check_enum(
                    "task_stage", LEGACY_STAGES, "enum_invalid_task_stage", "warning",
                )
            if counts:
                enum_invalid["task_stage"] = counts
        for col, allowed in (
            ("difficulty", DIFFICULTIES),
            ("split", set(SPLITS)),
            ("match_label", MATCH_LABELS),
        ):
            if col not in present:
                continue
            counts = self.check_enum(col, allowed, f"enum_invalid_{col}", "critical")
            if counts:
                enum_invalid[col] = counts

        # dataset_version 唯一性（v2）
        version_counts: Dict[str, int] = {}
        if "dataset_version" in present:
            counter = Counter(
                str(s["row"].get("dataset_version")).strip()
                for s in self.samples if not is_empty(s["row"].get("dataset_version"))
            )
            version_counts = counter_to_dist(counter, len(self.samples))
            if len(counter) > 1:
                self.add_issue(
                    "dataset_version_multiple", "warning", None,
                    f"dataset_version 存在多个取值: {dict(counter)}",
                )

        self.m["schema"] = {
            "mode": self.mode,
            "required_columns": required,
            "present_columns": list(self.cols),
            "missing_columns": missing,
            "extra_columns": extra,
            "column_types": column_types,
            "type_mismatches": type_mismatches,
            "enum_invalid": enum_invalid,
            "dataset_version_dist": version_counts,
            "allowed_empty_json": allowed_empty_json,
        }

    def check_enum(
        self,
        col: str,
        allowed: set,
        code: str,
        severity: str,
    ) -> Dict[str, int]:
        counter: Counter = Counter()
        offenders: List[Tuple[str, str]] = []
        for s in self.samples:
            value = s["row"].get(col)
            if is_empty(value):
                continue
            value_s = str(value).strip()
            if value_s not in allowed:
                counter[value_s] += 1
                offenders.append((s["sample_id"], f"{col} 非法取值 {value_s!r}"))
        self.emit_offenders(code, severity, offenders, f"{col} 存在非法枚举值")
        return {str(k): int(v) for k, v in counter.items()}

    # -- 2. BBox / Grounding ---------------------------------------------
    def check_bboxes(self) -> None:
        if "bboxes_norm_1000" not in self.cols:
            self.m["bbox"] = {"available": False}
            return

        counters = Counter()
        hard_issues: List[Tuple[str, str]] = []
        small_issues: List[Tuple[str, str]] = []
        entity_mismatch: List[Tuple[str, str]] = []
        grounding_errors: List[Tuple[str, str]] = []
        legacy_coord_mismatch: List[Tuple[str, str]] = []
        float_coord_rows: List[Tuple[str, str]] = []
        ref_bbox_issues: List[Tuple[str, str]] = []

        for s in self.samples:
            sid = s["sample_id"]
            bboxes = s.get("bboxes")
            row_problems: List[str] = []
            row_hard: List[str] = []
            row_small: List[str] = []

            num_entities = coerce_number(s["row"].get("num_entities"))
            if not isinstance(bboxes, list) or len(bboxes) == 0:
                if "bboxes_norm_1000" not in s["parse_errors"]:
                    if self.mode == "v2" and num_entities == 0:
                        # 负证据样本（absent/occluded）：允许没有任何可见 bbox
                        counters["rows_empty_bboxes_allowed"] += 1
                    else:
                        counters["rows_empty_bboxes"] += 1
                        row_hard.append("bboxes_norm_1000 为空或不是 JSON 数组")
                if row_hard:
                    hard_issues.append((sid, "; ".join(row_hard[:5])))
                continue

            bbox_set: set = set()
            for idx, raw_bbox in enumerate(bboxes):
                bb, err = coerce_bbox(raw_bbox)
                counters["boxes_checked"] += 1
                if err:
                    counters["boxes_invalid"] += 1
                    row_problems.append(f"bbox[{idx}] {err}")
                    continue
                ymin, xmin, ymax, xmax = bb
                if not all(0.0 <= v <= 1000.0 for v in bb):
                    counters["boxes_out_of_range"] += 1
                    row_problems.append(f"bbox[{idx}] 超出 0~1000 闭区间: {list(bb)}")
                if not ymin < ymax:
                    counters["boxes_bad_y_order"] += 1
                    row_problems.append(f"bbox[{idx}] ymin>=ymax: {list(bb)}")
                if not xmin < xmax:
                    counters["boxes_bad_x_order"] += 1
                    row_problems.append(f"bbox[{idx}] xmin>=xmax: {list(bb)}")
                area = max(0.0, ymax - ymin) * max(0.0, xmax - xmin)
                area_ratio = area / 1_000_000.0
                if area < self.min_bbox_area:
                    counters["boxes_below_critical"] += 1
                    msg = (
                        f"bbox[{idx}] 面积 {area:.1f}（{area_ratio * 100:.3f}% 画布）< "
                        f"critical 阈值 {self.args.min_bbox_area * 100:.2f}%"
                    )
                    row_problems.append(msg if self.mode == "v2" else msg + " (legacy 仅告警)")
                    if self.mode == "v2":
                        row_hard.append(msg)
                    else:
                        row_small.append(msg)
                elif area < self.warn_bbox_area:
                    counters["boxes_small_warning"] += 1
                    msg = (
                        f"bbox[{idx}] 面积 {area:.1f}（{area_ratio * 100:.3f}% 画布）< "
                        f"warning 阈值 {self.args.warn_bbox_area * 100:.2f}%"
                    )
                    row_problems.append(msg)
                    row_small.append(msg)
                if not all(float(v).is_integer() for v in bb):
                    counters["boxes_float_coords"] += 1
                    float_coord_rows.append((sid, f"bbox[{idx}] 含非整数坐标: {list(bb)}"))
                if all(0.0 <= v <= 1000.0 for v in bb) and ymin < ymax and xmin < xmax:
                    bbox_set.add(tuple(int(round(v)) for v in bb))

            # entities_json 中的 grounding bbox 必须落在 bboxes_norm_1000 内
            entities = s.get("entities")
            if isinstance(entities, list):
                for ent_idx, entity in enumerate(entities):
                    for path, raw_entity_bbox in iter_entity_bboxes(entity, f"$[{ent_idx}]"):
                        counters["entity_bboxes_checked"] += 1
                        eb, err = coerce_bbox(raw_entity_bbox)
                        if err:
                            counters["entity_bboxes_invalid"] += 1
                            problem = f"entities_json {path} {err}"
                            row_problems.append(problem)
                            row_hard.append(problem)
                            continue
                        ekey = tuple(int(round(v)) for v in eb)
                        if ekey not in bbox_set:
                            counters["entity_bbox_mismatch"] += 1
                            entity_mismatch.append(
                                (sid, f"entities_json {path} bbox {list(eb)} 不在 bboxes_norm_1000 中")
                            )

            # ref_candidates_json 的候选参考框（stage D 顶部参考带）：
            #   1) 契约修订：answer_bbox_1000 ∈ bboxes_norm_1000 ∪ ref_candidates[*].bbox_1000 均合法；
            #   2) 候选框应位于对应 pane 区域且互不重叠（info/warning 级，不 critical）。
            ref_bbox_set: set = set()
            ref_items: List[Dict[str, Any]] = []
            ref_candidates = s.get("ref_candidates")
            if isinstance(ref_candidates, list):
                for ref_idx, candidate in enumerate(ref_candidates):
                    if not isinstance(candidate, dict):
                        continue
                    raw_ref_bbox = candidate.get("bbox_1000", candidate.get("bbox"))
                    if raw_ref_bbox is None:
                        continue
                    counters["ref_bboxes_checked"] += 1
                    rb, err = coerce_bbox(raw_ref_bbox)
                    if err:
                        counters["ref_bboxes_invalid"] += 1
                        ref_bbox_issues.append((sid, f"ref_candidates_json[{ref_idx}] {err}"))
                        continue
                    rymin, rxmin, rymax, rxmax = rb
                    if not all(0.0 <= v <= 1000.0 for v in rb) or not rymin < rymax or not rxmin < rxmax:
                        counters["ref_bboxes_invalid"] += 1
                        ref_bbox_issues.append(
                            (sid, f"ref_candidates_json[{ref_idx}] bbox 越界/顺序非法: {list(rb)}")
                        )
                    else:
                        rkey = tuple(int(round(v)) for v in rb)
                        ref_bbox_set.add(rkey)
                        ref_items.append({
                            "idx": ref_idx,
                            "pane_index": candidate.get("pane_index"),
                            "bbox": rb,
                        })

                # 候选框 pane 区域校验（仅在 render_meta 提供画布/参考带信息时）
                render_meta = s.get("render_meta") if isinstance(s.get("render_meta"), dict) else None
                canvas = render_meta.get("canvas") if render_meta else None
                band_px = coerce_number(render_meta.get("product_offset_y")) if render_meta else None
                if ref_items and isinstance(canvas, list) and len(canvas) == 2 and band_px:
                    canvas_w, canvas_h = coerce_number(canvas[0]), coerce_number(canvas[1])
                    if canvas_w and canvas_h:
                        band_ymax = band_px / canvas_h * 1000.0
                        pane_total = len(ref_items)
                        tol = 15.0  # 归一化 0~1000 坐标容差（约 10 px）
                        for item in ref_items:
                            pane_idx = coerce_number(item["pane_index"])
                            if pane_idx is None or pane_idx < 0 or pane_idx >= pane_total:
                                continue
                            counters["ref_panes_checked"] += 1
                            rymin, rxmin, rymax, rxmax = item["bbox"]
                            pane_xmin = pane_idx / pane_total * 1000.0
                            pane_xmax = (pane_idx + 1) / pane_total * 1000.0
                            if (rymax > band_ymax + tol or rymin < -tol
                                    or rxmin < pane_xmin - tol or rxmax > pane_xmax + tol):
                                counters["ref_pane_mismatch"] += 1
                                ref_bbox_issues.append((
                                    sid,
                                    f"ref_candidates_json[{item['idx']}] 候选框 {list(item['bbox'])} "
                                    f"不在 pane {int(pane_idx)} 区域 "
                                    f"（x∈[{pane_xmin:.1f},{pane_xmax:.1f}], y≤{band_ymax:.1f}）",
                                ))
                        # 候选框互不重叠
                        for a_idx in range(len(ref_items)):
                            for b_idx in range(a_idx + 1, len(ref_items)):
                                ay0, ax0, ay1, ax1 = ref_items[a_idx]["bbox"]
                                by0, bx0, by1, bx1 = ref_items[b_idx]["bbox"]
                                iw = min(ax1, bx1) - max(ax0, bx0)
                                ih = min(ay1, by1) - max(ay0, by0)
                                if iw > 1.0 and ih > 1.0:
                                    counters["ref_bbox_overlaps"] += 1
                                    ref_bbox_issues.append((
                                        sid,
                                        f"候选框 {ref_items[a_idx]['idx']} 与 "
                                        f"{ref_items[b_idx]['idx']} 重叠 "
                                        f"({iw:.1f}×{ih:.1f} 归一化单位)",
                                    ))

            # QA grounding 答案 bbox 必须落在 bboxes_norm_1000 内
            qa_list = s.get("qa")
            if isinstance(qa_list, list):
                for qa_idx, qa_item in enumerate(qa_list):
                    qa_norm = normalize_qa(qa_item)
                    answer_bbox = qa_norm["answer_bbox"]
                    answer_type = str(qa_norm["answer_type"]).strip() if qa_norm["answer_type"] is not None else None
                    if answer_bbox is not None and not is_empty(answer_bbox):
                        counters["grounding_checked"] += 1
                        gb, err = coerce_bbox(answer_bbox)
                        if err:
                            counters["grounding_invalid"] += 1
                            grounding_errors.append(
                                (sid, f"qa[{qa_idx}] answer_bbox_1000 {err}: {answer_bbox!r}")
                            )
                        elif tuple(int(round(v)) for v in gb) not in bbox_set:
                            gkey = tuple(int(round(v)) for v in gb)
                            if gkey in ref_bbox_set:
                                # 契约修订：answer_bbox_1000 落在 ref_candidates_json 候选框内合法
                                counters["grounding_ref_answers"] += 1
                                if self.args.strict_ref_candidate_answer_bbox:
                                    counters["grounding_strict_violations"] += 1
                                    grounding_errors.append(
                                        (sid, f"qa[{qa_idx}] answer_bbox_1000 {list(gb)} 仅在 "
                                              f"ref_candidates_json 中（--strict-ref-candidate-answer-bbox 模式判 critical）")
                                    )
                            else:
                                counters["grounding_mismatch"] += 1
                                grounding_errors.append(
                                    (sid, f"qa[{qa_idx}] answer_bbox_1000 {list(gb)} 不在 bboxes_norm_1000 中")
                                )
                    elif answer_type == "bbox":
                        counters["grounding_missing"] += 1
                        grounding_errors.append(
                            (sid, f"qa[{qa_idx}] answer_type=bbox 但 answer_bbox_1000 为空")
                        )
                    # v1 legacy: 从中文答案中正则提取坐标做一致性检查（降级）
                    if self.mode == "legacy":
                        answer_text = qa_norm["answer"]
                        if isinstance(answer_text, str):
                            for match in COORD_RE.finditer(answer_text):
                                parsed = tuple(float(x) for x in match.groups())
                                counters["legacy_answer_coords_parsed"] += 1
                                if tuple(int(round(v)) for v in parsed) not in bbox_set:
                                    counters["legacy_answer_coords_mismatch"] += 1
                                    legacy_coord_mismatch.append(
                                        (sid, f"答案文本坐标 {list(parsed)} 不在 bboxes_norm_1000 中")
                                    )

            if row_problems:
                counters["rows_with_bbox_issues"] += 1
            if row_hard:
                hard_issues.append((sid, "; ".join(row_hard[:5])))
            elif row_small:
                small_issues.append((sid, "; ".join(row_small[:5])))

        # 记录问题（legacy 下面积阈值一律降级为 warning）
        self.emit_offenders("bbox_invalid", "critical", hard_issues,
                            "bbox 非法（越界/顺序/面积低于 critical 阈值等）")
        if self.mode == "v2":
            self.emit_offenders("bbox_small_warning", "warning", small_issues,
                                "bbox 面积处于 critical~warning 阈值之间（小目标，合法保留）")
        else:
            self.emit_offenders("bbox_too_small", "warning", small_issues,
                                "bbox 面积低于阈值（legacy 降级）")
            self.emit_offenders("legacy_answer_coord_mismatch", "warning", legacy_coord_mismatch,
                                "v1 答案文本中可解析坐标与 bboxes_norm_1000 不一致")
        if float_coord_rows:
            self.emit_offenders("bbox_float_coords", "info", float_coord_rows, "bbox 含非整数坐标")
        self.emit_offenders("entity_bbox_mismatch", "critical", entity_mismatch,
                            "entities_json ground truth bbox 与 bboxes_norm_1000 不一致")
        self.emit_offenders("qa_answer_bbox_mismatch", "critical", grounding_errors,
                            "QA answer_bbox_1000 与 bboxes_norm_1000 不一致或缺失")
        self.emit_offenders("ref_candidate_bbox_invalid", "warning", ref_bbox_issues,
                            "ref_candidates_json 候选参考框坐标非法")

        self.m["bbox"] = {
            "available": True,
            "min_area_threshold": self.min_bbox_area,
            "warn_area_threshold": self.warn_bbox_area,
            "min_area_ratio": self.args.min_bbox_area,
            "warn_area_ratio": self.args.warn_bbox_area,
            "strict_ref_candidate_answer_bbox": bool(self.args.strict_ref_candidate_answer_bbox),
            "boxes_checked": counters["boxes_checked"],
            "boxes_invalid": counters["boxes_invalid"],
            "boxes_out_of_range": counters["boxes_out_of_range"],
            "boxes_bad_y_order": counters["boxes_bad_y_order"],
            "boxes_bad_x_order": counters["boxes_bad_x_order"],
            "boxes_below_critical": counters["boxes_below_critical"],
            "boxes_small_warning": counters["boxes_small_warning"],
            "boxes_too_small": counters["boxes_below_critical"],  # 兼容别名
            "boxes_float_coords": counters["boxes_float_coords"],
            "rows_with_issues": counters["rows_with_bbox_issues"],
            "rows_empty_bboxes": counters["rows_empty_bboxes"],
            "rows_empty_bboxes_allowed": counters["rows_empty_bboxes_allowed"],
            "ref_bboxes_checked": counters["ref_bboxes_checked"],
            "ref_bboxes_invalid": counters["ref_bboxes_invalid"],
            "grounding_ref_answers": counters["grounding_ref_answers"],
            "grounding_strict_violations": counters["grounding_strict_violations"],
            "ref_panes_checked": counters["ref_panes_checked"],
            "ref_pane_mismatch": counters["ref_pane_mismatch"],
            "ref_bbox_overlaps": counters["ref_bbox_overlaps"],
            "entity_bboxes_checked": counters["entity_bboxes_checked"],
            "entity_bboxes_invalid": counters["entity_bboxes_invalid"],
            "entity_bbox_mismatch": counters["entity_bbox_mismatch"],
            "grounding_checked": counters["grounding_checked"],
            "grounding_invalid": counters["grounding_invalid"],
            "grounding_mismatch": counters["grounding_mismatch"] + counters["grounding_missing"],
            "legacy_answer_coords_parsed": counters["legacy_answer_coords_parsed"],
            "legacy_answer_coords_mismatch": counters["legacy_answer_coords_mismatch"],
        }

    # -- 3. 图片 ----------------------------------------------------------
    def _resolve_image(self, raw_path: str) -> Optional[Path]:
        p = Path(raw_path)
        candidates: List[Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([
                self.images_root / p,
                self.dataset_dir / p,
                self.project_root / p,
                self.cwd / p,
            ])
            if len(p.parts) > 1:
                candidates.append(self.images_root / Path(*p.parts[1:]))
                candidates.append(self.dataset_dir / Path(*p.parts[1:]))
            candidates.extend([
                self.images_root / "images" / p.name,
                self.dataset_dir / "images" / p.name,
                self.images_root / p.name,
                self.dataset_dir.parent / p,
            ])
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue
        return None

    def check_images(self) -> None:
        counters = Counter()
        empty_path_offenders: List[Tuple[str, str]] = []
        missing_offenders: List[Tuple[str, str]] = []
        decode_offenders: List[Tuple[str, str]] = []
        size_offenders: List[Tuple[str, str]] = []
        sha_offenders: List[Tuple[str, str]] = []
        dim_meta_offenders: List[Tuple[str, str]] = []
        path_reuse = Counter()
        sha_reuse = Counter()
        dimensions: Counter = Counter()
        resolved_paths: List[str] = []

        has_sha_col = "image_sha256" in self.cols
        has_dim_cols = "image_width" in self.cols and "image_height" in self.cols

        for s in self.samples:
            raw_path = s["row"].get("image_path")
            if is_empty(raw_path):
                counters["empty_path"] += 1
                empty_path_offenders.append((s["sample_id"], "image_path 为空"))
                continue
            raw_path_s = str(raw_path).strip()

            if self.args.skip_images:
                counters["skipped"] += 1
                continue

            resolved = self._resolve_image(raw_path_s)
            if resolved is None:
                counters["missing"] += 1
                missing_offenders.append(
                    (s["sample_id"], f"image_path 指向的文件不存在: {raw_path_s}")
                )
                continue

            counters["found"] += 1
            resolved_paths.append(str(resolved))
            path_reuse[str(resolved)] += 1

            # sha256
            image_sha = None
            try:
                image_sha = sha256_file(resolved)
            except OSError as exc:
                counters["read_error"] += 1
                decode_offenders.append((s["sample_id"], f"图片读取失败: {exc}"))
                continue

            # PIL 解码 + 尺寸
            try:
                from PIL import Image  # 延迟导入
                with Image.open(str(resolved)) as im:
                    im.load()
                    width, height = im.size
                    image_mode = im.mode
            except Exception as exc:  # noqa: BLE001
                counters["decode_fail"] += 1
                decode_offenders.append((s["sample_id"], f"PIL 无法解码: {exc}"))
                continue

            counters["decoded"] += 1
            dimensions[f"{width}x{height}"] += 1
            stage_name = str(s["row"].get("task_stage") or "").strip()
            if self.mode == "v2":
                expected_size = STAGE_IMAGE_SIZES.get(stage_name, EXPECTED_IMAGE_SIZE)
            else:
                expected_size = EXPECTED_IMAGE_SIZE
            if (width, height) != expected_size:
                counters["size_stage_mismatch"] += 1
                size_offenders.append(
                    (s["sample_id"],
                     f"{stage_name or 'legacy'} 图片尺寸 {width}x{height}，阶段期望 "
                     f"{expected_size[0]}x{expected_size[1]}")
                )
            if width != height:
                counters["non_square"] += 1

            if has_sha_col:
                expected_sha = s["row"].get("image_sha256")
                if not is_empty(expected_sha):
                    counters["sha_checked"] += 1
                    if str(expected_sha).strip().lower() != image_sha:
                        counters["sha_mismatch"] += 1
                        sha_offenders.append(
                            (s["sample_id"],
                             f"image_sha256 不一致: 记录 {str(expected_sha)[:16]}... 实际 {image_sha[:16]}...")
                        )
                    else:
                        counters["sha_ok"] += 1
                else:
                    counters["sha_missing"] += 1

            if has_dim_cols:
                meta_w = coerce_number(s["row"].get("image_width"))
                meta_h = coerce_number(s["row"].get("image_height"))
                if meta_w is not None and meta_h is not None:
                    if int(meta_w) != width or int(meta_h) != height:
                        counters["dim_meta_mismatch"] += 1
                        dim_meta_offenders.append(
                            (s["sample_id"],
                             f"image_width/height 记录 {int(meta_w)}x{int(meta_h)}，实际 {width}x{height}")
                        )
                else:
                    counters["dim_meta_missing"] += 1

            # 以 sha 统计图片复用（不同 sample 相同图片）
            sha_reuse[image_sha] += 1

        # 问题记录
        if empty_path_offenders:
            if self.mode == "legacy":
                self.add_issue(
                    "legacy_image_path_empty", "warning", None,
                    f"v1 有 {len(empty_path_offenders)}/{len(self.samples)} 行 image_path 为空"
                    f"（{pct(len(empty_path_offenders) / max(1, len(self.samples)))}，v1 现状，跳过图片校验）",
                )
            else:
                self.emit_offenders(
                    "image_path_empty", "critical", empty_path_offenders,
                    "v2 样本 image_path 为空", threshold=20,
                )
        if self.args.skip_images:
            if missing_offenders:
                self.emit_offenders("image_missing", "warning", missing_offenders, "图片缺失")
        else:
            missing_sev = "critical" if self.mode == "v2" else "warning"
            self.emit_offenders("image_missing", missing_sev, missing_offenders, "image_path 文件不存在")
            self.emit_offenders("image_decode_fail", missing_sev, decode_offenders, "图片无法读取/解码")
            self.emit_offenders(
                "image_size_stage_mismatch", "warning", size_offenders,
                "图片尺寸与阶段预期不一致（A/B/C/E=640x640，D=640x840）",
            )
            self.emit_offenders("image_sha_mismatch", "critical", sha_offenders,
                                "图片 sha256 与 image_sha256 字段不一致")
            self.emit_offenders("image_dim_meta_mismatch", "warning", dim_meta_offenders,
                                "image_width/image_height 字段与实际图片不符")

        # 路径复用 / 图片内容复用
        reused_paths = [(p, c) for p, c in path_reuse.items() if c > 1]
        if reused_paths:
            self.add_issue(
                "image_path_reused", "warning", None,
                f"同一图片文件被 {len(reused_paths)} 组样本复用: "
                + "; ".join(f"{p} ×{c}" for p, c in reused_paths[:5]),
            )
        reused_sha = [(h, c) for h, c in sha_reuse.items() if c > 1]
        if reused_sha:
            self.add_issue(
                "image_content_duplicated", "warning", None,
                f"不同样本存在完全相同的图片内容（sha256 相同）: {len(reused_sha)} 组，"
                f"例如 {reused_sha[0][0][:16]}... ×{reused_sha[0][1]}",
            )

        audited = len(self.samples)
        if self.args.skip_images:
            missing_rate = None
        else:
            missing_rate = (counters["empty_path"] + counters["missing"]) / audited if audited else None

        self.m["images"] = {
            "skipped": bool(self.args.skip_images),
            "images_root": str(self.images_root),
            "expected_sizes_by_stage": {
                stage: list(size) for stage, size in STAGE_IMAGE_SIZES.items()
            } if self.mode == "v2" else {"legacy": list(EXPECTED_IMAGE_SIZE)},
            "empty_path": counters["empty_path"],
            "checked": audited - counters["empty_path"],
            "found": counters["found"],
            "missing": counters["missing"],
            "read_error": counters["read_error"],
            "decode_fail": counters["decode_fail"],
            "decoded": counters["decoded"],
            "size_stage_mismatch": counters["size_stage_mismatch"],
            "non_square": counters["non_square"],
            "dimensions": dict(dimensions),
            "sha_checked": counters["sha_checked"],
            "sha_ok": counters["sha_ok"],
            "sha_mismatch": counters["sha_mismatch"],
            "sha_missing": counters["sha_missing"],
            "dim_meta_mismatch": counters["dim_meta_mismatch"],
            "dim_meta_missing": counters["dim_meta_missing"],
            "path_reused_groups": len(reused_paths),
            "content_dup_groups": len(reused_sha),
            "missing_rate": round(missing_rate, 6) if missing_rate is not None else None,
            "samples_with_image": counters["found"] if not self.args.skip_images else None,
        }

    # -- 4. 重复 ----------------------------------------------------------
    def check_duplicates(self) -> None:
        audited = len(self.samples)
        result: Dict[str, Any] = {"available": True}

        # sample_id 重复
        id_counter = Counter(
            str(s["row"].get("sample_id")).strip()
            for s in self.samples if not is_empty(s["row"].get("sample_id"))
        )
        id_dups = {k: v for k, v in id_counter.items() if v > 1}
        result["sample_id_duplicates"] = sum(v - 1 for v in id_dups.values())
        if id_dups:
            self.add_issue(
                "sample_id_duplicate", "critical", None,
                f"sample_id 重复 {result['sample_id_duplicates']} 行，例如: "
                + ", ".join(f"{k}×{v}" for k, v in list(id_dups.items())[:5]),
            )

        # content_hash 重复
        if "content_hash" in self.cols:
            hash_counter: Counter = Counter()
            hash_paths: Dict[str, set] = defaultdict(set)
            empty_hash: List[Tuple[str, str]] = []
            for s in self.samples:
                value = s["row"].get("content_hash")
                if is_empty(value):
                    empty_hash.append((s["sample_id"], "content_hash 为空"))
                    continue
                h = str(value).strip()
                hash_counter[h] += 1
                raw_img = s["row"].get("image_path")
                if not is_empty(raw_img):
                    hash_paths[h].add(str(raw_img).strip())
            dup_groups = {h: c for h, c in hash_counter.items() if c > 1}
            dup_rows = sum(c - 1 for c in dup_groups.values())
            multi_image_groups = {h: paths for h, paths in hash_paths.items() if len(paths) > 1}
            result.update({
                "content_hash_unique": len(hash_counter),
                "content_hash_dup_groups": len(dup_groups),
                "content_hash_dup_rows": dup_rows,
                "content_hash_dup_rate": round(dup_rows / audited, 6) if audited else None,
                "content_hash_empty": len(empty_hash),
                "content_hash_multi_image_groups": len(multi_image_groups),
            })
            if empty_hash:
                self.emit_offenders("content_hash_empty", "critical", empty_hash, "content_hash 为空")
            if dup_groups:
                examples = ", ".join(
                    f"{h[:12]}...×{c}" for h, c in list(dup_groups.items())[:5]
                )
                self.add_issue(
                    "content_hash_duplicate", "critical", None,
                    f"content_hash 存在 {len(dup_groups)} 组重复、共 {dup_rows} 行多余样本（示例: {examples}）",
                )
            if multi_image_groups:
                self.add_issue(
                    "content_hash_multiple_images", "warning", None,
                    f"同一 content_hash 对应多张不同图片: {len(multi_image_groups)} 组，"
                    f"例如 {list(multi_image_groups)[0][:12]}... → {list(multi_image_groups.values())[0]}",
                )
        else:
            result["content_hash_available"] = False

        # caption_zh 完全重复
        caption_counter: Counter = Counter()
        empty_caption: List[Tuple[str, str]] = []
        for s in self.samples:
            value = s["row"].get("caption_zh")
            if is_empty(value):
                empty_caption.append((s["sample_id"], "caption_zh 为空"))
                continue
            caption_counter[str(value).strip()] += 1
        cap_dup_rows = sum(c - 1 for c in caption_counter.values() if c > 1)
        cap_dup_rate = cap_dup_rows / audited if audited else 0.0
        result.update({
            "caption_zh_unique": len(caption_counter),
            "caption_zh_dup_rows": cap_dup_rows,
            "caption_zh_dup_rate": round(cap_dup_rate, 6),
        })
        if empty_caption:
            self.emit_offenders("caption_empty", "critical", empty_caption, "caption_zh 为空")
        if cap_dup_rows:
            severity = "warning" if cap_dup_rate >= CAPTION_DUP_WARN_RATE else "info"
            self.add_issue(
                "caption_zh_duplicate", severity, None,
                f"caption_zh 完全重复 {cap_dup_rows} 行（{pct(cap_dup_rate)}），"
                f"唯一值 {len(caption_counter)}/{audited}",
            )

        # caption 归一化近重复（去数字/标点/空白）
        norm_counter: Counter = Counter()
        for value in caption_counter:
            norm_counter[normalize_caption(value)] += 1
        # 注意: 这里以“唯一 caption 的归一化结果”计数会低估；改为逐样本归一化
        norm_counter = Counter()
        for s in self.samples:
            value = s["row"].get("caption_zh")
            if is_empty(value):
                continue
            norm_counter[normalize_caption(str(value))] += 1
        near_dup_rows = sum(c - 1 for c in norm_counter.values() if c > 1)
        near_dup_rate = near_dup_rows / audited if audited else 0.0
        result.update({
            "caption_zh_normalized_unique": len(norm_counter),
            "caption_zh_near_dup_rows": near_dup_rows,
            "caption_zh_near_dup_rate": round(near_dup_rate, 6),
        })
        if near_dup_rows:
            severity = "warning" if near_dup_rate >= CAPTION_NEAR_DUP_WARN_RATE else "info"
            self.add_issue(
                "caption_zh_near_duplicate", severity, None,
                f"caption_zh 归一化（去数字/标点）后重复 {near_dup_rows} 行"
                f"（{pct(near_dup_rate)}），唯一值 {len(norm_counter)}/{audited}",
            )

        self.m["duplicates"] = result

    # -- 5. Split 泄漏 ----------------------------------------------------
    def check_split_leakage(self) -> None:
        if "split" not in self.cols:
            self.m["split"] = {"available": False, "reason": "无 split 列（legacy 数据集）"}
            return

        split_counts: Counter = Counter()
        entity_sets: Dict[str, set] = defaultdict(set)          # 仅真实资产实体 + 监督目标 role
        substrate_sets: Dict[str, set] = defaultdict(set)       # 仅真实资产底板
        procedural_entity_sets: Dict[str, set] = defaultdict(set)
        procedural_substrate_sets: Dict[str, set] = defaultdict(set)
        missing_entity_json = 0

        def _is_supervision_role(item: Dict[str, Any]) -> bool:
            """泄漏判定只纳入监督目标实体（primary/printed_instance/queried_target 等）。"""
            role = str(item.get("role") or "").strip().lower()
            if not role:
                return True
            return not role.startswith(LEAK_EXCLUDED_ROLE_PREFIXES)

        for s in self.samples:
            split = s["row"].get("split")
            if is_empty(split):
                continue
            split = str(split).strip()
            split_counts[split] += 1

            substrate = s["row"].get("substrate_id")
            if not is_empty(substrate):
                substrate_id = str(substrate).strip()
                if substrate_id in PROCEDURAL_SUBSTRATES:
                    procedural_substrate_sets[split].add(substrate_id)
                else:
                    substrate_sets[split].add(substrate_id)

            entity_ids = s.get("entity_ids")
            if isinstance(entity_ids, list):
                for item in entity_ids:
                    if not isinstance(item, dict):
                        missing_entity_json += 1
                        continue
                    if not _is_supervision_role(item):
                        continue  # candidate / distractor / in-context 候选素材不参与泄漏判定
                    domain = item.get("domain")
                    stem = item.get("stem")
                    if domain is None or stem is None:
                        missing_entity_json += 1
                        continue
                    key = (str(domain).strip(), str(stem).strip())
                    if key[0] in PROCEDURAL_DOMAINS:
                        procedural_entity_sets[split].add(key)
                    else:
                        entity_sets[split].add(key)
            elif entity_ids is None:
                stage_name = str(s["row"].get("task_stage") or "")
                if not (self.mode == "v2" and stage_name == "stage_a_geometry"):
                    missing_entity_json += 1

        def _leak(train_set: set, test_set: set, code: str, label: str) -> Dict[str, Any]:
            overlap = train_set & test_set
            if overlap:
                examples = sorted(str(x) for x in list(overlap)[:10])
                self.add_issue(
                    code, "critical", None,
                    f"{label} 泄漏: train 与测试 split 交集 {len(overlap)} 个，"
                    f"示例: {examples}",
                )
            return {
                "overlap_count": len(overlap),
                "overlap_examples": sorted(str(x) for x in list(overlap)[:20]),
            }

        train_entities = entity_sets.get("train", set())
        train_substrates = substrate_sets.get("train", set())
        leak_checks: Dict[str, Any] = {
            "entity": {},
            "substrate": {},
        }
        for target in ("test_unseen_entity", "test_unseen_both"):
            leak_checks["entity"][target] = _leak(
                train_entities, entity_sets.get(target, set()),
                "split_entity_leak", f"实体 ({target})",
            )
        for target in ("test_unseen_substrate", "test_unseen_both"):
            leak_checks["substrate"][target] = _leak(
                train_substrates, substrate_sets.get(target, set()),
                "split_substrate_leak", f"底板 ({target})",
            )

        # 程序化图元 / procedural_canvas 共享属设计预期，不计入 critical 泄漏，仅 info 记录
        proc_entity_train = procedural_entity_sets.get("train", set())
        proc_entity_test: set = set()
        for target in ("test_unseen_entity", "test_unseen_both"):
            proc_entity_test |= procedural_entity_sets.get(target, set())
        proc_entity_overlap = proc_entity_train & proc_entity_test
        proc_substrate_train = procedural_substrate_sets.get("train", set())
        proc_substrate_test: set = set()
        for target in ("test_unseen_substrate", "test_unseen_both"):
            proc_substrate_test |= procedural_substrate_sets.get(target, set())
        proc_substrate_overlap = proc_substrate_train & proc_substrate_test
        if proc_entity_overlap or proc_substrate_overlap:
            self.add_issue(
                "procedural_shared_scope", "info", None,
                f"程序化图元/合成底板在 train 与测试 split 间共享（设计预期，不计入 critical 泄漏）："
                f"实体 {len(proc_entity_overlap)} 个、底板 {len(proc_substrate_overlap)} 个；"
                f"实体示例 {sorted(str(x) for x in list(proc_entity_overlap)[:5])}，"
                f"底板示例 {sorted(proc_substrate_overlap)[:5]}",
            )

        # 缺少 split 值由 schema 必需值检查负责；此处仅提示测试 split 为空导致泄漏检查不充分
        sampled = len(self.samples) < self.total_rows
        for target in ("test_unseen_entity", "test_unseen_substrate", "test_unseen_both"):
            if split_counts.get(target, 0) == 0 and not sampled:
                self.add_issue(
                    "split_test_empty", "warning", None,
                    f"split={target} 样本数为 0，对应泄漏检查无法生效",
                )
        if missing_entity_json:
            self.add_issue(
                "entity_ids_missing_for_leak_check", "warning", None,
                f"有 {missing_entity_json} 条（样本×实体条目）entity_ids_json 缺少 domain/stem，"
                f"实体级泄漏检查可能不完整",
            )

        split_entity_sizes = {name: len(vals) for name, vals in entity_sets.items()}
        split_substrate_sizes = {name: len(vals) for name, vals in substrate_sets.items()}
        self.m["split"] = {
            "available": True,
            "counts": dict(split_counts),
            "entity_set_sizes": split_entity_sizes,
            "substrate_set_sizes": split_substrate_sizes,
            "entity_leaks": leak_checks["entity"],
            "substrate_leaks": leak_checks["substrate"],
            "missing_entity_json": missing_entity_json,
            "procedural_shared": {
                "entity_overlap_count": len(proc_entity_overlap),
                "entity_overlap_examples": sorted(str(x) for x in list(proc_entity_overlap)[:20]),
                "substrate_overlap_count": len(proc_substrate_overlap),
                "substrate_overlap_examples": sorted(proc_substrate_overlap)[:20],
            },
            "policy": (
                "仅 entity_ids_json 中监督目标 role 参与实体泄漏判定；"
                "candidate/distractor 候选素材与 geometric_primitives/procedural_canvas 程序化对象不计入 critical；"
                "val 泄漏不在本版检查范围。"
            ),
        }

    # -- 6. 分布 ----------------------------------------------------------
    def compute_distributions(self) -> None:
        audited = len(self.samples)
        dists: Dict[str, Any] = {}

        for col in ("task_stage", "task_family", "difficulty", "split", "domain"):
            if col not in self.cols:
                continue
            counter: Counter = Counter()
            for s in self.samples:
                value = s["row"].get(col)
                counter["<null>" if is_empty(value) else str(value).strip()] += 1
            dists[col] = counter_to_dist(counter, audited, sort_by_count=(col == "domain"))

        # 阶段按设计顺序展示（v2），v1 按数量
        if "task_stage" in dists:
            if self.mode == "v2":
                ordered = {}
                for stage in STAGE_ORDER:
                    if stage in dists["task_stage"]:
                        ordered[stage] = dists["task_stage"][stage]
                for key, value in dists["task_stage"].items():
                    if key not in ordered:
                        ordered[key] = value
                dists["task_stage"] = ordered
            else:
                dists["task_stage"] = dict(
                    sorted(dists["task_stage"].items(), key=lambda kv: -kv[1]["count"])
                )

        # num_entities
        if "num_entities" in self.cols:
            counter = Counter()
            for s in self.samples:
                value = coerce_number(s["row"].get("num_entities"))
                counter[str(int(value)) if value is not None else "<null>"] += 1
            try:
                keys = sorted(counter, key=lambda k: (k == "<null>", int(k) if k != "<null>" else 0))
            except ValueError:
                keys = list(counter)
            dists["num_entities"] = {
                k: {"count": counter[k], "ratio": round(counter[k] / audited, 6)} for k in keys
            }

        # match_label + 负样本比例 / 正负比
        if "match_label" in self.cols:
            counter = Counter()
            for s in self.samples:
                value = s["row"].get("match_label")
                counter["<null>" if is_empty(value) else str(value).strip()] += 1
            dists["match_label"] = counter_to_dist(counter, audited, sort_by_count=False)
            pos = counter.get("match", 0)
            neg = counter.get("no_match", 0)
            insuff = counter.get("insufficient_evidence", 0)
            na = counter.get("not_applicable", 0)
            binary_denom = pos + neg
            dists["match_label_summary"] = {
                "match": pos,
                "no_match": neg,
                "insufficient_evidence": insuff,
                "not_applicable": na,
                "negative_ratio": round(neg / audited, 6) if audited else 0.0,
                "binary_negative_ratio": round(neg / binary_denom, 6) if binary_denom else None,
                "positive_negative_ratio": round(pos / neg, 4) if neg else None,
                "positive_negative_ratio_str": f"{pos}:{neg}",
            }

        # QA 数量 / 答案类型
        qa_count_counter: Counter = Counter()
        answer_type_counter: Counter = Counter()
        qa_total = 0
        for s in self.samples:
            qa_list = s.get("qa")
            if not isinstance(qa_list, list):
                qa_count_counter["<invalid>"] += 1
                continue
            qa_count_counter[str(len(qa_list))] += 1
            for item in qa_list:
                qa_total += 1
                qa_norm = normalize_qa(item)
                at = qa_norm["answer_type"]
                answer_type_counter["<null>" if is_empty(at) else str(at).strip()] += 1
        qa_keys = sorted(qa_count_counter, key=lambda k: (k == "<invalid>", int(k) if k.isdigit() else 0))
        dists["qa_per_sample"] = {
            k: {"count": qa_count_counter[k], "ratio": round(qa_count_counter[k] / audited, 6)}
            for k in qa_keys
        }
        known_answer_types = {k: v for k, v in answer_type_counter.items() if k != "<null>"}
        if self.mode == "v2" or known_answer_types:
            dists["qa_answer_type"] = counter_to_dist(
                Counter(known_answer_types), max(1, qa_total), sort_by_count=True,
            )
        dists["qa_total"] = qa_total

        # print_visibility 分位数（None 表示该样本不适用，如负证据/几何阶段）
        vis_values: List[float] = []
        vis_null = 0
        vis_values_total = 0
        samples_with_key = 0
        for s in self.samples:
            raws: List[Any] = []
            for key in ("quality", "render_meta"):
                obj = s.get(key)
                if obj is None:
                    continue
                find_key_values(obj, "print_visibility", raws)
            if raws:
                samples_with_key += 1
            for raw in raws:
                vis_values_total += 1
                f = coerce_number(raw)
                if f is None:
                    vis_null += 1
                else:
                    vis_values.append(f)
        dists["print_visibility"] = quantile_stats(vis_values)
        dists["print_visibility"].update({
            "found": bool(vis_values),
            "values_total": vis_values_total,
            "null_count": vis_null,
            "samples_with_key": samples_with_key,
        })
        if len(vis_values) < len(self.samples) and not vis_values:
            severity = "warning" if self.mode == "v2" else "info"
            self.add_issue(
                "print_visibility_missing", severity, None,
                "quality_json/render_meta_json 中未找到 print_visibility 字段，无法计算分位数"
                if self.mode == "v2" else
                "v1 无 quality_json/render_meta_json，print_visibility 检查跳过",
            )

        # 阶段配额对比（仅 v2 且未抽样时严格；抽样时按比例折算）
        quota_comparison: Dict[str, Any] = {}
        if self.mode == "v2" and "task_stage" in self.cols:
            stage_counter = Counter()
            for s in self.samples:
                value = s["row"].get("task_stage")
                if not is_empty(value):
                    stage_counter[str(value).strip()] += 1
            sampled = len(self.samples) < self.total_rows
            scale = len(self.samples) / float(self.total_rows) if sampled else 1.0
            for stage, meta in V2_STAGES.items():
                actual = stage_counter.get(stage, 0)
                quota = meta["quota"]
                expected = quota * scale
                delta = actual - expected
                tolerance = max(1.0, expected * QUOTA_TOLERANCE)
                ok = abs(delta) <= tolerance
                quota_comparison[stage] = {
                    "code": meta["code"],
                    "quota": quota,
                    "expected": round(expected, 2) if sampled else quota,
                    "actual": actual,
                    "delta": round(delta, 2),
                    "ok": ok,
                    "note": "抽样模式按比例折算" if sampled else "",
                }
                if not ok:
                    self.add_issue(
                        "distribution_quota_deviation", "warning", None,
                        f"阶段 {meta['code']}（{stage}）样本数 {actual}，"
                        f"设计配额 {quota}{'（抽样折算 ' + str(round(expected, 2)) + '）' if sampled else ''}，"
                        f"偏差 {delta:+.2f}",
                    )
            extra_stages = [
                k for k in stage_counter
                if k not in V2_STAGES
            ]
            for stage in extra_stages:
                self.add_issue(
                    "distribution_unexpected_stage", "warning", None,
                    f"task_stage 出现设计外取值 {stage!r}（{stage_counter[stage]} 条）",
                )
        dists["stage_quota"] = quota_comparison

        self.m["distributions"] = dists

    # -- 7. 质量启发式 ----------------------------------------------------
    def check_quality(self) -> None:
        audited = len(self.samples)
        quality: Dict[str, Any] = {}

        qa_total = 0
        empty_question: List[Tuple[str, str]] = []
        empty_answer: List[Tuple[str, str]] = []
        dup_within: List[Tuple[str, str]] = []
        missing_answer_type: List[Tuple[str, str]] = []
        invalid_answer_type: List[Tuple[str, str]] = []
        invalid_answer_type_counter: Counter = Counter()
        empty_evidence: List[Tuple[str, str]] = []
        qa_count_offenders: List[Tuple[str, str]] = []
        answer_counter: Counter = Counter()
        pair_counter: Counter = Counter()
        answer_type_counter: Counter = Counter()
        qa_rows = 0

        for s in self.samples:
            qa_list = s.get("qa")
            if not isinstance(qa_list, list):
                continue
            qa_rows += 1
            stage_name_for_qa = str(s["row"].get("task_stage") or "")
            if (self.mode == "v2" and stage_name_for_qa == "stage_e_negative_evidence"
                    and len(qa_list) not in (2, 3)):
                qa_count_offenders.append(
                    (s["sample_id"], f"E 阶段每样本 QA 数 {len(qa_list)} 不在契约要求的 2~3 条")
                )
            seen_pairs = set()
            for qa_idx, item in enumerate(qa_list):
                qa_total += 1
                qa_norm = normalize_qa(item)
                question = "" if is_empty(qa_norm["question"]) else str(qa_norm["question"]).strip()
                answer = "" if is_empty(qa_norm["answer"]) else str(qa_norm["answer"]).strip()
                if not question:
                    empty_question.append((s["sample_id"], f"qa[{qa_idx}] question 为空"))
                if not answer:
                    empty_answer.append((s["sample_id"], f"qa[{qa_idx}] answer 为空"))
                if answer:
                    answer_counter[answer] += 1
                if question and answer:
                    pair_counter[(question, answer)] += 1
                pair_key = (question, answer)
                if pair_key in seen_pairs:
                    dup_within.append((s["sample_id"], f"qa[{qa_idx}] 与样本内其他 QA 完全重复"))
                seen_pairs.add(pair_key)

                if self.mode == "v2":
                    at = qa_norm["answer_type"]
                    if is_empty(at):
                        missing_answer_type.append((s["sample_id"], f"qa[{qa_idx}] answer_type 为空"))
                    else:
                        at_s = str(at).strip()
                        answer_type_counter[at_s] += 1
                        if at_s not in ANSWER_TYPES:
                            invalid_answer_type_counter[at_s] += 1
                            invalid_answer_type.append(
                                (s["sample_id"], f"qa[{qa_idx}] answer_type 非法取值 {at_s!r}")
                            )
                    evidence = qa_norm["evidence"]
                    if is_empty(evidence):
                        empty_evidence.append((s["sample_id"], f"qa[{qa_idx}] evidence_zh 为空"))

        total_answers = sum(answer_counter.values())
        top_answers = [
            {"answer": str(ans), "count": int(cnt), "ratio": round(cnt / total_answers, 6)}
            for ans, cnt in answer_counter.most_common(5)
        ] if total_answers else []
        top3_share = (
            sum(item["count"] for item in top_answers[:3]) / total_answers
            if total_answers else 0.0
        )
        pair_total = sum(pair_counter.values())
        pair_dup_rows = sum(c - 1 for c in pair_counter.values() if c > 1)
        pair_dup_rate = pair_dup_rows / pair_total if pair_total else 0.0

        quality.update({
            "qa_total": qa_total,
            "qa_per_sample_distributed": qa_rows,
            "empty_question": len(empty_question),
            "empty_answer": len(empty_answer),
            "qa_dup_within_sample": len(dup_within),
            "qa_pair_unique": len(pair_counter),
            "qa_pair_dup_rows": pair_dup_rows,
            "qa_pair_dup_rate": round(pair_dup_rate, 6),
            "top_answers": top_answers,
            "top3_answer_share": round(top3_share, 6),
            "empty_evidence": len(empty_evidence),
            "qa_count_out_of_range": len(qa_count_offenders),
            "answer_type_counts": {str(k): int(v) for k, v in answer_type_counter.items()},
        })

        self.emit_offenders("qa_empty_question", "warning", empty_question, "QA 问题为空")
        self.emit_offenders("qa_empty_answer", "warning", empty_answer, "QA 答案为空")
        self.emit_offenders("qa_duplicate_within_sample", "warning", dup_within,
                            "同一图内 QA 对完全重复")
        self.emit_offenders("qa_answer_type_missing", "critical", missing_answer_type,
                            "v2 QA answer_type 为空")
        self.emit_offenders("qa_answer_type_invalid", "critical", invalid_answer_type,
                            "v2 QA answer_type 非法")
        self.emit_offenders("qa_evidence_empty", "warning", empty_evidence,
                            "v2 QA evidence_zh 为空")
        self.emit_offenders("qa_count_out_of_range", "warning", qa_count_offenders,
                            "E 阶段每样本 QA 数不在 2~3 条")
        if self.mode == "v2" and invalid_answer_type_counter:
            self.m.setdefault("schema", {}).setdefault("enum_invalid", {})["answer_type"] = {
                str(k): int(v) for k, v in invalid_answer_type_counter.items()
            }

        if top3_share >= QA_ANSWER_CONCENTRATION_WARN:
            self.add_issue(
                "qa_answer_template_concentration", "warning", None,
                f"QA 答案 Top3 占比 {pct(top3_share)}，存在明显模板复读："
                + "; ".join(f"{item['answer'][:30]!r}×{item['count']}" for item in top_answers[:3]),
            )
        if pair_total and pair_dup_rate >= QA_PAIR_DUP_WARN_RATE:
            self.add_issue(
                "qa_pair_template_duplicate", "warning", None,
                f"跨样本 QA 对重复率 {pct(pair_dup_rate)}（{pair_dup_rows}/{pair_total}），"
                f"模板化答案复用明显",
            )
        elif pair_total and pair_dup_rate >= 0.10:
            self.add_issue(
                "qa_pair_template_duplicate", "info", None,
                f"跨样本 QA 对重复率 {pct(pair_dup_rate)}（{pair_dup_rows}/{pair_total}）",
            )
        if self.mode == "v2" and total_answers and len(empty_evidence) / max(1, qa_total) >= EVIDENCE_EMPTY_WARN_RATE:
            self.add_issue(
                "qa_evidence_high_empty_rate", "warning", None,
                f"evidence_zh 为空比例 {pct(len(empty_evidence) / max(1, qa_total))}",
            )

        # num_entities 与 entities/bboxes 数量一致性
        ne_mismatch: List[Tuple[str, str]] = []
        for s in self.samples:
            num_entities = coerce_number(s["row"].get("num_entities"))
            entities = s.get("entities")
            bboxes = s.get("bboxes")
            if num_entities is not None and isinstance(entities, list) and len(entities) != int(num_entities):
                ne_mismatch.append(
                    (s["sample_id"],
                     f"len(entities_json)={len(entities)} != num_entities={int(num_entities)}")
                )
            if num_entities is not None and isinstance(bboxes, list) and len(bboxes) != int(num_entities):
                ne_mismatch.append(
                    (s["sample_id"],
                     f"len(bboxes_norm_1000)={len(bboxes)} != num_entities={int(num_entities)}")
                )
        self.emit_offenders("num_entities_mismatch", "warning", ne_mismatch,
                            "num_entities 与 entities/bboxes 数量不一致")
        quality["num_entities_mismatch"] = len(ne_mismatch)

        bad_num_entities = []
        for s in self.samples:
            raw_ne = s["row"].get("num_entities")
            if is_empty(raw_ne):
                continue
            num_entities = coerce_number(raw_ne)
            if num_entities is None or num_entities < 0 or not float(num_entities).is_integer():
                bad_num_entities.append((s["sample_id"], f"num_entities={raw_ne!r} 非法（应为 ≥0 整数）"))
        self.emit_offenders("num_entities_invalid", "critical", bad_num_entities,
                            "num_entities 非法")

        # bbox 长宽比异常（极端细长）
        aspect_offenders: List[Tuple[str, str]] = []
        for s in self.samples:
            bboxes = s.get("bboxes")
            if not isinstance(bboxes, list):
                continue
            for idx, raw_bbox in enumerate(bboxes):
                bb, err = coerce_bbox(raw_bbox)
                if err:
                    continue
                ymin, xmin, ymax, xmax = bb
                width = abs(xmax - xmin)
                height = abs(ymax - ymin)
                if width <= 0 or height <= 0:
                    continue
                aspect = max(width / height, height / width)
                if aspect >= BBOX_ASPECT_EXTREME:
                    aspect_offenders.append(
                        (s["sample_id"],
                         f"bbox[{idx}] 长宽比 {aspect:.1f}:1（{list(bb)}），疑似坐标异常")
                    )
        self.emit_offenders("bbox_aspect_extreme", "warning", aspect_offenders,
                            f"bbox 长宽比 >= {BBOX_ASPECT_EXTREME:.0f}:1，疑似异常")
        quality["bbox_aspect_extreme"] = len(aspect_offenders)

        # 图片长宽比：image_width/image_height 字段与画像本身（已在 images 检查）；此处检查非方图
        if "image_width" in self.cols and "image_height" in self.cols:
            non_square_meta: List[Tuple[str, str]] = []
            for s in self.samples:
                w = coerce_number(s["row"].get("image_width"))
                h = coerce_number(s["row"].get("image_height"))
                if w is not None and h is not None and w > 0 and h > 0:
                    ratio = max(w / h, h / w)
                    if ratio >= BBOX_ASPECT_EXTREME:
                        non_square_meta.append(
                            (s["sample_id"], f"图片宽高比异常 {int(w)}x{int(h)}")
                        )
            self.emit_offenders("image_aspect_extreme", "warning", non_square_meta,
                                "image_width/image_height 宽高比异常")

        # v2 引用候选：D 参考匹配 / E 负证据阶段应有候选
        if self.mode == "v2" and "ref_candidates_json" in self.cols:
            ref_counts_by_stage: Dict[str, List[int]] = defaultdict(list)
            empty_ref_d = []
            for s in self.samples:
                stage = s["row"].get("task_stage")
                candidates = s.get("ref_candidates")
                n = len(candidates) if isinstance(candidates, list) else 0
                if not is_empty(stage):
                    ref_counts_by_stage[str(stage)].append(n)
                if str(stage) == "stage_d_reference_match" and n == 0:
                    empty_ref_d.append((s["sample_id"], f"{stage} ref_candidates_json 为空"))
            quality["ref_candidates_by_stage"] = {
                stage: {
                    "samples": len(values),
                    "mean": round(sum(values) / len(values), 3) if values else 0.0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "zero_samples": sum(1 for v in values if v == 0),
                }
                for stage, values in ref_counts_by_stage.items()
            }
            self.emit_offenders("ref_candidates_empty", "warning", empty_ref_d,
                                "D 阶段 ref_candidates_json 为空")

        # entity_names 空值
        for col in ("entity_names_zh", "entity_names_en"):
            if col not in self.cols:
                continue
            offenders = [
                (s["sample_id"], f"{col} 为空")
                for s in self.samples if is_empty(s["row"].get(col))
            ]
            self.emit_offenders(f"empty_value_{col}", "warning", offenders, f"{col} 为空")

        quality["empty_evidence_rate"] = (
            round(len(empty_evidence) / qa_total, 6)
            if (self.mode == "v2" and qa_total) else None
        )
        self.m["quality"] = quality

    # -- 输出 -------------------------------------------------------------
    def _ds_sha256(self) -> Optional[str]:
        try:
            return sha256_file(self.dataset_path)
        except OSError:
            return None

    def finalize(self, started_at: float) -> None:
        severity_counter = Counter(issue["severity"] for issue in self.issues)
        code_counter = Counter(issue["code"] for issue in self.issues)
        issues_sorted = sorted(
            self.issues,
            key=lambda it: (
                SEVERITY_ORDER.get(it["severity"], 9),
                it["code"],
                it["sample_id"] or "",
            ),
        )
        self.issues_sorted = issues_sorted
        elapsed = time.time() - started_at
        config = {
            "dataset": str(self.dataset_path),
            "images_root": str(self.images_root),
            "out": str(self.out_dir),
            "skip_images": bool(self.args.skip_images),
            "sample_limit": self.args.sample_limit,
            "legacy_option": self.args.legacy,
            "min_bbox_area": self.args.min_bbox_area,
            "warn_bbox_area": self.args.warn_bbox_area,
            "strict_ref_candidate_answer_bbox": bool(self.args.strict_ref_candidate_answer_bbox),
        }
        self.m["meta"] = {
            "tool": "flywheel_audit.py",
            "tool_version": TOOL_VERSION,
            "generated_at": now_str(),
            "mode": self.mode,
            "dataset": str(self.dataset_path),
            "dataset_size_bytes": self.dataset_path.stat().st_size if self.dataset_path.is_file() else None,
            "dataset_sha256": self._ds_sha256(),
            "images_root": str(self.images_root),
            "skip_images": bool(self.args.skip_images),
            "sample_limit": self.args.sample_limit,
            "total_rows": self.total_rows,
            "audited_rows": len(self.samples),
            "sampled": len(self.samples) < self.total_rows,
            "elapsed_sec": round(elapsed, 3),
            "command": " ".join([sys.executable, *sys.argv]),
            "config": config,
        }
        self.m["counts"] = {
            "total": len(self.issues),
            "critical": severity_counter.get("critical", 0),
            "warning": severity_counter.get("warning", 0),
            "info": severity_counter.get("info", 0),
            "by_code": dict(sorted(code_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        }
        # 顶层冗余字段，方便下游快速消费
        self.m["total_samples"] = self.total_rows
        self.m["audited_samples"] = len(self.samples)
        self.m["critical_count"] = severity_counter.get("critical", 0)
        self.m["warning_count"] = severity_counter.get("warning", 0)
        self.m["info_count"] = severity_counter.get("info", 0)
        self.m["mode"] = self.mode
        self.m["exit_code"] = 1 if severity_counter.get("critical", 0) > 0 else 0

    def write_outputs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        issues_path = self.out_dir / "issues.jsonl"
        with issues_path.open("w", encoding="utf-8") as fh:
            for issue in self.issues_sorted:
                fh.write(json.dumps(issue, ensure_ascii=False) + "\n")

        metrics_path = self.out_dir / "audit_metrics.json"
        metrics_path.write_text(
            json.dumps(self.m, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        report_path = self.out_dir / "audit_report.md"
        report_path.write_text(self.build_report(), encoding="utf-8")

        self.output_paths = {
            "report": report_path,
            "metrics": metrics_path,
            "issues": issues_path,
        }

    def print_summary(self) -> None:
        counts = self.m["counts"]
        print(
            f"[flywheel_audit] dataset={self.dataset_path.name} mode={self.mode} "
            f"rows={self.total_rows} audited={len(self.samples)} "
            f"critical={counts['critical']} warning={counts['warning']} info={counts['info']}"
        )
        for name, path in getattr(self, "output_paths", {}).items():
            print(f"[flywheel_audit] {name}: {path}")
        print(f"[flywheel_audit] exit_code={self.m['exit_code']}")

    # -- 报告 -------------------------------------------------------------
    def build_report(self) -> str:
        m = self.m
        counts = m["counts"]
        meta = m["meta"]
        schema = m.get("schema", {})
        bbox = m.get("bbox", {})
        images = m.get("images", {})
        duplicates = m.get("duplicates", {})
        split = m.get("split", {})
        dists = m.get("distributions", {})
        quality = m.get("quality", {})
        audited = len(self.samples)
        total = self.total_rows
        mode_label = "v2 契约模式" if self.mode == "v2" else "v1 legacy 降级模式"

        lines: List[str] = []
        lines.append("# Datawheel 数据集质检审计报告")
        lines.append("")
        lines.append(
            "> 本报告由 `tools/flywheel_audit.py` 自动生成；只读校验，不修改数据集与图片。"
        )
        lines.append("")
        lines.append(md_table(
            ["项目", "值"],
            [
                ["数据集", f"`{self.dataset_path}`"],
                ["数据模式", mode_label],
                ["总行数", total],
                ["审计行数", f"{audited}（{'抽样' if audited < total else '全量'}）"],
                ["图片检查", "已跳过（--skip-images）" if self.args.skip_images else "已执行"],
                ["图片根目录", f"`{self.images_root}`"],
                ["数据集 sha256", meta.get("dataset_sha256") or "—"],
                ["生成时间", meta.get("generated_at")],
                ["耗时", f"{meta.get('elapsed_sec')} s"],
                ["退出码", m.get("exit_code")],
                ["命令", f"`{meta.get('command')}`"],
            ],
        ))
        lines.append("")

        # 0. 结论摘要
        lines.append("## 0. 结论摘要")
        lines.append("")
        lines.append(md_table(
            ["critical", "warning", "info", "问题总数"],
            [[counts["critical"], counts["warning"], counts["info"], counts["total"]]],
        ))
        lines.append("")
        if counts["critical"] == 0:
            lines.append(
                "**结论：未发现 critical 问题。** 数据集通过硬性契约校验，"
                "可进入训练数据导出 / 下游使用；仍需关注下方 warning 与风险提示。"
            )
        else:
            lines.append(
                f"**结论：发现 {counts['critical']} 个 critical 问题，数据集当前不建议直接用于训练。** "
                "请先修复 critical（见第 8 节问题汇总），修复后重跑本工具。"
            )
        lines.append("")
        # 快速指标卡
        quick_rows: List[List[Any]] = []
        if bbox.get("available"):
            invalid_boxes = (
                bbox.get("boxes_invalid", 0) + bbox.get("boxes_out_of_range", 0)
                + bbox.get("boxes_bad_y_order", 0) + bbox.get("boxes_bad_x_order", 0)
            )
            if self.mode == "v2":
                invalid_boxes += bbox.get("boxes_too_small", 0)
            grounding_issue_total = (
                bbox.get("grounding_mismatch", 0) + bbox.get("grounding_strict_violations", 0)
            )
            quick_rows.append([
                "bbox", f"检查 {bbox.get('boxes_checked', 0)} 个",
                f"非法 {invalid_boxes}",
                f"grounding 不一致 {grounding_issue_total}",
            ])
        if not images.get("skipped"):
            quick_rows.append([
                "图片", f"解码 {images.get('decoded', 0)} 张",
                f"缺失 {images.get('missing', 0) + images.get('empty_path', 0)}",
                f"sha 不一致 {images.get('sha_mismatch', 0)}",
            ])
        content_hash_quick = (
            "v1 无此列" if duplicates.get("content_hash_available") is False
            else f"content_hash 重复行 {duplicates.get('content_hash_dup_rows', 0)}"
        )
        quick_rows.append([
            "重复", content_hash_quick,
            f"caption 重复 {duplicates.get('caption_zh_dup_rows', '—')}",
            f"caption 近重复 {duplicates.get('caption_zh_near_dup_rows', '—')}",
        ])
        if split.get("available"):
            leak_entity = sum(v.get("overlap_count", 0) for v in split.get("entity_leaks", {}).values())
            leak_substrate = sum(v.get("overlap_count", 0) for v in split.get("substrate_leaks", {}).values())
            quick_rows.append([
                "split 泄漏", f"实体交集 {leak_entity}",
                f"底板交集 {leak_substrate}",
                f"split 数 {len(split.get('counts', {}))}",
            ])
        lines.append(md_table(["维度", "指标1", "指标2", "指标3"], quick_rows))
        lines.append("")

        # 1. Schema
        lines.append("## 1. Schema 检查")
        lines.append("")
        lines.append(md_table(
            ["检查项", "结果"],
            [
                ["模式", mode_label],
                ["必需列缺失", "无" if not schema.get("missing_columns") else
                 ", ".join(f"`{c}`" for c in schema["missing_columns"])],
                ["类型不匹配", "无" if not schema.get("type_mismatches") else
                 f"{len(schema['type_mismatches'])} 列（见 metrics.schema.type_mismatches）"],
                ["额外列", "无" if not schema.get("extra_columns") else
                 ", ".join(f"`{c}`" for c in schema["extra_columns"])],
            ],
        ))
        if self.mode == "legacy":
            lines.append("")
            lines.append(
                "> legacy 模式下仅 10 列为必需列，v1 自有的其余列（如 caption_en、visual_analysis 等）"
                "不参与 v2 契约，仅作为额外列报告。"
            )
        elif schema.get("allowed_empty_json"):
            lines.append("")
            lines.append(
                "> v2 合法空值（不记 critical）：stage_a_geometry 的 `entity_ids_json` 可为空；"
                "stage_e_negative_evidence 的 absent/occluded 样本允许 `num_entities=0` 且 "
                "`bboxes_norm_1000` / `entities_json` 为空。统计："
                + ", ".join(f"`{k}`×{v}" for k, v in schema["allowed_empty_json"].items())
                + "。"
            )
        lines.append("")
        if schema.get("enum_invalid"):
            lines.append("**枚举非法取值：**")
            lines.append("")
            enum_rows = [
                [col, ", ".join(f"`{val}`×{cnt}" for val, cnt in sorted(counts_.items(), key=lambda kv: -kv[1]))]
                for col, counts_ in schema["enum_invalid"].items() if counts_
            ]
            if enum_rows:
                lines.append(md_table(["字段", "非法取值"], enum_rows))
                lines.append("")
        lines.append(
            "<details><summary>列类型明细（点击展开）</summary>"
        )
        lines.append("")
        type_rows = [
            [col, schema.get("column_types", {}).get(col, "—")]
            for col in schema.get("present_columns", [])[:40]
        ]
        lines.append(md_table(["列名", "parquet 类型"], type_rows))
        lines.append("")
        lines.append("</details>")
        lines.append("")

        # 2. BBox
        lines.append("## 2. BBox 与 Grounding 检查")
        lines.append("")
        if not bbox.get("available"):
            lines.append("（无 bboxes_norm_1000 列，跳过）")
        else:
            lines.append(md_table(
                ["检查项", "计数", "说明"],
                [
                    ["bbox 总数", bbox.get("boxes_checked", 0), "顶层 bboxes_norm_1000"],
                    ["bbox 解析失败", bbox.get("boxes_invalid", 0), "非 4 元数值数组 / JSON 错误"],
                    ["越界（非 0~1000）", bbox.get("boxes_out_of_range", 0), "闭区间 [0, 1000]"],
                    ["ymin>=ymax", bbox.get("boxes_bad_y_order", 0), "纵向顺序错误"],
                    ["xmin>=xmax", bbox.get("boxes_bad_x_order", 0), "横向顺序错误"],
                    ["面积 < critical 阈值", bbox.get("boxes_below_critical", 0),
                     f"阈值 {bbox.get('min_area_threshold', 1000):.0f}（0~1000 单位面积，"
                     f"{bbox.get('min_area_ratio', 0.001) * 100:.2f}% 画布，--min-bbox-area）"],
                    ["面积在 critical~warning 之间", bbox.get("boxes_small_warning", 0),
                     f"{bbox.get('min_area_ratio', 0.001) * 100:.2f}%~"
                     f"{bbox.get('warn_area_ratio', 0.002) * 100:.2f}% 画布，合法小目标记 warning"],
                    ["非整数坐标", bbox.get("boxes_float_coords", 0), "契约建议整数"],
                    ["合法空 bboxes 样本", bbox.get("rows_empty_bboxes_allowed", 0), "negative evidence（num_entities=0）"],
                    ["entities_json bbox 数", bbox.get("entity_bboxes_checked", 0), "递归挖掘实体 bbox"],
                    ["实体 bbox 不在顶层列表", bbox.get("entity_bbox_mismatch", 0), "grounding 一致性"],
                    ["QA grounding 检查数", bbox.get("grounding_checked", 0), "answer_bbox_1000 非空"],
                    ["QA grounding 不一致/缺失", bbox.get("grounding_mismatch", 0), "answer_bbox_1000 ∉ bboxes_norm_1000（含缺失）"],
                    ["候选参考框答案（契约修订：合法）", bbox.get("grounding_ref_answers", 0),
                     "answer_bbox_1000 ∈ ref_candidates_json[*].bbox_1000，不计 critical"],
                    ["候选框严格模式违规", bbox.get("grounding_strict_violations", 0),
                     "仅 --strict-ref-candidate-answer-bbox 时>0"],
                    ["候选框 pane 区域检查", bbox.get("ref_panes_checked", 0),
                     "stage D 顶部参考带，按 pane_index 校验"],
                    ["候选框不在 pane 区域", bbox.get("ref_pane_mismatch", 0), "warning"],
                    ["候选框互相重叠", bbox.get("ref_bbox_overlaps", 0), "warning"],
                    ["ref_candidates 候选框数", bbox.get("ref_bboxes_checked", 0), "stage D 顶部参考带"],
                    ["候选框坐标非法", bbox.get("ref_bboxes_invalid", 0), "warning"],
                    ["legacy 答案文本坐标解析", bbox.get("legacy_answer_coords_parsed", 0), "v1 降级检查"],
                    ["legacy 答案坐标不一致", bbox.get("legacy_answer_coords_mismatch", 0), "v1 降级检查"],
                ],
            ))
            if self.mode == "legacy":
                lines.append("")
                lines.append(
                    "> v1 legacy 模式下「最小面积」降级为 warning；其余 bbox 顺序/越界/grounding 不一致仍为 critical。"
                )
        lines.append("")

        # 3. 图片
        lines.append("## 3. 图片检查")
        lines.append("")
        if images.get("skipped"):
            lines.append(
                f"已通过 `--skip-images` 跳过图片 IO 校验。仍统计到 **{images.get('empty_path', 0)}** 行 "
                f"`image_path` 为空（{pct(images.get('empty_path', 0) / max(1, audited))}）。"
            )
        else:
            lines.append(md_table(
                ["检查项", "计数 / 值", "说明"],
                [
                    ["空图片路径", images.get("empty_path", 0), "image_path 为空"],
                    ["已定位文件", images.get("found", 0), "存在且可读取"],
                    ["文件缺失", images.get("missing", 0), "image_path 指向文件不存在"],
                    ["读取/解码失败", images.get("decode_fail", 0) + images.get("read_error", 0), "PIL 无法解码"],
                    ["成功解码", images.get("decoded", 0), "—"],
                    ["尺寸分布", json.dumps(images.get("dimensions", {}), ensure_ascii=False),
                     "阶段期望：A/B/C/E 640x640；D 640x840"],
                    ["尺寸与阶段预期不符", images.get("size_stage_mismatch", 0), "warning"],
                    ["sha256 校验数", images.get("sha_checked", 0), "image_sha256 非空"],
                    ["sha256 不一致", images.get("sha_mismatch", 0), "critical（v2）"],
                    ["宽高元数据不一致", images.get("dim_meta_mismatch", 0), "image_width/height 与实际不符"],
                    ["图片文件复用组", images.get("path_reused_groups", 0), "同一文件被多样本引用"],
                    ["图片内容重复组", images.get("content_dup_groups", 0), "不同样本 sha256 相同"],
                    ["图片缺失率", pct(images.get("missing_rate")), "（空路径 + 文件缺失）/ 审计行数"],
                ],
            ))
        lines.append("")

        # 4. 重复
        lines.append("## 4. 重复与去重检查")
        lines.append("")
        dup_rows = [
            ["sample_id 重复行", duplicates.get("sample_id_duplicates", 0), "critical"],
            ["content_hash 重复组", duplicates.get("content_hash_dup_groups", "—" if duplicates.get("content_hash_available") is False else 0), "critical（v2）"],
            ["content_hash 重复行", duplicates.get("content_hash_dup_rows", 0), "critical（v2）"],
            ["content_hash 同 hash 多图", duplicates.get("content_hash_multi_image_groups", 0), "warning"],
            ["caption_zh 完全重复行", duplicates.get("caption_zh_dup_rows", 0), pct(duplicates.get("caption_zh_dup_rate"))],
            ["caption_zh 唯一值", duplicates.get("caption_zh_unique", "—"), f"共 {audited} 行"],
            ["caption 归一化后重复行", duplicates.get("caption_zh_near_dup_rows", 0), pct(duplicates.get("caption_zh_near_dup_rate"))],
            ["caption 归一化后唯一值", duplicates.get("caption_zh_normalized_unique", "—"), "去数字/标点/空白"],
        ]
        if duplicates.get("content_hash_available") is False:
            dup_rows = [row for row in dup_rows if "content_hash" not in row[0]]
            lines.append("> v1 legacy 无 `content_hash` 列，跳过 content_hash 重复检查（仅 caption 重复可查）。")
            lines.append("")
        lines.append(md_table(["指标", "计数", "备注"], dup_rows))
        lines.append("")

        # 5. Split
        lines.append("## 5. Split 泄漏检查")
        lines.append("")
        if not split.get("available"):
            lines.append(
                f"（{split.get('reason', '无 split 列')}，跳过。v1 不能用于 split 泄漏校验。）"
            )
        else:
            split_rows = []
            for name, count in split.get("counts", {}).items():
                split_rows.append([
                    name,
                    count,
                    split.get("entity_set_sizes", {}).get(name, 0),
                    split.get("substrate_set_sizes", {}).get(name, 0),
                ])
            lines.append(md_table(
                ["split", "样本数", "实体集合大小 (domain, stem)", "底板集合大小"],
                split_rows,
            ))
            lines.append("")
            leak_rows = []
            for target, info in split.get("entity_leaks", {}).items():
                leak_rows.append([f"train ↔ {target}", "实体", info.get("overlap_count", 0), ", ".join(info.get("overlap_examples", [])[:3]) or "无"])
            for target, info in split.get("substrate_leaks", {}).items():
                leak_rows.append([f"train ↔ {target}", "底板", info.get("overlap_count", 0), ", ".join(info.get("overlap_examples", [])[:3]) or "无"])
            lines.append(md_table(["检查对", "类型", "交集数量", "交集示例"], leak_rows))
            lines.append("")
            lines.append(
                "> 判定：仅 `entity_ids_json` 中监督目标 role（primary / printed_instance / queried_target 等）"
                "参与实体泄漏判定；candidate / distractor / in-context 候选素材不参与。"
                "train 与 test_unseen_entity / test_unseen_both 的实体集合必须不相交；"
                "train 与 test_unseen_substrate / test_unseen_both 的底板集合必须不相交。"
                "出现交集即 critical。val 与 train 的泄漏不在本版检查范围（建议后续补充）。"
            )
            proc_shared = split.get("procedural_shared") or {}
            if proc_shared.get("entity_overlap_count") or proc_shared.get("substrate_overlap_count"):
                lines.append("")
                lines.append(
                    "**程序化对象共享（设计预期，不计入 critical）：** "
                    f"实体交集 {proc_shared.get('entity_overlap_count', 0)} 个、"
                    f"底板交集 {proc_shared.get('substrate_overlap_count', 0)} 个。"
                    "stage_a_geometry 为纯程序化色块（entity_ids 为空、substrate=procedural_canvas），"
                    "不参与实体/底板泄漏判定。"
                )
        lines.append("")

        # 6. 分布
        lines.append("## 6. 分布与设计配额")
        lines.append("")
        stage_quota = dists.get("stage_quota") or {}
        if stage_quota:
            quota_rows = []
            for stage, info in stage_quota.items():
                quota_rows.append([
                    f"{info['code']}（{stage}）",
                    info["quota"],
                    info["expected"] if info.get("note") else info["quota"],
                    info["actual"],
                    f"{info['delta']:+.0f}" if isinstance(info["delta"], (int, float)) else info["delta"],
                    "✅" if info["ok"] else "⚠️",
                    info.get("note") or "—",
                ])
            lines.append("### 6.1 阶段配额对比（A 1000 / B 4000 / C 2500 / D 1500 / E 1000）")
            lines.append("")
            lines.append(md_table(
                ["阶段", "设计配额", "期望(折算)", "实际", "偏差", "状态", "备注"],
                quota_rows,
            ))
            lines.append("")
            lines.append(
                f"> 容差：±{QUOTA_TOLERANCE * 100:.0f}%（至少 ±1 行）。"
                "抽样模式下按抽样比例折算期望值，偏差仅供参考。"
            )
            lines.append("")

        section_no = [2 if stage_quota else 1]

        def add_dist_section(title: str, key: str, note: str = "", limit: int = 40) -> None:
            dist = dists.get(key)
            if not dist:
                return
            lines.append(f"### 6.{section_no[0]} {title}")
            section_no[0] += 1
            lines.append("")
            rows = []
            for value, info in list(dist.items())[:limit]:
                if isinstance(info, dict):
                    cnt = info.get("count", 0)
                    ratio = info.get("ratio")
                else:
                    cnt = info
                    ratio = cnt / audited if audited else 0
                rows.append([value, cnt, pct(ratio)])
            if len(dist) > limit:
                rows.append([f"…（其余 {len(dist) - limit} 项省略）", "", ""])
            lines.append(md_table(["取值", "样本数", "占比"], rows))
            if note:
                lines.append("")
                lines.append(f"> {note}")
            lines.append("")

        dist_order = [
            ("任务阶段 task_stage", "task_stage", ""),
            ("任务族 task_family", "task_family", ""),
            ("难度 difficulty", "difficulty", ""),
            ("split", "split", ""),
            ("领域 domain", "domain", f"共 {len(dists.get('domain', {}))} 个取值，此处最多展示 40 项"),
            ("实体数量 num_entities", "num_entities", ""),
            ("匹配标签 match_label", "match_label", ""),
            ("每样本 QA 数", "qa_per_sample", ""),
            ("QA 答案类型 answer_type", "qa_answer_type", "" if self.mode == "v2" else "v1 无 answer_type 字段，跳过"),
        ]
        for title, key, note in dist_order:
            add_dist_section(title, key, note)

        match_summary = dists.get("match_label_summary")
        if match_summary:
            lines.append(f"### 6.{section_no[0]} 负样本占比与参考匹配正负比")
            section_no[0] += 1
            lines.append("")
            lines.append(md_table(
                ["指标", "值"],
                [
                    ["match", match_summary.get("match", 0)],
                    ["no_match", match_summary.get("no_match", 0)],
                    ["insufficient_evidence", match_summary.get("insufficient_evidence", 0)],
                    ["not_applicable", match_summary.get("not_applicable", 0)],
                    ["负样本占比 no_match / 全量", pct(match_summary.get("negative_ratio"))],
                    ["二分类负样本占比 no_match / (match+no_match)", pct(match_summary.get("binary_negative_ratio"))],
                    ["参考匹配正负比 match:no_match", match_summary.get("positive_negative_ratio_str")],
                ],
            ))
            lines.append("")

        visibility = dists.get("print_visibility") or {}
        if visibility and visibility.get("count"):
            lines.append(f"### 6.{section_no[0]} print_visibility 分位数")
            section_no[0] += 1
            lines.append("")
            lines.append(md_table(
                ["有效值数", "null 数", "min", "p10", "p25", "p50", "p75", "p90", "max", "mean"],
                [[
                    visibility.get("count"), visibility.get("null_count", 0), visibility.get("min"),
                    visibility.get("p10"), visibility.get("p25"), visibility.get("p50"),
                    visibility.get("p75"), visibility.get("p90"), visibility.get("max"),
                    visibility.get("mean"),
                ]],
            ))
            lines.append("")
            lines.append(
                "> print_visibility 仅统计数值型有效值；null 表示该样本不适用（如 stage_a 几何 / "
                "stage_e absent-occluded），不计入分位数。"
            )
            lines.append("")
        elif dists:
            lines.append(
                "> print_visibility：未在 quality_json / render_meta_json 中找到该字段，"
                "无法计算分位数（已记 warning/info）。"
            )
            lines.append("")

        # 7. 质量启发式
        lines.append("## 7. 质量启发式检查")
        lines.append("")
        qual_rows = [
            ["QA 总数", quality.get("qa_total", "—")],
            ["空问题数", quality.get("empty_question", "—")],
            ["空答案数", quality.get("empty_answer", "—")],
            ["E 阶段 QA 数不在 2~3 条", quality.get("qa_count_out_of_range", "—")],
            ["样本内 QA 完全重复", quality.get("qa_dup_within_sample", "—")],
            ["跨样本 QA 对重复率", f"{pct(quality.get('qa_pair_dup_rate'))}（v2 目标 <15%；v1 基线 42.34%）"],
            ["答案 Top3 占比", pct(quality.get("top3_answer_share"))],
            ["num_entities 与实体/bbox 数量不符", quality.get("num_entities_mismatch", "—")],
            [f"bbox 长宽比 ≥ {BBOX_ASPECT_EXTREME:.0f}:1", quality.get("bbox_aspect_extreme", "—")],
            ["evidence_zh 为空比例", pct(quality.get("empty_evidence_rate"))],
        ]
        lines.append(md_table(["指标", "值"], qual_rows))
        lines.append("")
        top_answers = quality.get("top_answers") or []
        if top_answers:
            lines.append("**全局最高频 QA 答案（模板复读观察）：**")
            lines.append("")
            lines.append(md_table(
                ["答案（截断）", "出现次数", "占全部答案比例"],
                [[(item["answer"][:60] + ("…" if len(item["answer"]) > 60 else "")),
                  item["count"], pct(item["ratio"])] for item in top_answers],
            ))
            lines.append("")
        ref_stats = quality.get("ref_candidates_by_stage")
        if ref_stats:
            lines.append("**ref_candidates_json 数量（按阶段）：**")
            lines.append("")
            lines.append(md_table(
                ["阶段", "样本数", "均值", "最小", "最大", "0 候选样本数"],
                [[stage, info["samples"], info["mean"], info["min"], info["max"], info["zero_samples"]]
                 for stage, info in ref_stats.items()],
            ))
            lines.append("")

        # 8. 问题汇总
        lines.append("## 8. 问题汇总（按 code）")
        lines.append("")
        by_code = counts.get("by_code", {})
        code_severity: Dict[str, Counter] = defaultdict(Counter)
        for issue in self.issues_sorted:
            code_severity[issue["code"]][issue["severity"]] += 1
        issue_rows = []
        for code, count in by_code.items():
            sev_counter = code_severity.get(code, Counter())
            sev = "critical" if sev_counter.get("critical") else ("warning" if sev_counter.get("warning") else "info")
            issue_rows.append([sev, f"`{code}`", count])
        issue_rows.sort(key=lambda row: (SEVERITY_ORDER.get(row[0], 9), -row[2]))
        if issue_rows:
            lines.append("> 按严重性排序（critical → warning → info），同级按条数降序。")
            lines.append("")
            lines.append(md_table(["最高严重性", "code", "条数"], issue_rows))
        else:
            lines.append("未发现任何问题。")
        lines.append("")

        # 9. 风险
        lines.append("## 9. 风险")
        lines.append("")
        risk_lines: List[str] = []
        if counts["critical"] > 0:
            top_critical = sorted(
                [(code, cnt) for code, cnt in by_code.items()
                 if code_severity.get(code, Counter()).get("critical")],
                key=lambda kv: -kv[1],
            )[:5]
            risk_lines.append(
                "- **硬性契约未通过**：存在 "
                + "、".join(f"`{code}`×{cnt}" for code, cnt in top_critical)
                + " 等 critical 问题，直接训练会引入错误监督信号（错位 bbox / 错误 grounding / 泄漏）。"
            )
        if self.mode == "legacy":
            risk_lines.append(
                "- **v1 为 legacy 数据集**：仅 250/10000 行有图片，97.5% 样本无法做图像级校验；"
                "无 split / content_hash / difficulty 等字段，不能用于 v2 的多模态训练与泄漏评估。"
            )
        if warning_codes := [
            (code, cnt) for code, cnt in by_code.items()
            if code_severity.get(code, Counter()).get("warning")
        ]:
            top_warning = sorted(warning_codes, key=lambda kv: -kv[1])[:6]
            risk_lines.append(
                "- **常规质量风险**："
                + "、".join(f"`{code}`×{cnt}" for code, cnt in top_warning)
                + "，建议在训练前抽样人工复核。"
            )
        if duplicates.get("caption_zh_dup_rows"):
            risk_lines.append(
                f"- **Caption 重复**：{duplicates['caption_zh_dup_rows']} 行完全重复"
                f"（{pct(duplicates.get('caption_zh_dup_rate'))}），可能导致 loss 对模板化文本过拟合。"
            )
        if duplicates.get("content_hash_dup_rows"):
            risk_lines.append(
                f"- **样本内容重复**：content_hash 重复 {duplicates['content_hash_dup_rows']} 行，"
                "训练集/验证集可能出现同图不同描述，削弱评估可靠性。"
            )
        if split.get("available") and sum(
            v.get("overlap_count", 0)
            for v in list(split.get("entity_leaks", {}).values()) + list(split.get("substrate_leaks", {}).values())
        ):
            risk_lines.append(
                "- **数据泄漏**：train 与 test_unseen_* 的实体/底板集合存在交集，测试指标会虚高。"
            )
        if quality.get("qa_pair_dup_rate") and quality["qa_pair_dup_rate"] >= QA_PAIR_DUP_WARN_RATE:
            risk_lines.append(
                f"- **QA 模板复读**：跨样本 QA 对重复率 {pct(quality['qa_pair_dup_rate'])}，"
                "模型可能靠记忆高频答案而非视觉推理。"
            )
        if bbox.get("boxes_below_critical"):
            risk_lines.append(
                f"- **过小 bbox（critical）**：{bbox['boxes_below_critical']} 个 bbox 面积 < "
                f"{bbox.get('min_area_ratio', 0.001) * 100:.2f}% 画布，超出小目标容忍阈值，"
                "建议生成时重试放大或过滤。"
            )
        if bbox.get("boxes_small_warning"):
            risk_lines.append(
                f"- **小目标 bbox（warning）**：{bbox['boxes_small_warning']} 个 bbox 面积处于 "
                f"{bbox.get('min_area_ratio', 0.001) * 100:.2f}%~"
                f"{bbox.get('warn_area_ratio', 0.002) * 100:.2f}% 画布，属刻意保留的细粒度定位样本，"
                "建议抽样人工确认可辨识度。"
            )
        if bbox.get("ref_pane_mismatch") or bbox.get("ref_bbox_overlaps"):
            risk_lines.append(
                f"- **stage D 候选参考框布局**：pane 区域异常 {bbox.get('ref_pane_mismatch', 0)} 个、"
                f"候选框重叠 {bbox.get('ref_bbox_overlaps', 0)} 对（warning），"
                "需确认参考带布局与候选框坐标一致。"
            )
        if images.get("size_stage_mismatch"):
            risk_lines.append(
                f"- **图片尺寸**：{images['size_stage_mismatch']} 张图片尺寸与阶段预期不符"
                "（A/B/C/E=640x640，D=640x840）；请检查生成器画布配置与 image_width/height 字段。"
            )
        if not risk_lines:
            risk_lines.append("- 未识别到显著风险项。")
        lines.extend(risk_lines)
        lines.append("")

        # 10. 建议
        lines.append("## 10. 建议")
        lines.append("")
        suggestions: List[str] = []
        if counts["critical"] > 0:
            suggestions.append(
                "1. **先修复 critical**：优先处理 schema 缺列/类型、bbox/grounding、图片缺失/解码/sha、"
                "content_hash 重复与 split 泄漏；每轮修复后重跑 `tools/flywheel_audit.py`，"
                "直到 `exit_code == 0`。"
            )
        else:
            suggestions.append(
                "1. **保持门禁**：建议将本工具接入生成流水线末尾，以 `exit_code != 0` 作为阻断发布条件。"
            )
        suggestions.extend([
            "2. **抽样人工复核**：从 warning 命中的 code 中各抽 10~20 个样本，核对 caption / QA / bbox 的语义一致性。",
            "3. **补充 val 泄漏检查**：当前仅校验 train 与测试 split；建议对 val↔train、val↔test 同样做实体/底板不相交检查。",
            "4. **增强去重**：对 caption 增加语义/模板归一化去重，对图片增加感知哈希（pHash）近重复检测。",
            "5. **阈值参数化**：最小 bbox 面积、重复率、配额容差等阈值建议做成 CLI 参数，随数据规模调整。",
            "6. **回归基线**：保存每次审计的 audit_metrics.json，对比不同版本的 critical/warning 与分布漂移。",
        ])
        lines.extend(suggestions)
        lines.append("")

        # 附录 A: 规则
        lines.append("## 附录 A：检查规则与严重性分级")
        lines.append("")
        lines.append(md_table(
            ["类别", "检查", "severity"],
            [
                ["schema", "必需列缺失 / 类型错误 / 枚举非法 / 必需字段为空", "critical"],
                ["schema", "整数列被存为浮点、额外列、A 阶段 entity_ids 为空", "warning / 合法"],
                ["schema", "E 阶段 absent/occluded：num_entities=0 且 bboxes/entities 为空", "合法（不报）"],
                ["bbox", "越界 / 顺序错误 / 解析失败 / grounding 不一致", "critical"],
                ["bbox", "面积 < critical 阈值（默认 0.10% 画布，--min-bbox-area）", "critical（v2）/ warning（legacy）"],
                ["bbox", "面积处于 critical~warning（默认 0.10%~0.20% 画布）", "warning（合法小目标）"],
                ["bbox", "answer_bbox_1000 ∈ bboxes_norm_1000 ∪ ref_candidates 候选框", "合法（契约修订）"],
                ["bbox", "候选框坐标越界/顺序非法 / 不在 pane 区域 / 互相重叠", "warning"],
                ["bbox", "非整数坐标", "info"],
                ["图片", "文件缺失 / 无法解码 / sha256 不一致", "critical（v2）"],
                ["图片", "尺寸与阶段预期不符（A/B/C/E=640x640，D=640x840）", "warning"],
                ["图片", "image_width/height 字段与实际不符 / 图片复用", "warning"],
                ["重复", "sample_id 重复 / content_hash 重复", "critical"],
                ["重复", "caption 完全/近重复、同 hash 多图", "warning 或 info（按重复率）"],
                ["split", "train 与测试 split 的资产实体/底板交集", "critical"],
                ["split", "geometric_primitives / procedural_canvas 程序化对象共享", "info（设计预期）"],
                ["split", "测试 split 为空（泄漏检查失效）", "warning"],
                ["分布", "阶段配额偏差 / 设计外阶段", "warning"],
                ["质量", "空 QA 答案/问题、QA 模板重复、实体数不一致、bbox 长宽比异常", "warning"],
                ["质量", "E 阶段每样本 QA 数不在 2~3 条", "warning"],
                ["质量", "v2 answer_type 缺失/非法", "critical"],
            ],
        ))
        lines.append("")

        # 附录 B: 问题示例
        lines.append("## 附录 B：问题示例（每类最多 3 条）")
        lines.append("")
        if not self.issues_sorted:
            lines.append("无。")
        else:
            by_code_issues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for issue in self.issues_sorted:
                by_code_issues[issue["code"]].append(issue)
            for code, issue_list in sorted(
                by_code_issues.items(),
                key=lambda kv: (-len(kv[1]), kv[0]),
            ):
                first = issue_list[0]
                lines.append(
                    f"- `{code}`（{first['severity']}，{len(issue_list)} 条）"
                )
                for issue in issue_list[:3]:
                    sid = issue["sample_id"] or "<全局>"
                    lines.append(f"  - {sid}: {md_escape(issue['detail'])[:300]}")
            lines.append("")

        return "\n".join(lines) + "\n"

    # -- 主流程 -----------------------------------------------------------
    def run(self) -> int:
        started = time.time()
        self.load()
        self.detect_mode()
        self.build_samples()
        if not self.args.quiet:
            print(
                f"[flywheel_audit] 开始审计: {self.dataset_path} "
                f"(mode={self.mode}, rows={self.total_rows}, audited={len(self.samples)})",
                file=sys.stderr,
            )
        self.check_schema()
        self.check_bboxes()
        self.check_images()
        self.check_duplicates()
        self.check_split_leakage()
        self.compute_distributions()
        self.check_quality()
        self.finalize(started)
        self.write_outputs()
        self.print_summary()
        return 1 if self.m["counts"]["critical"] > 0 else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flywheel_audit.py",
        description="Datawheel 数据集质检审计工具（v2 契约模式 / v1 legacy 降级模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  .venv/bin/python tools/flywheel_audit.py \\\n"
            "      --dataset output/datawheel_v2/dataset.parquet \\\n"
            "      --images-root . --out output/datawheel_v2/audit/\n\n"
            "  # v1 legacy 快速冒烟（跳过图片 IO）\n"
            "  .venv/bin/python tools/flywheel_audit.py \\\n"
            "      --dataset output/legacy_v1/datawheel_10k_captions.parquet \\\n"
            "      --skip-images --out /tmp/audit_v1/\n\n"
            "退出码: 0=无 critical, 1=存在 critical, 2=运行错误"
        ),
    )
    parser.add_argument("--dataset", required=True, help="数据集 parquet 路径")
    parser.add_argument(
        "--images-root", default=".",
        help="图片路径解析根目录（相对当前工作目录）；默认 .（项目根）",
    )
    parser.add_argument(
        "--out", default=None,
        help="输出目录；默认 <dataset 所在目录>/audit/",
    )
    parser.add_argument(
        "--skip-images", action="store_true",
        help="跳过图片 IO 校验（仅统计空 image_path）",
    )
    parser.add_argument(
        "--sample-limit", type=int, default=None, metavar="N",
        help="只审计 N 个均匀抽样的样本（快速冒烟用；默认全量）",
    )
    parser.add_argument(
        "--legacy", nargs="?", const="on", default="auto",
        choices=["auto", "on", "off"],
        help="legacy 模式：auto=自动识别（默认），on=强制 v1 降级检查，off=强制 v2 契约检查",
    )
    parser.add_argument(
        "--min-bbox-area", type=float, default=0.001, metavar="R",
        help="bbox 面积 critical 阈值（占画布比例，默认 0.001 = 0.10%%；面积 < R×1e6 判 critical）",
    )
    parser.add_argument(
        "--warn-bbox-area", type=float, default=0.002, metavar="R",
        help="bbox 面积 warning 阈值（默认 0.002 = 0.20%%；R_min ≤ 面积 < R_warn 记 warning）",
    )
    parser.add_argument(
        "--strict-ref-candidate-answer-bbox", action="store_true",
        help="严格模式：QA answer_bbox_1000 必须在 bboxes_norm_1000 中（默认按修订契约，"
             "允许落在 ref_candidates_json 候选框内）",
    )
    parser.add_argument("--quiet", action="store_true", help="减少 stderr 进度输出")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit 必须为正整数")
    if not (0.0 <= args.min_bbox_area <= 1.0) or not (0.0 <= args.warn_bbox_area <= 1.0):
        parser.error("--min-bbox-area / --warn-bbox-area 必须在 [0, 1] 区间（0.001 = 0.1%% 画布）")
    if args.min_bbox_area > args.warn_bbox_area:
        parser.error("--min-bbox-area 不应大于 --warn-bbox-area")
    try:
        auditor = Auditor(args)
        return auditor.run()
    except AuditError as exc:
        print(f"[flywheel_audit] 错误: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("[flywheel_audit] 已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
