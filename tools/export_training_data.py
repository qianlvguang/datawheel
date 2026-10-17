#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Datawheel 多模态训练数据导出工具（SFT：ShareGPT / ms-swift）

将 Datawheel parquet 数据集中的 ``qa_pairs_json`` 展开为独立 SFT 训练行
（一条 QA = 一行），按 split 分别导出 ShareGPT 与 ms-swift JSONL，
并生成 ``export_manifest.json``（行数 / answer_type 分布 / sha256 / 参数）
与中文 ``dataset_card.md``（数据规模、课程阶段、split 说明、格式示例、
使用建议、许可证与来源隔离提醒）。

数据契约
--------
v2 schema（``output/datawheel_v2/dataset.parquet``）关键列::

    sample_id, dataset_version, task_stage, stage_title_zh, task_family,
    difficulty, split(train|val|test_unseen_entity|test_unseen_substrate|
    test_unseen_both), domain, entity_names_zh, ..., caption_zh, ...,
    qa_pairs_json(list[{question_zh, answer_zh, answer_type,
                        answer_bbox_1000, evidence_zh}]),
    image_path, image_sha256, ...

v1 旧 schema（``output/datawheel_10k_captions.parquet``）无 ``split`` /
``dataset_version`` 列，``qa_pairs_json`` 使用 ``q_zh`` / ``a_zh`` 键，
且仅 250 行预览样本带 ``image_path``。本工具自动降级兼容：
键名做别名映射，无 split 列时全部按 ``train`` 处理并告警。

输出格式
--------
ms-swift::

    {"messages": [{"role": "user", "content": "<image>\\n<问题>"},
                  {"role": "assistant", "content": "<答案>"}],
     "images": ["output/datawheel_v2/images/DW_V2_000001.jpg"],
     "answer_type": "bbox", "bbox_norm_1000": [ymin, xmin, ymax, xmax]}

ShareGPT::

    {"conversations": [{"from": "human", "value": "<image>\\n<问题>"},
                       {"from": "gpt", "value": "<答案>"}],
     "image": "output/datawheel_v2/images/DW_V2_000001.jpg",
     "answer_type": "text"}

图片路径统一写为「项目根目录相对路径」（如 ``output/...``），以保证数据集可
迁移；``answer_type`` / ``bbox_norm_1000`` 为附加结构化元数据，标准 ms-swift
加载器读取 ``messages`` / ``images`` 时通常忽略额外字段。

用法
----
::

    .venv/bin/python tools/export_training_data.py \
        --dataset output/datawheel_v2/dataset.parquet \
        --out-dir output/datawheel_v2/sft/ \
        --formats sharegpt,ms-swift --seed 42

v1 降级实测::

    .venv/bin/python tools/export_training_data.py \
        --dataset output/datawheel_10k_captions.parquet \
        --out-dir output/datawheel_v2/sft_v1_preview/

约束：100% 本地离线；只读输入 parquet 与图片，只写 ``--out-dir`` 下的
``*_sharegpt.jsonl`` / ``*_ms_swift.jsonl`` / ``export_manifest.json`` /
``dataset_card.md``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - 环境缺依赖时给出明确提示
    sys.stderr.write(
        "[FATAL] 缺少 pyarrow，请使用项目虚拟环境执行：\n"
        "        .venv/bin/python tools/export_training_data.py ...\n"
        f"        原始错误: {exc}\n"
    )
    raise SystemExit(2)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_ORDER: List[str] = [
    "train",
    "val",
    "test_unseen_entity",
    "test_unseen_substrate",
    "test_unseen_both",
]

SPLIT_ALIASES: Dict[str, str] = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test_unseen_entity": "test_unseen_entity",
    "test_unseen_substrate": "test_unseen_substrate",
    "test_unseen_both": "test_unseen_both",
    # v1 preview 子集没有 split 语义，统一归入 train
    "preview": "train",
}

SPLIT_DESCRIPTIONS: Dict[str, str] = {
    "train": "训练集：课程学习主数据，参与梯度更新。",
    "val": "验证集：用于模型选择、早停与超参微调，不参与梯度更新。",
    "test_unseen_entity": (
        "测试集（未见实体）：实体集合与 train 不相交，用于评测模型对"
        "新实体的泛化定位能力。"
    ),
    "test_unseen_substrate": (
        "测试集（未见底板）：商品底板与 train 不相交，用于评测跨材质 / "
        "跨商品底板的泛化能力。"
    ),
    "test_unseen_both": (
        "测试集（实体与底板双未见）：最严格的泛化评测，仅用于最终报告，"
        "禁止参与任何训练或调参。"
    ),
}

# v2 task_stage → 中文阶段标题（当数据缺少 stage_title_zh 时兜底）
STAGE_FALLBACK_TITLES: Dict[str, str] = {
    "stage_a_geometry": "阶段A：基础色块拼接与几何排布",
    "stage_a_color_blocks": "阶段A：基础色块拼接与几何排布",
    "stage_b_single_element": "阶段B：真实商品单图元物理贴合",
    "stage_c_multi_element": "阶段C：真实商品多图元组合与空间排布",
    "stage_c_multi_elements": "阶段C：真实商品多图元组合与空间排布",
    "stage_d_reference_match": "阶段D：模板对照与跨底板参考匹配",
    "stage_d_in_context_pairs": "阶段D：模板对照与跨底板参考匹配",
    "stage_e_negative_evidence": "阶段E：负证据与不确定性判定",
}

FORMAT_ALIASES: Dict[str, str] = {
    "sharegpt": "sharegpt",
    "share_gpt": "sharegpt",
    "ms-swift": "ms-swift",
    "ms_swift": "ms-swift",
    "msswift": "ms-swift",
    "swift": "ms-swift",
}

# 格式 → 输出文件名中的 token（train_sharegpt.jsonl / train_ms_swift.jsonl）
FORMAT_FILE_TOKENS: Dict[str, str] = {
    "sharegpt": "sharegpt",
    "ms-swift": "ms_swift",
}

CAPTION_INSTRUCTION = (
    "请仔细观察这张图片，对其中的视觉元素、颜色、布局、材质与细节"
    "给出详尽、准确、客观的描述。"
)

# qa_pairs_json 中问题 / 答案 / 结构化 bbox 的键名别名（v2 优先，v1 兼容）
QUESTION_KEYS = ("question_zh", "q_zh", "question", "instruction", "query")
ANSWER_KEYS = ("answer_zh", "a_zh", "answer", "response", "output")
BBOX_KEYS = ("answer_bbox_1000", "answer_bbox", "bbox_1000", "bbox_norm_1000", "bbox")
ANSWER_TYPE_KEYS = ("answer_type", "type")
EVIDENCE_KEYS = ("evidence_zh", "evidence")

BBOX_ANSWER_TYPES = {"bbox", "bbox_1000", "box", "bounding_box", "grounding"}

MSG_IMAGE_PREFIX = "<image>\n"

# 必需列（缺失则无法导出）
REQUIRED_COLUMNS = ("qa_pairs_json", "image_path")

# 统计 / 报告用到的可选列
OPTIONAL_COLUMNS = (
    "sample_id",
    "dataset_version",
    "split",
    "task_stage",
    "stage_title_zh",
    "task_family",
    "difficulty",
    "domain",
    "num_entities",
    "substrate_id",
    "substrate_category_zh",
    "caption_zh",
    "image_sha256",
    "image_width",
    "image_height",
)

VD2_SCHEMA_CONTRACT: Tuple[str, ...] = (
    "sample_id",
    "dataset_version",
    "task_stage",
    "stage_title_zh",
    "task_family",
    "difficulty",
    "split",
    "domain",
    "entity_names_zh",
    "entity_names_en",
    "num_entities",
    "substrate_id",
    "substrate_category_zh",
    "placement_zone",
    "bboxes_norm_1000",
    "entities_json",
    "ref_candidates_json",
    "match_label",
    "caption_zh",
    "caption_en",
    "visual_analysis",
    "knowledge_expansion",
    "qa_pairs_json",
    "image_path",
    "image_sha256",
    "image_width",
    "image_height",
    "render_meta_json",
    "quality_json",
    "content_hash",
)

MAX_WARNINGS = 50


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """返回带本地时区的 ISO 时间戳（秒精度）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    """计算文件 SHA256（分块读取，避免大文件占内存）。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """将路径尽量化为项目根相对路径（POSIX 风格），失败则返回绝对路径。"""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def canonical_image_ref(resolved: Path) -> str:
    """把已确认存在的图片路径转换为写进 JSONL 的规范引用。

    优先项目根相对路径（例如 ``output/datawheel_v2/images/DW_V2_000001.jpg``），
    当图片在项目目录之外时退化为绝对路径。
    """
    try:
        return resolved.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.resolve().as_posix()


def normalize_text(text: Any) -> str:
    """折叠空白，用于去重与空值判断。"""
    if text is None:
        return ""
    return " ".join(str(text).split())


def pct(value: int, total: int) -> str:
    if not total:
        return "0.0%"
    return f"{value / total * 100:.1f}%"


def pretty_counter(counter: Counter, total: Optional[int] = None) -> str:
    """把 Counter 输出为 ``key=count (pct), ...`` 的单行摘要。"""
    if not counter:
        return "（无）"
    if total is None:
        total = sum(counter.values())
    items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return ", ".join(f"{key}={count} ({pct(count, total)})" for key, count in items)


def add_warning(warnings: List[str], message: str) -> None:
    if message not in warnings and len(warnings) < MAX_WARNINGS:
        warnings.append(message)


def resolve_image_path(
    raw: Any,
    dataset_path: Path,
    image_root: Optional[Path] = None,
) -> Optional[Path]:
    """解析图片真实路径，返回存在的绝对路径；不存在返回 None。

    兼容：
    - 项目根相对路径（``output/datawheel_v2/images/xxx.jpg``）；
    - 数据集所在目录相对路径（``images/xxx.jpg``）；
    - 绝对路径；
    - ``--image-root`` 指定的额外根目录。
    """
    if raw is None:
        return None
    raw_str = str(raw).strip()
    if not raw_str:
        return None

    candidate = Path(raw_str).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        return resolved if resolved.is_file() else None

    bases: List[Path] = []
    if image_root is not None:
        bases.append(image_root)
    bases.extend(
        [
            PROJECT_ROOT,
            PROJECT_ROOT / "output",
            dataset_path.parent,
            dataset_path.parent.parent,
            Path.cwd(),
        ]
    )

    seen: set = set()
    for base in bases:
        try:
            resolved = (base / candidate).resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def coerce_bbox(value: Any) -> Optional[List[Any]]:
    """校验并规范化 bbox 为 ``[ymin, xmin, ymax, xmax]``（0~1000）。

    非法（长度不对 / 非数值 / 越界 / 顺序错误）返回 None。
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
            if len(numbers) != 4:
                return None
            value = numbers
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    numbers: List[Any] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        numbers.append(int(number) if number.is_integer() else round(number, 3))

    ymin, xmin, ymax, xmax = numbers
    if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
        return None
    return numbers


def extract_bbox_from_answer(answer: str) -> Optional[List[Any]]:
    """答案文本中若内嵌 4 元坐标（如 ``[667, 0, 1000, 1000]``），提取为结构化 bbox。"""
    text = answer.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        bbox = coerce_bbox(parsed)
        if bbox is not None:
            return bbox
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(
        r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*"
        r"(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[\]\)]",
        text,
    )
    if not match:
        return None
    return coerce_bbox([match.group(i) for i in range(1, 5)])


def get_first_value(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    """按候选键顺序取第一个非空值。"""
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
    return None


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


def parse_formats_arg(raw: str) -> List[str]:
    formats: List[str] = []
    for token in raw.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in FORMAT_ALIASES:
            valid = ", ".join(sorted(FORMAT_ALIASES))
            raise argparse.ArgumentTypeError(
                f"未知格式 {token!r}；可选：{valid}"
            )
        canonical = FORMAT_ALIASES[key]
        if canonical not in formats:
            formats.append(canonical)
    if not formats:
        raise argparse.ArgumentTypeError("--formats 不能为空")
    return formats


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Datawheel 数据集 → SFT 训练集导出工具"
            "（ShareGPT / ms-swift JSONL + manifest + dataset card）"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  .venv/bin/python tools/export_training_data.py \\\n"
            "      --dataset output/datawheel_v2/dataset.parquet \\\n"
            "      --out-dir output/datawheel_v2/sft/ \\\n"
            "      --formats sharegpt,ms-swift --seed 42\n"
            "\n"
            "v1 降级实测：\n"
            "  .venv/bin/python tools/export_training_data.py \\\n"
            "      --dataset output/datawheel_10k_captions.parquet \\\n"
            "      --out-dir output/datawheel_v2/sft_v1_preview/\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="输入 parquet 数据集（v2: output/datawheel_v2/dataset.parquet；"
        "v1 降级: output/datawheel_10k_captions.parquet）",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="输出目录（自动创建；只覆盖本工具生成的文件）",
    )
    parser.add_argument(
        "--formats",
        type=parse_formats_arg,
        default=parse_formats_arg("sharegpt,ms-swift"),
        help="导出格式，逗号分隔：sharegpt,ms-swift（默认两者都导出）",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="随机种子（split 内打乱、caption 抽样、去重顺序）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每个 split 最多使用的源样本数（过滤后按 seed 打乱截断，便于小样本冒烟）",
    )
    parser.add_argument(
        "--include-caption",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否为部分源样本额外导出一条 dense caption 行（默认开启；"
        "关闭用 --no-include-caption）",
    )
    parser.add_argument(
        "--caption-ratio",
        type=float,
        default=0.2,
        help="dense caption 行覆盖的源样本比例，0~1（默认 0.2）",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="图片相对路径的额外解析根目录（可选，默认自动尝试项目根 / "
        "output / 数据集目录）",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须 >= 1")
    if not (0.0 <= args.caption_ratio <= 1.0):
        parser.error("--caption-ratio 必须在 [0, 1] 区间")
    return args


# ---------------------------------------------------------------------------
# 数据加载与 schema 识别
# ---------------------------------------------------------------------------


def load_dataset(dataset_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """读取 parquet，返回 (列名列表, 行字典列表)。"""
    parquet = pq.ParquetFile(dataset_path)
    columns = list(parquet.schema_arrow.names)
    table = parquet.read()
    rows = table.to_pylist()
    return columns, rows


def detect_schema_variant(columns: Sequence[str]) -> str:
    """识别数据集 schema 变体：v2 / v2_partial / v1_preview_degraded。"""
    column_set = set(columns)
    if "split" in column_set and "dataset_version" in column_set:
        return "v2"
    if "split" in column_set:
        return "v2_partial"
    return "v1_preview_degraded"


def classify_split(record: Dict[str, Any], unknown_splits: Counter) -> Tuple[str, bool]:
    """把样本映射到标准 split；无法识别时降级为 train。返回 (split, 是否降级)。"""
    raw = record.get("split")
    if raw is None or not str(raw).strip():
        return "train", True
    key = str(raw).strip().lower()
    if key in SPLIT_ALIASES:
        return SPLIT_ALIASES[key], False
    unknown_splits[key] += 1
    return "train", True


# ---------------------------------------------------------------------------
# QA / caption 规范化
# ---------------------------------------------------------------------------


def parse_qa_pairs_json(
    raw: Any,
) -> Tuple[Optional[List[Any]], bool]:
    """解析 qa_pairs_json。返回 (pairs, ok)；ok=False 表示 JSON 解析失败。"""
    if raw is None:
        return [], True
    if not isinstance(raw, str):
        return (raw if isinstance(raw, list) else []), True
    text = raw.strip()
    if not text:
        return [], True
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None, False
    if isinstance(parsed, list):
        return parsed, True
    return None, False


def normalize_qa_pair(
    pair: Any,
    counters: Counter,
) -> Optional[Dict[str, Any]]:
    """把单条 QA 规范化为 {question, answer, answer_type, bbox}。

    空问题 / 空答案会被丢弃并计数；v1 缺少 answer_type 时，
    若答案内嵌 4 元坐标则推断为 bbox。
    """
    if not isinstance(pair, dict):
        counters["skipped_bad_qa_item"] += 1
        return None

    question = normalize_text(get_first_value(pair, QUESTION_KEYS))
    answer = normalize_text(get_first_value(pair, ANSWER_KEYS))

    if not question:
        counters["skipped_empty_question"] += 1
        return None
    if not answer:
        counters["skipped_empty_answer"] += 1
        return None

    raw_type_value = get_first_value(pair, ANSWER_TYPE_KEYS)
    raw_type = str(raw_type_value).strip().lower() if raw_type_value is not None else ""

    bbox = coerce_bbox(get_first_value(pair, BBOX_KEYS))
    bbox_inferred = False
    if bbox is None:
        bbox = extract_bbox_from_answer(answer)
        bbox_inferred = bbox is not None

    if raw_type in BBOX_ANSWER_TYPES:
        answer_type = "bbox"
        if bbox is None:
            counters["invalid_bbox"] += 1
    elif raw_type:
        # v2 已显式声明答案类型：以声明为准，避免把普通数值列表误标为 bbox
        answer_type = raw_type
        if bbox is not None:
            counters["bbox_dropped_non_bbox_type"] += 1
            bbox = None
            bbox_inferred = False
    else:
        # v1 旧 schema 无 answer_type：答案内嵌坐标时推断为 bbox
        answer_type = "bbox" if bbox is not None else "text"

    if bbox_inferred and answer_type == "bbox":
        counters["bbox_inferred"] += 1

    return {
        "question": question,
        "answer": answer,
        "answer_type": answer_type,
        "bbox": bbox,
        "bbox_inferred": bbox_inferred,
    }


# ---------------------------------------------------------------------------
# 源样本过滤
# ---------------------------------------------------------------------------


def prepare_source_records(
    parquet_rows: Iterable[Dict[str, Any]],
    split_raw_counts: Counter,
    counters: Counter,
    warnings: List[str],
    dataset_path: Path,
    image_root: Optional[Path],
    include_caption: bool,
    unknown_splits: Counter,
    stage_titles: Dict[str, str],
    schema_variant: str = "v2",
) -> Dict[str, List[Dict[str, Any]]]:
    """逐条源样本做图片存在性 / QA 解析 / 空值过滤，按 split 分组。"""
    grouped: Dict[str, List[Dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}
    missing_image_samples: List[Tuple[str, str]] = []
    bad_qa_samples: List[str] = []
    no_image_samples = 0

    for record in parquet_rows:
        counters["source_rows"] += 1
        split, degraded = classify_split(record, unknown_splits)
        split_raw_counts[split] += 1
        if degraded:
            counters["samples_split_degraded"] += 1

        task_stage = normalize_text(record.get("task_stage"))
        stage_title = normalize_text(record.get("stage_title_zh"))
        if task_stage and task_stage not in stage_titles:
            stage_titles[task_stage] = (
                stage_title or STAGE_FALLBACK_TITLES.get(task_stage, task_stage)
            )

        raw_image = record.get("image_path")
        if raw_image is None or not str(raw_image).strip():
            counters["skipped_no_image_path"] += 1
            no_image_samples += 1
            continue

        resolved_image = resolve_image_path(raw_image, dataset_path, image_root)
        if resolved_image is None:
            counters["skipped_missing_image_file"] += 1
            if len(missing_image_samples) < 20:
                missing_image_samples.append(
                    (normalize_text(record.get("sample_id")), str(raw_image))
                )
            continue

        pairs, qa_ok = parse_qa_pairs_json(record.get("qa_pairs_json"))
        if not qa_ok:
            counters["skipped_bad_qa_json"] += 1
            if len(bad_qa_samples) < 20:
                bad_qa_samples.append(normalize_text(record.get("sample_id")))
            continue

        qa_items: List[Dict[str, Any]] = []
        for pair in pairs or []:
            normalized = normalize_qa_pair(pair, counters)
            if normalized is not None:
                qa_items.append(normalized)

        caption_text = normalize_text(record.get("caption_zh"))

        if not qa_items and not (include_caption and caption_text):
            counters["skipped_no_content"] += 1
            continue

        grouped[split].append(
            {
                "sample_id": normalize_text(record.get("sample_id")),
                "task_stage": task_stage,
                "stage_title_zh": stage_titles.get(task_stage, task_stage),
                "difficulty": normalize_text(record.get("difficulty")),
                "domain": normalize_text(record.get("domain")),
                "image": canonical_image_ref(resolved_image),
                "qa": qa_items,
                "caption": caption_text if include_caption else "",
            }
        )

    if no_image_samples:
        note = (
            "（v1 仅保存预览图时属于预期）"
            if schema_variant == "v1_preview_degraded"
            else ""
        )
        add_warning(
            warnings,
            f"{no_image_samples:,} 条源样本无 image_path{note}，已跳过。",
        )
    if counters["skipped_missing_image_file"]:
        add_warning(
            warnings,
            f"{counters['skipped_missing_image_file']} 条源样本的图片文件不存在，已跳过"
            f"（示例：{missing_image_samples[:3]}）。",
        )
    if counters["skipped_bad_qa_json"]:
        add_warning(
            warnings,
            f"{counters['skipped_bad_qa_json']} 条源样本的 qa_pairs_json 解析失败，已跳过"
            f"（示例 sample_id：{bad_qa_samples[:5]}）。",
        )
    if not include_caption:
        add_warning(warnings, "已通过 --no-include-caption 关闭 dense caption 行。")
    return grouped


# ---------------------------------------------------------------------------
# 训练行展开
# ---------------------------------------------------------------------------


def make_training_row(fmt: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """把一条训练规格转换为目标格式的 JSONL 行。"""
    question = spec["question"]
    answer = spec["answer"]
    image = spec["image"]

    user_content = (
        question if "<image>" in question else MSG_IMAGE_PREFIX + question
    )

    if fmt == "ms-swift":
        row: Dict[str, Any] = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ],
            "images": [image],
        }
    elif fmt == "sharegpt":
        row = {
            "conversations": [
                {"from": "human", "value": user_content},
                {"from": "gpt", "value": answer},
            ],
            "image": image,
        }
    else:  # pragma: no cover - 参数已校验
        raise ValueError(f"未知格式: {fmt}")

    row["answer_type"] = spec["answer_type"]
    if spec.get("bbox") is not None:
        row["bbox_norm_1000"] = spec["bbox"]
    return row


def expand_split_rows(
    source_records: List[Dict[str, Any]],
    split: str,
    seed: int,
    limit: Optional[int],
    include_caption: bool,
    caption_ratio: float,
    counters: Counter,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """对单个 split 的源样本：确定性打乱 → limit → 展开 QA / caption → 去重 → 打乱。

    返回 (训练行规格列表, 实际使用的源样本列表)。
    """
    records = list(source_records)
    source_rng = random.Random(f"{seed}|{split}|source")
    source_rng.shuffle(records)
    if limit is not None and len(records) > limit:
        records = records[:limit]

    # 确定性抽取 caption 覆盖的源样本
    caption_indices: set = set()
    if include_caption and caption_ratio > 0 and records:
        caption_count = int(round(caption_ratio * len(records)))
        caption_count = max(0, min(caption_count, len(records)))
        caption_rng = random.Random(f"{seed}|{split}|caption")
        caption_indices = set(caption_rng.sample(range(len(records)), caption_count))

    specs: List[Dict[str, Any]] = []
    contributing_records: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        specs_before = len(specs)
        base = {
            "image": record["image"],
            "sample_id": record["sample_id"],
            "task_stage": record["task_stage"],
        }
        for qa in record["qa"]:
            spec = dict(base)
            spec.update(
                {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "answer_type": qa["answer_type"],
                    "bbox": qa["bbox"],
                    "kind": "qa",
                }
            )
            specs.append(spec)
        if index in caption_indices and record["caption"]:
            spec = dict(base)
            spec.update(
                {
                    "question": CAPTION_INSTRUCTION,
                    "answer": record["caption"],
                    "answer_type": "caption",
                    "bbox": None,
                    "kind": "caption",
                }
            )
            specs.append(spec)
        if len(specs) > specs_before:
            contributing_records.append(record)

    # 去重：question + answer + image
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for spec in specs:
        key = (
            normalize_text(spec["question"]),
            normalize_text(spec["answer"]),
            spec["image"],
        )
        if key in seen:
            counters["skipped_duplicate"] += 1
            continue
        seen.add(key)
        deduped.append(spec)

    row_rng = random.Random(f"{seed}|{split}|rows")
    row_rng.shuffle(deduped)
    return deduped, contributing_records


# ---------------------------------------------------------------------------
# 写出 JSONL
# ---------------------------------------------------------------------------


def write_jsonl(
    path: Path,
    specs: List[Dict[str, Any]],
    fmt: str,
) -> Tuple[int, int, str]:
    """写入 JSONL（原子替换），返回 (行数, 字节数, sha256)。"""
    templ = path.with_name(path.name + ".tmp")
    with templ.open("w", encoding="utf-8", newline="\n") as fh:
        for spec in specs:
            row = make_training_row(fmt, spec)
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    templ.replace(path)
    return len(specs), path.stat().st_size, sha256_file(path)


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------


def validate_jsonl_file(
    path: Path,
    fmt: str,
    dataset_path: Path,
    image_root: Optional[Path],
    sample_size: int = 20,
) -> Dict[str, Any]:
    """逐行 json.loads + 结构校验 + 抽样图片存在性校验。"""
    result: Dict[str, Any] = {
        "path": display_path(path),
        "fmt": fmt,
        "rows": 0,
        "json_errors": [],
        "structural_errors": [],
        "image_samples_checked": 0,
        "image_missing": [],
        "bbox_rows": 0,
        "caption_rows": 0,
        "answer_type_dist": {},
    }

    lines = path.read_text(encoding="utf-8").splitlines()
    objects: List[Dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            result["json_errors"].append(f"line {line_no}: {exc}")
            continue
        if not isinstance(obj, dict):
            result["structural_errors"].append(f"line {line_no}: 顶层不是 dict")
            continue
        objects.append(obj)

    result["rows"] = len(objects)
    answer_types: Counter = Counter()

    for index, obj in enumerate(objects, start=1):
        answer_type = obj.get("answer_type", "text")
        answer_types[str(answer_type)] += 1
        if "bbox_norm_1000" in obj:
            result["bbox_rows"] += 1
            if coerce_bbox(obj["bbox_norm_1000"]) is None:
                result["structural_errors"].append(
                    f"row {index}: bbox_norm_1000 非法 {obj['bbox_norm_1000']!r}"
                )
        if answer_type == "caption":
            result["caption_rows"] += 1

        if fmt == "ms-swift":
            messages = obj.get("messages")
            images = obj.get("images")
            problem = None
            if not isinstance(messages, list) or len(messages) != 2:
                problem = "messages 必须为 2 轮"
            elif [m.get("role") for m in messages] != ["user", "assistant"]:
                problem = "messages role 顺序必须为 user/assistant"
            elif not isinstance(images, list) or len(images) != 1:
                problem = "images 必须为长度 1 的列表"
            elif not str(messages[0].get("content", "")).startswith("<image>"):
                problem = "user content 缺少 <image> 占位"
            elif not str(images[0]).strip():
                problem = "images[0] 为空"
            if problem:
                result["structural_errors"].append(f"row {index}: {problem}")
        else:
            conversations = obj.get("conversations")
            problem = None
            if not isinstance(conversations, list) or len(conversations) != 2:
                problem = "conversations 必须为 2 轮"
            elif [c.get("from") for c in conversations] != ["human", "gpt"]:
                problem = "conversations from 顺序必须为 human/gpt"
            elif not str(conversations[0].get("value", "")).startswith("<image>"):
                problem = "human value 缺少 <image> 占位"
            elif not isinstance(obj.get("image"), str) or not obj["image"].strip():
                problem = "image 字段缺失或为空"
            if problem:
                result["structural_errors"].append(f"row {index}: {problem}")

    # 抽样 20 行做图片存在性校验（等距抽样，确定性）
    total = len(objects)
    if total:
        k = min(sample_size, total)
        if k == total:
            sample_indices = list(range(total))
        else:
            sample_indices = sorted(
                {round(i * (total - 1) / (k - 1)) for i in range(k)}
            )
        for index in sample_indices:
            obj = objects[index]
            ref = obj["images"][0] if fmt == "ms-swift" else obj.get("image", "")
            resolved = resolve_image_path(ref, dataset_path, image_root)
            result["image_samples_checked"] += 1
            if resolved is None:
                result["image_missing"].append(str(ref))

    result["answer_type_dist"] = dict(answer_types)
    return result


def validate_outputs(
    out_dir: Path,
    splits_present: List[str],
    formats: List[str],
    dataset_path: Path,
    image_root: Optional[Path],
) -> Dict[str, Any]:
    """校验全部输出 JSONL，返回汇总结果。"""
    file_results: Dict[str, Any] = {}
    json_error_count = 0
    structural_error_count = 0
    image_missing_count = 0
    image_checked_total = 0
    total_rows = 0

    for split in splits_present:
        for fmt in formats:
            path = out_dir / f"{split}_{FORMAT_FILE_TOKENS[fmt]}.jsonl"
            if not path.is_file():
                file_results[display_path(path)] = {
                    "path": display_path(path),
                    "fmt": fmt,
                    "rows": 0,
                    "error": "文件不存在",
                }
                structural_error_count += 1
                continue
            result = validate_jsonl_file(path, fmt, dataset_path, image_root)
            file_results[result["path"]] = result
            json_error_count += len(result["json_errors"])
            structural_error_count += len(result["structural_errors"])
            image_missing_count += len(result["image_missing"])
            image_checked_total += result["image_samples_checked"]
            total_rows += result["rows"]

    return {
        "files": file_results,
        "json_errors": json_error_count,
        "structural_errors": structural_error_count,
        "image_missing_total": image_missing_count,
        "image_samples_checked": image_checked_total,
        "rows_validated": total_rows,
        "passed": (
            json_error_count == 0
            and structural_error_count == 0
            and image_missing_count == 0
        ),
    }


# ---------------------------------------------------------------------------
# dataset_card 示例行提取
# ---------------------------------------------------------------------------


def shorten_for_card(obj: Any, max_str: int = 420) -> Any:
    if isinstance(obj, str):
        if len(obj) <= max_str:
            return obj
        return obj[:max_str] + "…（长文本已截断，完整内容见 JSONL）"
    if isinstance(obj, list):
        return [shorten_for_card(item, max_str) for item in obj]
    if isinstance(obj, dict):
        return {key: shorten_for_card(value, max_str) for key, value in obj.items()}
    return obj


def card_json_block(row: Dict[str, Any]) -> str:
    return json.dumps(shorten_for_card(row), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    args: argparse.Namespace,
    dataset_path: Path,
    dataset_columns: List[str],
    schema_variant: str,
    dataset_rows: int,
    split_stats: Dict[str, Dict[str, Any]],
    totals: Dict[str, Any],
    filtering: Dict[str, Any],
    validation: Dict[str, Any],
    card_path: Path,
    warnings: List[str],
    generated_at: str,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    splits_manifest: Dict[str, Any] = {}
    for split in SPLIT_ORDER:
        if split not in split_stats:
            continue
        stats = split_stats[split]
        files: Dict[str, Any] = {}
        for fmt in args.formats:
            file_info = stats["files"].get(fmt)
            if file_info is None:
                continue
            files[fmt] = {
                "file": file_info["path"],
                "rows": file_info["rows"],
                "bytes": file_info["bytes"],
                "sha256": file_info["sha256"],
                "answer_type_dist": file_info["answer_type_dist"],
            }
        splits_manifest[split] = {
            "source_rows": stats["source_rows"],
            "source_rows_used": stats["source_rows_used"],
            "unique_images": stats["unique_images"],
            "qa_rows": stats["qa_rows"],
            "caption_rows": stats["caption_rows"],
            "rows_per_format": {fmt: stats["files"][fmt]["rows"] for fmt in files},
            "answer_type_dist": stats["answer_type_dist"],
            "task_stage_dist": stats["task_stage_dist"],
            "difficulty_dist": stats.get("difficulty_dist", {}),
            "files": files,
        }

    return {
        "manifest_version": 1,
        "tool": "tools/export_training_data.py",
        "generated_at": generated_at,
        "command": " ".join(sys.argv),
        "dataset": {
            "path": display_path(dataset_path),
            "sha256": sha256_file(dataset_path),
            "rows": dataset_rows,
            "schema_variant": schema_variant,
            "is_degraded_v1": schema_variant == "v1_preview_degraded",
            "columns": dataset_columns,
            "dataset_version": totals.get("dataset_version"),
        },
        "parameters": {
            "formats": list(args.formats),
            "seed": args.seed,
            "limit": args.limit,
            "include_caption": bool(args.include_caption),
            "caption_ratio": float(args.caption_ratio),
            "caption_instruction": CAPTION_INSTRUCTION if args.include_caption else None,
            "image_root": str(args.image_root) if args.image_root else None,
            "out_dir": display_path(Path(args.out_dir).resolve()),
            "splits_expected": SPLIT_ORDER,
            "splits_written": list(split_stats.keys()),
            "image_path_style": "project-relative (相对项目根目录)",
            "message_image_placeholder": "<image>",
        },
        "filtering": filtering,
        "totals": totals,
        "splits": splits_manifest,
        "outputs": {
            "dataset_card": {
                "file": display_path(card_path),
                "bytes": card_path.stat().st_size if card_path.is_file() else 0,
                "sha256": sha256_file(card_path) if card_path.is_file() else None,
            },
            "format_files": {
                split: {
                    fmt: split_stats[split]["files"][fmt]["path"]
                    for fmt in split_stats[split]["files"]
                }
                for split in split_stats
            },
        },
        "validation": validation,
        "warnings": warnings,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }


# ---------------------------------------------------------------------------
# dataset_card.md
# ---------------------------------------------------------------------------


def build_dataset_card(
    *,
    args: argparse.Namespace,
    dataset_path: Path,
    dataset_rows: int,
    schema_variant: str,
    split_stats: Dict[str, Dict[str, Any]],
    totals: Dict[str, Any],
    filtering: Dict[str, Any],
    validation: Dict[str, Any],
    examples: Dict[str, Any],
    stage_titles: Dict[str, str],
    warnings: List[str],
    generated_at: str,
) -> str:
    lines: List[str] = []
    add = lines.append
    degraded = schema_variant == "v1_preview_degraded"

    add("# Datawheel 多模态 SFT 训练数据集卡")
    add("")
    add(
        "> 由 `tools/export_training_data.py` 自动生成 · "
        f"生成时间 `{generated_at}`"
    )
    add(
        f"> 数据源 `{display_path(dataset_path)}` · schema 变体 "
        f"`{schema_variant}` · 源样本 {dataset_rows:,} 行"
    )
    add("")

    if degraded:
        with_image_count = max(
            0,
            filtering["source_rows"]
            - filtering["skipped_no_image_path"]
            - filtering["skipped_missing_image_file"],
        )
        add("## ⚠️ 降级预览说明（重要）")
        add("")
        add(
            f"本卡片由 **v1 旧 schema 预览子集**（仅 {with_image_count:,} 行带预览图的样本）"
            "导出，**仅用于导出工具降级验证，不是正式训练集**。"
            "正式多模态 SFT 训练请使用 v2 数据集"
            "（`output/datawheel_v2/dataset.parquet`）重新导出。"
        )
        add("")
    for warning in warnings:
        if warning.startswith("导出数据仅包含") or "v1" in warning:
            add(f"> ⚠️ {warning}")
            add("")

    # 1. 数据规模 ---------------------------------------------------------
    add("## 1. 数据规模")
    add("")
    total_sft_rows = sum(stats["files"][next(iter(stats["files"]))]["rows"]
                         for stats in split_stats.values() if stats["files"])
    add(
        f"- 参与导出的源样本：**{sum(s['source_rows_used'] for s in split_stats.values()):,}** 行"
        f"（源 parquet 共 {dataset_rows:,} 行，过滤后使用）"
    )
    add(
        f"- SFT 训练行：**{total_sft_rows:,}** 行 / 每种格式"
        f"（QA {totals['qa_rows']:,} 行 + dense caption {totals['caption_rows']:,} 行）"
    )
    add(f"- 唯一图片：**{totals['unique_images']:,}** 张（图片按项目相对路径引用）")
    add(f"- 导出格式：{', '.join(args.formats)}；seed=`{args.seed}`；limit=`{args.limit}`")
    add("")

    format_cols = list(args.formats)
    header = "| split | 源样本(使用/全部) | 唯一图片 | QA 行 | caption 行 |"
    header += "".join(f" {fmt} 行 |" for fmt in format_cols)
    separator = "|---|---|---|---|---|" + "---|" * len(format_cols)
    add(header)
    add(separator)
    for split in SPLIT_ORDER:
        if split not in split_stats:
            continue
        stats = split_stats[split]
        cells = [
            f"`{split}`",
            f"{stats['source_rows_used']:,} / {stats['source_rows']:,}",
            f"{stats['unique_images']:,}",
            f"{stats['qa_rows']:,}",
            f"{stats['caption_rows']:,}",
        ]
        for fmt in format_cols:
            file_info = stats["files"].get(fmt)
            cells.append(f"{file_info['rows']:,}" if file_info else "—")
        add("| " + " | ".join(cells) + " |")
    add("")

    # 2. 课程阶段 ---------------------------------------------------------
    add("## 2. 课程阶段分布")
    add("")
    stage_counter: Counter = Counter()
    for stats in split_stats.values():
        stage_counter.update(stats["task_stage_dist"])
    total_stage = sum(stage_counter.values())
    if stage_counter:
        add("| task_stage | 阶段标题 | 源样本数 | 占比 |")
        add("|---|---|---|---|")
        for stage, count in sorted(
            stage_counter.items(), key=lambda kv: (-kv[1], str(kv[0]))
        ):
            title = stage_titles.get(stage, STAGE_FALLBACK_TITLES.get(stage, stage))
            add(f"| `{stage}` | {title} | {count:,} | {pct(count, total_stage)} |")
        add("")
        add(
            "> 课程学习建议：按 A → E 阶段由易到难安排训练轮次；阶段定义见项目 "
            "README 与 v2 生成脚本中的 `task_stage` 契约。"
        )
    else:
        add("（数据缺少 task_stage 列，无法统计课程阶段分布。）")
    add("")

    # 3. split 说明 -------------------------------------------------------
    add("## 3. Split 说明")
    add("")
    for split in SPLIT_ORDER:
        if split in split_stats:
            add(f"- **`{split}`**：{SPLIT_DESCRIPTIONS[split]}")
    missing_splits = [s for s in SPLIT_ORDER if s not in split_stats]
    if missing_splits:
        add("")
        add(
            f"- ⚠️ 本数据集未包含 split：`{', '.join(missing_splits)}`"
            f"（v1 降级或子集导出）。"
        )
    add("")
    add(
        "**来源隔离要求**：`test_unseen_entity` / `test_unseen_substrate` / "
        "`test_unseen_both` 的实体与底板集合按 v2 契约与 train 不相交，"
        "**只能用于最终评测，禁止混入训练或超参搜索**。"
    )
    add("")

    # 4. 格式示例 ---------------------------------------------------------
    add("## 4. 数据格式")
    add("")
    if examples.get("ms-swift") is not None:
        add("### ms-swift（JSONL，一行一个样本）")
        add("")
        add("```json")
        add(card_json_block(examples["ms-swift"]))
        add("```")
        add("")
        add(
            "- `messages`：user 内容以 `<image>\\n` 开头（图片占位），"
            "assistant 为答案；`images` 为项目相对图片路径（与占位一一对应）。"
        )
        add(
            "- `answer_type` / `bbox_norm_1000` 为附加结构化元数据；"
            "bbox 答案保留 `[ymin, xmin, ymax, xmax]`（0~1000 相对坐标）。"
        )
        add("")
    if examples.get("sharegpt") is not None:
        add("### ShareGPT（JSONL，一行一个样本）")
        add("")
        add("```json")
        add(card_json_block(examples["sharegpt"]))
        add("```")
        add("")
        add("- `conversations`：human / gpt 两轮；`image` 为项目相对图片路径。")
        add("")
    if examples.get("bbox") is not None:
        add("### bbox 答案行示例（结构化坐标）")
        add("")
        add("```json")
        add(card_json_block(examples["bbox"]))
        add("```")
        add("")
    if examples.get("caption") is not None:
        add("### dense caption 行示例")
        add("")
        add("```json")
        add(card_json_block(examples["caption"]))
        add("```")
        add("")

    # 5. 过滤与去重 -------------------------------------------------------
    add("## 5. 过滤、去重与导出参数")
    add("")
    add(f"- 过滤规则：无图 / 图片不存在 / 空答案 / 空问题 / 重复 content"
        f"（question + answer + image）自动剔除；`qa_pairs_json` 解析失败整条跳过。")
    add(
        f"- 去重：`{filtering['skipped_duplicate']:,}` 行重复 content 已剔除；"
        f"dense caption 覆盖源样本比例 `{args.caption_ratio}`"
        f"（`--caption-ratio`，可关闭：`--no-include-caption`）。"
    )
    add("")
    add("| 过滤 / 规范化统计项 | 数量 |")
    add("|---|---|")
    for label, key in (
        ("源样本总数", "source_rows"),
        ("无 image_path", "skipped_no_image_path"),
        ("图片文件不存在", "skipped_missing_image_file"),
        ("qa_pairs_json 解析失败", "skipped_bad_qa_json"),
        ("空问题 QA", "skipped_empty_question"),
        ("空答案 QA", "skipped_empty_answer"),
        ("无有效内容样本", "skipped_no_content"),
        ("重复 content 行", "skipped_duplicate"),
        ("非法 bbox", "invalid_bbox"),
        ("由答案文本推断的 bbox（v1 兼容）", "bbox_inferred"),
        ("因类型非 bbox 而丢弃的结构化坐标", "bbox_dropped_non_bbox_type"),
    ):
        add(f"| {label} | {filtering.get(key, 0):,} |")
    add("")
    add(
        f"- 导出参数：`--seed {args.seed}`、`--limit {args.limit}`、"
        f"`--formats {','.join(args.formats)}`、"
        f"`--caption-ratio {args.caption_ratio}`；"
        f"每个 split 内行顺序已按 seed 确定性打乱。"
    )
    add("")

    # 6. 使用建议 ---------------------------------------------------------
    add("## 6. 使用建议")
    add("")
    add(
        "1. **图片路径解析**：JSONL 中图片为项目根相对路径（如 "
        "`output/datawheel_v2/images/DW_V2_000001.jpg`）。训练时请在**项目根目录**"
        "启动，或将图片路径批量改写为绝对路径。"
    )
    add(
        "2. **课程学习**：建议按 `task_stage` A→E 由易到难采样或分阶段训练；"
        "bbox 答案行可作为定位能力强化数据（`bbox_norm_1000` 可计算 IoU 奖励）。"
    )
    add(
        "3. **ms-swift 示例**（路径与模型按实际环境调整）："
    )
    add("")
    add("```bash")
    add("swift sft \\")
    add("  --model <MODEL_PATH_OR_ID> \\")
    add(f"  --dataset {display_path(Path(args.out_dir).resolve())}/train_ms_swift.jsonl \\")
    if "val" in split_stats:
        add(f"  --val_dataset {display_path(Path(args.out_dir).resolve())}/val_ms_swift.jsonl \\")
    add("  --train_type lora --num_train_epochs 1 \\")
    add("  --per_device_train_batch_size 4 --learning_rate 1e-4")
    add("```")
    if "val" not in split_stats:
        add("")
        add("> 本次导出没有 `val` split（v1 降级/子集），示例中省略 `--val_dataset`。")
    add("")
    add(
        "4. **ShareGPT**：可直接用于 LLaMA-Factory / axolotl 等支持 ShareGPT + "
        "`image` 字段的框架；若框架对附加字段（`answer_type` / "
        "`bbox_norm_1000`）严格校验，可先剥离："
        "`jq -c 'del(.answer_type, .bbox_norm_1000)' train_sharegpt.jsonl`。"
    )
    add(
        "5. **评测**：`test_unseen_*` 仅用于最终报告；调参/早停使用 `val`；"
        "各 split 的 sha256 见 `export_manifest.json`，便于复现与版本追溯。"
    )
    add("")

    # 7. 许可证与来源隔离 -------------------------------------------------
    add("## 7. 许可证与来源隔离提醒")
    add("")
    add(
        "- 本数据集为 Datawheel **程序化本地合成数据**（100% 离线生成，无网络请求），"
        "合成图与 QA 标注版权归项目方；但底层图元资产受各自原始许可约束。"
    )
    add(
        "- 底层资产来源：ISO 3166-1 国旗 / 国徽 / 地方行政旗帜（公共领域或官方来源）、"
        "Simple Icons 商业品牌矢量（数据 CC0，商标权归各品牌方）、动漫角色 IP"
        "（版权归原权利方）、真实商品摄影底板（源自公开商品图）。"
        "**再分发或商用前必须逐项核查原始许可证并保留来源清单。**"
    )
    add(
        "- 动漫角色 IP 数据仅限内部研究与评测，不得对外分发或用于商业用途。"
    )
    add(
        "- 外部独立评测基准（LogoDet-3K / QMUL-OpenLogo / ZACI-20）与训练数据"
        "**物理隔离**，禁止混入本训练集；`test_unseen_*` split 同样禁止回流训练。"
    )
    add(
        "- 建议保留本卡片与 `export_manifest.json`（参数 + sha256）作为数据版本与"
        "合规追溯依据。"
    )
    add("")

    # 8. 校验记录 ---------------------------------------------------------
    add("## 8. 导出校验记录")
    add("")
    json_ok_text = (
        "✅ 全部通过"
        if validation["json_errors"] == 0
        else f"❌ {validation['json_errors']} 行失败"
    )
    struct_ok_text = (
        "✅ 通过"
        if validation["structural_errors"] == 0
        else f"❌ {validation['structural_errors']} 处失败"
    )
    image_ok_text = (
        "（✅ 通过）"
        if validation["image_missing_total"] == 0
        else "（❌ 请检查图片路径）"
    )
    add(f"- JSONL 逐行 `json.loads`：{json_ok_text}（共 {validation['rows_validated']:,} 行）")
    add(
        f"- 结构校验：{struct_ok_text}"
        f"（messages/images、conversations/image、bbox 合法性）"
    )
    add(
        f"- 图片存在性抽样：{validation['image_samples_checked']} 行抽检，"
        f"缺失 {validation['image_missing_total']} 张{image_ok_text}"
    )
    add("")
    add("| 文件 | 行数 | sha256 |")
    add("|---|---|---|")
    for split in SPLIT_ORDER:
        if split not in split_stats:
            continue
        for fmt in format_cols:
            file_info = split_stats[split]["files"].get(fmt)
            if not file_info:
                continue
            add(f"| `{file_info['path']}` | {file_info['rows']:,} | `{file_info['sha256']}` |")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    start_time = time.time()
    args = parse_args(argv)
    warnings: List[str] = []
    generated_at = now_iso()

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = (Path.cwd() / dataset_path).resolve()
    dataset_path = dataset_path.resolve()
    if not dataset_path.is_file():
        print(f"[FATAL] 数据集不存在：{dataset_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else None

    # 1. 加载与 schema 检查 ------------------------------------------------
    dataset_columns, parquet_rows = load_dataset(dataset_path)
    column_set = set(dataset_columns)
    missing_required = [col for col in REQUIRED_COLUMNS if col not in column_set]
    if missing_required:
        print(
            f"[FATAL] 数据集缺少必需列 {missing_required}；"
            f"实际列：{dataset_columns}",
            file=sys.stderr,
        )
        return 2

    if "caption_zh" not in column_set and args.include_caption:
        args.include_caption = False
        add_warning(warnings, "数据集缺少 caption_zh 列，已自动关闭 dense caption 行。")

    schema_variant = detect_schema_variant(dataset_columns)
    if schema_variant == "v1_preview_degraded":
        add_warning(
            warnings,
            "v1 旧 schema：无 split / dataset_version 列，仅有 image_path 的预览子集"
            "参与导出，全部按 train 处理。仅用于降级验证，正式训练请使用 v2 数据集。",
        )
    elif schema_variant == "v2_partial":
        add_warning(warnings, "数据集缺少 dataset_version 列（v2 部分字段），继续按 split 列导出。")

    dataset_version = next(
        (
            normalize_text(record.get("dataset_version"))
            for record in parquet_rows
            if normalize_text(record.get("dataset_version"))
        ),
        None,
    )

    # 2. 源样本过滤与 split 分组 ------------------------------------------
    counters: Counter = Counter()
    split_raw_counts: Counter = Counter()
    unknown_splits: Counter = Counter()
    stage_titles: Dict[str, str] = {}
    grouped = prepare_source_records(
        parquet_rows=parquet_rows,
        split_raw_counts=split_raw_counts,
        counters=counters,
        warnings=warnings,
        dataset_path=dataset_path,
        image_root=image_root,
        include_caption=args.include_caption,
        unknown_splits=unknown_splits,
        stage_titles=stage_titles,
        schema_variant=schema_variant,
    )

    if unknown_splits:
        add_warning(
            warnings,
            f"发现未知 split 值 {dict(unknown_splits)}，已降级按 train 处理。",
        )
    blank_split_count = sum(
        1
        for record in parquet_rows
        if "split" in column_set and not str(record.get("split") or "").strip()
    )
    if blank_split_count:
        add_warning(
            warnings,
            f"{blank_split_count:,} 条源样本 split 列为空，已按 train 处理。",
        )

    splits_present = [
        split for split in SPLIT_ORDER if split_raw_counts.get(split, 0) > 0
    ]
    if not splits_present:
        print("[FATAL] 没有可导出的源样本（全部被过滤或缺少 image_path）。", file=sys.stderr)
        return 2

    missing_splits = [split for split in SPLIT_ORDER if split not in splits_present]
    if missing_splits:
        add_warning(
            warnings,
            f"导出数据仅包含 split {splits_present}；缺失 "
            f"{missing_splits}（v1 降级或数据子集，未生成对应文件）。",
        )

    # 3. 展开训练行并输出 -------------------------------------------------
    split_stats: Dict[str, Dict[str, Any]] = {}
    examples: Dict[str, Any] = {}
    global_answer_types: Counter = Counter()
    global_images: set = set()
    total_qa_rows = 0
    total_caption_rows = 0
    format_rows_total: Counter = Counter()

    for split in splits_present:
        source_records = grouped.get(split, [])
        specs, used_records = expand_split_rows(
            source_records=source_records,
            split=split,
            seed=args.seed,
            limit=args.limit,
            include_caption=args.include_caption,
            caption_ratio=args.caption_ratio,
            counters=counters,
        )
        qa_rows = sum(1 for spec in specs if spec["kind"] == "qa")
        caption_rows = sum(1 for spec in specs if spec["kind"] == "caption")
        answer_type_dist: Counter = Counter(spec["answer_type"] for spec in specs)
        task_stage_dist: Counter = Counter()
        difficulty_dist: Counter = Counter()
        for record in used_records:
            task_stage_dist[record.get("task_stage") or "unknown"] += 1
            if record.get("difficulty"):
                difficulty_dist[record["difficulty"]] += 1

        files: Dict[str, Dict[str, Any]] = {}
        for fmt in args.formats:
            file_path = out_dir / f"{split}_{FORMAT_FILE_TOKENS[fmt]}.jsonl"
            rows, bytes_size, file_sha = write_jsonl(file_path, specs, fmt)
            files[fmt] = {
                "path": display_path(file_path),
                "rows": rows,
                "bytes": bytes_size,
                "sha256": file_sha,
                "answer_type_dist": dict(answer_type_dist),
            }
            format_rows_total[fmt] += rows
            # 卡片示例：取该 split 首行 + 首个 bbox 行 + 首个 caption 行
            for spec in specs:
                if fmt not in examples:
                    examples[fmt] = make_training_row(fmt, spec)
                if "bbox" not in examples and spec.get("bbox") is not None:
                    examples["bbox"] = make_training_row(fmt, spec)
                if "caption" not in examples and spec["kind"] == "caption":
                    examples["caption"] = make_training_row(fmt, spec)
            print(
                f"  [write] {display_path(file_path)}  rows={rows:,}  "
                f"bytes={bytes_size:,}"
            )

        unique_images = len({record["image"] for record in used_records})
        split_stats[split] = {
            "source_rows": split_raw_counts.get(split, 0),
            "source_rows_used": len(used_records),
            "unique_images": unique_images,
            "qa_rows": qa_rows,
            "caption_rows": caption_rows,
            "answer_type_dist": dict(answer_type_dist),
            "task_stage_dist": dict(task_stage_dist),
            "difficulty_dist": dict(difficulty_dist),
            "files": files,
        }
        global_answer_types.update(answer_type_dist)
        global_images.update(record["image"] for record in used_records)
        total_qa_rows += qa_rows
        total_caption_rows += caption_rows

        if not specs:
            add_warning(warnings, f"split `{split}` 过滤后为 0 行（文件为空）。")

    # 4. 导出 manifest / card ---------------------------------------------
    totals: Dict[str, Any] = {
        "sft_rows_per_format": {fmt: format_rows_total[fmt] for fmt in args.formats},
        "qa_rows": total_qa_rows,
        "caption_rows": total_caption_rows,
        "unique_images": len(global_images),
        "answer_type_dist": dict(global_answer_types),
        "splits_written": splits_present,
        "dataset_version": dataset_version,
    }
    filtering: Dict[str, Any] = {
        "source_rows": counters.get("source_rows", 0),
        "skipped_no_image_path": counters.get("skipped_no_image_path", 0),
        "skipped_missing_image_file": counters.get("skipped_missing_image_file", 0),
        "skipped_bad_qa_json": counters.get("skipped_bad_qa_json", 0),
        "skipped_empty_question": counters.get("skipped_empty_question", 0),
        "skipped_empty_answer": counters.get("skipped_empty_answer", 0),
        "skipped_bad_qa_item": counters.get("skipped_bad_qa_item", 0),
        "skipped_no_content": counters.get("skipped_no_content", 0),
        "skipped_duplicate": counters.get("skipped_duplicate", 0),
        "invalid_bbox": counters.get("invalid_bbox", 0),
        "bbox_inferred": counters.get("bbox_inferred", 0),
        "bbox_dropped_non_bbox_type": counters.get("bbox_dropped_non_bbox_type", 0),
        "samples_split_degraded": counters.get("samples_split_degraded", 0),
    }

    # 5. 校验输出 ---------------------------------------------------------
    print("\n[validate] 逐行 json.loads + 结构校验 + 抽样图片存在性检查 ...")
    validation = validate_outputs(
        out_dir=out_dir,
        splits_present=splits_present,
        formats=args.formats,
        dataset_path=dataset_path,
        image_root=image_root,
    )
    for file_path, result in validation["files"].items():
        status = "OK" if not (result.get("json_errors") or result.get("structural_errors")
                              or result.get("image_missing")) else "FAIL"
        print(
            f"  [{status}] {file_path}  rows={result.get('rows', 0):,}  "
            f"json_err={len(result.get('json_errors') or [])}  "
            f"struct_err={len(result.get('structural_errors') or [])}  "
            f"image_sample={result.get('image_samples_checked', 0)}  "
            f"image_missing={len(result.get('image_missing') or [])}"
        )

    if not validation["passed"]:
        print(
            "\n[ERROR] 输出校验失败："
            f"json_errors={validation['json_errors']}, "
            f"structural_errors={validation['structural_errors']}, "
            f"image_missing={validation['image_missing_total']}",
            file=sys.stderr,
        )
        for result in validation["files"].values():
            for error in (result.get("json_errors") or [])[:5]:
                print(f"  - {result['path']}: {error}", file=sys.stderr)
            for error in (result.get("structural_errors") or [])[:5]:
                print(f"  - {result['path']}: {error}", file=sys.stderr)
            for missing in (result.get("image_missing") or [])[:5]:
                print(f"  - {result['path']}: 图片缺失 {missing}", file=sys.stderr)

    if total_qa_rows + total_caption_rows == 0:
        print("[FATAL] 导出训练行为 0，请检查过滤条件。", file=sys.stderr)
        return 2

    # 6. dataset_card.md ---------------------------------------------------
    card_text = build_dataset_card(
        args=args,
        dataset_path=dataset_path,
        dataset_rows=len(parquet_rows),
        schema_variant=schema_variant,
        split_stats=split_stats,
        totals=totals,
        filtering=filtering,
        validation=validation,
        examples=examples,
        stage_titles=stage_titles,
        warnings=warnings,
        generated_at=generated_at,
    )
    card_path = out_dir / "dataset_card.md"
    card_path.write_text(card_text, encoding="utf-8")

    # 7. export_manifest.json ---------------------------------------------
    manifest = build_manifest(
        args=args,
        dataset_path=dataset_path,
        dataset_columns=dataset_columns,
        schema_variant=schema_variant,
        dataset_rows=len(parquet_rows),
        split_stats=split_stats,
        totals=totals,
        filtering=filtering,
        validation=validation,
        card_path=card_path,
        warnings=warnings,
        generated_at=generated_at,
        elapsed_seconds=time.time() - start_time,
    )
    manifest_path = out_dir / "export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 8. 打印统计 ---------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 88)
    print(f"{'✅ Datawheel SFT 导出完成' if validation['passed'] else '❌ 导出完成但校验未通过'}")
    print("=" * 88)
    print(f"数据源 : {display_path(dataset_path)}")
    print(f"         schema={schema_variant}, 源样本={len(parquet_rows):,} 行, "
          f"dataset_version={dataset_version!r}")
    print(f"输出目录: {display_path(out_dir)}")
    print(f"参数   : seed={args.seed}, limit={args.limit}, "
          f"include_caption={args.include_caption}, caption_ratio={args.caption_ratio}, "
          f"formats={args.formats}")
    print("\n按 split / 格式行数：")
    print(f"  {'split':<22}{'源样本(用/全)':>18}{'图片':>8}{'QA':>8}{'caption':>9}"
          + "".join(f"{fmt:>12}" for fmt in args.formats))
    for split in splits_present:
        stats = split_stats[split]
        print(
            f"  {split:<22}"
            f"{stats['source_rows_used']:>9,}/{stats['source_rows']:<8,}"
            f"{stats['unique_images']:>8,}{stats['qa_rows']:>8,}"
            f"{stats['caption_rows']:>9,}"
            + "".join(f"{stats['files'][fmt]['rows']:>12,}" for fmt in args.formats)
        )
    print(f"\nanswer_type 分布: {pretty_counter(global_answer_types)}")
    print(
        "过滤统计: "
        f"无图路径 {filtering['skipped_no_image_path']:,}, "
        f"图片不存在 {filtering['skipped_missing_image_file']:,}, "
        f"空问题 {filtering['skipped_empty_question']:,}, "
        f"空答案 {filtering['skipped_empty_answer']:,}, "
        f"坏 QA JSON {filtering['skipped_bad_qa_json']:,}, "
        f"无内容样本 {filtering['skipped_no_content']:,}, "
        f"重复行 {filtering['skipped_duplicate']:,}"
    )
    print(
        f"校验: json_errors={validation['json_errors']}, "
        f"structural_errors={validation['structural_errors']}, "
        f"抽样图片 {validation['image_samples_checked']} 行, "
        f"缺失 {validation['image_missing_total']}"
    )
    print(f"产物: {display_path(manifest_path)}  +  {display_path(card_path)}")
    if warnings:
        print("\n告警：")
        for warning in warnings:
            print(f"  [WARN] {warning}")
    print(f"\n耗时: {elapsed:.2f}s")
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
