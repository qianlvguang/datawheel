#!/usr/bin/env python3
"""V6 中训练 parquet 校验器（Datawheel → Qwen3.5-27B ms-swift 4.3.2 / Megatron）。

对 V6 通道 parquet（以及合并后的 dataset.parquet）做行级 + 数据集级校验，输出
validation_report.md + validation_metrics.json；存在 critical 时 exit code=1。

校验项：
  1) schema：必需列 / 列类型（messages=list<struct<role,content>>、images=list<string>、
     enable_thinking=bool、sample_id/cate_id=string、variant=int）；
  2) 行级：messages 非空且 user/assistant 严格交替（首 user 尾 assistant）；
     <image> 数量 == len(images) 且都在第一条 user 消息中；图片相对路径存在、
     位于 image-root 内且可被 PIL 解码；enable_thinking is False；variant==6；
     sample_id 唯一；cate_id 非空；行级键 (sample_id, cate_id, images) 唯一；
  3) 长度：估算长度 = 文本 token（中文 chars/1.6 + 其余 chars/4）+ images×image_token_estimate，
     超过 --max-len 记为 critical；
  4) 统计：cate 分布、行数、图片数/尺寸、长度分位数、最长行。

用法（在 Coding/datawheel/ 下）：
  .venv/bin/python tools/validate_swift_v6.py \
      --dataset output/datawheel_v6_pilot/parts/single_image.parquet \
      --image-root . --image-token-estimate 512 --max-len 8192 \
      --out output/datawheel_v6_pilot/parts/validation/

  # 同时校验多个 part（或合并后的 dataset.parquet）
  .venv/bin/python tools/validate_swift_v6.py --dataset \
      output/datawheel_v6_pilot/parts/single_image.parquet \
      output/datawheel_v6_pilot/parts/reference_multi.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "output" / "datawheel_v6_pilot" / "parts" / "validation"

REQUIRED_COLUMNS = (
    "messages", "images", "enable_thinking", "sample_id", "cate_id", "variant",
)
VARIANT_EXPECTED = 6
EXPECTED_ROLE = {0: "user", 1: "assistant"}
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ffff]")
TOP_LONGEST = 5


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def estimate_text_tokens(text: str) -> float:
    """中文 chars/1.6 + 其余 chars/4（与 tools/build_v6_single_image.py 一致）。"""
    if not text:
        return 0.0
    cjk = len(CJK_RE.findall(text))
    return cjk / 1.6 + (len(text) - cjk) / 4.0


def stats_summary(values: Sequence[float]) -> Dict[str, float]:
    values = [float(v) for v in values]
    if not values:
        return {"count": 0, "min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "min": round(min(values), 2),
        "median": round(statistics.median(values), 2),
        "mean": round(statistics.fmean(values), 2),
        "max": round(max(values), 2),
    }


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


class IssueCollector:
    """按 code 聚合 critical / warning 问题（最多保留 8 条示例）。"""

    def __init__(self) -> None:
        self._items: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def add(self, severity: str, code: str, check: str,
            message: str, example: Optional[str] = None) -> None:
        key = (severity, code)
        item = self._items.get(key)
        if item is None:
            item = {"code": code, "severity": severity, "check": check,
                    "message": message, "count": 0, "examples": []}
            self._items[key] = item
        item["count"] += 1
        if example is not None and len(item["examples"]) < 8:
            item["examples"].append(example)

    def critical(self, code: str, check: str, message: str,
                 example: Optional[str] = None) -> None:
        self.add("critical", code, check, message, example)

    def warning(self, code: str, check: str, message: str,
                example: Optional[str] = None) -> None:
        self.add("warning", code, check, message, example)

    def as_list(self) -> List[Dict[str, Any]]:
        order = {"critical": 0, "warning": 1}
        return sorted(self._items.values(),
                      key=lambda item: (order.get(item["severity"], 9), item["code"]))

    def count(self, severity: str) -> int:
        return sum(item["count"] for item in self._items.values()
                   if item["severity"] == severity)


# ---------------------------------------------------------------------------
# schema 校验
# ---------------------------------------------------------------------------
def check_schema(table: pa.Table) -> List[Dict[str, Any]]:
    """返回 schema 问题列表（critical）。"""
    rows: List[Dict[str, Any]] = []
    names = set(table.column_names)

    for column in REQUIRED_COLUMNS:
        if column not in names:
            rows.append({"code": f"schema_missing_{column}", "severity": "critical",
                         "check": "schema", "message": f"缺少必需列 {column}"})
    if "messages" in names:
        messages_type = table.schema.field("messages").type
        ok = pa.types.is_list(messages_type) and pa.types.is_struct(messages_type.value_type)
        if ok:
            struct_type = messages_type.value_type
            fields = {struct_type.field(i).name: struct_type.field(i).type for i in range(struct_type.num_fields)}
            ok = (set(fields) >= {"role", "content"}
                  and all(pa.types.is_string(fields[f]) for f in ("role", "content")))
        if not ok:
            rows.append({"code": "schema_messages_type", "severity": "critical", "check": "schema",
                         "message": "messages 类型应为 list<struct<role:string,content:string>>，"
                                    f"实际为 {messages_type}"})
    if "images" in names:
        images_type = table.schema.field("images").type
        if not (pa.types.is_list(images_type) and pa.types.is_string(images_type.value_type)):
            rows.append({"code": "schema_images_type", "severity": "critical", "check": "schema",
                         "message": f"images 类型应为 list<string>，实际为 {images_type}"})
    if "enable_thinking" in names and not pa.types.is_boolean(table.schema.field("enable_thinking").type):
        rows.append({"code": "schema_enable_thinking_type", "severity": "critical", "check": "schema",
                     "message": f"enable_thinking 应为 bool，实际为 {table.schema.field('enable_thinking').type}"})
    for column in ("sample_id", "cate_id"):
        if column in names and not pa.types.is_string(table.schema.field(column).type):
            rows.append({"code": f"schema_{column}_type", "severity": "critical", "check": "schema",
                         "message": f"{column} 应为 string，实际为 {table.schema.field(column).type}"})
    if "variant" in names and not pa.types.is_integer(table.schema.field("variant").type):
        rows.append({"code": "schema_variant_type", "severity": "critical", "check": "schema",
                     "message": f"variant 应为整数，实际为 {table.schema.field('variant').type}"})
    return rows


# ---------------------------------------------------------------------------
# 单数据集校验
# ---------------------------------------------------------------------------
def validate_dataset(
    dataset_path: Path,
    image_root: Path,
    image_token_estimate: float,
    max_len: int,
    image_cache: Dict[Path, Tuple[bool, Optional[Tuple[int, int]]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """校验单个 parquet，返回 (metrics, issues, cross_state)。"""
    collector = IssueCollector()
    metrics: Dict[str, Any] = {
        "path": rel_path(dataset_path),
        "rows": 0,
        "bytes": dataset_path.stat().st_size if dataset_path.is_file() else None,
        "schema": {"columns": [], "ok": True},
        "cate_dist": {},
        "variant_dist": {},
        "enable_thinking_dist": {},
        "image_count_dist": {},
        "images": {},
        "length": {},
        "examples": {},
    }
    if not dataset_path.is_file():
        metrics["schema"]["ok"] = False
        collector.critical("dataset_missing", "io", f"文件不存在：{dataset_path}")
        return metrics, collector.as_list(), {}

    try:
        table = pq.read_table(dataset_path)
    except Exception as exc:  # noqa: BLE001 - 报告解析失败而不是崩溃
        metrics["schema"]["ok"] = False
        collector.critical("dataset_unreadable", "io", f"parquet 读取失败：{type(exc).__name__}: {exc}")
        return metrics, collector.as_list(), {}

    rows = table.to_pylist()
    metrics["rows"] = len(rows)
    metrics["schema"]["columns"] = [{"name": field.name, "type": str(field.type)}
                                    for field in table.schema]
    schema_issues = check_schema(table)
    for issue in schema_issues:
        metrics["schema"]["ok"] = False
        collector.critical(issue["code"], issue["check"], issue["message"])
    extra_columns = [name for name in table.column_names if name not in REQUIRED_COLUMNS]
    if extra_columns:
        collector.warning("schema_extra_columns", "schema",
                          f"存在额外列（ms-swift 会忽略）：{extra_columns}")
    if not rows:
        collector.critical("dataset_empty", "row_count", "数据集 0 行")
        return metrics, collector.as_list(), {}

    names = set(table.column_names)
    has_messages = "messages" in names
    has_images = "images" in names

    cate_counter: Counter = Counter()
    variant_counter: Counter = Counter()
    thinking_counter: Counter = Counter()
    image_count_counter: Counter = Counter()
    sample_id_counter: Counter = Counter()
    row_key_counter: Counter = Counter()
    image_path_counter: Counter = Counter()

    text_chars: List[int] = []
    text_tokens: List[float] = []
    total_tokens: List[float] = []
    longest: List[Tuple[float, str]] = []
    over_limit: List[str] = []
    unique_image_paths: set = set()
    image_sizes: set = set()

    def sid_of(index: int) -> str:
        if "sample_id" not in names:
            return f"row#{index}"
        value = rows[index].get("sample_id")
        return str(value) if value else f"row#{index}"

    for index, row in enumerate(rows):
        sid = sid_of(index)

        # ---- 唯一性 / 基础字段 ----
        if "sample_id" in names:
            sample_id = row.get("sample_id")
            if sample_id is None or not str(sample_id).strip():
                collector.critical("sample_id_empty", "uniqueness", "sample_id 为空", sid)
            else:
                sample_id_counter[str(sample_id)] += 1
        cate_id = row.get("cate_id")
        if "cate_id" in names:
            if cate_id is None or not str(cate_id).strip():
                collector.critical("cate_id_empty", "cate_id", "cate_id 为空", sid)
            else:
                cate_counter[str(cate_id)] += 1
        if "variant" in names:
            variant_counter[str(row.get("variant"))] += 1
            if row.get("variant") != VARIANT_EXPECTED:
                collector.critical("variant_not_6", "variant",
                                   f"variant 应为 {VARIANT_EXPECTED}", f"{sid} variant={row.get('variant')}")
        if "enable_thinking" in names:
            thinking_counter[str(row.get("enable_thinking"))] += 1
            if row.get("enable_thinking") is not False:
                collector.critical("enable_thinking_not_false", "enable_thinking",
                                   "enable_thinking 必须为 False",
                                   f"{sid} enable_thinking={row.get('enable_thinking')!r}")

        # ---- messages ----
        messages = row.get("messages") or []
        images = row.get("images") or []
        if has_messages:
            if not isinstance(messages, list) or not messages:
                collector.critical("messages_empty", "messages", "messages 为空", sid)
                messages = []
            else:
                if len(messages) % 2 != 0:
                    collector.critical("messages_odd", "messages", "messages 条数应为偶数（user/assistant 成对）", sid)
                for position, message in enumerate(messages):
                    role = message.get("role") if isinstance(message, dict) else None
                    content = message.get("content") if isinstance(message, dict) else None
                    expected = EXPECTED_ROLE[position % 2]
                    if role != expected:
                        collector.critical("messages_role_alternation", "messages",
                                           "messages 角色应严格 user/assistant 交替",
                                           f"{sid} 第 {position} 条 role={role!r}（期望 {expected}）")
                    if content is None or not str(content).strip():
                        collector.critical("messages_empty_content", "messages",
                                           "messages 存在空 content", f"{sid} 第 {position} 条")
        if has_images:
            if not isinstance(images, list) or not images:
                collector.critical("images_empty", "images", "images 为空（V6 行应至少 1 张图）", sid)
                images = []
            else:
                image_count_counter[len(images)] += 1

        # ---- 图片占位符与图片清单 ----
        if has_messages and has_images and messages:
            contents = [str(message.get("content") or "") for message in messages]
            placeholder_total = sum(content.count("<image>") for content in contents)
            placeholder_first = contents[0].count("<image>")
            if placeholder_total != len(images):
                collector.critical("image_placeholder_mismatch", "images",
                                   "<image> 总数应与 images 长度一致",
                                   f"{sid} placeholders={placeholder_total} images={len(images)}")
            if placeholder_first != len(images):
                collector.critical("image_placeholder_not_first_user", "images",
                                   "<image> 应全部位于第一条 user 消息中",
                                   f"{sid} first_user={placeholder_first} images={len(images)}")

        # ---- 图片文件 ----
        if has_images:
            for image_value in images:
                image_rel = "" if image_value is None else str(image_value)
                image_path_counter[image_rel] += 1
                if not image_rel:
                    collector.critical("image_path_empty", "images", "images 中存在空路径", sid)
                    continue
                image_path = Path(image_rel)
                if image_path.is_absolute():
                    collector.critical("image_path_absolute", "images", "images 必须是仓库相对路径",
                                       f"{sid} {image_rel}")
                    continue
                resolved = (image_root / image_path).resolve()
                try:
                    resolved.relative_to(image_root)
                except ValueError:
                    collector.critical("image_path_escape", "images", "图片路径逃逸出 image-root",
                                       f"{sid} {image_rel}")
                    continue
                if not resolved.is_file():
                    collector.critical("image_missing", "images", "图片文件不存在",
                                       f"{sid} {image_rel}")
                    continue
                unique_image_paths.add(str(resolved))
                cached = image_cache.get(resolved)
                if cached is None:
                    try:
                        with Image.open(resolved) as image:
                            image.load()
                            size = image.size
                        cached = (True, size)
                        image_sizes.add(size)
                    except Exception as exc:  # noqa: BLE001 - 解码失败要报告而不是崩溃
                        cached = (False, None)
                        collector.critical("image_undecodable", "images", "图片无法解码",
                                           f"{sid} {image_rel}: {type(exc).__name__}")
                    image_cache[resolved] = cached
                elif not cached[0]:
                    collector.critical("image_undecodable", "images", "图片无法解码",
                                       f"{sid} {image_rel}")

        # ---- 行级键唯一性 ----
        if "sample_id" in names and "cate_id" in names and has_images:
            key = (str(row.get("sample_id")), str(row.get("cate_id")),
                   tuple(str(value) for value in images))
            row_key_counter[key] += 1

        # ---- 长度估算 ----
        if has_messages:
            text = "\n".join(str(message.get("content") or "") for message in messages)
            chars = sum(len(str(message.get("content") or "")) for message in messages)
            token_est = estimate_text_tokens(text)
            length_est = token_est + len(images) * image_token_estimate
            text_chars.append(chars)
            text_tokens.append(token_est)
            total_tokens.append(length_est)
            longest.append((length_est, sid))
            if length_est > max_len:
                over_limit.append(sid)
                collector.critical("length_over_max", "length",
                                   f"估算长度超过 max_len={max_len}",
                                   f"{sid} est={length_est:.1f} (> {max_len})")

    # ---- 聚合重复项 ----
    duplicates = {key: count for key, count in sample_id_counter.items() if count > 1}
    if duplicates:
        example = ", ".join(f"{key}×{count}" for key, count in list(duplicates.items())[:5])
        collector.critical("sample_id_duplicate", "uniqueness",
                           "sample_id 必须全局唯一", example)
    row_key_duplicates = {key: count for key, count in row_key_counter.items() if count > 1}
    if row_key_duplicates:
        collector.critical("row_key_duplicate", "uniqueness",
                           "行级键 (sample_id, cate_id, images) 必须唯一",
                           str(list(row_key_duplicates)[:2]))
    reused_images = {path: count for path, count in image_path_counter.items() if count > 1}
    if reused_images:
        collector.warning("image_reuse", "images",
                          "同一图片被多行引用（多图参考行正常；单图行应为 0）",
                          ", ".join(f"{path}×{count}" for path, count in list(reused_images.items())[:3]))

    # ---- 长度/图片统计 ----
    longest.sort(key=lambda pair: pair[0], reverse=True)
    metrics["length"] = {
        "max_len": max_len,
        "image_token_estimate": image_token_estimate,
        "text_chars": stats_summary(text_chars),
        "text_tokens_est": stats_summary(text_tokens),
        "total_tokens_est": stats_summary(total_tokens),
        "total_tokens_est_sum": round(sum(total_tokens), 1),
        "over_max_len": len(over_limit),
        "over_max_len_examples": over_limit[:5],
        "top_longest": [{"sample_id": sid, "est_tokens": round(value, 1)}
                        for value, sid in longest[:TOP_LONGEST]],
    }
    metrics["images"] = {
        "references": sum(count * n for n, count in image_count_counter.items()),
        "unique_files": len(unique_image_paths),
        "reused_files": len(reused_images),
        "sizes": sorted(f"{width}x{height}" for width, height in image_sizes),
    }
    metrics["cate_dist"] = dict(sorted(cate_counter.items()))
    metrics["variant_dist"] = dict(sorted(variant_counter.items()))
    metrics["enable_thinking_dist"] = dict(sorted(thinking_counter.items()))
    metrics["image_count_dist"] = {str(n): count for n, count in sorted(image_count_counter.items())}
    metrics["checks"] = collector.as_list()
    metrics["critical"] = collector.count("critical")
    metrics["warning"] = collector.count("warning")
    metrics["ok"] = metrics["critical"] == 0
    cross_state = {"sample_ids": set(sample_id_counter), "rows": len(rows)}
    return metrics, collector.as_list(), cross_state


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------
def render_report(metrics_list: Sequence[Dict[str, Any]],
                  issues_by_dataset: Dict[str, List[Dict[str, Any]]],
                  summary: Dict[str, Any],
                  config: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# V6 ms-swift 中训练数据校验报告",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 数据集数量：{summary['datasets']}；总行数：{summary['rows_total']}",
        f"- image-root：`{config['image_root']}`；image-token-estimate：{config['image_token_estimate']}；"
        f"max-len：{config['max_len']}",
        f"- 结论：**{'PASS ✅' if summary['exit_code'] == 0 else 'FAIL ❌'}** "
        f"（critical={summary['critical_total']}，warning={summary['warning_total']}）",
        "",
        "## 汇总",
        "",
        "| dataset | rows | critical | warning | over-max-len | 最长估算 token |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metrics in metrics_list:
        lines.append(
            f"| `{metrics['path']}` | {metrics['rows']} | {metrics.get('critical', 0)} | "
            f"{metrics.get('warning', 0)} | {metrics.get('length', {}).get('over_max_len', 0)} | "
            f"{metrics.get('length', {}).get('total_tokens_est', {}).get('max', 0)} |"
        )

    for metrics in metrics_list:
        lines.extend([
            "",
            f"## {metrics['path']}",
            "",
            f"- 行数：{metrics['rows']}；文件大小："
            f"{(metrics['bytes'] or 0) / 1024:.1f} KB；verdict："
            f"{'PASS ✅' if metrics.get('ok') else 'FAIL ❌'}",
            f"- schema 检查：{'通过 ✅' if metrics['schema']['ok'] else '失败 ❌'}；"
            f"columns = {[column['name'] for column in metrics['schema']['columns']]}",
            "",
            "### cate 分布",
            "",
            "| cate_id | rows |",
            "|---|---:|",
        ])
        for cate, count in metrics.get("cate_dist", {}).items():
            lines.append(f"| `{cate}` | {count} |")
        lines.extend([
            "",
            "### 长度估算（token）",
            "",
            "| 指标 | min | median | mean | max |",
            "|---|---:|---:|---:|---:|",
        ])
        length = metrics.get("length", {})
        for label, key in (("文本 chars", "text_chars"),
                           ("文本 token（chars 估算）", "text_tokens_est"),
                           ("总 token（含 images×estimate）", "total_tokens_est")):
            value = length.get(key, {})
            lines.append(f"| {label} | {value.get('min', 0)} | {value.get('median', 0)} | "
                         f"{value.get('mean', 0)} | {value.get('max', 0)} |")
        lines.extend([
            "",
            f"- 总 token（含图片估算）合计：{length.get('total_tokens_est_sum', 0)}；"
            f"超限行数（> {config['max_len']}）：{length.get('over_max_len', 0)}",
        ])
        top = length.get("top_longest", [])
        if top:
            lines.append("- 最长行：" + "；".join(
                f"`{item['sample_id']}`={item['est_tokens']}" for item in top))
        image_metrics = metrics.get("images", {})
        lines.extend([
            "",
            "### 图片",
            "",
            f"- 引用总数：{image_metrics.get('references', 0)}；唯一文件："
            f"{image_metrics.get('unique_files', 0)}；跨行复用文件："
            f"{image_metrics.get('reused_files', 0)}",
            f"- 每行图片数分布：{metrics.get('image_count_dist', {})}",
            f"- 尺寸集合：{image_metrics.get('sizes', [])}",
            "",
            "### 分布信息",
            "",
            f"- variant 分布：{metrics.get('variant_dist', {})}",
            f"- enable_thinking 分布：{metrics.get('enable_thinking_dist', {})}",
            "",
            "### 检查项",
            "",
            "| severity | code | check | count | 说明 |",
            "|---|---|---|---:|---|",
        ])
        checks = issues_by_dataset.get(metrics["path"], [])
        if not checks:
            lines.append("| — | — | — | 0 | 全部通过 |")
        for issue in checks:
            lines.append(
                f"| {issue['severity']} | `{issue['code']}` | {issue['check']} | "
                f"{issue['count']} | {issue['message']} |"
            )
        critical_issues = [issue for issue in checks if issue["severity"] == "critical"]
        if critical_issues:
            lines.extend(["", "#### critical 示例", ""])
            for issue in critical_issues:
                if issue["examples"]:
                    lines.append(f"- `{issue['code']}`：" + "；".join(str(e) for e in issue["examples"][:5]))
        lines.append("")

    lines.extend([
        "## 复现命令",
        "",
        "```bash",
        ".venv/bin/python tools/validate_swift_v6.py \\",
        "    --dataset " + " ".join(m["path"] for m in metrics_list) + " \\",
        f"    --image-root {config['image_root']} --image-token-estimate {config['image_token_estimate']} \\",
        f"    --max-len {config['max_len']} --out {config['out']}",
        "```",
        "",
        "> 文本 token 估算公式：中文 chars/1.6 + 其余 chars/4；总长度再加 images×image_token_estimate。",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate V6 ms-swift parquet datasets")
    parser.add_argument("--dataset", type=Path, nargs="+", required=True,
                        help="一个或多个 V6 parquet（parts 或合并后的 dataset.parquet）")
    parser.add_argument("--image-root", default=".", help="图片相对路径的解析根（默认仓库根 .）")
    parser.add_argument("--image-token-estimate", type=float, default=512.0,
                        help="每张图片折算的 token 数（默认 512）")
    parser.add_argument("--max-len", type=int, default=8192, help="ms-swift MAX_LENGTH（默认 8192）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="报告输出目录")
    args = parser.parse_args(argv)

    started = time.time()
    image_root = Path(args.image_root)
    image_root = (image_root if image_root.is_absolute() else (PROJECT_ROOT / image_root)).resolve()
    out_dir = args.out if args.out.is_absolute() else (PROJECT_ROOT / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_cache: Dict[Path, Tuple[bool, Optional[Tuple[int, int]]]] = {}
    metrics_list: List[Dict[str, Any]] = []
    issues_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    cross_states: List[Dict[str, Any]] = []

    for dataset in args.dataset:
        dataset_path = dataset if dataset.is_absolute() else (PROJECT_ROOT / dataset)
        metrics, issues, cross_state = validate_dataset(
            dataset_path=dataset_path,
            image_root=image_root,
            image_token_estimate=args.image_token_estimate,
            max_len=args.max_len,
            image_cache=image_cache,
        )
        metrics_list.append(metrics)
        issues_by_dataset[metrics["path"]] = issues
        cross_states.append(cross_state)

    # 多数据集同时校验时，检查跨文件 sample_id 交集（parts + merged 同传时会命中）
    if len(cross_states) > 1:
        for i in range(len(cross_states)):
            for j in range(i + 1, len(cross_states)):
                overlap = cross_states[i]["sample_ids"] & cross_states[j]["sample_ids"]
                if overlap:
                    path_i, path_j = metrics_list[i]["path"], metrics_list[j]["path"]
                    issue = {
                        "code": "cross_dataset_sample_id_overlap",
                        "severity": "warning",
                        "check": "uniqueness",
                        "message": f"`{path_i}` 与 `{path_j}` 的 sample_id 有交集"
                                   "（同时传入 part 与合并文件时属预期）",
                        "count": len(overlap),
                        "examples": sorted(overlap)[:5],
                    }
                    issues_by_dataset[path_i].append(issue)
                    issues_by_dataset[path_j].append(issue)
                    metrics_list[i]["warning"] = metrics_list[i].get("warning", 0) + len(overlap)
                    metrics_list[j]["warning"] = metrics_list[j].get("warning", 0) + len(overlap)

    summary = {
        "datasets": len(metrics_list),
        "rows_total": sum(metrics["rows"] for metrics in metrics_list),
        "critical_total": sum(metrics.get("critical", 0) for metrics in metrics_list),
        "warning_total": sum(metrics.get("warning", 0) for metrics in metrics_list),
    }
    summary["exit_code"] = 1 if summary["critical_total"] > 0 else 0
    config = {
        "image_root": args.image_root,
        "image_root_resolved": str(image_root),
        "image_token_estimate": args.image_token_estimate,
        "max_len": args.max_len,
        "out": str(out_dir),
        "datasets": [metrics["path"] for metrics in metrics_list],
    }
    elapsed = round(time.time() - started, 3)

    metrics_payload = {
        "validator": "tools/validate_swift_v6.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "config": config,
        "summary": summary,
        "datasets": [
            {**metrics, "checks": issues_by_dataset.get(metrics["path"], [])}
            for metrics in metrics_list
        ],
    }
    metrics_path = out_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=1), encoding="utf-8")

    report = render_report(metrics_list, issues_by_dataset, summary, config)
    report_path = out_dir / "validation_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"[validate_swift_v6] datasets={summary['datasets']} rows={summary['rows_total']} "
          f"critical={summary['critical_total']} warning={summary['warning_total']} "
          f"exit={summary['exit_code']}")
    for metrics in metrics_list:
        length = metrics.get("length", {})
        print(f"  - {metrics['path']}: rows={metrics['rows']} "
              f"critical={metrics.get('critical', 0)} warning={metrics.get('warning', 0)} "
              f"max_tokens={length.get('total_tokens_est', {}).get('max', 0)} "
              f"cate={metrics.get('cate_dist', {})}")
    print(f"  report: {rel_path(report_path)}")
    print(f"  metrics: {rel_path(metrics_path)}")
    return summary["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
