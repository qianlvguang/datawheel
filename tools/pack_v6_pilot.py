#!/usr/bin/env python3
"""合并 V6 试点 parts → 训练用 parquet，并生成统计与远端上传包。

输入：output/datawheel_v6_pilot/parts/*.parquet
  - single_image.parquet   （单图多轮 QA / 结构化 caption / 文本一致性）
  - reference_multi.parquet（多图参考匹配）
输出：
  - output/datawheel_v6_pilot/dataset.parquet       （ms-swift 7 列 schema）
  - output/datawheel_v6_pilot/pilot_manifest.json
  - output/datawheel_v6_pilot/pilot_stats.md
  - 可选 --upload：output/datawheel_v6_pilot/upload/
      dataset.parquet（images 路径改写为远端前缀）
      images/*（去重后的图片）
      upload_manifest.json / README.md（rsync + 训练接入说明）

用法：
  .venv/bin/python tools/pack_v6_pilot.py
  .venv/bin/python tools/pack_v6_pilot.py --upload --image-prefix algo4/tpz/test/datawheel_v6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PILOT_DIR = PROJECT_ROOT / "output" / "datawheel_v6_pilot"
PARTS_DIR = PILOT_DIR / "parts"
DEFAULT_OUT = PILOT_DIR / "dataset.parquet"

MESSAGES_SCHEMA = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
SCHEMA = pa.schema([
    ("messages", MESSAGES_SCHEMA),
    ("images", pa.list_(pa.string())),
    ("enable_thinking", pa.bool_()),
    ("sample_id", pa.string()),
    ("cate_id", pa.string()),
    ("variant", pa.int32()),
])


def estimate_text_tokens(text: str) -> int:
    """粗略文本 token：中文汉字/1.6 + 英文数字字符/4。"""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_chars = len(re.findall(r"[A-Za-z0-9_\-\[\]{}:,\".<>=/]", text))
    return int(round(cjk / 1.6 + ascii_chars / 4.0)) + 1


def row_text(row: Dict[str, Any]) -> str:
    return "\n".join(m.get("content", "") for m in row["messages"])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_parts(parts_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    per_part: Dict[str, int] = {}
    for p in sorted(parts_dir.glob("*.parquet")):
        if p.name.startswith("dataset"):
            continue
        t = pq.read_table(p)
        if t.schema.names != SCHEMA.names:
            raise ValueError(f"{p} schema mismatch: {t.schema.names} != {SCHEMA.names}")
        part_rows = t.to_pylist()
        per_part[p.name] = len(part_rows)
        rows.extend(part_rows)
    if not rows:
        raise RuntimeError(f"no parquet parts under {parts_dir}")
    return rows, per_part


def dedupe_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen = set()
    out: List[Dict[str, Any]] = []
    dropped = 0
    for r in rows:
        key = (r["sample_id"], r["cate_id"], tuple(r["images"]))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(r)
    return out, dropped


def compute_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter
    cate_dist = Counter(r["cate_id"] for r in rows)
    image_counts = Counter(len(r["images"]) for r in rows)
    variants = Counter(r["variant"] for r in rows)
    texts = [row_text(r) for r in rows]
    text_tokens = [estimate_text_tokens(t) for t in texts]
    image_tokens = {str(k): sum(len(r["images"]) * k for r in rows) for k in (256, 512, 1024)}
    total_tokens_512 = sum(text_tokens) + image_tokens["512"]
    total_tokens_1024 = sum(text_tokens) + image_tokens["1024"]
    # 单行最长估算（1024/图）用于长度门禁
    max_len_1024 = max(t + len(r["images"]) * 1024 for t, r in zip(text_tokens, rows))
    over_len = sum(1 for t, r in zip(text_tokens, rows) if t + len(r["images"]) * 1024 > 8192)
    unique_images = sorted({img for r in rows for img in r["images"]})
    missing = [img for img in unique_images if not (PROJECT_ROOT / img).exists()]
    return {
        "rows": len(rows),
        "cate_dist": dict(cate_dist),
        "image_count_dist": {str(k): v for k, v in sorted(image_counts.items())},
        "variant_dist": dict(variants),
        "unique_images": len(unique_images),
        "missing_images": len(missing),
        "missing_examples": missing[:5],
        "text_tokens_sum": int(sum(text_tokens)),
        "text_tokens_mean": round(sum(text_tokens) / len(rows), 1),
        "text_tokens_p95": int(sorted(text_tokens)[int(len(text_tokens) * 0.95)]),
        "image_tokens_by_assumption": image_tokens,
        "total_tokens_512": int(total_tokens_512),
        "total_tokens_1024": int(total_tokens_1024),
        "est_windows_8114": int(round(total_tokens_512 / 8114)),
        "est_windows_8114_conservative": int(round(total_tokens_1024 / 8114)),
        "max_row_len_1024": int(max_len_1024),
        "rows_over_8192_conservative": int(over_len),
        "two_turn_ratio": round(sum(1 for r in rows if len(r["messages"]) == 4) / len(rows), 3),
    }


def write_upload(rows: List[Dict[str, Any]], upload_dir: Path, image_prefix: str) -> Dict[str, Any]:
    upload_dir = upload_dir.resolve()
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    images_dir = upload_dir / "images"
    images_dir.mkdir(parents=True)
    mapping: Dict[str, str] = {}
    used = set()
    for img in sorted({img for r in rows for img in r["images"]}):
        src = PROJECT_ROOT / img
        base = Path(img).name
        if base in used:  # 防重名
            base = f"{hashlib.md5(img.encode()).hexdigest()[:8]}_{base}"
        used.add(base)
        shutil.copy2(src, images_dir / base)
        mapping[img] = f"{image_prefix}/images/{base}"
        # 远端路径校验：图片可解码
        Image.open(src).verify()

    rewritten: List[Dict[str, Any]] = []
    for r in rows:
        r2 = dict(r)
        r2["images"] = [mapping[x] for x in r["images"]]
        rewritten.append(r2)
    table = pa.Table.from_pylist(rewritten, schema=SCHEMA)
    pq.write_table(table, upload_dir / "dataset.parquet", compression="zstd")
    manifest = {
        "rows": len(rewritten),
        "images": len(mapping),
        "image_prefix": image_prefix,
        "parquet": "dataset.parquet",
        "parquet_sha256": sha256_file(upload_dir / "dataset.parquet"),
        "example_row": {"messages": rewritten[0]["messages"][:2], "images": rewritten[0]["images"]},
        "access_base": "/data/autovision-cfs/autovision-notebook",
        "rsync_hint": f"rsync -av {upload_dir}/ <remote>:/data/autovision-cfs/autovision-notebook/{image_prefix}/",
        "train_files_snippet": f'    "${{TEST_DIR}}/datawheel_v6/dataset.parquet"  # V6 Datawheel pilot',
    }
    (upload_dir / "upload_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (upload_dir / "README.md").write_text(
        "# V6 试点上传包\n\n"
        f"- 图片前缀（相对 /data/autovision-cfs/autovision-notebook）：`{image_prefix}`\n"
        f"- 行数：{manifest['rows']}；图片：{manifest['images']}\n"
        f"- parquet sha256：`{manifest['parquet_sha256']}`\n\n"
        "上传：\n```bash\n" + manifest["rsync_hint"] + "\n```\n\n"
        "接入训练：把下面一行追加到对应阶段脚本的 `TRAIN_FILES` 数组末尾（先 smoke）：\n"
        "```bash\n" + manifest["train_files_snippet"] + "\n```\n\n"
        "smoke 检查：strict 不丢行、日志出现 `loss_V6`、无 MaxLengthError。\n",
        encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack V6 pilot dataset for mid-training")
    parser.add_argument("--parts-dir", type=Path, default=PARTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--upload-dir", type=Path, default=PILOT_DIR / "upload")
    parser.add_argument("--image-prefix", default="algo4/tpz/test/datawheel_v6")
    args = parser.parse_args()

    parts = {p.name: p for p in sorted(args.parts_dir.glob("*.parquet"))}
    if not parts:
        raise SystemExit(f"no parts found in {args.parts_dir}; run build_v6_*.py first")
    print(f"[pack] parts: { {k: v.stat().st_size for k, v in parts.items()} }")
    rows, per_part = load_parts(args.parts_dir)
    print(f"[pack] loaded {len(rows)} rows from {per_part}")
    rows, dropped = dedupe_rows(rows)
    print(f"[pack] dedupe dropped {dropped} rows -> {len(rows)}")

    # 基本结构检查（validator 会做完整版）
    problems: List[str] = []
    for r in rows:
        if not r["messages"] or r["messages"][0]["role"] != "user":
            problems.append(f"{r['sample_id']}: messages 首条不是 user")
        if r["variant"] != 6:
            problems.append(f"{r['sample_id']}: variant != 6")
        if r["enable_thinking"] is not False:
            problems.append(f"{r['sample_id']}: enable_thinking != False")
        n_img = r["messages"][0]["content"].count("<image>")
        if n_img != len(r["images"]):
            problems.append(f"{r['sample_id']}: <image>×{n_img} != images×{len(r['images'])}")
    if problems:
        print(f"[pack] {len(problems)} structural problem(s), first 5: {problems[:5]}")
    else:
        print("[pack] structural checks passed")

    stats = compute_stats(rows)
    stats["per_part_rows"] = per_part
    stats["dedupe_dropped"] = dropped
    stats["structural_problems"] = problems[:20]
    if args.out.parent.exists() is False:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, args.out, compression="zstd")
    print(f"[pack] wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB, {table.num_rows} rows)")

    manifest = {"dataset": str(args.out.relative_to(PROJECT_ROOT)), "stats": stats}
    if args.upload:
        upload_manifest = write_upload(rows, args.upload_dir, args.image_prefix)
        manifest["upload"] = upload_manifest
        print(f"[pack] upload package -> {args.upload_dir} ({upload_manifest['rows']} rows, "
              f"{upload_manifest['images']} images)")

    (PILOT_DIR / "pilot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    md = [
        "# V6 试点集统计",
        "",
        f"- 行数：{stats['rows']}（parts: {per_part}）",
        f"- cate 分布：{json.dumps(stats['cate_dist'], ensure_ascii=False)}",
        f"- 每行图片数分布：{json.dumps(stats['image_count_dist'], ensure_ascii=False)}",
        f"- 唯一图片：{stats['unique_images']}；缺失：{stats['missing_images']}",
        f"- 文本 token 估算：mean {stats['text_tokens_mean']} / p95 {stats['text_tokens_p95']} / sum {stats['text_tokens_sum']}",
        f"- 图片 token 估算：256/512/1024 每图 → {json.dumps(stats['image_tokens_by_assumption'])}",
        f"- 总 token（512/图）：{stats['total_tokens_512']:,} → 估算 windows≈{stats['est_windows_8114']}",
        f"- 总 token（1024/图保守）：{stats['total_tokens_1024']:,} → 估算 windows≈{stats['est_windows_8114_conservative']}",
        f"- 单行最长保守估算：{stats['max_row_len_1024']} tokens；超 8192 行数：{stats['rows_over_8192_conservative']}",
        f"- 两轮对话行占比：{stats['two_turn_ratio']}",
        f"- 结构问题：{len(problems)}",
        "",
        "> token 为字符估算，正式长度需远端 tokenizer/template 编码确认。",
    ]
    (PILOT_DIR / "pilot_stats.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
