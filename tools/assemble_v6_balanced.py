#!/usr/bin/env python3
"""组装 V6 balanced 扩数据：union 43,893 行 → S1/S2/S3 stage 文件 + 上传包。

设计（见 docs/MID_TRAIN_DATAWHEEL_PLAN.md §4.6）：
- 先生成跨阶段 union（每 cate 取各阶段需求的最大值，共 43,893 行）；
- 各阶段按 profile 从 union 中抽取自己的 cate 行，组成 stage-specific parquet：
  S1 = 18,438 行（感知：apparel/flag-hard/reference/misleading）
  S2 = 15,656 行（知识：structured_caption/text-compliance/ip-brand/anime-ip）
  S3 = 19,164 行（hard+回放：reference/misleading/quantity/text-compliance）
- 同一行可同时出现在 S1/S3（顺序训练的两次曝光），但 union 内 sample_id 唯一。
- 可选 --upload：把图片复制到 upload/images 并把 images 改写为远端前缀。

用法：
  .venv/bin/python tools/assemble_v6_balanced.py --tolerant
  .venv/bin/python tools/assemble_v6_balanced.py --upload
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.pack_v6_pilot import (estimate_text_tokens, row_text, sha256_file,  # noqa: E402
                                 SCHEMA)

BAL_DIR = PROJECT_ROOT / "output" / "datawheel_v6_balanced"
PARTS_DIR = BAL_DIR / "parts"
DEFAULT_OUT = BAL_DIR / "stage_files"
DEFAULT_UPLOAD = BAL_DIR / "upload"
IMAGE_PREFIX = "algo4/tpz/test/datawheel_v6_balanced"

# 各阶段 cate 行配额（由 mix_plan.json expansion_targets 推导）
STAGE_PROFILES: Dict[str, Dict[str, int]] = {
    "S1": {"apparel": 6453, "flag-hard": 6453, "reference_match": 2766, "misleading": 2766},
    "S2": {"structured_caption": 5480, "text-compliance": 3914, "ip-brand": 3131, "anime-ip": 3131},
    "S3": {"reference_match": 6707, "misleading": 4791, "quantity": 3833, "text-compliance": 3833},
}
UNION_QUOTAS: Dict[str, int] = {
    "apparel": 6453, "flag-hard": 6453, "reference_match": 6707, "misleading": 4791,
    "quantity": 3833, "text-compliance": 3914, "structured_caption": 5480,
    "ip-brand": 3131, "anime-ip": 3131,
}
BASELINE = {
    "S1": {"windows": 84217, "tok_per_window": 8104.29, "epochs": 2, "iters_total_old": 5263},
    "S2": {"windows": 58936, "tok_per_window": 8114.23, "epochs": 2, "iters_total_old": 3683},
    "S3": {"windows": 119366, "tok_per_window": 8110.0, "epochs": 1, "iters_total_old": 3730},
}
GBS = 32


def load_parts(parts_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in ("single_rows.parquet", "reference_rows.parquet"):
        path = parts_dir / name
        if not path.exists():
            raise SystemExit(f"missing part: {path}")
        t = pq.read_table(path)
        if t.schema.names != SCHEMA.names:
            raise SystemExit(f"{path} schema mismatch: {t.schema.names}")
        part = t.to_pylist()
        print(f"[assemble] {name}: {len(part)} rows")
        rows.extend(part)
    return rows


def dedupe(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen = set()
    out = []
    dropped = 0
    for r in rows:
        key = (r["sample_id"], r["cate_id"], tuple(r["images"]))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(r)
    return out, dropped


def select_stage(rows: List[Dict[str, Any]], profile: Dict[str, int], stage: str,
                 seed: int = 42) -> List[Dict[str, Any]]:
    by_cate: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cate.setdefault(r["cate_id"], []).append(r)
    selected: List[Dict[str, Any]] = []
    for cate, need in profile.items():
        pool = sorted(by_cate.get(cate, []), key=lambda r: r["sample_id"])
        random.Random(f"{seed}|{stage}|{cate}").shuffle(pool)
        if len(pool) < need:
            raise SystemExit(f"{stage}/{cate}: need {need} but only {len(pool)} available")
        selected.extend(pool[:need])
    random.Random(f"{seed}|{stage}|all").shuffle(selected)
    return selected


def stage_stats(rows: List[Dict[str, Any]], stage: str) -> Dict[str, Any]:
    base = BASELINE[stage]
    text_tokens = [estimate_text_tokens(row_text(r)) for r in rows]
    n_img = sum(len(r["images"]) for r in rows)
    total512 = sum(text_tokens) + n_img * 512
    total1024 = sum(text_tokens) + n_img * 1024
    windows_added = total512 / base["tok_per_window"] * 0.95
    ipe_old = base["windows"] / GBS
    ipe_new = (base["windows"] + windows_added) / GBS
    from collections import Counter
    return {
        "rows": len(rows),
        "cate_dist": dict(Counter(r["cate_id"] for r in rows)),
        "image_slots": n_img,
        "unique_images": len({i for r in rows for i in r["images"]}),
        "text_tokens": int(sum(text_tokens)),
        "total_tokens_512": int(total512),
        "total_tokens_1024": int(total1024),
        "token_share_512": round(total512 / (base["windows"] * base["tok_per_window"]), 5),
        "token_share_1024": round(total1024 / (base["windows"] * base["tok_per_window"]), 5),
        "windows_added_est": round(windows_added, 1),
        "iters_per_epoch_old": round(ipe_old, 2),
        "iters_per_epoch_new": round(ipe_new, 2),
        "iters_delta_pct": round((ipe_new / ipe_old - 1) * 100, 3),
        "total_iters_new": int(round(ipe_new * base["epochs"])),
        "save_steps_suggested": int(round(ipe_new / 3)),
        "max_row_len_1024": int(max(t + len(r["images"]) * 1024 for t, r in zip(text_tokens, rows))),
    }


def write_parquet(rows: List[Dict[str, Any]], path: Path, image_prefix: str | None = None) -> None:
    if image_prefix:
        mapping = {img: f"{image_prefix}/images/{Path(img).name}"
                   for r in rows for img in r["images"]}
        rows = [{**r, "images": [mapping[i] for i in r["images"]]} for r in rows]
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path, compression="zstd")


def package_upload(stage_files: Dict[str, Path], upload_dir: Path) -> Dict[str, Any]:
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    images_dir = upload_dir / "images"
    images_dir.mkdir(parents=True)
    mapping: Dict[str, str] = {}
    used = set()
    all_images = set()
    for path in stage_files.values():
        for r in pq.read_table(path).to_pylist():
            all_images.update(r["images"])
    for img in sorted(all_images):
        src = PROJECT_ROOT / img
        base = Path(img).name
        if base in used:
            base = hashlib.md5(img.encode()).hexdigest()[:8] + "_" + base
        used.add(base)
        shutil.copy2(src, images_dir / base)
        mapping[img] = f"{IMAGE_PREFIX}/images/{base}"
    manifest = {"rows": {}, "images": len(mapping), "image_prefix": IMAGE_PREFIX,
                "parquet_sha256": {}, "train_files_snippet": {}}
    for stage, path in stage_files.items():
        rows = pq.read_table(path).to_pylist()
        rewritten = [{**r, "images": [mapping[i] for i in r["images"]]} for r in rows]
        dst = upload_dir / f"{stage.lower()}.parquet"
        pq.write_table(pa.Table.from_pylist(rewritten, schema=SCHEMA), dst, compression="zstd")
        manifest["rows"][stage] = len(rewritten)
        manifest["parquet_sha256"][stage] = sha256_file(dst)
        manifest["train_files_snippet"][stage] = (
            f'    "${{TEST_DIR}}/datawheel_v6_balanced/{stage.lower()}.parquet"'
            f"  # V6 Datawheel balanced fast, {len(rewritten):,} rows")
    (upload_dir / "upload_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (upload_dir / "README.md").write_text(
        "# V6 balanced 上传包\n\n"
        f"- 图片前缀：`{IMAGE_PREFIX}`（base=/data/autovision-cfs/autovision-notebook）\n"
        f"- 图片数：{manifest['images']}；每阶段文件行数：{manifest['rows']}\n\n"
        "上传：\n```bash\n"
        f"rsync -av {upload_dir}/ <remote>:/data/autovision-cfs/autovision-notebook/{IMAGE_PREFIX}/\n"
        "```\n\n"
        "TRAIN_FILES 追加（每个阶段只加自己那一行）：\n```bash\n"
        + "\n".join(manifest["train_files_snippet"].values())
        + "\n```\n\n"
        "**先 smoke 再正式跑**；smoke 后按真实 ipe 重算 SAVE_STEPS，并确认 S3 行数口径（是否含 V5）。\n",
        encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble V6 balanced stage datasets")
    parser.add_argument("--parts-dir", type=Path, default=PARTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD)
    parser.add_argument("--tolerant", action="store_true", help="cate 配额不足时继续并报告缺口")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    rows = load_parts(args.parts_dir)
    rows, dropped = dedupe(rows)
    print(f"[assemble] union rows={len(rows)} (dedupe dropped {dropped})")
    from collections import Counter
    cate_counts = Counter(r["cate_id"] for r in rows)
    print(f"[assemble] cate counts: {dict(cate_counts)}")

    quota_issues = {c: {"need": n, "have": cate_counts.get(c, 0)}
                    for c, n in UNION_QUOTAS.items() if cate_counts.get(c, 0) != n}
    if quota_issues:
        print(f"[assemble] union quota issues: {json.dumps(quota_issues, ensure_ascii=False)}")
        if not args.tolerant:
            raise SystemExit("union quotas mismatch; use --tolerant to continue")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    union_manifest = {
        "union_rows": len(rows),
        "cate_counts": dict(cate_counts),
        "union_quotas": UNION_QUOTAS,
        "quota_issues": quota_issues,
    }
    (args.out_dir / "union_manifest.json").write_text(
        json.dumps(union_manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    manifest: Dict[str, Any] = {"stages": {}, "union": union_manifest}
    stage_files: Dict[str, Path] = {}
    for stage, profile in STAGE_PROFILES.items():
        selected = select_stage(rows, profile, stage, args.seed)
        out = args.out_dir / f"{stage.lower()}.parquet"
        write_parquet(selected, out)
        st = stage_stats(selected, stage)
        try:
            st["parquet"] = str(out.relative_to(PROJECT_ROOT))
        except ValueError:
            st["parquet"] = str(out)
        st["parquet_sha256"] = sha256_file(out)
        st["parquet_bytes"] = out.stat().st_size
        manifest["stages"][stage] = st
        stage_files[stage] = out
        print(f"[assemble] {stage}: rows={st['rows']} share={st['token_share_512']:.3%} "
              f"ipe {st['iters_per_epoch_old']}->{st['iters_per_epoch_new']} "
              f"(+{st['iters_delta_pct']}%) save_steps~{st['save_steps_suggested']}")

    (args.out_dir / "balanced_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    if not args.skip_validate:
        cmd = [sys.executable, str(PROJECT_ROOT / "tools" / "validate_swift_v6.py"),
               "--dataset", *[str(p) for p in stage_files.values()],
               "--image-root", str(PROJECT_ROOT),
               "--out", str(args.out_dir / "validation")]
        print("[assemble] " + " ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
        print(f"[assemble] validator exit={rc}")
        if rc != 0:
            raise SystemExit(rc)

    if args.upload:
        up = package_upload(stage_files, args.upload_dir)
        manifest["upload"] = up
        (args.out_dir / "balanced_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[assemble] upload package -> {args.upload_dir} "
              f"({up['images']} images, {up['rows']})")


if __name__ == "__main__":
    main()
