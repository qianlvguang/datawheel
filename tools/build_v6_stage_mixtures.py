#!/usr/bin/env python3
"""把 V6 试点集按 mid-training light 档导出为 S1/S2/S3 三份 stage parquet。

背景（见 docs/MID_TRAIN_DATAWHEEL_PLAN.md §4.3）：
- 试点集仅 2,500 行，要达到 1% 级 token 份额（light 档）需要按阶段重复：
  S1 ×3（2 epochs → 每行 6 次曝光）、S2 ×2（4 次）、S3 ×4（1 epoch → 4 次）。
- 重复副本 sample_id 追加 _rN 后缀，保证行级键 (sample_id, cate_id, images) 唯一。
- 每份文件独立加入对应阶段脚本的 TRAIN_FILES；variant=6 由插件自动映射 loss_V6。

用法：
  .venv/bin/python tools/build_v6_stage_mixtures.py
  .venv/bin/python tools/build_v6_stage_mixtures.py --package-upload output/datawheel_v6_pilot/upload
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.pack_v6_pilot import (estimate_text_tokens, row_text,  # noqa: E402
                                 sha256_file, SCHEMA)

PILOT_DIR = PROJECT_ROOT / "output" / "datawheel_v6_pilot"
DEFAULT_DATASET = PILOT_DIR / "dataset.parquet"
DEFAULT_OUT = PILOT_DIR / "stage_files"
DEFAULT_UPLOAD = PILOT_DIR / "upload"
DEFAULT_PREFIX = "algo4/tpz/test/datawheel_v6"

# light 档（来自 mix_plan.json.recommended；以 smoke 实测 ipe 为准复核）
STAGE_CONFIG = {
    "S1": {"repeat": 3, "epochs": 2, "baseline_windows": 84217, "tok_per_window": 8104.29,
           "iters_total_old": 5263, "save_steps_hint": 887},
    "S2": {"repeat": 2, "epochs": 2, "baseline_windows": 58936, "tok_per_window": 8114.23,
           "iters_total_old": 3683, "save_steps_hint": 620},
    "S3": {"repeat": 4, "epochs": 1, "baseline_windows": 119366, "tok_per_window": 8110.0,
           "iters_total_old": 3730, "save_steps_hint": 1256},
}
GBS = 32


def replicate_rows(rows: List[Dict[str, Any]], repeat: int, stage: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for n in range(repeat):
        for r in rows:
            r2 = dict(r)
            r2["messages"] = [dict(m) for m in r["messages"]]
            r2["images"] = list(r["images"])
            if n > 0:
                r2["sample_id"] = f"{r['sample_id']}_r{n + 1}"
            out.append(r2)
    return out


def stats_for(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    from collections import Counter
    text_tokens = [estimate_text_tokens(row_text(r)) for r in rows]
    n_images = sum(len(r["images"]) for r in rows)
    total512 = sum(text_tokens) + n_images * 512
    total1024 = sum(text_tokens) + n_images * 1024
    windows_added = total512 / cfg["tok_per_window"] * 0.95  # 多图行 packing 折减
    ipe_old = cfg["baseline_windows"] / GBS
    ipe_new = (cfg["baseline_windows"] + windows_added) / GBS
    total_iters_new = ipe_new * cfg["epochs"]
    return {
        "rows": len(rows),
        "unique_source_rows": len(rows) // cfg["repeat"],
        "repeat_per_epoch": cfg["repeat"],
        "epochs": cfg["epochs"],
        "exposures_per_row": cfg["repeat"] * cfg["epochs"],
        "cate_dist": dict(Counter(r["cate_id"] for r in rows)),
        "image_slots": n_images,
        "text_tokens": int(sum(text_tokens)),
        "total_tokens_512": int(total512),
        "total_tokens_1024": int(total1024),
        "token_share_512": round(total512 / (cfg["baseline_windows"] * cfg["tok_per_window"]), 5),
        "token_share_1024": round(total1024 / (cfg["baseline_windows"] * cfg["tok_per_window"]), 5),
        "windows_added_est": round(windows_added, 1),
        "iters_per_epoch_old": round(ipe_old, 2),
        "iters_per_epoch_new": round(ipe_new, 2),
        "iters_per_epoch_delta_pct": round((ipe_new / ipe_old - 1) * 100, 3),
        "total_iters_old": cfg["iters_total_old"],
        "total_iters_new": int(round(total_iters_new)),
        "save_steps_suggested": int(round(ipe_new / 3)),
        "save_steps_old_hint": cfg["save_steps_hint"],
        "max_row_len_1024": int(max(t + len(r["images"]) * 1024 for t, r in zip(text_tokens, rows))),
    }


def write_stage_parquet(rows: List[Dict[str, Any]], path: Path,
                        image_prefix: str | None = None) -> None:
    if image_prefix:
        mapping = {}
        for img in sorted({i for r in rows for i in r["images"]}):
            mapping[img] = f"{image_prefix}/images/{Path(img).name}"
        rows = [{**r, "images": [mapping[i] for i in r["images"]]} for r in rows]
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, path, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V6 light-stage mixtures for S1/S2/S3")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--package-upload", type=Path, default=None,
                        help="把 s1/s2/s3.parquet 写入该上传目录（images 需已存在）")
    parser.add_argument("--image-prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    base_rows = pq.read_table(args.dataset).to_pylist()
    if not base_rows:
        raise SystemExit(f"empty dataset: {args.dataset}")
    print(f"[stage] base rows={len(base_rows)} cates={len({r['cate_id'] for r in base_rows})}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "base_dataset": str(args.dataset.relative_to(PROJECT_ROOT)),
        "base_rows": len(base_rows),
        "stage_config": STAGE_CONFIG,
        "stages": {},
    }
    for stage, cfg in STAGE_CONFIG.items():
        rows = replicate_rows(base_rows, cfg["repeat"], stage)
        # 行级键唯一性检查
        keys = {(r["sample_id"], r["cate_id"], tuple(r["images"])) for r in rows}
        if len(keys) != len(rows):
            raise RuntimeError(f"{stage}: row key collision {len(rows) - len(keys)}")
        out = args.out_dir / f"{stage.lower()}.parquet"
        write_stage_parquet(rows, out)
        st = stats_for(rows, cfg)
        st["parquet"] = str(out.relative_to(PROJECT_ROOT))
        st["parquet_sha256"] = sha256_file(out)
        st["parquet_bytes"] = out.stat().st_size
        manifest["stages"][stage] = st
        print(f"[stage] {stage}: rows={st['rows']} windows+{st['windows_added_est']} "
              f"ipe {st['iters_per_epoch_old']}->{st['iters_per_epoch_new']} "
              f"(+{st['iters_per_epoch_delta_pct']}%) save_steps~{st['save_steps_suggested']}")

    (args.out_dir / "stage_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[stage] manifest -> {args.out_dir / 'stage_manifest.json'}")

    if args.package_upload:
        upload = args.package_upload
        images_dir = upload / "images"
        if not images_dir.exists():
            raise SystemExit(f"upload images missing: {images_dir}; run pack_v6_pilot.py --upload first")
        uploaded = {}
        for stage in STAGE_CONFIG:
            rows = pq.read_table(args.out_dir / f"{stage.lower()}.parquet").to_pylist()
            missing = []
            for r in rows:
                for img in r["images"]:
                    if not (PROJECT_ROOT / img).exists():
                        missing.append(img)
            if missing:
                raise RuntimeError(f"{stage}: {len(set(missing))} local images missing")
            dst = upload / f"{stage.lower()}.parquet"
            write_stage_parquet(rows, dst, image_prefix=args.image_prefix)
            uploaded[stage] = {
                "parquet": dst.name,
                "rows": len(rows),
                "sha256": sha256_file(dst),
                "bytes": dst.stat().st_size,
            }
        up_manifest_path = upload / "upload_manifest.json"
        up = json.loads(up_manifest_path.read_text(encoding="utf-8")) if up_manifest_path.exists() else {}
        up["stage_files"] = uploaded
        up["train_files_snippet"] = {
            "S1": '    "${TEST_DIR}/datawheel_v6/s1.parquet"  # V6 Datawheel light, 2 epochs, r=3',
            "S2": '    "${TEST_DIR}/datawheel_v6/s2.parquet"  # V6 Datawheel light, 2 epochs, r=2',
            "S3": '    "${TEST_DIR}/datawheel_v6/s3.parquet"  # V6 Datawheel light, 1 epoch, r=4',
        }
        up_manifest_path.write_text(json.dumps(up, ensure_ascii=False, indent=1), encoding="utf-8")
        (upload / "STAGE_FILES.md").write_text(
            "# V6 light 档 stage 文件（每份只加到对应阶段）\n\n"
            "| 阶段 | 文件 | 行数 | 建议 SAVE_STEPS |\n|---|---|---:|---:|\n"
            + "\n".join(
                f"| {s} | `{uploaded[s]['parquet']}` | {uploaded[s]['rows']:,} | "
                f"{manifest['stages'][s]['save_steps_suggested']} |" for s in STAGE_CONFIG)
            + "\n\n"
            "TRAIN_FILES 追加（对应阶段脚本各加一行，不要全部加到同一次训练）：\n"
            "```bash\n"
            + "\n".join(up["train_files_snippet"][s] for s in STAGE_CONFIG)
            + "\n```\n\n"
            "smoke 后以真实 `iters_per_epoch` 重算 `SAVE_STEPS = round(ipe/3)`。\n",
            encoding="utf-8")
        print(f"[stage] packaged stage files -> {upload} "
              f"({ {s: uploaded[s]['rows'] for s in uploaded} })")


if __name__ == "__main__":
    main()
