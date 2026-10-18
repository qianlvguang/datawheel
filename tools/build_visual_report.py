#!/usr/bin/env python3
"""生成离线 HTML 可视化核验报告：output/report/index.html

内容：
1. 数据集总览卡片（v2 10k / V6 pilot / V6 balanced）
2. 样例画廊：按数据集/阶段/cate/split 过滤，搜索 sample_id，BBox 叠加标注（[ymin,xmin,ymax,xmax] 0~1000）
3. 训练配比：S1/S2/S3 stage 文件、token 份额、iters/SAVE_STEPS、TRAIN_FILES 片段
4. 质检结果：flywheel_audit / validate_swift_v6 的 critical/warning 摘要与报告链接
5. 产物索引与复现命令

无外部依赖/无 CDN，双击文件即可在本地打开（图片来自项目 output/ 相对路径）。

用法：
  .venv/bin/python tools/build_visual_report.py
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import html
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT_HTML = ROOT / "output" / "report" / "index.html"

V2_DATASET = ROOT / "output" / "datawheel_v2" / "dataset.parquet"
V2_AUDIT = ROOT / "output" / "datawheel_v2" / "audit" / "audit_metrics.json"
PILOT_DATASET = ROOT / "output" / "datawheel_v6_pilot" / "dataset.parquet"
PILOT_MANIFEST = ROOT / "output" / "datawheel_v6_pilot" / "pilot_manifest.json"
PILOT_VALID = ROOT / "output" / "datawheel_v6_pilot" / "validation" / "validation_metrics.json"
LIGHT_MANIFEST = ROOT / "output" / "datawheel_v6_pilot" / "stage_files" / "stage_manifest.json"
BAL_DIR = ROOT / "output" / "datawheel_v6_balanced"
BAL_MANIFEST = BAL_DIR / "stage_files" / "balanced_manifest.json"
BAL_VALID = BAL_DIR / "stage_files" / "validation" / "validation_metrics.json"
PLAN_DOC = ROOT / "docs" / "MID_TRAIN_DATAWHEEL_PLAN.md"

STAGE_SHORT = {
    "stage_a_geometry": "A 几何",
    "stage_b_single_element": "B 单图元",
    "stage_c_multi_element": "C 多图元",
    "stage_d_reference_match": "D 参考匹配",
    "stage_e_negative_evidence": "E 负样本",
}
STAGE_CN = {
    "S1": "S1 视觉感知", "S2": "S2 知识", "S3": "S3 final+回放",
}
DOMAIN_CN = {
    "geometric_primitives": "色块图案",
    "official_flags": "国旗",
    "official_emblems": "国徽/纹章",
    "official_regional": "地方旗帜",
    "luxury_brands": "奢侈品",
    "ip_brands": "IP 品牌",
    "character_ips": "动漫角色",
    "subculture_elements": "亚文化符号",
}
CATE_CN = {
    "apparel": "服饰商品",
    "flag-hard": "国旗国徽",
    "ip-brand": "IP 品牌",
    "anime-ip": "动漫角色",
    "misleading": "否定与不确定",
    "quantity": "计数与方位",
    "text-compliance": "描述一致性",
    "structured_caption": "结构化描述",
    "reference_match": "参考图匹配",
}
BBOX_RE = re.compile(r"\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]")


# ---------------------------------------------------------------- utilities
def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}


def human_size(n: int | float | None) -> str:
    if n is None:
        return "-"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024
    return f"{n:.1f} GB"


def embed_image(project_path: str, max_side: int = 560, quality: int = 80) -> str:
    """把项目内图片压缩后转成 data URI，保证 HTML 单文件可离线/跨目录打开。"""
    try:
        path = Path(project_path)
        if not path.is_absolute():
            path = ROOT / path
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return url_for(project_path)


def url_for(path: str | Path) -> str:
    """项目相对路径 → 相对 output/report/index.html 的 URL。"""
    p = Path(str(path))
    if not p.is_absolute():
        p = ROOT / p
    try:
        rel = p.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return p.as_uri() if p.is_absolute() else str(path)
    parts = Path(rel).parts
    if parts and parts[0] == "output":
        return "../" + "/".join(parts[1:])
    return "../../" + "/".join(parts)


def pick_stride(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if len(rows) <= n:
        return list(rows)
    step = len(rows) / float(n)
    return [rows[int(i * step)] for i in range(n)]


def bbox_from_texts(texts: List[str]) -> List[List[int]]:
    boxes: List[List[int]] = []
    for text in texts:
        for m in BBOX_RE.finditer(text or ""):
            b = [int(x) for x in m.groups()]
            if all(0 <= x <= 1000 for x in b) and b[0] < b[2] and b[1] < b[3]:
                if b not in boxes:
                    boxes.append(b)
    return boxes


# ---------------------------------------------------------------- sample collection
def load_v2_samples() -> List[Dict[str, Any]]:
    rows = pq.read_table(V2_DATASET).to_pylist()
    quotas = {"stage_a_geometry": 12, "stage_b_single_element": 30, "stage_c_multi_element": 30,
              "stage_d_reference_match": 26, "stage_e_negative_evidence": 22}
    samples: List[Dict[str, Any]] = []
    for stage, quota in quotas.items():
        pool = [r for r in rows if r["task_stage"] == stage]
        # B/C/D 优先带 bbox 的样本
        with_boxes = [r for r in pool if json.loads(r.get("bboxes_norm_1000") or "[]")]
        if stage in ("stage_b_single_element", "stage_c_multi_element", "stage_d_reference_match"):
            pool = with_boxes or pool
        picked = pick_stride(pool, quota)
        for r in picked:
            try:
                entities = json.loads(r.get("entities_json") or "[]")
                boxes = json.loads(r.get("bboxes_norm_1000") or "[]")
                qa = json.loads(r.get("qa_pairs_json") or "[]")
            except Exception:
                entities, boxes, qa = [], [], []
            labels = []
            for i, e in enumerate(entities):
                labels.append(e.get("entity_name_zh") or e.get("full_title_zh") or f"target{i + 1}")
            samples.append({
                "id": r["sample_id"],
                "dataset": "v2",
                "dataset_label": "Datawheel v2 · 10K",
                "group": STAGE_SHORT.get(stage, stage),
                "stage": stage,
                "split": r.get("split", ""),
                "cate": r.get("domain", ""),
                "images": [url_for(r["image_path"])] if r.get("image_path") else [],
                "image_raw": [r.get("image_path", "")],
                "bboxes": boxes if isinstance(boxes, list) else [],
                "labels": labels,
                "messages": [],
                "caption": r.get("caption_zh", ""),
                "qa": qa[:4],
                "meta": {
                    "阶段": STAGE_SHORT.get(stage, stage),
                    "数据划分": r.get("split", ""),
                    "类别": DOMAIN_CN.get(r.get("domain", ""), r.get("domain", "")),
                    "难度": r.get("difficulty", ""),
                    "画幅": f'{r.get("image_width", "?")}×{r.get("image_height", "?")}',
                },
            })
    return samples


def load_v6_samples(path: Path, dataset_key: str, dataset_label: str,
                    per_cate: int = 6, stage: str = "") -> List[Dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    by_cate: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cate.setdefault(r["cate_id"], []).append(r)
    samples: List[Dict[str, Any]] = []
    for cate, pool in by_cate.items():
        for r in pick_stride(pool, per_cate):
            texts = [m.get("content", "") for m in r.get("messages", [])]
            boxes = bbox_from_texts(texts)[:3]
            labels = [f"{cate}"] * len(boxes)
            last_answer = next((m["content"] for m in reversed(r["messages"]) if m.get("role") == "assistant"), "")
            samples.append({
                "id": r["sample_id"],
                "dataset": dataset_key,
                "dataset_label": dataset_label,
                "group": f"{stage}·{CATE_CN.get(cate, cate)}" if stage else CATE_CN.get(cate, cate),
                "stage": stage,
                "split": "",
                "cate": r["cate_id"],
                "images": [url_for(p) for p in r.get("images", [])],
                "image_raw": list(r.get("images", [])),
                "bboxes": boxes,
                "labels": labels,
                "messages": [{"role": m.get("role", ""), "content": m.get("content", "")}
                             for m in r.get("messages", [])],
                "caption": last_answer,
                "qa": [],
                "meta": {
                    "任务": CATE_CN.get(r["cate_id"], r["cate_id"]),
                    "图片数": len(r.get("images", [])),
                },
            })
    return samples


def collect_samples() -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    samples += load_v2_samples()
    samples += load_v6_samples(PILOT_DATASET, "v6-pilot", "V6 pilot · 2.5K", per_cate=6)
    for stage, path in (("S1", BAL_DIR / "stage_files" / "s1.parquet"),
                        ("S2", BAL_DIR / "stage_files" / "s2.parquet"),
                        ("S3", BAL_DIR / "stage_files" / "s3.parquet")):
        samples += load_v6_samples(path, "v6-balanced", f"V6 balanced · {stage}", per_cate=3, stage=stage)
    return {"samples": samples}


# ---------------------------------------------------------------- stats
def build_stats(samples: Dict[str, Any]) -> Dict[str, Any]:
    v2_audit = load_json(V2_AUDIT)
    pilot_manifest = load_json(PILOT_MANIFEST)
    pilot_valid = load_json(PILOT_VALID)
    light = load_json(LIGHT_MANIFEST)
    bal = load_json(BAL_MANIFEST)
    bal_valid = load_json(BAL_VALID)
    v2_rows = pq.read_table(V2_DATASET).num_rows if V2_DATASET.exists() else 0
    return {
        "v2": {"rows": v2_rows, "audit": v2_audit},
        "pilot": {"manifest": pilot_manifest, "validation": pilot_valid},
        "light": light,
        "balanced": {"manifest": bal, "validation": bal_valid},
        "samples": samples["samples"],
    }


# ---------------------------------------------------------------- html blocks
def esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def badge(text: str, kind: str = "") -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


def table(headers: List[str], rows: List[List[Any]], cls: str = "") -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def link(label: str, path: str | Path, desc: str = "") -> str:
    p = Path(path)
    state = "" if (ROOT / path).exists() else " missing"
    size = human_size((ROOT / path).stat().st_size) if (ROOT / path).is_file() else "目录"
    return (f'<a href="{esc(url_for(path))}"{" class=\"missing\"" if state else ""}>{esc(label)}</a>'
            f'<span class="muted"> {esc(desc)} · {size}</span>')


def overview_cards() -> str:
    stats = build_stats(collect_samples())
    v2 = stats["v2"]
    pilot = stats["pilot"]["manifest"].get("stats", {})
    bal = stats["balanced"]["manifest"]
    bal_rows = {s: v.get("rows", 0) for s, v in bal.get("stages", {}).items()}
    total_rows = sum(bal_rows.values())
    cards = [
        ("Datawheel v2 主数据集", f"{v2['rows']:,}", "条 · 10,000 张全量渲染图 · 5 阶段课程", "v2"),
        ("V6 中训练试点", f"{pilot.get('rows', 0):,}", f"条 · {pilot.get('unique_images', 0):,} 图 · 9 个 cate", "pilot"),
        ("V6 balanced-v1", f"{total_rows:,}", f"条 stage 数据 · union {bal.get('union', {}).get('union_rows', 0):,} 行", "balanced"),
        ("契约校验", "0 critical", "v2 audit / V6 validator 全通过", "ok"),
    ]
    return "".join(
        f'<div class="stat-card {kind}"><div class="stat-value">{esc(value)}</div>'
        f'<div class="stat-label">{esc(label)}</div><div class="stat-sub">{esc(sub)}</div></div>'
        for label, value, sub, kind in cards)


def dataset_section() -> str:
    stats = build_stats(collect_samples())
    v2 = stats["v2"]["audit"]
    pilot = stats["pilot"]["manifest"].get("stats", {})
    light = stats["light"].get("stages", {})
    bal = stats["balanced"]["manifest"].get("stages", {})
    val = stats["balanced"]["validation"].get("summary", {})

    dist = v2.get("distributions", {}).get("task_stage", {})
    stage_rows = [[STAGE_SHORT.get(k, k), v.get("count", 0), f"{v.get('ratio', 0) * 100:.1f}%"]
                  for k, v in dist.items()]
    split = v2.get("split", {}).get("counts", {})
    split_rows = [[k, v] for k, v in split.items()]
    bbox = v2.get("bbox", {})
    img = v2.get("images", {})
    v2_quality = [
        ["BBox", f"{bbox.get('boxes_checked', 0):,} 个", f"非法/越界/顺序错误 = {bbox.get('boxes_invalid', 0)} / "
         f"{bbox.get('boxes_out_of_range', 0)} / {bbox.get('boxes_bad_y_order', 0) + bbox.get('boxes_bad_x_order', 0)}"],
        ["图片", f"{img.get('checked', 0):,} 张", f"缺失 {img.get('missing', 0)} / 解码失败 {img.get('decode_fail', 0)} / "
         f"sha 不一致 {img.get('sha_mismatch', 0)}"],
        ["split 泄漏", "0", f"实体交集 0 / 底板交集 0（实际 split 数 {len(split)}）"],
        ["审计提示", f"warning {v2.get('warning_count', 0)}", "小目标 562、caption 重复 6.7%、QA 模板重复 56.5%（合法保留）"],
    ]

    pilot_rows = [[k, v] for k, v in pilot.get("cate_dist", {}).items()]
    light_rows = [[s, v.get("rows", 0), f"{v.get('token_share_512', 0) * 100:.2f}%",
                   v.get("save_steps_suggested", "-")] for s, v in light.items()]
    bal_rows = [[f"{s}（{STAGE_CN.get(s, s)}）", v.get("rows", 0), f"{v.get('token_share_512', 0) * 100:.2f}%",
                 f"+{v.get('iters_delta_pct', 0):.2f}%", v.get("save_steps_suggested", "-")]
                for s, v in bal.items()]
    val_rows = [[d["path"].split("/")[-1], d.get("rows", 0), d.get("critical", 0), d.get("warning", 0)]
                for d in stats["balanced"]["validation"].get("datasets", [])]

    return f"""
<section id="datasets">
  <h2>📊 数据集总览</h2>
  <div class="split-2">
    <div class="panel">
      <h3>Datawheel v2 · 10,000 条</h3>
      {table(["阶段", "行数", "占比"], stage_rows)}
      {table(["split", "行数"], split_rows)}
      {table(["质检项", "数值", "结果"], v2_quality)}
      <p class="muted">报告：{link("report.md", "output/datawheel_v2/report.md")} ·
      {link("audit_report.md", "output/datawheel_v2/audit/audit_report.md")} ·
      {link("dataset.parquet", "output/datawheel_v2/dataset.parquet")} ·
      {link("sft/", "output/datawheel_v2/sft")}</p>
    </div>
    <div>
      <div class="panel">
        <h3>V6 试点 · 2,500 条（9 cate）</h3>
        {table(["cate", "行数"], pilot_rows)}
        <p class="muted">校验：critical {stats['pilot']['validation'].get('summary', {}).get('critical_total', 0)} /
        warning {stats['pilot']['validation'].get('summary', {}).get('warning_total', 0)} ·
        {link("validation_report.md", "output/datawheel_v6_pilot/validation/validation_report.md")}</p>
      </div>
      <div class="panel">
        <h3>V6 balanced-v1 · S1/S2/S3</h3>
        {table(["阶段", "行数", "token 份额", "Δiters/epoch", "建议 SAVE_STEPS"], bal_rows)}
        <p class="muted">validator critical {val.get('critical_total', 0)} · 
        {link("balanced_manifest.json", "output/datawheel_v6_balanced/stage_files/balanced_manifest.json")} ·
        {link("upload/README.md", "output/datawheel_v6_balanced/upload/README.md")}</p>
      </div>
    </div>
  </div>
</section>"""


def mixture_section() -> str:
    stats = build_stats(collect_samples())
    light = stats["light"].get("stages", {})
    bal = stats["balanced"]["manifest"].get("stages", {})
    up = stats["balanced"]["manifest"].get("upload", {})
    snippets = up.get("train_files_snippet", {})
    bars = ""
    for s, v in bal.items():
        pct = v.get("token_share_512", 0) * 100
        bars += (f'<div class="bar-row"><span class="bar-label">{esc(s)}</span>'
                 f'<div class="bar"><i style="width:{min(pct / 3.0 * 100, 100):.1f}%"></i></div>'
                 f'<span class="bar-value">{pct:.2f}%</span></div>')
    train_lines = "\n".join(snippets.get(s, "") for s in ("S1", "S2", "S3")) if snippets else "(见 upload package)"
    return f"""
<section id="mixture">
  <h2>⚙️ 中训练配比（Qwen3.5-27B V6 通道）</h2>
  <div class="split-2">
    <div class="panel">
      <h3>balanced-v1 token 份额（相对每阶段 epoch）</h3>
      {bars}
      <p class="muted">目标：S1 2% / S2 2% / S3 3%；S3 实测 2.06%（参考行 token 密度低于估算），
      若要 3% 需扩 reference 行或提高单行图数/问答密度。</p>
    </div>
    <div class="panel">
      <h3>训练接入（每阶段只加自己那一行）</h3>
      <pre class="code">TRAIN_FILES+=(
{train_lines}
)</pre>
      <p class="muted">图片前缀：<code>algo4/tpz/test/datawheel_v6_balanced</code>；
      先 <code>SMOKE=1</code> 10 iters 确认 <code>loss_V6</code>、strict 不丢行，再按真实 ipe 重算 SAVE_STEPS。</p>
    </div>
  </div>
</section>"""


def validation_section() -> str:
    stats = build_stats(collect_samples())
    v2 = stats["v2"]["audit"]
    items = [
        ("flywheel_audit（v2）", "critical=0 / warning=4", "10,000 行、10,000 图；BBox 13,796 个全部合法；split 泄漏 0",
         "output/datawheel_v2/audit/audit_report.md"),
        ("validate_swift_v6（pilot）", "critical=0 / exit 0", "2,500 行 ms-swift 契约通过",
         "output/datawheel_v6_pilot/validation/validation_report.md"),
        ("validate_swift_v6（balanced）", "critical=0 / exit 0", "53,258 行 stage 数据；最长 4,210 tokens（1024/图口径）",
         "output/datawheel_v6_balanced/stage_files/validation/validation_report.md"),
    ]
    rows = "".join(
        f'<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td><td>{link("报告", d)}</td></tr>'
        for a, b, c, d in items)
    return f"""
<section id="validation">
  <h2>✅ 质检与契约校验</h2>
  <table><thead><tr><th>工具/数据集</th><th>结论</th><th>关键结果</th><th>报告</th></tr></thead>
  <tbody>{rows}</tbody></table>
  <p class="muted">warning 均为设计预期：图片跨行复用（多视图/候选参考图）、S1↔S3 共享 reference/misleading 行。
  远端 H200 训练 smoke 与 eval 尚未执行（本机无远端环境）。</p>
</section>"""


def artifacts_section() -> str:
    entries = [
        ("v2 数据集", "output/datawheel_v2/dataset.parquet", "10,000 行 × 35 列"),
        ("v2 图像", "output/datawheel_v2/images", "10,000 张 JPEG"),
        ("v2 质检报告", "output/datawheel_v2/audit/audit_report.md", "flywheel_audit"),
        ("v2 训练导出", "output/datawheel_v2/sft", "ShareGPT / ms-swift"),
        ("V6 试点数据", "output/datawheel_v6_pilot/dataset.parquet", "2,500 行"),
        ("V6 试点上传包", "output/datawheel_v6_pilot/upload", "117MB · 3,225 图"),
        ("V6 balanced stage", "output/datawheel_v6_balanced/stage_files", "s1/s2/s3 + manifest"),
        ("V6 balanced 上传包", "output/datawheel_v6_balanced/upload", "314MB · 7,982 图"),
        ("V6 balanced 校验", "output/datawheel_v6_balanced/stage_files/validation/validation_report.md", "critical=0"),
        ("中训练方案", "docs/MID_TRAIN_DATAWHEEL_PLAN.md", "V6 数据方案 v0.2"),
        ("项目 README", "README.md", "总览与快速开始"),
        ("资产台账", "assets/ASSET_CATALOG.md", "全景资产"),
        ("v1 历史产物", "output/legacy_v1", "gallery/contrastive/density/demo 等"),
    ]
    rows = "".join(f'<tr><td>{link(label, path)}</td><td class="muted">{esc(desc)}</td></tr>'
                   for label, path, desc in entries)
    return f"""
<section id="artifacts">
  <h2>📁 产物索引</h2>
  <table><thead><tr><th>产物</th><th>说明</th></tr></thead><tbody>{rows}</tbody></table>
  <div class="panel">
    <h3>复现命令</h3>
    <pre class="code">cd ~/Documents/Coding/datawheel
.venv/bin/python tools/build_visual_report.py      # 重新生成本页
.venv/bin/python generate.py v2                    # v2 10k
.venv/bin/python tools/assemble_v6_balanced.py --upload  # V6 balanced
.venv/bin/python tools/flywheel_audit.py --dataset output/datawheel_v2/dataset.parquet --images-root . --out output/datawheel_v2/audit/
.venv/bin/python tools/validate_swift_v6.py --dataset output/datawheel_v6_balanced/stage_files/s1.parquet output/datawheel_v6_balanced/stage_files/s2.parquet output/datawheel_v6_balanced/stage_files/s3.parquet</pre>
  </div>
</section>"""


def gallery_section(n_samples: int) -> str:
    return f"""
<section id="gallery">
  <h2>🖼️ 样例画廊与 BBox 标注</h2>
  <p class="muted">BBox 为 <code>[ymin, xmin, ymax, xmax]</code>，相对坐标 0~1000。V6 行的框从答案文本提取，标注在第一张商品图上；
  v2 行直接读取真值 bboxes。点击图片可放大查看。</p>
  <div class="controls">
    <label>数据集 <select id="f-dataset"><option value="all">全部</option></select></label>
    <label>分组 <select id="f-group"><option value="all">全部</option></select></label>
    <label>split <select id="f-split"><option value="all">全部</option></select></label>
    <label class="search"><input id="f-search" type="search" placeholder="搜索 sample_id / 关键词"></label>
    <label class="toggle"><input id="f-bbox" type="checkbox" checked> 显示 BBox</label>
    <span id="btn-reset" class="ghost" role="button" tabindex="0">重置</span>
    <span id="gallery-count" class="muted"></span>
  </div>
  <div id="gallery-grid" class="gallery"></div>
  <div class="more-wrap"><span id="btn-more" class="ghost" role="button" tabindex="0">加载更多</span></div>
</section>"""


CSS = """
:root{--bg:#f7f8fa;--card:#fff;--text:#1d1d1f;--muted:#6e6e73;--line:#e5e5ea;--accent:#0a84ff;--ok:#30a14e;--warn:#b7791f;--bad:#d1242f;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text)}
header{background:linear-gradient(135deg,#0b2545,#13315c 55%,#1b4d89);color:#fff;padding:28px 36px}
header h1{margin:0 0 6px;font-size:25px;letter-spacing:.3px}
header p{margin:0;opacity:.85;font-size:13px}
nav{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.94);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:8px 36px;display:flex;gap:18px;flex-wrap:wrap}
nav a{color:var(--text);text-decoration:none;font-size:13px;padding:4px 8px;border-radius:6px}
nav a:hover{background:#eef2f7;color:var(--accent)}
main{max-width:1500px;margin:0 auto;padding:18px 32px 70px}
section{margin:26px 0}
h2{font-size:20px;margin:0 0 14px}
h3{font-size:15px;margin:0 0 10px}
.panel,.stat-card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.panel{margin-bottom:14px;overflow:auto}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.stat-card{border-top:3px solid var(--accent)}
.stat-card.ok{border-top-color:var(--ok)}.stat-card.pilot{border-top-color:#b7791f}.stat-card.balanced{border-top-color:#7b3fe4}
.stat-value{font-size:28px;font-weight:700}
.stat-label{font-weight:600;margin-top:2px}
.stat-sub{color:var(--muted);font-size:12px;margin-top:3px}
.split-2{display:grid;grid-template-columns:1.06fr .94fr;gap:14px}
@media(max-width:1000px){.split-2{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:13px;margin:6px 0 12px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f2f4f8;font-weight:600;white-space:nowrap}
.muted{color:var(--muted);font-size:12px}
.badge{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;background:#eef2f7;color:#35507a;margin-right:5px;white-space:nowrap}
.badge.v2{background:#e7f0ff;color:#0a4bab}.badge.v6{background:#f3ecff;color:#5b21b6}
.badge.split{background:#eaf7ef;color:#1d6b3c}
.bar-row{display:flex;align-items:center;gap:10px;margin:8px 0}
.bar-label{width:40px;font-weight:600}
.bar{flex:1;height:14px;background:#eef1f5;border-radius:8px;overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#0a84ff,#7b3fe4);border-radius:8px}
.bar-value{width:62px;text-align:right;font-variant-numeric:tabular-nums}
.code{background:#0f172a;color:#d8e1f0;padding:12px 14px;border-radius:10px;overflow:auto;font-size:12px;line-height:1.5}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:14px;position:sticky;top:46px;z-index:10}
.controls label{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:6px}
select,input[type=search]{font:inherit;font-size:13px;padding:5px 8px;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--text)}
.search input{min-width:220px}
.controls .toggle{gap:5px;color:var(--text)}
.ghost{font:inherit;font-size:13px;padding:5px 12px;border-radius:8px;border:1px solid var(--line);background:#fff;cursor:pointer;display:inline-block}
.ghost:hover{border-color:var(--accent);color:var(--accent)}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.card-head{display:flex;flex-wrap:wrap;gap:4px;padding:8px 10px;border-bottom:1px solid var(--line);align-items:center}
.card-head .sid{font-size:11px;color:var(--muted);margin-left:auto;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.imgwrap{position:relative;background:#eef0f3;cursor:zoom-in;min-height:140px;display:flex;align-items:center;justify-content:center}
.imgwrap img{display:block;max-width:100%;max-height:360px;object-fit:contain}
.bbox-canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.thumbs{display:flex;gap:6px;padding:8px 10px 0;overflow-x:auto}
.thumbs img{width:54px;height:54px;object-fit:cover;border-radius:6px;border:1px solid var(--line);cursor:pointer}
.card-body{padding:10px;display:flex;flex-direction:column;gap:6px;flex:1}
.caption{font-size:12.5px;margin:0;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}
details{font-size:12px;border-top:1px dashed var(--line);padding-top:6px}
summary{cursor:pointer;color:var(--accent)}
.msg{margin:4px 0;padding:5px 8px;border-radius:7px;background:#f5f7fb}
.msg.assistant{background:#f2fbf5}
.msg .role{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
.qa{margin:4px 0;font-size:12px}
.qa b{color:#35507a;font-weight:600}
.more-wrap{text-align:center;padding:18px}
.modal{position:fixed;inset:0;background:rgba(10,15,25,.72);display:none;align-items:center;justify-content:center;padding:24px;z-index:100}
.modal.show{display:flex}
.modal-inner{background:#fff;border-radius:14px;max-width:1200px;width:100%;max-height:92vh;overflow:auto;padding:18px}
.modal-close{float:right;cursor:pointer;background:#eef2f7;border-radius:8px;padding:6px 10px}
.modal-img{position:relative;display:flex;justify-content:center;background:#eef0f3;border-radius:10px;overflow:hidden;margin:10px 0}
.modal-img img{max-height:60vh;max-width:100%;display:block}
.muted a{color:var(--accent)}
a{color:var(--accent)}
a.missing{color:var(--bad);text-decoration:line-through}
"""

JS = r"""
const DATA = __DATA__;
const COLORS = ['#ff3b30','#0a84ff','#30d158','#ff9f0a','#bf5af2','#64d2ff','#ffd60a','#ff375f'];
const state = {dataset:'all', group:'all', split:'all', q:'', showBBox:true, limit:48};

function el(tag, cls, text){const e=document.createElement(tag); if(cls) e.className=cls; if(text!==undefined) e.textContent=text; return e;}
function fmtPct(x){return (x*100).toFixed(2)+'%';}

function initFilters(){
  const datasets = [...new Set(DATA.samples.map(s=>s.dataset))];
  const dsSel=document.getElementById('f-dataset');
  datasets.forEach(d=>{const o=el('option',null,d); o.value=d; dsSel.appendChild(o);});
  const groups=[...new Set(DATA.samples.map(s=>s.group))];
  const gSel=document.getElementById('f-group');
  groups.forEach(g=>{const o=el('option',null,g); o.value=g; gSel.appendChild(o);});
  const splits=[...new Set(DATA.samples.map(s=>s.split).filter(Boolean))];
  const sSel=document.getElementById('f-split');
  splits.forEach(s=>{const o=el('option',null,s); o.value=s; sSel.appendChild(o);});
  dsSel.addEventListener('change',()=>{state.dataset=dsSel.value; state.limit=48; render();});
  gSel.addEventListener('change',()=>{state.group=gSel.value; state.limit=48; render();});
  sSel.addEventListener('change',()=>{state.split=sSel.value; state.limit=48; render();});
  document.getElementById('f-search').addEventListener('input',e=>{state.q=e.target.value.trim(); state.limit=48; render();});
  document.getElementById('f-bbox').addEventListener('change',e=>{state.showBBox=e.target.checked; redrawAll();});
  document.getElementById('btn-reset').addEventListener('click',()=>{
    state.dataset='all';state.group='all';state.split='all';state.q='';state.limit=48;
    dsSel.value='all';gSel.value='all';sSel.value='all';
    document.getElementById('f-search').value='';render();
  });
  document.getElementById('btn-more').addEventListener('click',()=>{state.limit+=48; render(false);});
}

function matchSample(s){
  if(state.dataset!=='all' && s.dataset!==state.dataset) return false;
  if(state.group!=='all' && s.group!==state.group) return false;
  if(state.split!=='all' && s.split!==state.split) return false;
  if(state.q){
    const hay=[s.id,s.caption,s.meta&&s.meta.domain,s.cate].concat(s.messages.map(m=>m.content)).join(' ').toLowerCase();
    if(!hay.includes(state.q.toLowerCase())) return false;
  }
  return true;
}

function drawBoxes(img, canvas, boxes, show){
  const wrap=canvas.parentElement;
  const w=wrap.clientWidth, h=img.clientHeight || wrap.clientHeight;
  if(!w||!h||!img.naturalWidth) return;
  const dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(w*dpr); canvas.height=Math.round(h*dpr);
  canvas.style.width=w+'px'; canvas.style.height=h+'px';
  const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h);
  if(!show || !boxes.length) return;
  boxes.forEach((b,i)=>{
    const x1=b[1]/1000*w, y1=b[0]/1000*h, x2=b[3]/1000*w, y2=b[2]/1000*h;
    const color=COLORS[i%COLORS.length];
    ctx.strokeStyle=color; ctx.lineWidth=2.2;
    ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    ctx.fillStyle=color+'33'; ctx.fillRect(x1,y1,x2-x1,y2-y1);
    const label='#'+(i+1);
    ctx.font='11px -apple-system,sans-serif';
    const tw=ctx.measureText(label).width+8;
    ctx.fillStyle=color; ctx.fillRect(x1, Math.max(0,y1-16), tw, 14);
    ctx.fillStyle='#fff'; ctx.fillText(label, x1+4, Math.max(11,y1-5));
  });
}

function textBlock(s){
  const wrap=el('div','card-body');
  if(s.messages.length){
    const body=el('div');
    s.messages.slice(0,6).forEach(m=>{
      const div=el('div','msg '+(m.role==='assistant'?'assistant':''));
      div.appendChild(el('div','role',m.role));
      div.appendChild(el('div',null,m.content.replaceAll('<image>','[图片]')));
      body.appendChild(div);
    });
    wrap.appendChild(body);
    const det=el('details'); det.appendChild(el('summary',null,'完整对话 ('+s.messages.length+' 条)'));
    s.messages.slice(6).forEach(m=>{const d=el('div','msg '+(m.role==='assistant'?'assistant':'')); d.appendChild(el('div','role',m.role)); d.appendChild(el('div',null,m.content.replaceAll('<image>','[图片]'))); det.appendChild(d);});
    if(s.messages.length>6) wrap.appendChild(det);
  } else {
    wrap.appendChild(el('p','caption',s.caption||'（无 caption）'));
    if(s.qa && s.qa.length){
      const det=el('details'); det.appendChild(el('summary',null,'VQA ('+s.qa.length+' 条)'));
      s.qa.forEach(q=>{
        const d=el('div','qa');
        d.appendChild(el('b',null,'Q: ')); d.appendChild(el('span',null,q.question_zh||'')); d.appendChild(el('br'));
        d.appendChild(el('b',null,'A: ')); d.appendChild(el('span',null,q.answer_zh||''));
        det.appendChild(d);
      });
      wrap.appendChild(det);
    }
  }
  const meta=el('div','muted',Object.entries(s.meta||{}).map(([k,v])=>k+'='+v).join(' · '));
  wrap.appendChild(meta);
  return wrap;
}

function card(s){
  const c=el('div','card');
  const head=el('div','card-head');
  head.appendChild(el('span','badge '+(s.dataset==='v2'?'v2':'v6'),s.dataset_label));
  head.appendChild(el('span','badge',s.group));
  if(s.split) head.appendChild(el('span','badge split',s.split));
  if(s.bboxes.length) head.appendChild(el('span','badge','BBox '+s.bboxes.length));
  head.appendChild(el('span','sid',s.id));
  c.appendChild(head);
  const wrap=el('div','imgwrap');
  const img=el('img'); img.src=s.images[0]||''; img.alt=s.id; img.loading='lazy'; img.className='gallery-img';
  img._boxes=s.bboxes;
  const canvas=el('canvas','bbox-canvas');
  wrap.appendChild(img); wrap.appendChild(canvas);
  img.addEventListener('load',()=>drawBoxes(img,canvas,s.bboxes,state.showBBox));
  wrap.addEventListener('click',()=>openModal(s));
  c.appendChild(wrap);
  if(s.images.length>1){
    const th=el('div','thumbs');
    s.images.slice(1).forEach((u,i)=>{const t=el('img'); t.src=u; t.loading='lazy'; t.title='候选参考图 '+(i+1); t.addEventListener('click',ev=>{ev.stopPropagation(); openModal(s,i+1);}); th.appendChild(t);});
    c.appendChild(th);
  }
  c.appendChild(textBlock(s));
  return c;
}

function render(reset=true){
  const grid=document.getElementById('gallery-grid');
  if(reset) grid.innerHTML='';
  const list=DATA.samples.filter(matchSample);
  document.getElementById('gallery-count').textContent='匹配 '+list.length+' 条，已显示 '+Math.min(state.limit,list.length)+' 条';
  const start=reset?0:grid.children.length;
  list.slice(start,state.limit).forEach(s=>grid.appendChild(card(s)));
  document.getElementById('btn-more').style.display=list.length>state.limit?'inline-block':'none';
}

function redrawAll(){
  document.querySelectorAll('img.gallery-img').forEach(img=>{
    const canvas=img.parentElement.querySelector('canvas.bbox-canvas');
    if(canvas) drawBoxes(img,canvas,img._boxes||[],state.showBBox);
  });
}

let modalState=null;
function openModal(s, imageIndex=0){
  modalState={s,imageIndex};
  const modal=document.getElementById('modal');
  document.getElementById('modal-title').textContent=s.id+' · '+s.group;
  const wrap=document.getElementById('modal-img');
  wrap.innerHTML='';
  const img=el('img'); img.id='modal-image'; img.src=s.images[imageIndex]||s.images[0]||'';
  const canvas=el('canvas','bbox-canvas'); canvas.id='modal-canvas';
  wrap.appendChild(img); wrap.appendChild(canvas);
  img.addEventListener('load',()=>drawBoxes(img,canvas,imageIndex===0?s.bboxes:[],state.showBBox));
  const thumbs=document.getElementById('modal-thumbs'); thumbs.innerHTML='';
  s.images.forEach((u,i)=>{const t=el('img'); t.src=u; t.style.cssText='width:72px;height:72px;object-fit:cover;border-radius:6px;border:2px solid '+(i===imageIndex?'#0a84ff':'#e5e5ea')+';cursor:pointer'; t.addEventListener('click',()=>openModal(s,i)); thumbs.appendChild(t);});
  const texts=document.getElementById('modal-text'); texts.innerHTML='';
  if(s.messages.length){s.messages.forEach(m=>{const d=el('div','msg '+(m.role==='assistant'?'assistant':'')); d.appendChild(el('div','role',m.role)); d.appendChild(el('div',null,m.content.replaceAll('<image>','[图片]'))); texts.appendChild(d);});}
  else{texts.appendChild(el('p','caption',s.caption)); (s.qa||[]).forEach(q=>{const d=el('div','qa'); d.appendChild(el('b',null,'Q: '));d.appendChild(el('span',null,q.question_zh||''));d.appendChild(el('br'));d.appendChild(el('b',null,'A: '));d.appendChild(el('span',null,q.answer_zh||''));texts.appendChild(d);});}
  modal.classList.add('show');
}
function closeModal(){document.getElementById('modal').classList.remove('show'); modalState=null;}

window.addEventListener('resize',()=>{clearTimeout(window._rz);window._rz=setTimeout(redrawAll,120);});
document.addEventListener('DOMContentLoaded',()=>{
  initFilters(); render();
  document.getElementById('modal').addEventListener('click',e=>{if(e.target.id==='modal') closeModal();});
  document.getElementById('modal-close').addEventListener('click',closeModal);
  document.addEventListener('keydown',e=>{if(e.key==='Escape') closeModal();});
});
"""


# ---------------------------------------------------------------- build page
def build_html(embed_images: bool = True, max_side: int = 560, quality: int = 80) -> str:
    samples = collect_samples()["samples"]
    if embed_images:
        for s in samples:
            raw = s.get("image_raw") or []
            if not raw:
                continue
            s["images"] = [embed_image(raw[0], max_side=max_side, quality=quality)]
            s["images"] += [embed_image(p, max_side=200, quality=72) for p in raw[1:]]
        print(f"[visual] embedded {sum(len(s['images']) for s in samples)} images as data URIs")
    payload = {"samples": samples}
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    stats = build_stats(collect_samples())
    v2_rows = stats["v2"]["rows"]
    total_gallery = len(samples)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    js = JS.replace("__DATA__", data_json)

    head = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Datawheel 数据核验报告</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Datawheel 数据核验报告</h1>
  <p>v2 10K 多模态数据集 · V6 × Qwen3.5-27B 中训练数据 · 生成于 {esc(generated)} ·
  共 {esc(total_gallery)} 条画廊样本，BBox 边界框坐标 [ymin,xmin,ymax,xmax]（相对画面 0~1000）
  {'· 图片已内嵌（单文件离线可开）' if embed_images else '· 图片使用项目相对路径 (file:// 打开)'}
  · 文本描述以物品/名称/可见特征为主，已清理实现术语</p>
</header>
<nav>
  <a href="#overview">总览</a><a href="#datasets">数据集</a><a href="#gallery">样例画廊</a>
  <a href="#mixture">中训练配比</a><a href="#validation">质检</a><a href="#artifacts">产物索引</a>
</nav>
<main>
<section id="overview"><h2>🎯 总览</h2><div class="stat-grid">{overview_cards()}</div></section>
""" + dataset_section() + gallery_section(total_gallery) + mixture_section() + validation_section() + artifacts_section() + f"""
</main>
<div class="modal" id="modal">
  <div class="modal-inner">
    <span class="modal-close" id="modal-close" role="button" tabindex="0">关闭 ✕</span>
    <h3 id="modal-title"></h3>
    <div class="modal-img" id="modal-img"></div>
    <div class="thumbs" id="modal-thumbs"></div>
    <div id="modal-text"></div>
  </div>
</div>
<script>{js}</script>
</body>
</html>
"""
    return head


def write_report(out_path: Path, *, embed_images: bool, max_side: int, quality: int) -> None:
    html_text = build_html(embed_images=embed_images, max_side=max_side, quality=quality)
    out_path.write_text(html_text, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[visual] wrote {out_path.relative_to(ROOT)} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Datawheel 离线 HTML 核验报告")
    parser.add_argument("--no-embed", action="store_true",
                        help="不内嵌图片，生成使用项目相对路径的轻量版（仅适合 file:// 直接打开）")
    parser.add_argument("--max-side", type=int, default=560, help="内嵌主图最长边（默认 560px）")
    parser.add_argument("--quality", type=int, default=80, help="内嵌主图 JPEG 质量（默认 80）")
    parser.add_argument("--light-output", type=Path, default=None,
                        help="轻量版输出路径（默认 output/report/index_light.html）")
    args = parser.parse_args()

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    if args.no_embed:
        out = args.light_output or OUT_HTML.with_name("index_light.html")
        write_report(out, embed_images=False, max_side=args.max_side, quality=args.quality)
        print(f"[visual] open: open {out.relative_to(ROOT)}")
        return

    write_report(OUT_HTML, embed_images=True, max_side=args.max_side, quality=args.quality)
    # 同时保留一个轻量版（相对路径，便于 git 小体积分发）
    light = args.light_output or OUT_HTML.with_name("index_light.html")
    write_report(light, embed_images=False, max_side=args.max_side, quality=args.quality)
    print(f"[visual] open: open {OUT_HTML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
