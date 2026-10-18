#!/usr/bin/env python3
"""Qwen-Image 2.1 批量生成/编辑客户端（离线局域网调用）。

支持两种服务协议：
1. json   —— 自定义/自建 FastAPI：POST {base}/generate、POST {base}/edit
2. openai —— OpenAI 兼容：/v1/images/generations、/v1/images/edits

设计目标：给 Datawheel 生成"空白商品 / 背景 / 光照"素材，图案仍由确定性
compositor 合成；编辑时可保护指定 bbox 区域，防止模型改坏商品或印花。

用法示例：
  # 生成空白商品底图
  .venv/bin/python tools/qwen_image_ops.py generate \
      --base-url http://127.0.0.1:8000 --protocol json \
      --plan plans/blank_products.jsonl --out-dir output/qwen_gen/blank_products

  # 背景/光照编辑（mask 白色=可编辑，黑色=保护）
  .venv/bin/python tools/qwen_image_ops.py edit \
      --base-url http://127.0.0.1:8000 --protocol json \
      --plan plans/bg_edits.jsonl --out-dir output/qwen_edit/bg \
      --protect-bbox 0,0,1000,1000

  # 质检
  .venv/bin/python tools/qwen_image_ops.py verify \
      --manifest output/qwen_gen/blank_products/manifest.jsonl --min-size 512
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageChops

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "qwen-image-2.1"
DEFAULT_GENERATE_PROMPT = (
    "professional e-commerce product photo of a blank product, front view, "
    "no logo, no text, no graphic, no print, clean white background, "
    "softbox studio lighting, sharp focus, high detail"
)
DEFAULT_NEGATIVE = (
    "logo, text, watermark, letters, brand mark, graphic, print, pattern, "
    "distorted anatomy, extra fingers, mannequin face, low quality, blurry"
)
_LOCK = threading.Lock()


# ---------------------------------------------------------------- common IO
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def b64_file(path: Path) -> Tuple[str, str]:
    data = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return base64.b64encode(data).decode("ascii"), mime


def resolve_path(value: str, plan_path: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    # 优先相对当前工作目录，其次相对 plan 文件目录，最后相对项目根
    if p.exists():
        return p.resolve()
    cand = (plan_path.parent / p).resolve()
    if cand.exists():
        return cand
    return (PROJECT_ROOT / p).resolve()


def parse_bbox_list(values: Optional[List[str]]) -> List[Tuple[int, int, int, int]]:
    boxes = []
    for raw in values or []:
        parts = [int(x.strip()) for x in raw.split(",")]
        if len(parts) != 4 or not all(0 <= x <= 1000 for x in parts):
            raise ValueError(f"非法 protect-bbox: {raw}（应为 ymin,xmin,ymax,xmax，0~1000）")
        if parts[0] >= parts[2] or parts[1] >= parts[3]:
            raise ValueError(f"非法 protect-bbox 顺序: {raw}")
        boxes.append(tuple(parts))
    return boxes


def image_from_bytes(data: bytes) -> Image.Image:
    from io import BytesIO
    return Image.open(BytesIO(data)).convert("RGB")


def mean_abs_diff(image_a: Image.Image, image_b: Image.Image,
                  bbox: Optional[Tuple[int, int, int, int]] = None) -> float:
    if image_a.size != image_b.size:
        return 1.0
    if bbox:
        w, h = image_a.size
        x1 = int(bbox[1] * w / 1000)
        y1 = int(bbox[0] * h / 1000)
        x2 = int(bbox[3] * w / 1000)
        y2 = int(bbox[2] * h / 1000)
        image_a = image_a.crop((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)))
        image_b = image_b.crop((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)))
    diff = ImageChops.difference(image_a, image_b)
    # 用灰度平均差，避免单通道噪声放大
    gray = diff.convert("L")
    return sum(gray.getdata()) / (255.0 * max(1, gray.width * gray.height))


# ---------------------------------------------------------------- HTTP
def _request(url: str, data: Optional[bytes], headers: Dict[str, str],
             timeout: float, method: Optional[str] = None) -> Tuple[bytes, str]:
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"连接失败 {url}: {exc}") from exc


def http_json(url: str, payload: Dict[str, Any], api_key: Optional[str],
              timeout: float) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    raw, _ = _request(url, body, headers, timeout)
    return json.loads(raw.decode("utf-8"))


def _multipart_body(fields: Dict[str, str],
                    files: List[Tuple[str, str, bytes, str]]) -> Tuple[bytes, str]:
    boundary = "----qwenimage" + uuid.uuid4().hex
    parts: List[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(str(value).encode("utf-8") + b"\r\n")
    for name, filename, data, mime in files:
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        parts.append(f"Content-Type: {mime}\r\n\r\n".encode())
        parts.append(data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def http_multipart(url: str, fields: Dict[str, str],
                   files: List[Tuple[str, str, bytes, str]],
                   api_key: Optional[str], timeout: float) -> Any:
    body, boundary = _multipart_body(fields, files)
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    raw, ctype = _request(url, body, headers, timeout)
    if ctype.startswith("image/"):
        return {"__image_bytes__": raw}
    return json.loads(raw.decode("utf-8"))


def endpoint(base_url: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return base_url.rstrip("/") + "/" + path.lstrip("/")


# ---------------------------------------------------------------- response parsing
def _iter_images(obj: Any):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in ("b64", "b64_json", "base64", "image_base64") and isinstance(value, str):
                yield ("b64", value)
            elif key.lower() == "url" and isinstance(value, str):
                yield ("url", value)
            elif key == "__image_bytes__" and isinstance(value, (bytes, bytearray)):
                yield ("bytes", bytes(value))
            elif isinstance(value, (dict, list)):
                yield from _iter_images(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_images(item)


def extract_first_image(payload: Any) -> Tuple[str, Any]:
    seen = set()
    for kind, value in _iter_images(payload):
        if kind == "b64":
            try:
                raw = base64.b64decode(value)
            except Exception:
                continue
            key = hashlib.sha1(raw).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            return "bytes", raw
        if kind == "url":
            if value in seen:
                continue
            seen.add(value)
            return kind, value
        if kind == "bytes":
            return kind, value
    raise RuntimeError(f"响应中未找到图片: {str(payload)[:500]}")


def download_bytes(url: str, timeout: float) -> bytes:
    raw, ctype = _request(url, None, {}, timeout)
    if not ctype.startswith("image/") and not raw:
        raise RuntimeError(f"下载图片为空或非图片: {url}")
    return raw


def save_image(payload: Any, out_path: Path, timeout: float) -> bytes:
    kind, value = extract_first_image(payload)
    if kind == "url":
        raw = download_bytes(value, timeout)
    elif kind == "b64":
        raw = base64.b64decode(value)
    else:
        raw = value
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)
    image_from_bytes(raw)  # 解码校验
    return raw


# ---------------------------------------------------------------- request builders
def build_generate_payload(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "prompt": row.get("prompt", DEFAULT_GENERATE_PROMPT),
        "negative_prompt": row.get("negative_prompt", DEFAULT_NEGATIVE),
        "width": int(row.get("width", 1024)),
        "height": int(row.get("height", 1024)),
        "steps": int(row.get("steps", args.steps)),
        "guidance": float(row.get("guidance", args.guidance)),
        "seed": int(row.get("seed", args.seed)),
        "model": row.get("model", args.model),
    }


def build_edit_payload(path: Path, plan_path: Path, row: Dict[str, Any],
                       args: argparse.Namespace) -> Dict[str, Any]:
    image_path = resolve_path(row["image"], plan_path)
    image_b64, image_mime = b64_file(image_path)
    payload: Dict[str, Any] = {
        "image_b64": image_b64,
        "image_mime": image_mime,
        "prompt": row["prompt"],
        "negative_prompt": row.get("negative_prompt", DEFAULT_NEGATIVE),
        "steps": int(row.get("steps", args.steps)),
        "guidance": float(row.get("guidance", args.guidance)),
        "seed": int(row.get("seed", args.seed)),
        "model": row.get("model", args.model),
    }
    if row.get("mask"):
        mask_path = resolve_path(row["mask"], plan_path)
        mask_b64, mask_mime = b64_file(mask_path)
        payload["mask_b64"] = mask_b64
        payload["mask_mime"] = mask_mime
    return payload


# ---------------------------------------------------------------- workers
def _retry(fn, args: argparse.Namespace, label: str):
    last = None
    for attempt in range(1, args.retries + 1):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if attempt < args.retries:
                time.sleep(min(2 ** attempt, 10))
                print(f"[retry] {label} attempt {attempt} failed: {exc}", file=sys.stderr)
    raise RuntimeError(f"{label} 失败: {last}")


def process_generate(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    plan_path = Path(args.plan)
    sid = str(row.get("id") or f"gen_{uuid.uuid4().hex[:8]}")
    out_path = Path(args.out_dir) / "images" / f"{sid}.png"
    if args.skip_existing and out_path.exists():
        return {"id": sid, "status": "skipped", "path": str(out_path)}
    payload = build_generate_payload(row, args)
    if args.dry_run:
        return {"id": sid, "status": "dry_run", "payload": {k: v for k, v in payload.items() if k != "image_b64"}}
    started = time.time()
    if args.protocol == "openai":
        url = endpoint(args.base_url, args.generate_path or "/v1/images/generations")
        body = {
            "model": args.model,
            "prompt": payload["prompt"],
            "n": 1,
            "size": f'{payload["width"]}x{payload["height"]}',
            "response_format": "b64_json",
        }
        for key in ("negative_prompt", "steps", "guidance", "seed"):
            body[key] = payload[key]
        resp = _retry(lambda: http_json(url, body, args.api_key, args.timeout), args, sid)
    else:
        url = endpoint(args.base_url, args.generate_path or "/generate")
        resp = _retry(lambda: http_json(url, payload, args.api_key, args.timeout), args, sid)
    raw = save_image(resp, out_path, args.timeout)
    return {
        "id": sid, "status": "ok", "path": str(out_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "prompt": payload["prompt"], "negative_prompt": payload["negative_prompt"],
        "seed": payload["seed"], "elapsed_sec": round(time.time() - started, 2),
    }


def process_edit(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    plan_path = Path(args.plan)
    sid = str(row.get("id") or f"edit_{uuid.uuid4().hex[:8]}")
    out_path = Path(args.out_dir) / "images" / f"{sid}.png"
    if args.skip_existing and out_path.exists():
        return {"id": sid, "status": "skipped", "path": str(out_path)}
    image_path = resolve_path(row["image"], plan_path)
    payload = build_edit_payload(out_path, plan_path, row, args)
    if args.dry_run:
        return {"id": sid, "status": "dry_run",
                "image": str(image_path), "prompt": payload["prompt"]}
    started = time.time()
    if args.protocol == "openai":
        url = endpoint(args.base_url, args.edit_path or "/v1/images/edits")
        fields = {
            "model": args.model,
            "prompt": payload["prompt"],
            "size": row.get("size", "1024x1024"),
            "n": "1",
        }
        for key in ("negative_prompt", "steps", "guidance", "seed"):
            if key in payload:
                fields[key] = str(payload[key])
        files = [("image", image_path.name, image_path.read_bytes(),
                  mimetypes.guess_type(str(image_path))[0] or "image/png")]
        if row.get("mask"):
            mask_path = resolve_path(row["mask"], plan_path)
            files.append(("mask", mask_path.name, mask_path.read_bytes(),
                          mimetypes.guess_type(str(mask_path))[0] or "image/png"))
        resp = _retry(lambda: http_multipart(url, fields, files, args.api_key, args.timeout), args, sid)
    else:
        url = endpoint(args.base_url, args.edit_path or "/edit")
        resp = _retry(lambda: http_json(url, payload, args.api_key, args.timeout), args, sid)
    raw = save_image(resp, out_path, args.timeout)

    quality: Dict[str, Any] = {}
    try:
        src = image_from_bytes(image_path.read_bytes())
        edited = image_from_bytes(raw)
        quality["source_size"] = list(src.size)
        quality["edited_size"] = list(edited.size)
        quality["size_match"] = src.size == edited.size
        if args.protect_bbox and src.size == edited.size:
            diffs = [round(mean_abs_diff(src, edited, b), 4) for b in args.protect_bbox]
            quality["protect_diff"] = diffs
            quality["protect_ok"] = all(d <= args.protect_threshold for d in diffs)
    except Exception as exc:
        quality["error"] = str(exc)

    if args.protect_bbox and not quality.get("protect_ok", False):
        rejected = Path(args.out_dir) / "rejected"
        rejected.mkdir(parents=True, exist_ok=True)
        reject_path = rejected / out_path.name
        if out_path.exists():
            out_path.replace(reject_path)
        return {"id": sid, "status": "rejected", "reason": "protected_region_changed",
                "path": str(reject_path), "quality": quality,
                "source_image": str(image_path), "prompt": payload["prompt"]}
    return {
        "id": sid, "status": "ok", "path": str(out_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_image": str(image_path), "source_sha256": file_sha256(image_path),
        "prompt": payload["prompt"], "seed": payload["seed"],
        "elapsed_sec": round(time.time() - started, 2), "quality": quality,
    }


# ---------------------------------------------------------------- commands
def run_batch(args: argparse.Namespace) -> None:
    plan = Path(args.plan)
    if not plan.exists():
        raise SystemExit(f"plan 不存在: {plan}")
    rows = load_jsonl(plan)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("plan 为空")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    worker = process_generate if args.command == "generate" else process_edit
    results: List[Dict[str, Any]] = []
    if args.concurrency <= 1:
        for row in rows:
            results.append(worker(row, args))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(worker, row, args): row for row in rows}
            for fut in as_completed(futures):
                row = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    results.append({"id": row.get("id"), "status": "error", "error": str(exc)})
                with _LOCK:
                    ok = sum(1 for r in results if r.get("status") == "ok")
                    print(f"[qwen] {len(results)}/{len(rows)} done, ok={ok}", end="\r")
        print()
    manifest = out_dir / "manifest.jsonl"
    write_jsonl(manifest, results)
    ok = sum(1 for r in results if r.get("status") == "ok")
    rejected = sum(1 for r in results if r.get("status") == "rejected")
    errors = sum(1 for r in results if r.get("status") == "error")
    print(f"[qwen] plan={len(rows)} ok={ok} rejected={rejected} errors={errors}")
    print(f"[qwen] manifest -> {manifest}")
    if errors:
        raise SystemExit(1)


def run_verify(args: argparse.Namespace) -> None:
    rows = load_jsonl(Path(args.manifest))
    errors = []
    for row in rows:
        path = Path(row.get("path", ""))
        if row.get("status") != "ok":
            continue
        if not path.exists():
            errors.append({"id": row.get("id"), "error": "missing_file", "path": str(path)})
            continue
        try:
            img = Image.open(path)
            if args.min_size and min(img.size) < args.min_size:
                errors.append({"id": row.get("id"), "error": "too_small", "size": img.size})
        except Exception as exc:
            errors.append({"id": row.get("id"), "error": "decode_error", "detail": str(exc)})
    print(f"[verify] checked={sum(1 for r in rows if r.get('status')=='ok')} errors={len(errors)}")
    for err in errors[:20]:
        print(" -", err)
    if errors:
        raise SystemExit(1)


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen-Image 2.1 batch client for Datawheel")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL", ""))
        p.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY"))
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--protocol", choices=["json", "openai"], default="json")
        p.add_argument("--generate-path", default=None)
        p.add_argument("--edit-path", default=None)
        p.add_argument("--plan", required=True)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--concurrency", type=int, default=2)
        p.add_argument("--retries", type=int, default=3)
        p.add_argument("--timeout", type=float, default=300.0)
        p.add_argument("--steps", type=int, default=30)
        p.add_argument("--guidance", type=float, default=4.0)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--skip-existing", action="store_true")
        p.add_argument("--dry-run", action="store_true")

    gen = sub.add_parser("generate")
    common(gen)
    gen.set_defaults(func=run_batch)

    edit = sub.add_parser("edit")
    common(edit)
    edit.add_argument("--protect-bbox", action="append", default=None,
                      help="保护区域 ymin,xmin,ymax,xmax（0~1000），可重复")
    edit.add_argument("--protect-threshold", type=float, default=0.02,
                      help="保护区域平均像素差阈值（0~1，默认 0.02）")
    edit.set_defaults(func=run_batch)

    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--min-size", type=int, default=512)
    verify.set_defaults(func=run_verify)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command in ("generate", "edit"):
        if not args.base_url:
            parser.error("--base-url 或环境变量 QWEN_BASE_URL 必填")
        args.protect_bbox = parse_bbox_list(getattr(args, "protect_bbox", None))
        if args.concurrency < 1:
            parser.error("--concurrency 必须 >=1")
    args.func(args)


if __name__ == "__main__":
    main()
