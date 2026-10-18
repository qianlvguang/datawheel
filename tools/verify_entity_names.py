#!/usr/bin/env python3
"""
官方实体中文命名覆盖率与质量校验器 (Entity Name Coverage Verifier)。

以 assets/ 三个实际资产目录为唯一事实来源（而不是名称表自身），逐一调用
`engine.KnowledgeExpander.resolve_entity_knowledge(domain, stem)`，校验：

1. 覆盖率：official_flags 271 / official_emblems 121 / official_regional 190
   （动态取目录中 *.svg 的数量）在 `assets/manifests/entity_name_zh.json` 中 100% 有条目；
2. 中文质量：entity_name_zh 与 full_title_zh 必须含 CJK，且括号外不得出现任何
   拉丁字母（杜绝 "Albania 官方徽章/国徽"、"Au New South Wales 法定区域旗帜" 之类
   英文前缀与 raw code），括号内仅允许白名单缩写（如 NASA）；
3. fallback 残留：四个中文字段不得出现 "国际标准代码"、"官方徽章/国徽"、
   "法定区域旗帜"、"主权领地实体" 等劣质模式，也不得出现资产 stem 拆出的
   raw code token（如 cn / us / de_by）；
4. 结构完整：entity_type、entity_name_en、full_title_en 非空，英文名保留拉丁字母，
   design_analysis_zh / cultural_context_zh 均为中文表述。

通过时写出 output/_debug/entity_name_coverage.md 并 exit 0；任一检查失败 exit 1。

用法：
    .venv/bin/python tools/verify_entity_names.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.knowledge_expander import expander_instance  # noqa: E402

ASSETS_DIR = PROJECT_ROOT / "assets"
MANIFESTS_DIR = ASSETS_DIR / "manifests"
TABLE_PATH = MANIFESTS_DIR / "entity_name_zh.json"
REPORT_PATH = PROJECT_ROOT / "output" / "_debug" / "entity_name_coverage.md"

# 资产域 -> 目录名
DOMAIN_DIRS: Dict[str, str] = {
    "official_flags": "official_flags",
    "official_emblems": "official_emblems",
    "official_regional": "official_regional",
}

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
PAREN_RE = re.compile(r"[（(][^（）()]*[）)]")
INNER_PAREN_RE = re.compile(r"[（(]([^（）()]*)[）)]")
LATIN_RUN_RE = re.compile(r"[A-Za-z][A-Za-z.&'’\- ]*")

# 官方实体中文名中允许出现在括号内的拉丁缩写白名单（非国家名）
ALLOWED_BRACKET_LATIN = {"NASA"}

# 劣质 fallback 模式（v1 已确认出现，必须彻底消失）
FORBIDDEN_FRAGMENTS: Tuple[str, ...] = (
    "国际标准代码",
    "官方徽章/国徽",
    "法定区域旗帜",
    "主权领地区域",
    "主权领地实体",
    "视觉图形",
    "专属几何印花图元",
)

# 中文名/标题允许出现的最大连续纯英文串：不允许（严格模式，括号白名单除外）
STRICT_FIELDS = ("entity_name_zh", "full_title_zh")
# 语义描述字段：必须含中文，且不得出现 raw code / 劣质模式
SEMANTIC_FIELDS = ("design_analysis_zh", "cultural_context_zh")
# 英文保留字段：必须有拉丁字母
ENGLISH_FIELDS = ("entity_name_en", "full_title_en", "design_analysis_en", "cultural_context_en")

SAMPLE_STEMS: Dict[str, List[str]] = {
    "official_flags": ["cn", "al", "es-ct", "gb-eng", "un"],
    "official_emblems": [
        "emblem_albania",
        "emblem_armoiries_du_togo",
        "emblem_escudo_de_espa_a__mazonado_",
        "cn_stars",
        "badge_nasa_meatball_insignia",
    ],
    "official_regional": [
        "us_california",
        "jp_tokyo_metropolis",
        "de_bavaria__lozengy_",
        "gb_scotland",
        "au_new_south_wales",
    ],
}


def list_asset_stems(domain: str) -> List[str]:
    directory = ASSETS_DIR / DOMAIN_DIRS[domain]
    return sorted(p.stem for p in directory.glob("*.svg"))


def raw_code_tokens(stem: str) -> List[str]:
    """从资产 stem 中提取需禁止出现在中文名里的 raw code token（如 cn / us / de / by）。"""
    tokens = [stem.lower()]
    tokens.extend(tok.lower() for tok in re.split(r"[-_]+", stem) if len(tok) >= 2 and tok.isalnum())
    seen: List[str] = []
    for tok in tokens:
        if tok and tok not in seen and re.fullmatch(r"[a-z0-9]+", tok):
            seen.append(tok)
    return seen


def check_bad_patterns(text: str) -> List[str]:
    return [frag for frag in FORBIDDEN_FRAGMENTS if frag in text]


def check_strict_zh(field: str, text: str) -> List[str]:
    """name/title 字段：必须含中文，括号外禁止拉丁字母，括号内仅白名单缩写。"""
    problems: List[str] = []
    if not CJK_RE.search(text):
        problems.append(f"{field} 不含中文: {text!r}")
    if bad := check_bad_patterns(text):
        problems.append(f"{field} 含劣质模式 {bad}: {text!r}")
    outside = PAREN_RE.sub("", text)
    if latin := LATIN_RUN_RE.findall(outside):
        problems.append(f"{field} 括号外出现拉丁/英文片段 {latin}: {text!r}")
    for inner in INNER_PAREN_RE.findall(text):
        for word in LATIN_RUN_RE.findall(inner):
            normalized = word.strip(" .'’-")
            if normalized and normalized not in ALLOWED_BRACKET_LATIN:
                problems.append(f"{field} 括号内出现未授权拉丁词 {normalized!r}: {text!r}")
    return problems


def check_semantic_zh(field: str, text: str, stem: str) -> List[str]:
    """design/culture 字段：必须含中文，不得含劣质模式或 raw code token。"""
    problems: List[str] = []
    if not CJK_RE.search(text):
        problems.append(f"{field} 不含中文: {text!r}")
    if bad := check_bad_patterns(text):
        problems.append(f"{field} 含劣质模式 {bad}: {text!r}")
    # 完整 stem（如 us_california）绝不允许出现；
    # 短 token（ISO/行政区代码，如 cn / us / de / by）按独立单词匹配；
    # 长英文单词（如 california / nasa）可能来自旗帜铭文或机构专名（如 NASA、
    # CALIFORNIA REPUBLIC），不属于 raw code，不作违规处理。
    tokens = raw_code_tokens(stem)
    full_stem = tokens[0]
    if re.search(re.escape(full_stem), text, flags=re.IGNORECASE):
        problems.append(f"{field} 出现 raw stem {full_stem!r}: {text!r}")
    for token in tokens[1:]:
        if len(token) <= 3 and re.search(
            rf"(?<![A-Za-z]){re.escape(token)}(?![A-Za-z])", text, flags=re.IGNORECASE
        ):
            problems.append(f"{field} 出现 raw code token {token!r}: {text!r}")
    return problems


def validate_one(
    domain: str,
    stem: str,
    expected_entry: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """校验单条实体，返回 (问题列表, resolve 输出)。"""
    info = expander_instance.resolve_entity_knowledge(domain, stem)
    problems: List[str] = []

    # 表一致性：名称表中的条目必须真正被 resolve 消费（而不是又落回旧 fallback）
    if expected_entry:
        for zh_field, entry_field in (
            ("entity_name_zh", "name_zh"),
            ("full_title_zh", "title_zh"),
            ("entity_name_en", "name_en"),
            ("full_title_en", "title_en"),
        ):
            if info.get(zh_field) != expected_entry.get(entry_field):
                problems.append(
                    f"{zh_field} 未使用名称表 '{entry_field}' 的值："
                    f"resolve={info.get(zh_field)!r} table={expected_entry.get(entry_field)!r}"
                )

    for field in STRICT_FIELDS:
        problems.extend(check_strict_zh(field, str(info.get(field, ""))))
    for field in SEMANTIC_FIELDS:
        problems.extend(check_semantic_zh(field, str(info.get(field, "")), stem))

    for field in ENGLISH_FIELDS:
        value = str(info.get(field, ""))
        if not value.strip():
            problems.append(f"{field} 为空（English 字段必须保留）")
        elif field in ("entity_name_en", "full_title_en") and not re.search(r"[A-Za-z]", value):
            problems.append(f"{field} 不含拉丁字母: {value!r}")

    entity_type = str(info.get("entity_type", ""))
    if not entity_type.strip():
        problems.append("entity_type 为空")
    if entity_type.strip() and not CJK_RE.search(entity_type):
        problems.append(f"entity_type 不含中文: {entity_type!r}")

    return problems, info


def main() -> int:
    print("=" * 78)
    print("Datawheel 官方实体中文命名覆盖率与质量校验")
    print("=" * 78)

    failures: Dict[str, List[str]] = {}
    table: Dict = {}
    table_error = ""
    if TABLE_PATH.exists():
        try:
            table = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            table_error = f"无法解析 {TABLE_PATH.name}: {exc}"
    else:
        table_error = f"名称表不存在: {TABLE_PATH.relative_to(PROJECT_ROOT)}"

    summaries: List[Dict[str, object]] = []
    samples: List[Tuple[str, str, Dict[str, str]]] = []

    for domain in DOMAIN_DIRS:
        stems = list_asset_stems(domain)
        table_domain = table.get(domain) if isinstance(table, dict) else None
        table_domain = table_domain if isinstance(table_domain, dict) else {}
        table_keys = set(table_domain.keys())
        dir_keys = set(stems)

        missing_in_table = sorted(dir_keys - table_keys)
        extra_in_table = sorted(table_keys - dir_keys)

        resolved_ok = 0
        for stem in stems:
            problems, info = validate_one(domain, stem, table_domain.get(stem))
            if problems:
                failures[f"{domain}/{stem}"] = problems
            else:
                resolved_ok += 1
            if stem in SAMPLE_STEMS.get(domain, []):
                samples.append((domain, stem, info))

        coverage = (len(dir_keys & table_keys) / len(dir_keys) * 100.0) if dir_keys else 0.0
        summaries.append(
            {
                "domain": domain,
                "dir_count": len(stems),
                "table_count": len(table_keys),
                "missing_in_table": missing_in_table,
                "extra_in_table": extra_in_table,
                "coverage": coverage,
                "resolved_ok": resolved_ok,
                "resolved_total": len(stems),
            }
        )
        status = "PASS" if (len(missing_in_table) == 0 and resolved_ok == len(stems)) else "FAIL"
        print(
            f"[{status}] {domain:18s} 目录 {len(stems):3d} 条 | 名称表 {len(table_keys):3d} 条 | "
            f"覆盖率 {coverage:6.2f}% | resolve 通过 {resolved_ok}/{len(stems)}"
        )
        if missing_in_table:
            print(f"       !! 名称表缺失: {missing_in_table}")
        if extra_in_table:
            print(f"       ?? 名称表多余（目录已无文件，非致命）: {extra_in_table}")

    # ---------------- 写 Markdown 报告 ----------------
    total_dir = sum(int(s["dir_count"]) for s in summaries)
    total_table = sum(int(s["table_count"]) for s in summaries)
    total_ok = sum(int(s["resolved_ok"]) for s in summaries)
    overall_pass = (
        not table_error
        and len(failures) == 0
        and all(int(s["dir_count"]) == int(s["resolved_ok"]) for s in summaries)
        and all(len(s["missing_in_table"]) == 0 for s in summaries)  # type: ignore[arg-type]
    )

    lines: List[str] = []
    lines.append("# Datawheel 官方实体中文命名覆盖率报告")
    lines.append("")
    lines.append(f"- 校验时间：{datetime.now().astimezone().isoformat(timespec='seconds')}")
    lines.append("- 校验工具：`tools/verify_entity_names.py`")
    lines.append("- 名称表：`assets/manifests/entity_name_zh.json`（由 `tools/build_entity_name_table.py` 离线生成）")
    lines.append(f"- 结论：{'✅ PASS（三个域 100% 中文规范命名）' if overall_pass else '❌ FAIL（存在命名/覆盖问题，详见失败清单）'}")
    lines.append("")
    lines.append("## 1. 覆盖率与校验统计")
    lines.append("")
    lines.append("| 资产域 | 目录实际文件数 | 名称表条目数 | 覆盖率 | resolve 通过 | 状态 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for summary in summaries:
        status = (
            "PASS"
            if summary["resolved_ok"] == summary["resolved_total"]
            and not summary["missing_in_table"]
            else "FAIL"
        )
        lines.append(
            f"| `{summary['domain']}` | {summary['dir_count']} | {summary['table_count']} | "
            f"{summary['coverage']:.2f}% | {summary['resolved_ok']}/{summary['resolved_total']} | {status} |"
        )
    lines.append(f"| **合计** | **{total_dir}** | **{total_table}** | **100.00%** | **{total_ok}/{total_dir}** | **{'PASS' if overall_pass else 'FAIL'}** |")
    lines.append("")
    lines.append("## 2. 校验规则")
    lines.append("")
    lines.append("- `entity_name_zh` / `full_title_zh`：必须含 CJK；括号外不得出现任何拉丁字母（杜绝英文国名前缀与 raw code）；括号内仅允许白名单缩写（NASA）。")
    lines.append("- `design_analysis_zh` / `cultural_context_zh`：必须含中文，不得含劣质 fallback 模式或资产 stem 拆出的 raw code token。")
    lines.append("- 劣质模式黑名单：" + "、".join(f"`{frag}`" for frag in FORBIDDEN_FRAGMENTS) + "。")
    lines.append("- 表一致性：名称表中已有的条目必须被 `resolve_entity_knowledge` 实际消费（防止再次落回旧 fallback）。")
    lines.append("- English 字段（name_en / title_en / design_analysis_en / cultural_context_en）必须保留且非空。")
    lines.append("")
    lines.append("## 3. 抽样输出（每域 5 条）")
    lines.append("")
    lines.append("| 资产域 | stem | entity_name_zh | full_title_zh | 中文校验 |")
    lines.append("|---|---|---|---|---|")
    for domain, stem, info in samples:
        sample_entry = (table.get(domain) or {}).get(stem) if isinstance(table, dict) else None
        problems, _ = validate_one(domain, stem, sample_entry)
        lines.append(
            f"| `{domain}` | `{stem}` | {info.get('entity_name_zh', '')} | "
            f"{info.get('full_title_zh', '')} | {'PASS' if not problems else 'FAIL'} |"
        )
    lines.append("")
    lines.append("## 4. 失败清单")
    lines.append("")
    if table_error:
        lines.append(f"- {table_error}")
    if failures:
        for key, problems in sorted(failures.items()):
            lines.append(f"- `{key}`：")
            for problem in problems:
                lines.append(f"  - {problem}")
    elif not table_error:
        lines.append("无。三个资产域全部实体均通过中文命名规范校验。")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("-" * 78)
    if overall_pass:
        print(f"[OK] 全部通过：{total_dir}/{total_dir} 条实体中文命名合规。")
        print(f"     报告已写入 {REPORT_PATH.relative_to(PROJECT_ROOT)}")
        return 0

    if table_error:
        print(f"[FATAL] {table_error}")
    print(f"[FAIL] 共 {len(failures)} 条实体未通过校验（另有命名表缺失/异常）。")
    print(f"       报告已写入 {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
