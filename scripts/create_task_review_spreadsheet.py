#!/usr/bin/env python3
"""Create the compact task-review spreadsheet for benchmark tasks.

The output is intended for Google Sheets import / sharing with reviewers. Most
columns are deterministic metadata projections; human-readable annotation
columns can be overlaid from JSONL files with objects keyed by source+task_id.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK = REPO_ROOT / "benchmark"
DEFAULT_GRADER_RESULTS = REPO_ROOT / "grader_results_0504.json"
DEFAULT_CSV = REPO_ROOT / "docs" / "task_review_spreadsheet.csv"
DEFAULT_XLSX = REPO_ROOT / "docs" / "task_review_spreadsheet.xlsx"
DEFAULT_CWE_NAMES = REPO_ROOT / "docs" / "cwe_names_used.json"
DEFAULT_ANNOTATIONS = (
    REPO_ROOT / "docs" / "task_review_annotations" / "task_review_annotations_cybench_cybergym.jsonl",
    REPO_ROOT / "docs" / "task_review_annotations" / "task_review_annotations_exploitdb.jsonl",
    REPO_ROOT / "docs" / "task_review_annotations" / "task_review_annotations_vulhub_a_j.jsonl",
    REPO_ROOT / "docs" / "task_review_annotations" / "task_review_annotations_vulhub_k_z.jsonl",
)
DEFAULT_EXCLUDED_TASKS = {"craftcms_CVE-2025-32432"}

FIELDS = [
    "source",
    "task_id",
    "task_type",
    "target",
    "short_summary",
    "cves",
    "cwe",
    "cwe_is_nvd",
    "tws_scores",
    "tws_revealed",
    "poc_shape",
    "validation_coverage",
    "validation_method",
    "reference_links",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def join_values(values: list[Any]) -> str:
    return "; ".join(str(v) for v in values if v not in (None, ""))


def first_sentence(text: str, *, max_chars: int = 220) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    cut = len(text)
    for marker in (". ", "? ", "! "):
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx + 1)
    if cut == len(text) and len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut == -1:
            cut = max_chars
        return text[:cut].rstrip(" ,;:") + "..."
    return text[:cut].strip()


def read_vulnerability_summary(task_dir: Path) -> str:
    vuln_path = task_dir / "vulnerability.txt"
    if not vuln_path.exists():
        return ""
    return first_sentence(vuln_path.read_text(encoding="utf-8", errors="replace"))


def target_from_metadata(meta: dict[str, Any]) -> str:
    if meta.get("product"):
        return str(meta["product"])
    if meta.get("project"):
        return str(meta["project"])
    if meta.get("original_name"):
        return str(meta["original_name"])
    if meta.get("description"):
        return str(meta["description"])
    task_id = str(meta.get("task_id") or "")
    cves = as_list(meta.get("cves"))
    if meta.get("source") == "vulhub" and cves:
        suffix = "_" + str(cves[0])
        if task_id.endswith(suffix):
            return task_id[: -len(suffix)]
    return task_id


def summary_from_metadata(task_dir: Path, meta: dict[str, Any]) -> str:
    effect = ((meta.get("vuln_class") or {}).get("effect_summary") or "").strip()
    if effect:
        return first_sentence(effect)
    if meta.get("description"):
        return first_sentence(str(meta["description"]))
    return read_vulnerability_summary(task_dir)


def poc_shape_from_metadata(meta: dict[str, Any]) -> str:
    summary = (meta.get("poc_summary") or meta.get("poc_outline") or "").strip()
    if summary:
        return first_sentence(summary, max_chars=180)
    validation = meta.get("validation") or {}
    categories: list[str] = []
    for phase in ("poc", "exploit"):
        if isinstance(validation.get(phase), dict) and validation[phase].get("category"):
            categories.append(str(validation[phase]["category"]))
    if validation.get("category"):
        categories.append(str(validation["category"]))
    if categories:
        return "Validator evidence: " + "; ".join(categories)
    return ""


DEPRECATED_CWE_NAMES = {
    "CWE-19": "Data Processing Errors",
    "CWE-189": "Numeric Errors",
    "CWE-264": "Permissions, Privileges, and Access Controls",
    "CWE-310": "Cryptographic Issues",
    "CWE-399": "Resource Management Errors",
    "CWE-OTHER": "Other",
}


def load_cwe_names(path: Path) -> dict[str, str]:
    names = dict(DEPRECATED_CWE_NAMES)
    if path.exists():
        data = load_json(path)
        if isinstance(data, dict):
            loaded = data.get("names", data)
            if isinstance(loaded, dict):
                names.update({str(k): str(v) for k, v in loaded.items() if v})
    return names


def format_cwe(cwe_id: str, cwe_names: dict[str, str]) -> str:
    name = cwe_names.get(cwe_id)
    return f"{cwe_id} ({name})" if name else cwe_id


def cwe_columns(meta: dict[str, Any], cwe_names: dict[str, str]) -> tuple[str, str]:
    gt = meta.get("ground_truth_cwes") or {}
    nvd_cwes = [c for c in as_list(gt.get("task_cwes")) if isinstance(c, str) and c.startswith("CWE-")]
    if nvd_cwes:
        return join_values([format_cwe(c, cwe_names) for c in sorted(set(nvd_cwes))]), "true"
    benchmark_cwe = ((meta.get("vuln_class") or {}).get("cwe_id") or "").strip()
    return format_cwe(benchmark_cwe, cwe_names) if benchmark_cwe else "", "false"


def load_tws_scores(path: Path) -> dict[tuple[str, str], list[int]]:
    if not path.exists():
        return {}
    report = load_json(path)
    scores: dict[tuple[str, str], list[int]] = {}
    for task in report.get("tasks") or []:
        source = str(task.get("source") or "")
        task_id = str(task.get("task_id") or "")
        if not source or not task_id:
            continue
        task_scores: list[int] = []
        for label in task.get("human_labels") or []:
            if label.get("uncertain"):
                continue
            score = label.get("score")
            if isinstance(score, int) and not isinstance(score, bool):
                task_scores.append(score)
        if task_scores:
            scores[(source, task_id)] = task_scores
    return scores


def tws_columns(meta: dict[str, Any], raw_scores: dict[tuple[str, str], list[int]]) -> tuple[str, str]:
    key = (str(meta.get("source") or ""), str(meta.get("task_id") or ""))
    scores = raw_scores.get(key, [])
    expert = meta.get("tws_classification_expert") or {}
    if scores:
        avg = sum(scores) / len(scores)
        scores_cell = f"{scores} (avg {avg:.2f})"
    elif expert.get("num_scores"):
        avg = expert.get("average_score")
        scores_cell = f"expert n={expert.get('num_scores')} (avg {float(avg):.2f})"
    else:
        scores_cell = ""
    revealed = expert.get("revealed_score")
    if revealed is None:
        revealed = (meta.get("tws_classification") or {}).get("score")
    return scores_cell, "" if revealed is None else str(revealed)


def phase_validated(meta: dict[str, Any], phase: str) -> bool:
    phase_accuracy = (meta.get("accuracy") or {}).get(phase) or {}
    return phase_accuracy.get("skip") is False


def validation_coverage(meta: dict[str, Any]) -> str:
    return "/".join("Y" if phase_validated(meta, phase) else "N" for phase in ("analysis", "poc", "exploit"))


def validation_method(meta: dict[str, Any]) -> str:
    validation = meta.get("validation") or {}
    parts: list[str] = []
    if phase_validated(meta, "analysis"):
        parts.append("analysis: deterministic_oracle")
    for phase in ("poc", "exploit"):
        block = validation.get(phase) if isinstance(validation, dict) else None
        if isinstance(block, dict) and block.get("category"):
            parts.append(f"{phase}: {block['category']}")
    if not parts and isinstance(validation, dict) and validation.get("category"):
        parts.append(f"exploit: {validation['category']}")
    return "; ".join(parts)


def reference_links(meta: dict[str, Any]) -> str:
    links: list[str] = []
    for cve in as_list(meta.get("cves")):
        if isinstance(cve, str) and cve.startswith("CVE-"):
            links.append(f"https://nvd.nist.gov/vuln/detail/{cve}")
    if meta.get("edb_id"):
        links.append(f"https://www.exploit-db.com/exploits/{meta['edb_id']}")
    if meta.get("source") == "vulhub":
        task_id = str(meta.get("task_id") or "")
        links.append(f"https://github.com/vulhub/vulhub/tree/master/{task_id}")
    return "; ".join(dict.fromkeys(links))


def load_annotations(paths: tuple[Path, ...]) -> dict[tuple[str, str], dict[str, str]]:
    annotations: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            source = str(obj.get("source") or "")
            task_id = str(obj.get("task_id") or "")
            if not source or not task_id:
                raise ValueError(f"{path}:{lineno}: annotation missing source/task_id")
            annotations[(source, task_id)] = {
                k: str(obj.get(k) or "")
                for k in ("target", "short_summary", "poc_shape", "reference_links")
            }
    return annotations


def build_rows(
    benchmark_root: Path,
    *,
    excluded_tasks: set[str],
    raw_scores: dict[tuple[str, str], list[int]],
    annotations: dict[tuple[str, str], dict[str, str]],
    cwe_names: dict[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for meta_path in sorted(benchmark_root.glob("*/*/metadata.json")):
        meta = load_json(meta_path)
        source = str(meta.get("source") or meta_path.parent.parent.name)
        task_id = str(meta.get("task_id") or meta_path.parent.name)
        if source not in {"cybench", "cybergym", "exploitdb", "vulhub"}:
            continue
        if task_id in excluded_tasks:
            continue

        cwe, cwe_is_nvd = cwe_columns(meta, cwe_names)
        tws_scores, tws_revealed = tws_columns(meta, raw_scores)
        row = {
            "source": source,
            "task_id": task_id,
            "task_type": str(meta.get("category") or ""),
            "target": target_from_metadata(meta),
            "short_summary": summary_from_metadata(meta_path.parent, meta),
            "cves": join_values(as_list(meta.get("cves"))),
            "cwe": cwe,
            "cwe_is_nvd": cwe_is_nvd,
            "tws_scores": tws_scores,
            "tws_revealed": tws_revealed,
            "poc_shape": poc_shape_from_metadata(meta),
            "validation_coverage": validation_coverage(meta),
            "validation_method": validation_method(meta),
            "reference_links": reference_links(meta),
        }
        overlay = annotations.get((source, task_id))
        if overlay:
            for key, value in overlay.items():
                if value:
                    row[key] = value
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def column_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def sheet_xml(rows: list[dict[str, str]]) -> str:
    matrix = [FIELDS] + [[row.get(field, "") for field in FIELDS] for row in rows]
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"',
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>',
        '<sheetFormatPr defaultRowHeight="15"/>',
        '<cols>',
    ]
    widths = {
        "A": 12,
        "B": 34,
        "C": 16,
        "D": 28,
        "E": 58,
        "F": 28,
        "G": 24,
        "H": 10,
        "I": 20,
        "J": 10,
        "K": 54,
        "L": 12,
        "M": 42,
        "N": 58,
    }
    for idx in range(1, len(FIELDS) + 1):
        width = widths.get(column_name(idx), 18)
        lines.append(f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>')
    lines.append("</cols><sheetData>")
    for r_idx, row in enumerate(matrix, start=1):
        lines.append(f'<row r="{r_idx}">')
        for c_idx, value in enumerate(row, start=1):
            cell = f"{column_name(c_idx)}{r_idx}"
            escaped = html.escape(str(value), quote=False)
            style = ' s="1"' if r_idx == 1 else ""
            lines.append(f'<c r="{cell}" t="inlineStr"{style}><is><t>{escaped}</t></is></c>')
        lines.append("</row>")
    lines.append('</sheetData><autoFilter ref="A1:N1"/></worksheet>')
    return "".join(lines)


def write_xlsx(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    files = {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>""",
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
        "docProps/app.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>CyberDualEval</Application></Properties>""",
        "docProps/core.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:creator>CyberDualEval</dc:creator><cp:lastModifiedBy>CyberDualEval</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>""",
        "xl/workbook.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="tasks" sheetId="1" r:id="rId1"/></sheets></workbook>""",
        "xl/_rels/workbook.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>""",
        "xl/styles.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>""",
        "xl/worksheets/sheet1.xml": sheet_xml(rows),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--grader-results", type=Path, default=DEFAULT_GRADER_RESULTS)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--xlsx-out", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--cwe-names", type=Path, default=DEFAULT_CWE_NAMES)
    parser.add_argument("--annotation", type=Path, action="append", default=[])
    parser.add_argument("--exclude-task", action="append", default=sorted(DEFAULT_EXCLUDED_TASKS))
    args = parser.parse_args()

    annotation_paths = tuple(args.annotation) if args.annotation else DEFAULT_ANNOTATIONS
    rows = build_rows(
        args.benchmark_root,
        excluded_tasks=set(args.exclude_task),
        raw_scores=load_tws_scores(args.grader_results),
        annotations=load_annotations(annotation_paths),
        cwe_names=load_cwe_names(args.cwe_names),
    )
    write_csv(args.csv_out, rows)
    write_xlsx(args.xlsx_out, rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "csv": str(args.csv_out),
                "xlsx": str(args.xlsx_out),
                "annotations_loaded": sum(1 for p in annotation_paths if p.exists()),
                "excluded_tasks": sorted(set(args.exclude_task)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
