"""Backfill NVD CWE mappings into benchmark task metadata.

Walks CVE-backed ExploitDB and Vulhub tasks, fetches their official NVD
weakness labels, and writes a non-overwriting `ground_truth_cwes` block. This
does not modify the model-derived `vuln_class` field.

Usage:
    uv run python scripts/backfill_ground_truth_cwes.py --dry-run
    uv run python scripts/backfill_ground_truth_cwes.py --source vulhub
    uv run python scripts/backfill_ground_truth_cwes.py --source exploitdb
    uv run python scripts/backfill_ground_truth_cwes.py --force
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from nvd_helpers import NVD_API_URL, fetch_cwes, load_cache, save_cache  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(SCRIPT_DIR).parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "ground-truth-cwe-report.json"
NUMERIC_CWE_RE = re.compile(r"^CWE-\d+$")
NOTES = (
    "Fetched from NVD CVE weakness descriptions. Does not overwrite "
    "model-derived vuln_class."
)


def _metadata_paths(source: str) -> list[Path]:
    return sorted((BENCHMARK_DIR / source).glob("*/metadata.json"))


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def _normalise_cves(cves: Any) -> list[str]:
    if not isinstance(cves, list):
        return []
    return sorted({str(cve).strip().upper() for cve in cves if str(cve).strip()})


def _split_cwe_labels(labels: list[str]) -> tuple[list[str], list[str]]:
    numeric = sorted({label for label in labels if NUMERIC_CWE_RE.fullmatch(label)})
    non_specific = sorted({label for label in labels if not NUMERIC_CWE_RE.fullmatch(label)})
    return numeric, non_specific


def _fetch_all_nvd_weakness_labels(
    cve_id: str,
    api_key: str | None,
    cache: dict,
    cache_path: str,
) -> list[str]:
    """Fetch all NVD weakness labels, preserving NVD-CWE-noinfo style values.

    The existing `fetch_cwes` helper intentionally returns values beginning
    with `CWE-`. This companion cache entry preserves the full NVD weakness
    labels so non-specific labels remain auditable.
    """
    cache_key = f"{cve_id}:cwe_labels"
    if cache_key in cache:
        return cache[cache_key]

    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    labels: list[str] = []
    try:
        resp = requests.get(
            NVD_API_URL,
            params={"cveId": cve_id},
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            vulns = data.get("vulnerabilities", [])
            if vulns:
                for weakness in vulns[0].get("cve", {}).get("weaknesses", []):
                    for desc in weakness.get("description", []):
                        value = str(desc.get("value", "")).strip()
                        if value:
                            labels.append(value)
        else:
            log.warning("  NVD %d while fetching weakness labels for %s", resp.status_code, cve_id)
    except requests.RequestException as exc:
        log.warning("  NVD request error while fetching weakness labels for %s: %s", cve_id, exc)

    labels = sorted(set(labels))
    cache[cache_key] = labels
    cache.setdefault(f"{cve_id}:cwes", sorted(label for label in labels if label.startswith("CWE-")))
    save_cache(cache_path, cache)
    return labels


def _fetch_task_cwes(
    cves: list[str],
    api_key: str | None,
    cache: dict,
    cache_path: str,
    rate_delay: float,
) -> tuple[dict[str, list[str]], int]:
    per_cve: dict[str, list[str]] = {}
    api_calls = 0

    for cve_id in cves:
        cwes_key = f"{cve_id}:cwes"
        labels_key = f"{cve_id}:cwe_labels"
        cwes_was_cached = cwes_key in cache
        labels_was_cached = labels_key in cache

        labels = _fetch_all_nvd_weakness_labels(cve_id, api_key, cache, cache_path)
        if not labels_was_cached:
            api_calls += 1
            time.sleep(rate_delay)

        helper_cwes = fetch_cwes(cve_id, api_key, cache, cache_path)

        # Preserve any standard CWE labels returned by the helper even if the
        # full-label call failed transiently.
        per_cve[cve_id] = sorted(set(labels) | set(helper_cwes))

        if labels_was_cached and not cwes_was_cached:
            api_calls += 1
            time.sleep(rate_delay)

    return per_cve, api_calls


def _build_ground_truth_block(per_cve: dict[str, list[str]], fetched_at: str) -> dict[str, Any]:
    all_labels = [label for labels in per_cve.values() for label in labels]
    task_cwes, non_specific = _split_cwe_labels(all_labels)
    block: dict[str, Any] = {
        "source": "nvd",
        "fetched_at": fetched_at,
        "per_cve": {cve: sorted(set(labels)) for cve, labels in sorted(per_cve.items())},
        "task_cwes": task_cwes,
        "notes": NOTES,
    }
    if non_specific:
        block["non_specific"] = non_specific
    return block


def _empty_source_stats() -> dict[str, int]:
    return {
        "total_tasks": 0,
        "cve_backed_tasks": 0,
        "updated": 0,
        "skipped_existing": 0,
        "with_numeric_cwe": 0,
        "only_non_specific": 0,
        "missing": 0,
        "api_calls": 0,
    }


def _count_block_quality(stats: dict[str, int], block: dict[str, Any]) -> None:
    if block.get("task_cwes"):
        stats["with_numeric_cwe"] += 1
    elif block.get("non_specific"):
        stats["only_non_specific"] += 1
    else:
        stats["missing"] += 1


def _selected_existing_quality(source: str, stats: dict[str, int]) -> None:
    for meta_path in _metadata_paths(source):
        meta = _load_json(meta_path)
        if not _normalise_cves(meta.get("cves")):
            continue
        block = meta.get("ground_truth_cwes")
        if isinstance(block, dict):
            _count_block_quality(stats, block)


def _prepare_cache(source: str, dry_run: bool, temp_dir: tempfile.TemporaryDirectory[str] | None) -> tuple[dict, str]:
    real_cache_path = BENCHMARK_DIR / source / ".nvd_cache.json"
    cache = load_cache(str(real_cache_path))
    if not dry_run:
        return cache, str(real_cache_path)

    assert temp_dir is not None
    dry_cache_path = Path(temp_dir.name) / f"{source}.nvd_cache.json"
    if cache:
        _write_json(dry_cache_path, cache)
    return copy.deepcopy(cache), str(dry_cache_path)


def backfill_source(
    source: str,
    api_key: str | None,
    rate_delay: float,
    force: bool,
    dry_run: bool,
    fetched_at: str,
    temp_dir: tempfile.TemporaryDirectory[str] | None,
) -> dict[str, int]:
    stats = _empty_source_stats()
    cache, cache_path = _prepare_cache(source, dry_run, temp_dir)
    log.info("  cache: %d entries", len(cache))

    for meta_path in _metadata_paths(source):
        stats["total_tasks"] += 1
        meta = _load_json(meta_path)
        cves = _normalise_cves(meta.get("cves"))
        if not cves:
            continue

        stats["cve_backed_tasks"] += 1
        existing_block = meta.get("ground_truth_cwes")
        if not force and existing_block is not None:
            stats["skipped_existing"] += 1
            if isinstance(existing_block, dict):
                _count_block_quality(stats, existing_block)
            continue

        per_cve, api_calls = _fetch_task_cwes(cves, api_key, cache, cache_path, rate_delay)
        stats["api_calls"] += api_calls
        block = _build_ground_truth_block(per_cve, fetched_at)
        _count_block_quality(stats, block)
        stats["updated"] += 1

        if dry_run:
            log.info(
                "  [dry-run] would write %s: task_cwes=%s non_specific=%s",
                meta.get("task_id", meta_path.parent.name),
                block["task_cwes"],
                block.get("non_specific", []),
            )
        else:
            meta["ground_truth_cwes"] = block
            _write_json(meta_path, meta)

    if not dry_run:
        save_cache(cache_path, cache)

    return stats


def _merge_report_stats(per_source: dict[str, dict[str, int]], dry_run: bool, sources: list[str]) -> dict[str, Any]:
    totals = {
        "total_cve_backed_tasks": sum(stats["cve_backed_tasks"] for stats in per_source.values()),
        "updated": sum(stats["updated"] for stats in per_source.values()),
        "skipped_existing": sum(stats["skipped_existing"] for stats in per_source.values()),
        "with_numeric_cwe": sum(stats["with_numeric_cwe"] for stats in per_source.values()),
        "only_non_specific": sum(stats["only_non_specific"] for stats in per_source.values()),
        "missing": sum(stats["missing"] for stats in per_source.values()),
        "api_calls": sum(stats["api_calls"] for stats in per_source.values()),
        "dry_run": dry_run,
        "sources": sources,
        "per_source": per_source,
    }
    return totals


def validate_backfill(sources: list[str], vuln_class_before: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    for source in sources:
        for meta_path in _metadata_paths(source):
            rel_path = str(meta_path.relative_to(PROJECT_ROOT))
            meta = _load_json(meta_path)
            cves = _normalise_cves(meta.get("cves"))

            if meta.get("vuln_class") != vuln_class_before.get(rel_path):
                errors.append(f"{rel_path}: vuln_class changed")

            if not cves:
                continue

            block = meta.get("ground_truth_cwes")
            if not isinstance(block, dict):
                errors.append(f"{rel_path}: missing ground_truth_cwes")
                continue

            per_cve = block.get("per_cve")
            if not isinstance(per_cve, dict):
                errors.append(f"{rel_path}: ground_truth_cwes.per_cve is not an object")
                continue

            extra_keys = sorted(set(per_cve) - set(cves))
            if extra_keys:
                errors.append(f"{rel_path}: per_cve has keys not in metadata.cves: {extra_keys}")

            missing_keys = sorted(set(cves) - set(per_cve))
            if missing_keys:
                errors.append(f"{rel_path}: per_cve missing metadata.cves keys: {missing_keys}")

            expected_task_cwes = sorted(
                {
                    label
                    for labels in per_cve.values()
                    if isinstance(labels, list)
                    for label in labels
                    if isinstance(label, str) and NUMERIC_CWE_RE.fullmatch(label)
                }
            )
            if block.get("task_cwes") != expected_task_cwes:
                errors.append(
                    f"{rel_path}: task_cwes {block.get('task_cwes')} != union {expected_task_cwes}"
                )

    return errors


def _snapshot_vuln_class(sources: list[str]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for source in sources:
        for meta_path in _metadata_paths(source):
            rel_path = str(meta_path.relative_to(PROJECT_ROOT))
            snapshot[rel_path] = copy.deepcopy(_load_json(meta_path).get("vuln_class"))
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["exploitdb", "vulhub", "both"],
        default="both",
        help="Which benchmark source(s) to backfill (default: both)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing ground_truth_cwes blocks (skips by default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write metadata.json, cache files, or report files",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help=f"Where to write the JSON report (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write the JSON report",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        except ImportError:
            pass

    api_key = os.environ.get("NVD_API_KEY")
    rate_delay = 0.6 if api_key else 6.0
    sources = ["exploitdb", "vulhub"] if args.source == "both" else [args.source]
    fetched_at = datetime.now(timezone.utc).date().isoformat()
    temp_dir = tempfile.TemporaryDirectory() if args.dry_run else None
    vuln_class_before = _snapshot_vuln_class(sources)

    log.info(
        "NVD API key: %s (rate delay %.1fs)",
        "yes" if api_key else "no",
        rate_delay,
    )

    try:
        per_source: dict[str, dict[str, int]] = {}
        for source in sources:
            source_dir = BENCHMARK_DIR / source
            if not source_dir.is_dir():
                log.warning("skipping %s: directory not found", source_dir)
                continue
            log.info("\n=== %s ===", source)
            stats = backfill_source(
                source=source,
                api_key=api_key,
                rate_delay=rate_delay,
                force=args.force,
                dry_run=args.dry_run,
                fetched_at=fetched_at,
                temp_dir=temp_dir,
            )
            per_source[source] = stats
            log.info(
                "  cve_backed=%d updated=%d skipped_existing=%d "
                "with_numeric=%d only_non_specific=%d missing=%d api_calls=%d",
                stats["cve_backed_tasks"],
                stats["updated"],
                stats["skipped_existing"],
                stats["with_numeric_cwe"],
                stats["only_non_specific"],
                stats["missing"],
                stats["api_calls"],
            )

        if not args.dry_run:
            validation_errors = validate_backfill(sources, vuln_class_before)
            if validation_errors:
                log.error("\nValidation failed:")
                for error in validation_errors[:50]:
                    log.error("  %s", error)
                if len(validation_errors) > 50:
                    log.error("  ... %d more", len(validation_errors) - 50)
                return 1
            log.info("\nValidation passed.")

        report = _merge_report_stats(per_source, args.dry_run, sources)
        log.info("\n=== REPORT ===")
        log.info(json.dumps(report, indent=2))

        if not args.dry_run and not args.no_report:
            report_path = Path(args.report_path)
            if not report_path.is_absolute():
                report_path = PROJECT_ROOT / report_path
            _write_json(report_path, report)
            log.info("Wrote report to %s", report_path.relative_to(PROJECT_ROOT))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
