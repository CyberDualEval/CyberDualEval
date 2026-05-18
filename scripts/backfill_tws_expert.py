#!/usr/bin/env python3
"""Backfill expert TWS classifications from grader_results_0504.json.

For each report task, average non-uncertain human TWS labels and write:

    tws_classification_expert = {
      "average_score": <float>,
      "num_scores": <int>,
      "revealed_score": <nearest integer, half-up>,
    }

Labels marked ``uncertain`` or with ``score: null`` are excluded.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "grader_results_0504.json"
DEFAULT_BENCHMARK = REPO_ROOT / "benchmark"
FIELD_NAME = "tws_classification_expert"


@dataclass(frozen=True)
class ExpertClassification:
    average_score: float
    num_scores: int
    revealed_score: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "average_score": self.average_score,
            "num_scores": self.num_scores,
            "revealed_score": self.revealed_score,
        }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def round_half_up(value: float) -> int:
    """Round positive TWS averages to the nearest integer, with .5 rounding up."""
    return int(math.floor(value + 0.5))


def classification_from_labels(task: dict[str, Any]) -> ExpertClassification | None:
    scores: list[int] = []
    for label in task.get("human_labels") or []:
        if label.get("uncertain"):
            continue
        score = label.get("score")
        if score is None:
            continue
        if not isinstance(score, int) or isinstance(score, bool) or score not in (1, 2, 3, 4):
            task_id = task.get("task_id", "?")
            raise ValueError(f"{task_id}: invalid human label score {score!r}")
        scores.append(score)

    if not scores:
        return None

    average = sum(scores) / len(scores)
    return ExpertClassification(
        average_score=round(average, 6),
        num_scores=len(scores),
        revealed_score=round_half_up(average),
    )


def index_metadata_paths(benchmark_root: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for meta_path in sorted(benchmark_root.glob("*/*/metadata.json")):
        meta = _load_json(meta_path)
        source = str(meta.get("source") or meta_path.parent.parent.name)
        task_id = str(meta.get("task_id") or meta_path.parent.name)
        key = (source, task_id)
        if key in index:
            raise ValueError(f"Duplicate metadata task key {key}: {index[key]} and {meta_path}")
        index[key] = meta_path
    return index


def apply_report(
    results_path: Path,
    benchmark_root: Path,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    report = _load_json(results_path)
    tasks = report.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"{results_path} must contain a top-level tasks list")

    metadata_index = index_metadata_paths(benchmark_root)
    stats = {
        "report_tasks": len(tasks),
        "updated": 0,
        "unchanged": 0,
        "missing_metadata": 0,
        "uncertain_only": 0,
    }

    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("report tasks entries must be objects")
        source = str(task.get("source") or "")
        task_id = str(task.get("task_id") or "")
        if not source or not task_id:
            raise ValueError(f"report task missing source/task_id: {task!r}")

        classification = classification_from_labels(task)
        if classification is None:
            stats["uncertain_only"] += 1
            continue

        meta_path = metadata_index.get((source, task_id))
        if meta_path is None:
            stats["missing_metadata"] += 1
            continue

        meta = _load_json(meta_path)
        new_value = classification.as_dict()
        if meta.get(FIELD_NAME) == new_value:
            stats["unchanged"] += 1
            continue

        meta[FIELD_NAME] = new_value
        if dry_run:
            stats["updated"] += 1
        else:
            _write_json(meta_path, meta)
            stats["updated"] += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grader-results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Path to expert grader results JSON.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="Benchmark root containing source/task metadata.json files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize only.")
    args = parser.parse_args()

    stats = apply_report(
        args.grader_results.resolve(),
        args.benchmark_root.resolve(),
        dry_run=args.dry_run,
    )
    prefix = "DRY RUN " if args.dry_run else ""
    print(prefix + json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
