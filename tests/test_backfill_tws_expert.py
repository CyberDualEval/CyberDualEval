from __future__ import annotations

import json
from pathlib import Path

from scripts.backfill_tws_expert import (
    apply_report,
    classification_from_labels,
    round_half_up,
)


def test_classification_excludes_uncertain_labels():
    result = classification_from_labels({
        "task_id": "demo",
        "human_labels": [
            {"labeler": "a", "score": 2},
            {"labeler": "b", "score": None, "uncertain": True},
            {"labeler": "c", "score": 4},
        ],
    })

    assert result is not None
    assert result.as_dict() == {
        "average_score": 3.0,
        "num_scores": 2,
        "revealed_score": 3,
    }


def test_classification_returns_none_for_uncertain_only():
    result = classification_from_labels({
        "task_id": "demo",
        "human_labels": [{"labeler": "a", "score": None, "uncertain": True}],
    })

    assert result is None


def test_round_half_up_for_decision_boundaries():
    assert round_half_up(1.49) == 1
    assert round_half_up(1.5) == 2
    assert round_half_up(2.5) == 3
    assert round_half_up(3.5) == 4


def test_apply_report_writes_metadata_field(tmp_path):
    benchmark = tmp_path / "benchmark"
    task_dir = benchmark / "exploitdb" / "edb-1"
    task_dir.mkdir(parents=True)
    meta_path = task_dir / "metadata.json"
    meta_path.write_text(json.dumps({
        "task_id": "edb-1",
        "source": "exploitdb",
        "category": "remote",
    }))

    report_path = tmp_path / "grader_results.json"
    report_path.write_text(json.dumps({
        "tasks": [
            {
                "task_id": "edb-1",
                "source": "exploitdb",
                "human_labels": [
                    {"labeler": "a", "score": 2},
                    {"labeler": "b", "score": 3},
                ],
            }
        ],
    }))

    stats = apply_report(report_path, benchmark)

    assert stats["updated"] == 1
    meta = json.loads(meta_path.read_text())
    assert meta["tws_classification_expert"] == {
        "average_score": 2.5,
        "num_scores": 2,
        "revealed_score": 3,
    }
