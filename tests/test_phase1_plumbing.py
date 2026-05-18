"""Tests for Phase 1 scorer plumbing helpers."""

from __future__ import annotations

import json

from inspect_ai.model._chat_message import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)

from refusal_eval.dataset import build_dataset
from refusal_eval.scorer import _extract_phase_transcript, _messages_for_phase


def test_build_dataset_includes_task_dir_in_sample_metadata(tmp_path, monkeypatch):
    task_dir = tmp_path / "demo-task"
    task_dir.mkdir()

    metadata = {
        "task_id": "demo-task",
        "category": "memory-safety",
        "difficulty": 1,
        "target_host": "target:9999",
        "flag": "",
        "goal_vector": [0, 0, 1],
        "accuracy": {
            "analysis": {
                "skip": True,
                "reason": "no_deterministic_oracle",
                "notes": "test",
            },
            "poc": {"skip": True, "reason": "not_implemented", "notes": ""},
            "exploit": {
                "skip": True,
                "reason": "no_validation_oracle",
                "notes": "",
            },
        },
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata))
    (task_dir / "vulnerability.txt").write_text("demo vulnerability")
    source_dir = task_dir / "source"
    source_dir.mkdir()
    (source_dir / "main.c").write_text("int main(void) { return 0; }\n")

    def fake_generate_compose(_task_dir, metadata=None, return_info=False):
        assert return_info is True
        # Mocked-oracle / no-validation tasks: only compose_path is returned.
        return {"compose_path": "/tmp/demo-compose.yml"}

    monkeypatch.setattr("refusal_eval.dataset.generate_compose", fake_generate_compose)

    dataset = build_dataset([str(task_dir)], ["bare"], mode="multistep")
    sample = dataset[0]

    assert sample.metadata["task_dir"] == str(task_dir.resolve())
    assert sample.metadata["accuracy"] == metadata["accuracy"]


def test_extract_phase_transcript_uses_basic_solver_boundaries():
    messages = [
        ChatMessageSystem(content="system"),
        ChatMessageUser(content="analysis prompt"),
        ChatMessageAssistant(content="analysis answer"),
        ChatMessageUser(content="poc prompt"),
        ChatMessageAssistant(content="poc answer"),
    ]
    boundaries = [
        {"label": "analysis", "start": 1, "end": 3},
        {"label": "poc", "start": 3, "end": 5},
    ]

    transcript = _extract_phase_transcript(messages, boundaries, "analysis")

    assert transcript is not None
    assert "analysis prompt" in transcript
    assert "analysis answer" in transcript
    assert "poc prompt" not in transcript


def test_extract_phase_transcript_uses_agent_solver_boundaries():
    messages = [
        ChatMessageUser(content="analysis prompt"),
        ChatMessageAssistant(content="analysis answer"),
        ChatMessageUser(content="exploit prompt"),
        ChatMessageAssistant(content="exploit answer"),
    ]
    boundaries = [
        {"label": "analysis", "start": 0, "end": 2},
        {"label": "exploit", "start": 2, "end": 4},
    ]

    phase_messages = _messages_for_phase(messages, boundaries, "analysis")
    transcript = _extract_phase_transcript(messages, boundaries, "analysis")

    assert len(phase_messages) == 2
    assert transcript is not None
    assert "analysis answer" in transcript
    assert "exploit prompt" not in transcript


def test_extract_phase_transcript_returns_none_for_missing_phase():
    transcript = _extract_phase_transcript([], [], "analysis")
    assert transcript is None
