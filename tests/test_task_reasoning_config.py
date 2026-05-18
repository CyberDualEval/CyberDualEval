from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from refusal_eval.task import _reasoning_effort_from_config, refusal_eval
from refusal_eval.solver import _contains_nc_command, _is_api_refusal_error


def _write_minimal_config(
    tmp_path: Path,
    *,
    reasoning: object,
    agent: str = "basic",
    tool_timeout: int | None = None,
    nc_timeout: int | None = None,
    assistant_turn_limit: int | None = None,
    phases: object | None = None,
    sandbox_resources: dict | None = None,
) -> Path:
    task_dir = tmp_path / "task1"
    task_dir.mkdir()
    source_dir = task_dir / "source"
    source_dir.mkdir()
    (source_dir / "main.c").write_text("int main(void) { return 0; }\n")
    (task_dir / "metadata.json").write_text(json.dumps({
        "task_id": "task1",
        "category": "pwn",
        "difficulty": 1,
        "target_host": "target:1337",
        "flag": "",
        "goal_vector": [0, 0, 1],
    }))

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump({
        "version": 2,
        "source": "test",
        "tasks": [{"task_id": "task1", "dir": "task1"}],
    }))

    cfg = {
        "agent": agent,
        "mode": "multistep",
        "framings": ["bare"],
        "task": {
            "benchmark": [
                {"source": "test", "manifest": str(manifest_path)},
            ],
        },
    }
    if isinstance(reasoning, dict):
        cfg["reasoning"] = reasoning
    else:
        cfg["reasoning_effort"] = reasoning
    if tool_timeout is not None:
        cfg["tool_timeout"] = tool_timeout
    if nc_timeout is not None:
        cfg["nc_timeout"] = nc_timeout
    if assistant_turn_limit is not None:
        cfg["assistant_turn_limit"] = assistant_turn_limit
    if phases is not None:
        cfg["phases"] = phases
    if sandbox_resources is not None:
        cfg["sandbox_resources"] = sandbox_resources

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return config_path


def test_reasoning_effort_config_sets_task_generate_config(tmp_path):
    config_path = _write_minimal_config(tmp_path, reasoning="high")

    task = refusal_eval(config_path=str(config_path))

    assert task.config.reasoning_effort == "high"


def test_reasoning_effort_accepts_reasoning_effort_alias_dict():
    assert _reasoning_effort_from_config({"reasoning": {"effort": "xhigh"}}) == "xhigh"


def test_reasoning_effort_normalizes_case_and_whitespace():
    assert _reasoning_effort_from_config({"reasoning_effort": " Medium "}) == "medium"


def test_reasoning_effort_rejects_invalid_value():
    with pytest.raises(ValueError, match="Unsupported reasoning_effort"):
        _reasoning_effort_from_config({"reasoning_effort": "maximum"})


def test_tool_timeout_config_flows_to_solver_params(tmp_path, monkeypatch):
    config_path = _write_minimal_config(
        tmp_path, reasoning="medium", tool_timeout=45, nc_timeout=7,
    )
    captured = {}

    def fake_multistep_solver(
        *,
        message_limit_per_phase: int,
        tool_timeout: int,
        nc_timeout: int | None,
        assistant_turn_limit_per_phase: int | None,
    ):
        captured["message_limit_per_phase"] = message_limit_per_phase
        captured["tool_timeout"] = tool_timeout
        captured["nc_timeout"] = nc_timeout
        captured["assistant_turn_limit_per_phase"] = assistant_turn_limit_per_phase

        async def solve(state, generate):
            return state

        return solve

    monkeypatch.setattr(
        "refusal_eval.task.multistep_solver", fake_multistep_solver,
    )

    task = refusal_eval(config_path=str(config_path))

    assert task.solver is not None
    assert captured["tool_timeout"] == 45
    assert captured["nc_timeout"] == 7
    assert captured["assistant_turn_limit_per_phase"] is None


def test_assistant_turn_limit_config_flows_to_basic_multistep_solver(
    tmp_path, monkeypatch
):
    config_path = _write_minimal_config(
        tmp_path, reasoning="medium", assistant_turn_limit=40,
    )
    captured = {}

    def fake_multistep_solver(
        *,
        message_limit_per_phase: int,
        tool_timeout: int,
        nc_timeout: int | None,
        assistant_turn_limit_per_phase: int | None,
    ):
        captured["message_limit_per_phase"] = message_limit_per_phase
        captured["assistant_turn_limit_per_phase"] = assistant_turn_limit_per_phase

        async def solve(state, generate):
            return state

        return solve

    monkeypatch.setattr(
        "refusal_eval.task.multistep_solver", fake_multistep_solver,
    )

    task = refusal_eval(config_path=str(config_path))

    assert task.solver is not None
    assert captured["message_limit_per_phase"] == 30
    assert captured["assistant_turn_limit_per_phase"] == 40


def test_assistant_turn_limit_config_flows_to_claude_code_multistep_solver(
    tmp_path, monkeypatch
):
    config_path = _write_minimal_config(
        tmp_path,
        reasoning="medium",
        agent="claude-code",
        assistant_turn_limit=50,
    )
    captured = {}

    def fake_claude_code_multistep_solver(
        *,
        message_limit_per_phase: int,
        assistant_turn_limit_per_phase: int | None,
    ):
        captured["message_limit_per_phase"] = message_limit_per_phase
        captured["assistant_turn_limit_per_phase"] = assistant_turn_limit_per_phase

        async def solve(state, generate):
            return state

        return solve

    monkeypatch.setattr(
        "refusal_eval.claude_code_solver.claude_code_multistep_solver",
        fake_claude_code_multistep_solver,
    )

    task = refusal_eval(config_path=str(config_path))

    assert task.solver is not None
    assert captured["message_limit_per_phase"] == 30
    assert captured["assistant_turn_limit_per_phase"] == 50


def test_phases_config_flows_to_dataset(tmp_path):
    config_path = _write_minimal_config(
        tmp_path, reasoning="medium", phases=["analysis"],
    )

    task = refusal_eval(config_path=str(config_path))

    sample = task.dataset.samples[0]
    assert sample.metadata["selected_phases"] == ["analysis"]
    assert [p["label"] for p in sample.metadata["phase_prompts"]] == ["analysis"]


def test_phases_task_param_overrides_config(tmp_path):
    config_path = _write_minimal_config(
        tmp_path, reasoning="medium", phases=["analysis"],
    )

    task = refusal_eval(
        config_path=str(config_path), phases=["analysis", "poc", "exploit"],
    )

    sample = task.dataset.samples[0]
    assert sample.metadata["selected_phases"] == ["analysis", "poc", "exploit"]
    assert [p["label"] for p in sample.metadata["phase_prompts"]] == [
        "analysis",
        "poc",
        "exploit",
    ]


def test_phases_config_rejects_unsupported_subset(tmp_path):
    config_path = _write_minimal_config(
        tmp_path, reasoning="medium", phases=["poc"],
    )

    with pytest.raises(ValueError, match="Unsupported phases selection"):
        refusal_eval(config_path=str(config_path))


def test_sandbox_resources_config_flows_to_generated_compose(tmp_path):
    config_path = _write_minimal_config(
        tmp_path,
        reasoning="medium",
        sandbox_resources={
            "default": {"memory": "6g", "cpus": 3, "pids_limit": 1024},
            "oracle": {"memory": "512m", "cpus": 0.5},
        },
    )

    task = refusal_eval(config_path=str(config_path))

    sample = task.dataset.samples[0]
    _, (_, compose_path) = sample.sandbox
    compose = yaml.safe_load(Path(compose_path).read_text())
    assert compose["services"]["default"]["mem_limit"] == "6g"
    assert compose["services"]["default"]["cpus"] == 3
    assert compose["services"]["default"]["pids_limit"] == 1024
    assert compose["services"]["target"]["mem_limit"] == "512m"
    assert compose["services"]["target"]["cpus"] == 0.5


def test_nc_command_detection_targets_netcat_tokens_only():
    assert _contains_nc_command("nc challenge 1337")
    assert _contains_nc_command("printf hi | ncat host 80")
    assert _contains_nc_command("netcat -w 1 host 80")
    assert not _contains_nc_command("sync")
    assert not _contains_nc_command("python3 exploit.py")


def test_api_refusal_error_detection_targets_provider_filters():
    assert _is_api_refusal_error(
        RuntimeError("BadRequestError: code='cyber_policy'")
    )
    assert _is_api_refusal_error(
        RuntimeError("This content was flagged for possible cybersecurity risk")
    )
    assert _is_api_refusal_error(RuntimeError("stop_reason=content_filter"))
    assert not _is_api_refusal_error(RuntimeError("rate limit exceeded"))
