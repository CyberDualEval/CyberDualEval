"""Inspect task definitions for CyberAgent-Bench refusal evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from refusal_eval.dataset import build_dataset, normalize_phases, resolve_manifest
from refusal_eval.model_costs import register_all_costs
from refusal_eval.scorer import refusal_scorer
from refusal_eval.solver import multistep_solver, oneshot_solver

# Project root: one level up from refusal_eval/
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

# Register model pricing from bundled LiteLLM data so Inspect can compute
# total_cost across providers, including AWS Bedrock.
register_all_costs()

_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def _load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load config file with defaults."""
    # Resolve relative to project root
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    if os.path.isfile(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _reasoning_effort_from_config(cfg: dict[str, Any]) -> str | None:
    """Return the configured reasoning effort, accepting concise YAML shapes."""
    value = cfg.get("reasoning_effort")
    if value is None and "reasoning" in cfg:
        reasoning = cfg["reasoning"]
        if isinstance(reasoning, dict):
            value = reasoning.get("effort")
        else:
            value = reasoning

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("reasoning_effort must be a string if configured.")

    value = value.strip().lower()
    if value not in _REASONING_EFFORTS:
        allowed = ", ".join(sorted(_REASONING_EFFORTS))
        raise ValueError(
            f"Unsupported reasoning_effort '{value}'. Expected one of: {allowed}."
        )
    return value


def _task_has_validation(task_dir: str) -> bool:
    """Return whether ``task_dir`` has a committed ``validation`` block."""
    meta_path = Path(task_dir) / "metadata.json"
    if not meta_path.is_file():
        return False
    try:
        with meta_path.open() as f:
            meta = json.load(f)
    except Exception:
        return False
    return bool(meta.get("validation"))


def _order_task_dirs(task_dirs: list[str], task_order: str) -> list[str]:
    """Apply optional ordering to the resolved task list.

    ``validated_first`` is useful for large benchmark runs where we want the
    real graded Vulhub environments to execute before the mocked-oracle tasks,
    so environment bugs surface early.
    """
    if task_order == "validated_first":
        return sorted(
            task_dirs,
            key=lambda task_dir: (0 if _task_has_validation(task_dir) else 1, task_dir),
        )
    return task_dirs



@task
def refusal_eval(
    framings: str | list[str] | None = None,
    mode: str | None = None,
    agent: str | None = None,
    judge_model: str | None = None,
    refusal_token: bool | list[bool] | None = None,
    message_limit: int = 30,
    assistant_turn_limit: int | None = None,
    tool_timeout: int = 180,
    nc_timeout: int | None = None,
    config_path: str = "config.yaml",
    validated: bool | None = None,
    phases: str | list[str] | None = None,
) -> Task:
    """CyberAgent-Bench refusal evaluation task.

    Measures whether models refuse cybersecurity tasks across different
    prompt framings and phases.

    Args:
        framings: Framing names (e.g. "ctf", "bare"). Defaults to config.yaml.
        mode: "oneshot" or "multistep". Defaults to config.yaml.
        agent: "basic" (default), "claude-code", or "codex-cli". Defaults to config.yaml.
        judge_model: Model for LLM judge (stage 3). Defaults to config.yaml.
        refusal_token: Whether to include <<<REFUSE>>> token instruction in system
            prompt and use token scan in scoring. Pass a list (e.g. [true, false])
            to cross-product both conditions. Defaults to config.yaml (true).
        message_limit: Max messages per phase (multistep) or total (oneshot).
        assistant_turn_limit: Optional max assistant/model responses per phase
            (multistep) or total (oneshot). When set for the basic agent, this
            is used instead of message_limit for loop termination while
            preserving a larger internal message safety cap for tool results.
        tool_timeout: Timeout for bash/python tool calls in seconds.
        nc_timeout: Optional shorter timeout for bash commands containing nc/netcat.
        config_path: Path to config.yaml for defaults.
        validated: When True (default), per-task ``validation:`` blocks and
            ``accuracy`` fields are honored as authored — grader/controller
            sidecars are spliced into the compose for tasks that opted in,
            and each phase is scored against its authored predicates.
            When False, bypass all in-eval validation: drop every task's
            ``validation`` block before sandbox composition and force
            ``accuracy.{analysis,poc,exploit}.skip=True`` for every sample
            (capability-only refusal-rate run). Defaults to config.yaml
            (which itself defaults to True).
        phases: Optional multistep phase selection. Omit for full
            analysis/poc/exploit. Currently supports ["analysis"] for
            analysis-only experiments.
    """
    cfg = _load_config(config_path)

    # Resolve tasks from benchmark manifests
    task_dirs: list[str] = []
    for source_cfg in cfg.get("task", {}).get("benchmark", []):
        manifest = source_cfg["manifest"]
        if not os.path.isabs(manifest):
            manifest = os.path.join(PROJECT_ROOT, manifest)
        task_filter = source_cfg.get("tasks")
        exclude_tasks = source_cfg.get("exclude_tasks")
        if os.path.isfile(manifest):
            task_dirs.extend(
                resolve_manifest(
                    manifest,
                    task_ids=task_filter,
                    exclude_task_ids=exclude_tasks,
                )
            )
    task_order = cfg.get("task_order", "manifest")
    task_dirs = _order_task_dirs(task_dirs, task_order)

    if framings is None:
        framings = cfg.get("framings", ["ctf"])
    elif isinstance(framings, str):
        framings = [framings]

    if mode is None:
        mode = cfg.get("mode", "multistep")

    if judge_model is None:
        judge_model = cfg.get("judge", {}).get("model", "openai/gpt-5-mini")

    if refusal_token is None:
        refusal_token = cfg.get("refusal_token", True)

    # message_limit: CLI -T flag overrides config, config overrides default
    if message_limit == 30:  # default unchanged by caller
        message_limit = cfg.get("message_limit", 30)

    if assistant_turn_limit is None:
        assistant_turn_limit = cfg.get("assistant_turn_limit")

    # tool_timeout: CLI -T flag overrides config, config overrides default
    if tool_timeout == 180:  # default unchanged by caller
        tool_timeout = cfg.get("tool_timeout", 180)

    if nc_timeout is None:
        nc_timeout = cfg.get("nc_timeout")

    # Normalize refusal_token to a list for cross-product
    if isinstance(refusal_token, bool):
        rt_values = [refusal_token]
    else:
        rt_values = list(refusal_token)

    if validated is None:
        validated = bool(cfg.get("validated", True))

    if phases is None:
        phases = cfg.get("phases")
    selected_phases = normalize_phases(phases)

    # Build dataset
    dataset = build_dataset(
        task_dirs, framings, mode,
        refusal_token_values=rt_values,
        validated=validated,
        phases=selected_phases,
        sandbox_resources=cfg.get("sandbox_resources"),
    )

    # Resolve agent type
    if agent is None:
        agent = cfg.get("agent", "basic")

    # Select solver
    if agent == "claude-code":
        from refusal_eval.claude_code_solver import (
            claude_code_multistep_solver,
            claude_code_oneshot_solver,
        )

        if mode == "multistep":
            slvr = claude_code_multistep_solver(
                message_limit_per_phase=message_limit,
                assistant_turn_limit_per_phase=assistant_turn_limit,
            )
        else:
            slvr = claude_code_oneshot_solver()
    elif agent == "codex-cli":
        from refusal_eval.codex_cli_solver import (
            codex_cli_multistep_solver,
            codex_cli_oneshot_solver,
        )

        model_config = cfg.get("model_config", "gpt-5.1")
        if mode == "multistep":
            slvr = codex_cli_multistep_solver(
                message_limit_per_phase=message_limit,
                model_config=model_config,
                assistant_turn_limit_per_phase=assistant_turn_limit,
            )
        else:
            slvr = codex_cli_oneshot_solver(model_config=model_config)
    elif mode == "multistep":
        slvr = multistep_solver(
            message_limit_per_phase=message_limit,
            tool_timeout=tool_timeout,
            nc_timeout=nc_timeout,
            assistant_turn_limit_per_phase=assistant_turn_limit,
        )
    else:
        slvr = oneshot_solver(
            message_limit=message_limit,
            tool_timeout=tool_timeout,
            nc_timeout=nc_timeout,
            assistant_turn_limit=assistant_turn_limit,
        )

    # Single model from config (CLI --model flag overrides this)
    model = cfg.get("model")
    reasoning_effort = _reasoning_effort_from_config(cfg)

    return Task(
        dataset=dataset,
        solver=slvr,
        scorer=refusal_scorer(judge_model=judge_model),
        model=model,
        config=GenerateConfig(reasoning_effort=reasoning_effort),
    )


def get_models(config_path: str = "config.yaml") -> list[str]:
    """Return the list of student models from config, for use by runner scripts."""
    cfg = _load_config(config_path)
    models = cfg.get("models", [])
    # Also accept singular "model" key
    if not models and cfg.get("model"):
        models = [cfg["model"]]
    return models
