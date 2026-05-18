#!/usr/bin/env python3
"""Run inspect eval for each student model listed in config.

Usage:
    python scripts/run_models.py config-robin.yaml
    python scripts/run_models.py config-robin.yaml --max-tasks 4 --epochs 3
    python scripts/run_models.py config-robin.yaml --no-prewarm

Any extra arguments are forwarded to `inspect eval`.

Image prewarm: when the config references the ``vulhub`` source, this script
invokes ``prewarm_vulhub_images.py`` once before the first model runs so the
host docker cache is hot. Back-to-back model evaluations then reuse the
cached images (docker compose's default ``pull_policy: missing`` skips pulls
when an image with the requested tag is already local). Pass ``--no-prewarm``
to skip the warm-up.
"""

from __future__ import annotations

import os
import subprocess
import sys

import yaml

sys.path.insert(0, ".")
from refusal_eval.task import get_models

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def _validate_model_strings(models: list[str]) -> None:
    """Fail early when a configured model omits its provider prefix."""
    for model in models:
        if "/" not in model:
            print(
                f"ERROR: Model '{model}' is missing a provider prefix "
                f"(e.g. 'openai/', 'anthropic/', 'bedrock/')."
            )
            sys.exit(1)


def _load_config(config_path: str) -> dict:
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    if not os.path.isfile(config_path):
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _config_sources(config_path: str) -> set[str]:
    """Return the set of benchmark source names referenced by the config."""
    cfg = _load_config(config_path)
    sources: set[str] = set()
    for entry in (cfg.get("task") or {}).get("benchmark", []) or []:
        if isinstance(entry, dict) and entry.get("source"):
            sources.add(entry["source"])
    return sources


def _vulhub_task_filters(config_path: str) -> tuple[set[str], set[str]]:
    """Return task include/exclude filters for Vulhub prewarm."""
    cfg = _load_config(config_path)
    includes: set[str] = set()
    excludes: set[str] = set()
    for entry in (cfg.get("task") or {}).get("benchmark", []) or []:
        if not isinstance(entry, dict) or entry.get("source") != "vulhub":
            continue
        includes.update(entry.get("tasks") or [])
        excludes.update(entry.get("exclude_tasks") or [])
    return includes, excludes


def _prewarm_images(config_path: str) -> None:
    """Ensure docker images are cached before the model loop kicks off.

    Currently warms only ``vulhub`` (the heaviest source by image count and
    cold-pull latency). Skips silently for configs that don't include vulhub.
    """
    sources = _config_sources(config_path)
    if "vulhub" not in sources:
        print("[prewarm] vulhub not in config; skipping image prewarm.")
        return
    print("[prewarm] Running prewarm_vulhub_images.py to populate docker cache once.")
    cmd = [sys.executable, "scripts/prewarm_vulhub_images.py"]
    include_tasks, exclude_tasks = _vulhub_task_filters(config_path)
    if include_tasks:
        cmd.extend(["--only-tasks", ",".join(sorted(include_tasks))])
    if exclude_tasks:
        cmd.extend(["--exclude-tasks", ",".join(sorted(exclude_tasks))])
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(
            "[prewarm] WARN: prewarm exited non-zero; continuing — uncached "
            "images may be pulled on demand by docker compose.",
        )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_models.py <config.yaml> [--no-prewarm] [inspect eval flags...]")
        sys.exit(1)

    config_path = sys.argv[1]
    raw_extra = sys.argv[2:]
    prewarm = True
    extra_args: list[str] = []
    for arg in raw_extra:
        if arg == "--no-prewarm":
            prewarm = False
        else:
            extra_args.append(arg)

    models = get_models(config_path)
    if not models:
        print(f"No models found in {config_path}. Add a 'models' list.")
        sys.exit(1)

    _validate_model_strings(models)

    if prewarm:
        _prewarm_images(config_path)

    print(f"Running {len(models)} model(s) from {config_path}")
    for i, model in enumerate(models, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(models)}] {model}")
        print(f"{'='*60}")
        cmd = [
            "inspect", "eval", "refusal_eval/task.py",
            "--model", model,
            "-T", f"config_path={config_path}",
            *extra_args,
        ]
        print(f"  {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  WARN: {model} exited with code {result.returncode}")

    print(f"\nDone. Ran {len(models)} model(s).")


if __name__ == "__main__":
    main()
