"""Build inspect Sample list from benchmark tasks × framings."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import yaml
from inspect_ai.dataset import MemoryDataset, Sample

from refusal_eval.prompts import build_phase_prompts, build_task_prompt
from refusal_eval.sandbox import generate_compose

ALL_PHASES = ("analysis", "poc", "exploit")


@dataclass
class TaskInfo:
    """Parsed info from a benchmark task directory."""

    task_id: str
    category: str
    difficulty: int
    target_host: str
    flag: str
    task_dir: str
    has_metadata: bool
    exploit_goal: str = ""
    goal_vector: list[int] | None = None


def parse_task(task_dir: str) -> tuple[TaskInfo, dict]:
    """Read standardized metadata.json from benchmark task root.

    Returns (TaskInfo, raw_metadata_dict) so callers can pass the raw
    dict downstream without re-reading the file.
    """
    task_dir = os.path.abspath(task_dir)
    meta_path = os.path.join(task_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    info = TaskInfo(
        task_id=meta["task_id"],
        category=meta["category"],
        difficulty=meta.get("difficulty", 0),
        target_host=meta.get("target_host", "target:8080"),
        flag=meta.get("flag", ""),
        task_dir=task_dir,
        has_metadata=True,
        exploit_goal=meta.get("exploit_goal", ""),
        goal_vector=meta.get("goal_vector", [0, 0, 1]),
    )
    return info, meta


def resolve_manifest(
    manifest_path: str,
    task_ids: list[str] | None = None,
    exclude_task_ids: list[str] | None = None,
) -> list[str]:
    """Resolve a v2 manifest to a list of task directory paths."""
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    excluded = set(exclude_task_ids or [])
    with open(manifest_path) as f:
        data = yaml.safe_load(f)
    dirs = []
    for task in data.get("tasks", []):
        tid = task["task_id"]
        if task_ids and tid not in task_ids:
            continue
        if tid in excluded:
            continue
        task_dir = os.path.join(manifest_dir, task["dir"])
        if os.path.isdir(task_dir):
            dirs.append(task_dir)
    return dirs


def _collect_files(task_dir: str) -> dict[str, str]:
    """Map task files to sandbox paths: {sandbox_path: local_path}.

    Reads directly from the task directory — no intermediate copy.
    Includes vulnerability.txt and everything under source/.
    """
    files: dict[str, str] = {}
    vuln = os.path.join(task_dir, "vulnerability.txt")
    if os.path.isfile(vuln):
        files["/root/vulnerability.txt"] = vuln
    src = os.path.join(task_dir, "source")
    if os.path.isdir(src):
        for root, _dirs, filenames in os.walk(src):
            for fname in filenames:
                local_path = os.path.join(root, fname)
                rel_path = os.path.relpath(local_path, task_dir)
                files[f"/root/{rel_path}"] = local_path
    return files


def _partition_files(
    files: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Split files into source-only (Phase 1) and all files (Phases 2-3).

    Phase 1 (analysis) receives only source code — no vulnerability description.
    Phases 2-3 (PoC, exploit) receive everything including vulnerability.txt.
    """
    vuln_names = {"vulnerability.txt", "vuln_description.txt"}
    source_files = {
        k: v for k, v in files.items()
        if os.path.basename(k) not in vuln_names
    }
    return source_files, files


_VALIDATION_DISABLED_ACCURACY = {
    "analysis": {"skip": True, "reason": "validation_disabled_by_config"},
    "poc": {"skip": True, "reason": "validation_disabled_by_config"},
    "exploit": {"skip": True, "reason": "validation_disabled_by_config"},
}

_PHASE_NOT_RUN_ACCURACY = {
    "skip": True,
    "reason": "phase_not_run",
    "notes": "",
}


def normalize_phases(phases: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    """Normalize supported phase-selection forms.

    The first supported subset is intentionally narrow: analysis-only for
    Phase 1 experiments, plus the default/full three-phase path.
    """
    if phases is None:
        return None

    if isinstance(phases, str):
        selected = [part.strip() for part in phases.split(",") if part.strip()]
    elif isinstance(phases, (list, tuple)):
        selected = []
        for phase in phases:
            if not isinstance(phase, str):
                raise ValueError("phases entries must be strings.")
            selected.append(phase.strip())
    else:
        raise ValueError("phases must be a string or list of strings.")

    if selected == ["analysis"] or selected == list(ALL_PHASES):
        return selected

    raise ValueError(
        "Unsupported phases selection. Expected ['analysis'] for analysis-only "
        "or ['analysis', 'poc', 'exploit'] for the full multistep run."
    )


def _accuracy_for_selected_phases(
    accuracy: dict | None,
    selected_phases: list[str] | None,
) -> dict | None:
    """Return accuracy metadata with unrun phases explicitly skipped."""
    if selected_phases is None or selected_phases == list(ALL_PHASES):
        return accuracy

    out = dict(accuracy or {})
    selected = set(selected_phases)
    for phase in ALL_PHASES:
        if phase not in selected:
            out[phase] = dict(_PHASE_NOT_RUN_ACCURACY)
    return out


def build_dataset(
    task_dirs: list[str],
    framings: list[str],
    mode: str = "multistep",
    refusal_token_values: list[bool] | None = None,
    validated: bool = True,
    phases: str | list[str] | tuple[str, ...] | None = None,
    sandbox_resources: dict | None = None,
) -> MemoryDataset:
    """Build a MemoryDataset of Samples from task directories × framings × refusal_token.

    Parameters
    ----------
    task_dirs : list[str]
        Paths to benchmark task directories.
    framings : list[str]
        Framing names (e.g. "ctf", "bare", "authorized").
    mode : str
        "oneshot" or "multistep".
    refusal_token_values : list[bool] | None
        List of refusal_token settings to cross-product. Defaults to [True].
    validated : bool
        Default True. Controls whether per-task validation runs in this
        evaluation:
          - True:  honor each task's ``validation:`` block and ``accuracy``
                   field as authored in metadata. Grader/controller sidecars
                   are spliced into the compose for tasks that opted in.
          - False: bypass all in-eval validation. Drops every task's
                   ``validation`` block before sandbox composition (so neither
                   grader nor controller sidecars are spliced in) and overrides
                   ``accuracy`` so every phase reports ``skip=True`` with reason
                   ``validation_disabled_by_config``. Useful for capability-only
                   runs against datasets whose validation infrastructure is
                   missing or intentionally bypassed.
    """
    if refusal_token_values is None:
        refusal_token_values = [True]
    selected_phases = normalize_phases(phases)
    analysis_only = selected_phases == ["analysis"]
    if selected_phases is not None and mode != "multistep":
        raise ValueError("phases is only supported for multistep evaluations.")

    samples: list[Sample] = []

    for task_dir in task_dirs:
        task, raw_meta = parse_task(task_dir)

        # Build file maps directly from task directory (no temp copy)
        files = _collect_files(task.task_dir)
        source_files, all_files = _partition_files(files)

        if not validated:
            compose_meta = {k: v for k, v in raw_meta.items() if k != "validation"}
            sample_validation = None
            sample_accuracy = dict(_VALIDATION_DISABLED_ACCURACY)
        elif analysis_only:
            compose_meta = {k: v for k, v in raw_meta.items() if k != "validation"}
            sample_validation = None
            sample_accuracy = _accuracy_for_selected_phases(
                raw_meta.get("accuracy"), selected_phases,
            )
        else:
            compose_meta = raw_meta
            sample_validation = raw_meta.get("validation")
            sample_accuracy = _accuracy_for_selected_phases(
                raw_meta.get("accuracy"), selected_phases,
            )

        # Generate sandbox compose file (pass metadata to avoid re-reading).
        # When the task has a ``validation:`` block, both grader and
        # controller sidecars are spliced in and we get a controller_token
        # back; the solver authenticates /restart calls with it. The grader's
        # HMAC token is rotated per phase by the controller and never lives
        # in sample metadata.
        compose_info = generate_compose(
            task.task_dir,
            metadata=compose_meta,
            return_info=True,
            sandbox_resources=sandbox_resources,
        )
        compose_path = compose_info["compose_path"]
        controller_token = compose_info.get("controller_token")

        for framing in framings:
            # Build prompts
            if mode == "multistep":
                phase_prompts = build_phase_prompts(
                    task.category, task.target_host, framing,
                    exploit_goal=task.exploit_goal,
                )
                if selected_phases is not None:
                    selected_set = set(selected_phases)
                    phase_prompts = [
                        phase for phase in phase_prompts
                        if phase["label"] in selected_set
                    ]
                # Phase 1 prompt is the initial input
                prompt = phase_prompts[0]["prompt"]
            else:
                phase_prompts = []
                prompt = build_task_prompt(
                    task.category, task.target_host, framing,
                    exploit_goal=task.exploit_goal,
                )

            for rt in refusal_token_values:
                rt_suffix = "token" if rt else "notoken"
                sample = Sample(
                    id=f"{task.task_id}_{framing}_{mode}_{rt_suffix}",
                    input=prompt,
                    target=task.flag,
                    sandbox=("docker", compose_path),
                    metadata={
                        "task_dir": task.task_dir,
                        "task_id": task.task_id,
                        "category": task.category,
                        "difficulty": task.difficulty,
                        "framing": framing,
                        "mode": mode,
                        "target_host": task.target_host,
                        "has_metadata": task.has_metadata,
                        "phase_prompts": phase_prompts,
                        "selected_phases": selected_phases or list(ALL_PHASES),
                        "refusal_token": rt,
                        "source_files": source_files,
                        "all_files": all_files,
                        "goal_vector": task.goal_vector,
                        "accuracy": sample_accuracy,
                        "validation": sample_validation,
                        "controller_token": controller_token,
                        "flag": task.flag,
                    },
                )
                samples.append(sample)

    return MemoryDataset(samples, name="cyberagent-bench")
