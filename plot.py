"""Generate plots from Inspect eval logs.

Usage:
    python3 plot.py logs/2026-03-24T20-06-52.eval
    python3 plot.py logs/                              # all .eval files in dir
    python3 plot.py logs/run1.eval logs/run2.eval      # compare multiple runs
    python3 plot.py logs/ --out plots/ --error-bars
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from inspect_ai.log import EvalLog, read_eval_log


PLOT_RECORD_CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def _extract_records(log: EvalLog) -> list[dict]:
    """Extract flat per-sample records from an EvalLog."""
    model = str(log.eval.model) if log.eval.model else "unknown"
    records = []

    for sample in log.samples or []:
        meta = sample.metadata or {}
        score_obj = sample.scores.get("refusal_scorer") if sample.scores else None
        score_meta = score_obj.metadata if score_obj and score_obj.metadata else {}

        # Token usage
        input_tokens = 0
        output_tokens = 0
        if sample.model_usage:
            for usage in sample.model_usage.values():
                input_tokens += usage.input_tokens or 0
                output_tokens += usage.output_tokens or 0

        records.append({
            "model": model,
            "task_id": meta.get("task_id", sample.id or "unknown"),
            "category": meta.get("category", "unknown"),
            "framing": meta.get("framing", "unknown"),
            "mode": meta.get("mode", "oneshot"),
            "refused": score_meta.get("refused", False),
            "refusal_source": score_meta.get("refusal_source"),
            "phase_results": score_meta.get("phase_results", []),
            "goal_vector": meta.get("goal_vector", [0, 0, 1]),
            "refusal_vector": score_meta.get("refusal_vector", []),
            "alignment": score_meta.get("alignment"),
            "total_time": sample.total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

    return records


def _file_stamp(path: str) -> dict[str, int]:
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _load_record_cache(cache_path: str | None) -> dict:
    if not cache_path or not os.path.isfile(cache_path):
        return {"version": PLOT_RECORD_CACHE_VERSION, "logs": {}}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"version": PLOT_RECORD_CACHE_VERSION, "logs": {}}
    if cache.get("version") != PLOT_RECORD_CACHE_VERSION:
        return {"version": PLOT_RECORD_CACHE_VERSION, "logs": {}}
    if not isinstance(cache.get("logs"), dict):
        return {"version": PLOT_RECORD_CACHE_VERSION, "logs": {}}
    return cache


def _write_record_cache(cache_path: str | None, cache: dict) -> None:
    if not cache_path:
        return
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
        f.write("\n")


def load_records(
    paths: list[str],
    latest: bool = False,
    *,
    cache_path: str | None = None,
    refresh_cache: bool = False,
) -> list[dict]:
    """Load records from one or more .eval files or directories."""
    eval_files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            eval_files.extend(sorted(glob.glob(os.path.join(p, "*.eval"))))
        elif os.path.isfile(p):
            eval_files.append(p)
        else:
            print(f"WARN: {p} not found, skipping", file=sys.stderr)

    if not eval_files:
        print("No .eval files found.", file=sys.stderr)
        sys.exit(1)

    if latest:
        eval_files = [eval_files[-1]]

    cache = _load_record_cache(cache_path)
    cached_logs = cache.setdefault("logs", {})
    records: list[dict] = []
    for f in eval_files:
        abs_path = os.path.abspath(f)
        stamp = _file_stamp(abs_path)
        cached = cached_logs.get(abs_path)
        if (
            not refresh_cache
            and isinstance(cached, dict)
            and cached.get("size") == stamp["size"]
            and cached.get("mtime_ns") == stamp["mtime_ns"]
            and isinstance(cached.get("records"), list)
        ):
            extracted = cached["records"]
            records.extend(extracted)
            print(f"  Cached {f} ({len(extracted)} samples, model={cached.get('model', 'unknown')})")
            continue

        log = read_eval_log(f)
        extracted = _extract_records(log)
        records.extend(extracted)
        cached_logs[abs_path] = {
            **stamp,
            "model": str(log.eval.model) if log.eval.model else "unknown",
            "records": extracted,
        }
        print(f"  Loaded {f} ({len(log.samples or [])} samples, model={log.eval.model})")

    if cache_path:
        live_paths = {os.path.abspath(f) for f in eval_files}
        for cached_path in list(cached_logs):
            if cached_path not in live_paths and not os.path.isfile(cached_path):
                del cached_logs[cached_path]
        _write_record_cache(cache_path, cache)
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def short_task_id(task_id: str) -> str:
    for prefix in ("[Very Easy] ", "[Easy] ", "[Medium] ", "[Hard] ", "[Insane] "):
        if task_id.startswith(prefix):
            return task_id[len(prefix):]
    # CyberGym IDs (e.g. oss-fuzz_42535201) are already short
    return task_id


def _task_label(task_id: str, framing: str, multi_framing: bool) -> str:
    name = short_task_id(task_id)
    if multi_framing:
        return f"{name}\n({framing})"
    return name


def _se_proportion(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return (p * (1 - p) / n) ** 0.5


def _has_multi_framing(records: list[dict]) -> bool:
    return len(set(r["framing"] for r in records)) > 1


def _has_multi_model(records: list[dict]) -> bool:
    return len(set(r["model"] for r in records)) > 1


def _title_suffix(records: list[dict]) -> str:
    models = sorted(set(_short_model_name(r["model"]) for r in records))
    return ", ".join(models)


# ---------------------------------------------------------------------------
# Plot 1: Per-phase refusal rate (aggregated across all tasks)
# ---------------------------------------------------------------------------


def plot_phase_refusal_aggregate(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    multistep = [r for r in records if r["mode"] == "multistep" and r["phase_results"]]
    if not multistep:
        return

    phase_refused: dict[str, int] = defaultdict(int)
    phase_total: dict[str, int] = defaultdict(int)

    for r in multistep:
        for p in r["phase_results"]:
            phase_refused[p["label"]] += int(p["refused"])
            phase_total[p["label"]] += 1

    labels = list(dict.fromkeys(p["label"] for p in multistep[0]["phase_results"]))
    rates = [phase_refused[l] / phase_total[l] if phase_total[l] else 0 for l in labels]
    ses = [_se_proportion(r, phase_total[l]) for r, l in zip(rates, labels)]
    counts = [f"{phase_refused[l]}/{phase_total[l]}" for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    eb = {"yerr": ses, "capsize": 4} if error_bars else {}
    bars = ax.bar(labels, rates, **eb, color=["#4CAF50", "#FF9800", "#F44336"])
    ax.set_ylabel("Refusal Rate")
    ax.set_title(f"Refusal Rate by Phase (all tasks)\n{_title_suffix(records)}")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    for bar, count, se in zip(bars, counts, ses):
        offset = (se + 0.02) if error_bars else 0.02
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                count, ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_refusal_aggregate.png"), dpi=150)
    plt.close(fig)
    print(f"  phase_refusal_aggregate.png")


# ---------------------------------------------------------------------------
# Plot 2: Per-phase refusal rate broken down by task
# ---------------------------------------------------------------------------


def plot_phase_refusal_per_task(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    multistep = [r for r in records if r["mode"] == "multistep" and r["phase_results"]]
    if not multistep:
        return

    labels = list(dict.fromkeys(p["label"] for p in multistep[0]["phase_results"]))
    mf = _has_multi_framing(records)

    # Group by (task_id, framing)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in multistep:
        groups[(r["task_id"], r["framing"])].append(r)

    group_keys = list(groups.keys())
    task_labels = [_task_label(tid, fr, mf) for tid, fr in group_keys]
    n_tasks = len(group_keys)
    n_phases = len(labels)

    x = np.arange(n_tasks)
    width = 0.8 / n_phases
    colors = ["#4CAF50", "#FF9800", "#F44336", "#2196F3", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(max(6, n_tasks * 2), 4))

    for i, label in enumerate(labels):
        rates = []
        ses = []
        for key in group_keys:
            samples = groups[key]
            n = len(samples)
            refused = sum(
                1 for s in samples
                for p in s["phase_results"] if p["label"] == label and p["refused"]
            )
            r = refused / n if n else 0
            rates.append(r)
            ses.append(_se_proportion(r, n))
        offset = (i - n_phases / 2 + 0.5) * width
        eb = {"yerr": ses, "capsize": 3} if error_bars else {}
        ax.bar(x + offset, rates, width, **eb, label=label, color=colors[i % len(colors)])

    ax.set_ylabel("Refusal Rate")
    ax.set_title(f"Refusal Rate by Phase per Task\n{_title_suffix(records)}")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_refusal_per_task.png"), dpi=150)
    plt.close(fig)
    print(f"  phase_refusal_per_task.png")


# ---------------------------------------------------------------------------
# Plot 3: Token vs Judge refusal detection
# ---------------------------------------------------------------------------


def plot_token_vs_judge(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    """Stacked bar showing refusals caught by token vs only by judge, aggregated across all tasks."""
    token_count = 0
    judge_only_count = 0
    api_count = 0
    total = 0

    for r in records:
        if r["mode"] == "multistep" and r["phase_results"]:
            for p in r["phase_results"]:
                total += 1
                src = p.get("refusal_source")
                if src == "api_error":
                    api_count += 1
                elif src == "token":
                    token_count += 1
                elif src == "judge":
                    judge_only_count += 1
        else:
            total += 1
            src = r.get("refusal_source")
            if src == "api_error":
                api_count += 1
            elif src == "token":
                token_count += 1
            elif src == "judge":
                judge_only_count += 1

    if total == 0:
        return

    not_refused = total - token_count - judge_only_count - api_count
    categories = ["Token\n(<<<REFUSE>>>)", "Judge only\n(LLM classifier)", "API error\n(content filter)"]
    values = [token_count, judge_only_count, api_count]
    colors = ["#2196F3", "#FF9800", "#9E9E9E"]

    # Filter out categories with zero counts
    active = [(c, v, col) for c, v, col in zip(categories, values, colors) if v > 0]
    if not active:
        return

    cats, vals, cols = zip(*active)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(cats, vals, color=cols)

    for bar, v in zip(bars, vals):
        pct = v / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v} ({pct:.0f}%)", ha="center", va="bottom", fontsize=10)

    total_refused = token_count + judge_only_count + api_count
    ax.set_ylabel("Refusal Count")
    ax.set_title(
        f"Refusal Detection Method\n"
        f"{total_refused} refusals out of {total} phases — "
        f"{_title_suffix(records)}"
    )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "token_vs_judge.png"), dpi=150)
    plt.close(fig)
    print(f"  token_vs_judge.png")


# ---------------------------------------------------------------------------
# Plot 4: Overall refusal rate by task
# ---------------------------------------------------------------------------


def plot_refusal_by_task(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    mf = _has_multi_framing(records)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["task_id"], r["framing"])].append(r)

    task_labels = []
    rates = []
    ses = []
    counts = []

    for (tid, fr), samples in groups.items():
        task_labels.append(_task_label(tid, fr, mf))
        n = len(samples)
        refused = sum(1 for s in samples if s["refused"])
        rate = refused / n if n else 0
        rates.append(rate)
        ses.append(_se_proportion(rate, n))
        counts.append(f"{refused}/{n}")

    if not task_labels:
        return

    x = np.arange(len(task_labels))
    fig, ax = plt.subplots(figsize=(max(6, len(task_labels) * 1.5), 4))
    eb = {"yerr": ses, "capsize": 4} if error_bars else {}
    bars = ax.bar(x, rates, **eb, color="#F44336")
    ax.set_ylabel("Refusal Rate")
    ax.set_title(f"Overall Refusal Rate by Task\n{_title_suffix(records)}")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    for bar, count, se in zip(bars, counts, ses):
        offset = (se + 0.02) if error_bars else 0.02
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                count, ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "refusal_by_task.png"), dpi=150)
    plt.close(fig)
    print(f"  refusal_by_task.png")


# ---------------------------------------------------------------------------
# Plot 5: Time per sample
# ---------------------------------------------------------------------------


def plot_time_per_sample(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    mf = _has_multi_framing(records)

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        if r["total_time"] is not None:
            groups[(r["task_id"], r["framing"])].append(r["total_time"])

    if not groups:
        return

    task_labels = []
    time_data = []
    for (tid, fr), times in groups.items():
        task_labels.append(_task_label(tid, fr, mf))
        time_data.append(times)

    fig, ax = plt.subplots(figsize=(max(6, len(task_labels) * 1.5), 4))
    bp = ax.boxplot(time_data, patch_artist=True)
    ax.set_xticks(range(1, len(task_labels) + 1))
    ax.set_xticklabels(task_labels, rotation=30, ha="right")
    for patch in bp["boxes"]:
        patch.set_facecolor("#E3F2FD")

    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"Time per Sample\n{_title_suffix(records)}")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "time_per_sample.png"), dpi=150)
    plt.close(fig)
    print(f"  time_per_sample.png")


# ---------------------------------------------------------------------------
# Plot 6: Refusal source breakdown
# ---------------------------------------------------------------------------


def plot_refusal_sources(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    mf = _has_multi_framing(records)
    source_labels = ["api_error", "token", "judge"]
    source_colors = {
        "api_error": "#9E9E9E",
        "token": "#2196F3",
        "judge": "#FF9800",
    }

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["task_id"], r["framing"])].append(r)

    task_labels: list[str] = []
    source_counts: dict[str, list[int]] = defaultdict(list)

    for (tid, fr), samples in groups.items():
        task_labels.append(_task_label(tid, fr, mf))
        counts: dict[str, int] = defaultdict(int)

        for s in samples:
            if s["mode"] == "multistep":
                for p in s["phase_results"]:
                    src = p.get("refusal_source")
                    if src:
                        counts[src] += 1
            else:
                src = s.get("refusal_source")
                if src:
                    counts[src] += 1

        for src in source_labels:
            source_counts[src].append(counts.get(src, 0))

    if not task_labels:
        return

    active_sources = [s for s in source_labels if any(c > 0 for c in source_counts[s])]
    if not active_sources:
        return

    x = np.arange(len(task_labels))
    fig, ax = plt.subplots(figsize=(max(6, len(task_labels) * 1.5), 4))

    bottom = np.zeros(len(task_labels))
    for src in active_sources:
        values = np.array(source_counts[src], dtype=float)
        ax.bar(x, values, bottom=bottom, label=src, color=source_colors[src])
        bottom += values

    ax.set_ylabel("Refusal Count")
    ax.set_title(f"Refusal Detection Sources\n{_title_suffix(records)}")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=30, ha="right")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "refusal_sources.png"), dpi=150)
    plt.close(fig)
    print(f"  refusal_sources.png")


# ---------------------------------------------------------------------------
# Plot 7: Phase refusal rate by framing
# ---------------------------------------------------------------------------


def plot_phase_refusal_by_framing(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    framings = sorted(set(r["framing"] for r in records))
    if len(framings) < 2:
        return

    multistep = [r for r in records if r["mode"] == "multistep" and r["phase_results"]]

    if not multistep:
        # Oneshot fallback: single bar per framing
        x = np.arange(len(framings))
        fig, ax = plt.subplots(figsize=(max(6, len(framings) * 2), 4))
        rates = []
        ses = []
        counts = []
        for framing in framings:
            fr_records = [r for r in records if r["framing"] == framing]
            n = len(fr_records)
            refused = sum(1 for r in fr_records if r["refused"])
            r = refused / n if n else 0
            rates.append(r)
            ses.append(_se_proportion(r, n))
            counts.append(f"{refused}/{n}")

        eb = {"yerr": ses, "capsize": 4} if error_bars else {}
        bars = ax.bar(framings, rates, **eb, color="#F44336")
        for bar, count, se in zip(bars, counts, ses):
            offset = (se + 0.02) if error_bars else 0.02
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    count, ha="center", va="bottom", fontsize=10)

        ax.set_ylabel("Refusal Rate")
        ax.set_title(f"Refusal Rate by Framing\n{_title_suffix(records)}")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "phase_refusal_by_framing.png"), dpi=150)
        plt.close(fig)
        print(f"  phase_refusal_by_framing.png")
        return

    # Multistep: grouped bars per framing, one bar per phase
    phase_labels = list(dict.fromkeys(p["label"] for p in multistep[0]["phase_results"]))
    n_framings = len(framings)
    n_phases = len(phase_labels)
    colors = ["#4CAF50", "#FF9800", "#F44336", "#2196F3", "#9C27B0"]

    x = np.arange(n_framings)
    width = 0.8 / n_phases

    fig, ax = plt.subplots(figsize=(max(6, n_framings * 2.5), 4))

    for i, phase in enumerate(phase_labels):
        rates = []
        ses = []
        for framing in framings:
            fr_records = [r for r in multistep if r["framing"] == framing]
            n = len(fr_records)
            refused = sum(
                1 for r in fr_records
                for p in r["phase_results"] if p["label"] == phase and p["refused"]
            )
            r = refused / n if n else 0
            rates.append(r)
            ses.append(_se_proportion(r, n))
        offset = (i - n_phases / 2 + 0.5) * width
        eb = {"yerr": ses, "capsize": 3} if error_bars else {}
        ax.bar(x + offset, rates, width, **eb, label=phase, color=colors[i % len(colors)])

    ax.set_ylabel("Refusal Rate")
    ax.set_title(f"Phase Refusal Rate by Framing\n{_title_suffix(records)}")
    ax.set_xticks(x)
    ax.set_xticklabels(framings)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_refusal_by_framing.png"), dpi=150)
    plt.close(fig)
    print(f"  phase_refusal_by_framing.png")


# ---------------------------------------------------------------------------
# Plot 8: Refusal rate by category (pwn vs memory-safety)
# ---------------------------------------------------------------------------


def plot_refusal_by_category(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    """Compare refusal rates across task categories (e.g. pwn vs memory-safety)."""
    categories = sorted(set(r["category"] for r in records))
    if len(categories) < 2:
        return

    multistep = [r for r in records if r["mode"] == "multistep" and r["phase_results"]]

    if not multistep:
        # Oneshot: single bar per category
        x = np.arange(len(categories))
        fig, ax = plt.subplots(figsize=(max(6, len(categories) * 2), 4))
        rates = []
        ses = []
        counts = []
        for cat in categories:
            cat_records = [r for r in records if r["category"] == cat]
            n = len(cat_records)
            refused = sum(1 for r in cat_records if r["refused"])
            r = refused / n if n else 0
            rates.append(r)
            ses.append(_se_proportion(r, n))
            counts.append(f"{refused}/{n}")

        eb = {"yerr": ses, "capsize": 4} if error_bars else {}
        bars = ax.bar(categories, rates, **eb, color=["#4CAF50", "#F44336", "#FF9800", "#2196F3"])
        for bar, count, se in zip(bars, counts, ses):
            offset = (se + 0.02) if error_bars else 0.02
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    count, ha="center", va="bottom", fontsize=10)

        ax.set_ylabel("Refusal Rate")
        ax.set_title(f"Refusal Rate by Category\n{_title_suffix(records)}")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "refusal_by_category.png"), dpi=150)
        plt.close(fig)
        print(f"  refusal_by_category.png")
        return

    # Multistep: grouped bars per category, one bar per phase
    phase_labels = list(dict.fromkeys(p["label"] for p in multistep[0]["phase_results"]))
    n_categories = len(categories)
    n_phases = len(phase_labels)
    colors = ["#4CAF50", "#FF9800", "#F44336", "#2196F3", "#9C27B0"]

    x = np.arange(n_categories)
    width = 0.8 / n_phases

    fig, ax = plt.subplots(figsize=(max(6, n_categories * 2.5), 4))

    for i, phase in enumerate(phase_labels):
        rates = []
        ses = []
        for cat in categories:
            cat_records = [r for r in multistep if r["category"] == cat]
            n = len(cat_records)
            refused = sum(
                1 for r in cat_records
                for p in r["phase_results"] if p["label"] == phase and p["refused"]
            )
            r = refused / n if n else 0
            rates.append(r)
            ses.append(_se_proportion(r, n))
        offset = (i - n_phases / 2 + 0.5) * width
        eb = {"yerr": ses, "capsize": 3} if error_bars else {}
        ax.bar(x + offset, rates, width, **eb, label=phase, color=colors[i % len(colors)])

    ax.set_ylabel("Refusal Rate")
    ax.set_title(f"Phase Refusal Rate by Category\n{_title_suffix(records)}")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "refusal_by_category.png"), dpi=150)
    plt.close(fig)
    print(f"  refusal_by_category.png")


# ---------------------------------------------------------------------------
# Plot 9: Refusal rate by source (CyBench vs CyberGym vs Exploit-DB)
# ---------------------------------------------------------------------------

# Map task category to data source
_CATEGORY_SOURCE = {
    "pwn": "CyBench",
    "web": "CyBench",
    "crypto": "CyBench",
    "forensics": "CyBench",
    "reversing": "CyBench",
    "misc": "CyBench",
    "memory-safety": "CyberGym",
    "remote": "Exploit-DB",
    "dos": "Exploit-DB",
}


def _get_source(record: dict) -> str:
    return _CATEGORY_SOURCE.get(record["category"], "Unknown")


def plot_refusal_by_source(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    """Compare refusal rates across task sources (CyBench vs CyberGym vs Exploit-DB)."""
    sources = sorted(set(_get_source(r) for r in records))
    if len(sources) < 2:
        return

    multistep = [r for r in records if r["mode"] == "multistep" and r["phase_results"]]

    source_colors = {
        "CyBench": "#4CAF50",
        "CyberGym": "#2196F3",
        "Exploit-DB": "#F44336",
        "Unknown": "#9E9E9E",
    }

    if not multistep:
        # Oneshot: single bar per source
        x = np.arange(len(sources))
        fig, ax = plt.subplots(figsize=(max(6, len(sources) * 2), 4))
        rates = []
        ses = []
        counts = []
        for src in sources:
            src_records = [r for r in records if _get_source(r) == src]
            n = len(src_records)
            refused = sum(1 for r in src_records if r["refused"])
            r = refused / n if n else 0
            rates.append(r)
            ses.append(_se_proportion(r, n))
            counts.append(f"{refused}/{n}")

        colors = [source_colors.get(s, "#9E9E9E") for s in sources]
        eb = {"yerr": ses, "capsize": 4} if error_bars else {}
        bars = ax.bar(sources, rates, **eb, color=colors)
        for bar, count, se in zip(bars, counts, ses):
            offset = (se + 0.02) if error_bars else 0.02
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    count, ha="center", va="bottom", fontsize=10)

        ax.set_ylabel("Refusal Rate")
        ax.set_title(f"Refusal Rate by Source\n{_title_suffix(records)}")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "refusal_by_source.png"), dpi=150)
        plt.close(fig)
        print(f"  refusal_by_source.png")
        return

    # Multistep: grouped bars per source, one bar per phase
    phase_labels = list(dict.fromkeys(p["label"] for p in multistep[0]["phase_results"]))
    n_sources = len(sources)
    n_phases = len(phase_labels)
    phase_colors = ["#4CAF50", "#FF9800", "#F44336", "#2196F3", "#9C27B0"]

    x = np.arange(n_sources)
    width = 0.8 / n_phases

    fig, ax = plt.subplots(figsize=(max(6, n_sources * 2.5), 4))

    for i, phase in enumerate(phase_labels):
        rates = []
        ses = []
        for src in sources:
            src_records = [r for r in multistep if _get_source(r) == src]
            n = len(src_records)
            refused = sum(
                1 for r in src_records
                for p in r["phase_results"] if p["label"] == phase and p["refused"]
            )
            r = refused / n if n else 0
            rates.append(r)
            ses.append(_se_proportion(r, n))
        offset = (i - n_phases / 2 + 0.5) * width
        eb = {"yerr": ses, "capsize": 3} if error_bars else {}
        ax.bar(x + offset, rates, width, **eb, label=phase, color=phase_colors[i % len(phase_colors)])

    ax.set_ylabel("Refusal Rate")
    ax.set_title(f"Phase Refusal Rate by Source\n{_title_suffix(records)}")
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "refusal_by_source.png"), dpi=150)
    plt.close(fig)
    print(f"  refusal_by_source.png")


# ---------------------------------------------------------------------------
# Aggregate metrics (situational awareness, utility, safety)
# ---------------------------------------------------------------------------

_EPS = 1e-7


def _compute_aggregate_metrics(records: list[dict]) -> dict[str, dict]:
    """Compute per-model aggregate metrics from multistep records.

    Returns {model: {utility, safety, situational_awareness, refusal_rates}}.
    refusal_rates is [a, b, c] — per-phase refusal rates.
    """
    multistep = [r for r in records if r["mode"] == "multistep" and r["phase_results"]]
    if not multistep:
        return {}

    phase_labels = list(dict.fromkeys(p["label"] for p in multistep[0]["phase_results"]))

    models = sorted(set(r["model"] for r in multistep))
    result: dict[str, dict] = {}

    for model in models:
        model_records = [r for r in multistep if r["model"] == model]
        n = len(model_records)
        if n == 0:
            continue

        # Per-phase refusal rates
        phase_rates: list[float] = []
        for label in phase_labels:
            refused = sum(
                1 for r in model_records
                for p in r["phase_results"] if p["label"] == label and p["refused"]
            )
            phase_rates.append(refused / n)

        # Average goal vector across all samples
        goal_vectors = [r.get("goal_vector", [0, 0, 1]) for r in model_records]
        avg_goal = [
            sum(gv[i] for gv in goal_vectors) / len(goal_vectors)
            for i in range(min(len(phase_labels), 3))
        ]

        # Cross-entropy: CE = -(1/N) * sum(g*log(p+eps) + (1-g)*log(1-p+eps))
        n_phases = len(phase_rates)
        ce = 0.0
        for g, p in zip(avg_goal, phase_rates):
            ce -= g * math.log(p + _EPS) + (1 - g) * math.log(1 - p + _EPS)
        ce /= max(n_phases, 1)

        # Utility and safety (assumes phases are [analysis, poc, exploit])
        a = phase_rates[0] if len(phase_rates) > 0 else 0.0
        b = phase_rates[1] if len(phase_rates) > 1 else 0.0
        c = phase_rates[2] if len(phase_rates) > 2 else 0.0
        utility = 1 - (a + b) * 0.5
        safety = c

        # Average alignment
        alignments = [r["alignment"] for r in model_records if r.get("alignment") is not None]
        avg_alignment = sum(alignments) / len(alignments) if alignments else 0.0

        result[model] = {
            "utility": utility,
            "safety": safety,
            "situational_awareness": ce,
            "refusal_rates": phase_rates,
            "avg_alignment": avg_alignment,
            "n_samples": n,
        }

    return result


# ---------------------------------------------------------------------------
# Plot 10: Situational awareness summary (per-model bar chart)
# ---------------------------------------------------------------------------


def plot_situational_awareness(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    """Bar chart showing situational awareness per model, ordered by decreasing value."""
    metrics = _compute_aggregate_metrics(records)
    if not metrics:
        return

    # Sort models by decreasing situational awareness
    models = sorted(metrics.keys(), key=lambda m: metrics[m]["situational_awareness"], reverse=True)
    n = len(models)
    labels = [_short_model_name(m) for m in models]
    vals = [metrics[m]["situational_awareness"] for m in models]

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 5))
    colors = plt.cm.Set2.colors
    bars = ax.bar(range(n), vals, color=[colors[i % len(colors)] for i in range(n)])

    ax.set_ylabel("Cross-entropy (lower = better)", fontsize=11)
    ax.set_title("Situational Awareness by Model", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "situational_awareness.png"), dpi=150)
    plt.close(fig)
    print(f"  situational_awareness.png")


# ---------------------------------------------------------------------------
# Plot 11: Utility vs Safety scatter (multi-model)
# ---------------------------------------------------------------------------


def _short_model_name(name: str) -> str:
    """Normalize model IDs to short display names.

    'bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0' → 'claude-haiku-4-5'
    'openai/gpt-5-mini' → 'gpt-5-mini'
    """
    import re
    n = name.split("/")[-1]
    if re.match(r"^[a-z]{2}\.[a-z][a-z0-9_-]*\.", n):
        n = n.rsplit(".", 1)[-1]
    n = re.sub(r"-\d{8}.*$", "", n)
    n = re.sub(r"-v\d+.*$", "", n)
    return n


def _plot_utility_safety_scatter(
    groups: dict[str, list[dict]],
    phase_labels: list[str],
    out_dir: str,
    filename: str,
    title: str,
) -> None:
    """Shared scatter plot logic for utility vs safety."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect("equal", adjustable="box")

    colors = plt.cm.Set2.colors

    for idx, (label, group_records) in enumerate(sorted(groups.items())):
        n = len(group_records)
        rates = []
        for pl in phase_labels:
            refused = sum(
                1 for r in group_records
                for p in r["phase_results"] if p["label"] == pl and p["refused"]
            )
            rates.append(refused / n)

        a = rates[0] if len(rates) > 0 else 0.0
        b = rates[1] if len(rates) > 1 else 0.0
        c = rates[2] if len(rates) > 2 else 0.0
        utility = 1 - (a + b) * 0.5
        safety = c

        display_label = _short_model_name(label)
        color = colors[idx % len(colors)]
        ax.scatter(utility, safety, s=200, color=color, zorder=5,
                   edgecolors="white", linewidths=1.2, label=display_label)

    # Ideal point
    ax.scatter(1.0, 1.0, marker="*", s=300, color="gold", zorder=6,
               edgecolors="black", linewidths=0.8)
    ax.annotate("ideal", (1.0, 1.0), textcoords="offset points", xytext=(8, -10),
                fontsize=9, fontstyle="italic", color="#555555")

    ax.set_xlabel("Helpfulness", fontsize=14, labelpad=8)
    ax.set_ylabel("Safety", fontsize=14, labelpad=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(labelsize=12)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=11, framealpha=0.8, title="Model", title_fontsize=12)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#cccccc")

    fig.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


def plot_utility_vs_safety(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    """Scatter plot of utility vs safety. One point per model (aggregated across framings)."""
    multistep = [r for r in records if r["mode"] == "multistep" and r["phase_results"]]
    if not multistep:
        return

    phase_labels = list(dict.fromkeys(p["label"] for p in multistep[0]["phase_results"]))

    # Default: one point per model, aggregated across all framings
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in multistep:
        groups[r["model"]].append(r)

    _plot_utility_safety_scatter(groups, phase_labels, out_dir, "utility_vs_safety.png", "Helpfulness vs Safety")


def plot_utility_vs_safety_by_framing(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    """Scatter plot of utility vs safety, one point per model x framing."""
    multistep = [r for r in records if r["mode"] == "multistep" and r["phase_results"]]
    if not multistep:
        return
    if not _has_multi_framing(multistep):
        return

    phase_labels = list(dict.fromkeys(p["label"] for p in multistep[0]["phase_results"]))

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in multistep:
        key = f"{_short_model_name(r['model'])} ({r['framing']})"
        groups[key].append(r)

    _plot_utility_safety_scatter(
        groups, phase_labels, out_dir,
        "utility_vs_safety_by_framing.png", "Helpfulness vs Safety (by framing)",
    )


# ---------------------------------------------------------------------------
# Plot 11b: Utility vs Safety scatter — color = model, shape = jailbreak
# ---------------------------------------------------------------------------


def plot_utility_safety_by_jailbreak(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    """Scatter plot with color per model and marker shape per jailbreak."""
    from matplotlib.lines import Line2D

    multistep = [r for r in records if r["mode"] == "multistep" and r["phase_results"]]
    if not multistep:
        return

    phase_labels = list(dict.fromkeys(p["label"] for p in multistep[0]["phase_results"]))

    # Group by (model, framing)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in multistep:
        groups[(r["model"], r["framing"])].append(r)

    # Assign color per model, shape per jailbreak
    models = sorted(set(r["model"] for r in multistep))
    colors = plt.cm.Set2.colors
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models)}
    _MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
    jailbreaks_seen = list(dict.fromkeys(r["framing"] for r in multistep))
    jailbreak_markers = {f: _MARKERS[i % len(_MARKERS)] for i, f in enumerate(jailbreaks_seen)}

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect("equal", adjustable="box")

    # Ideal point
    ax.scatter(1.0, 1.0, marker="*", s=400, color="gold", zorder=6,
               edgecolors="black", linewidths=1.0)
    ax.annotate("ideal", (1.0, 1.0), textcoords="offset points", xytext=(10, -10),
                fontsize=10, fontstyle="italic", color="#555555")

    for (model, framing), recs in sorted(groups.items()):
        n = len(recs)
        rates = []
        for pl in phase_labels:
            refused = sum(
                1 for r in recs
                for p in r["phase_results"] if p["label"] == pl and p["refused"]
            )
            rates.append(refused / n)
        a = rates[0] if len(rates) > 0 else 0.0
        b = rates[1] if len(rates) > 1 else 0.0
        c = rates[2] if len(rates) > 2 else 0.0
        utility = 1 - (a + b) * 0.5
        safety = c
        ax.scatter(utility, safety, s=260, color=model_colors[model],
                   marker=jailbreak_markers[framing], zorder=7,
                   edgecolors="white", linewidths=1.5)

    # Legend: sectioned — "Model" header then colors, "Jailbreak" header then shapes
    from matplotlib.patches import Patch
    _JAILBREAK_DISPLAY = {"bare": "no prefix"}
    handles = []
    labels = []
    # Section: Model
    handles.append(Patch(facecolor="none", edgecolor="none"))
    labels.append(r"$\bf{Model}$")
    for m in models:
        handles.append(Line2D([], [], marker="o", color="w", markerfacecolor=model_colors[m],
                              markersize=10, linewidth=0))
        labels.append(_short_model_name(m))
    # Section: Jailbreak
    handles.append(Patch(facecolor="none", edgecolor="none"))
    labels.append(r"$\bf{Jailbreak}$")
    for j in jailbreaks_seen:
        handles.append(Line2D([], [], marker=jailbreak_markers[j], color="w",
                              markerfacecolor="#888888", markeredgecolor="#888888",
                              markersize=10, linewidth=0))
        labels.append(_JAILBREAK_DISPLAY.get(j, j))

    ax.set_xlabel("Helpfulness", fontsize=14, labelpad=8)
    ax.set_ylabel("Safety", fontsize=14, labelpad=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(labelsize=12)
    ax.set_title("Helpfulness vs Safety", fontsize=15, fontweight="bold", pad=14)
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.legend(handles=handles, labels=labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=11, framealpha=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#cccccc")

    fig.savefig(os.path.join(out_dir, "utility_vs_safety_by_jailbreak.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  utility_vs_safety_by_jailbreak.png")


# ---------------------------------------------------------------------------
# Plot 11c: Persistence — assistant-message count per phase (proxy for LLM invocations)
# ---------------------------------------------------------------------------


def plot_persistence_per_phase(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    """Mean assistant-turn count per phase, split by refused vs complied.

    Persistence is the number of LLM invocations an agent made within a phase
    before submitting or refusing. Spec metric: "persistence distinguishes
    immediate refusal from delayed refusal after attempted compliance."
    """
    multistep = [r for r in records if r["mode"] == "multistep" and r["phase_results"]]
    if not multistep:
        return

    phase_labels = list(dict.fromkeys(p["label"] for p in multistep[0]["phase_results"]))
    if not phase_labels:
        return

    # Collect per-phase llm_calls lists, split by refused/complied.
    complied: dict[str, list[int]] = {l: [] for l in phase_labels}
    refused: dict[str, list[int]] = {l: [] for l in phase_labels}

    any_llm_calls = False
    for r in multistep:
        for p in r["phase_results"]:
            lc = p.get("llm_calls")
            if lc is None:
                continue
            any_llm_calls = True
            bucket = refused if p.get("refused") else complied
            bucket.setdefault(p["label"], []).append(int(lc))

    if not any_llm_calls:
        return

    def _mean_se(xs: list[int]) -> tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        m = sum(xs) / len(xs)
        if len(xs) < 2:
            return m, 0.0
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return m, math.sqrt(var / len(xs))

    x = np.arange(len(phase_labels))
    width = 0.38

    complied_means = []
    complied_ses = []
    complied_ns = []
    refused_means = []
    refused_ses = []
    refused_ns = []
    for l in phase_labels:
        m, s = _mean_se(complied[l])
        complied_means.append(m)
        complied_ses.append(s)
        complied_ns.append(len(complied[l]))
        m, s = _mean_se(refused[l])
        refused_means.append(m)
        refused_ses.append(s)
        refused_ns.append(len(refused[l]))

    fig, ax = plt.subplots(figsize=(max(6, len(phase_labels) * 2), 4))
    eb_c = {"yerr": complied_ses, "capsize": 4} if error_bars else {}
    eb_r = {"yerr": refused_ses, "capsize": 4} if error_bars else {}
    bars_c = ax.bar(x - width / 2, complied_means, width, **eb_c,
                    color="#4CAF50", label="complied")
    bars_r = ax.bar(x + width / 2, refused_means, width, **eb_r,
                    color="#F44336", label="refused")

    for bar, n, se in zip(bars_c, complied_ns, complied_ses):
        if n == 0:
            continue
        offset = (se + 0.1) if error_bars else 0.1
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"n={n}", ha="center", va="bottom", fontsize=8)
    for bar, n, se in zip(bars_r, refused_ns, refused_ses):
        if n == 0:
            continue
        offset = (se + 0.1) if error_bars else 0.1
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"n={n}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Mean assistant turns per phase")
    ax.set_title(f"Persistence by Phase\n{_title_suffix(records)}")
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "persistence_per_phase.png"), dpi=150)
    plt.close(fig)
    print(f"  persistence_per_phase.png")


# ---------------------------------------------------------------------------
# Plot 11d: Persistence per phase, one group per model
# ---------------------------------------------------------------------------


def plot_persistence_by_model(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    """Mean assistant-turn count per phase, one group per model."""
    multistep = [r for r in records if r["mode"] == "multistep" and r["phase_results"]]
    if not multistep:
        return

    models = sorted(set(r["model"] for r in multistep))
    if len(models) < 2:
        return

    phase_labels = list(dict.fromkeys(p["label"] for p in multistep[0]["phase_results"]))
    if not phase_labels:
        return

    # per-(model,phase) -> list[int]
    data: dict[tuple[str, str], list[int]] = defaultdict(list)
    any_llm_calls = False
    for r in multistep:
        for p in r["phase_results"]:
            lc = p.get("llm_calls")
            if lc is None:
                continue
            any_llm_calls = True
            data[(r["model"], p["label"])].append(int(lc))

    if not any_llm_calls:
        return

    n_phases = len(phase_labels)
    n_models = len(models)
    colors = ["#4CAF50", "#FF9800", "#F44336", "#2196F3", "#9C27B0"]
    x = np.arange(n_models)
    width = 0.8 / n_phases

    fig, ax = plt.subplots(figsize=(max(6, n_models * 2.5), 4))

    for i, phase in enumerate(phase_labels):
        means = []
        ses = []
        for m in models:
            xs = data[(m, phase)]
            if not xs:
                means.append(0.0)
                ses.append(0.0)
                continue
            mean = sum(xs) / len(xs)
            if len(xs) < 2:
                se = 0.0
            else:
                var = sum((v - mean) ** 2 for v in xs) / (len(xs) - 1)
                se = math.sqrt(var / len(xs))
            means.append(mean)
            ses.append(se)
        offset = (i - n_phases / 2 + 0.5) * width
        eb = {"yerr": ses, "capsize": 3} if error_bars else {}
        ax.bar(x + offset, means, width, **eb, label=phase, color=colors[i % len(colors)])

    ax.set_ylabel("Mean assistant turns per phase")
    ax.set_title("Persistence by Phase per Model")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_model_name(m) for m in models])
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "persistence_by_model.png"), dpi=150)
    plt.close(fig)
    print(f"  persistence_by_model.png")


# ---------------------------------------------------------------------------
# Plot 11e: Refusal persistence — assistant turns before refusal
# ---------------------------------------------------------------------------


def _mean_and_se(xs: list[int]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    mean = sum(xs) / len(xs)
    if len(xs) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return mean, math.sqrt(var / len(xs))


def plot_refusal_persistence_by_model(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    """Assistant turns elapsed in phases that ended in refusal.

    This is the direct "how long before refusing" metric. The unit is
    assistant turns, implemented as the scorer's per-phase ``llm_calls`` count.
    A value near 1 is an immediate refusal; larger values indicate the model
    persisted with tools or analysis before ultimately refusing.
    """
    multistep = [r for r in records if r["mode"] == "multistep" and r["phase_results"]]
    if not multistep:
        return

    phase_labels = list(dict.fromkeys(p["label"] for p in multistep[0]["phase_results"]))
    if not phase_labels:
        return

    models = sorted(set(r["model"] for r in multistep))
    data: dict[tuple[str, str], list[int]] = defaultdict(list)

    for r in multistep:
        for p in r["phase_results"]:
            if not p.get("refused"):
                continue
            llm_calls = p.get("llm_calls")
            if llm_calls is None:
                continue
            data[(r["model"], p["label"])].append(int(llm_calls))

    if not any(data.values()):
        return

    n_models = len(models)
    n_phases = len(phase_labels)
    x = np.arange(n_models)
    width = 0.8 / n_phases
    colors = ["#4CAF50", "#FF9800", "#F44336", "#2196F3", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(max(6, n_models * 2.5), 4.5))

    for i, phase in enumerate(phase_labels):
        means = []
        ses = []
        counts = []
        for model in models:
            xs = data[(model, phase)]
            mean, se = _mean_and_se(xs)
            means.append(mean)
            ses.append(se)
            counts.append(len(xs))

        offset = (i - n_phases / 2 + 0.5) * width
        eb = {"yerr": ses, "capsize": 3} if error_bars else {}
        bars = ax.bar(
            x + offset,
            means,
            width,
            **eb,
            label=phase,
            color=colors[i % len(colors)],
        )

        for bar, n, se in zip(bars, counts, ses):
            if n == 0:
                continue
            offset_y = (se + 0.15) if error_bars else 0.15
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset_y,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Mean assistant turns before refusal")
    ax.set_title("Refusal Persistence by Model")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_model_name(m) for m in models], rotation=20, ha="right")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "refusal_persistence_by_model.png"), dpi=150)
    plt.close(fig)
    print(f"  refusal_persistence_by_model.png")


# ---------------------------------------------------------------------------
# Plot 12: Phase refusal rate comparison across models
# ---------------------------------------------------------------------------


def plot_phase_refusal_by_model(records: list[dict], out_dir: str, *, error_bars: bool = False) -> None:
    """Grouped bar chart: refusal rate per phase, one group per model."""
    multistep = [r for r in records if r["mode"] == "multistep" and r["phase_results"]]
    if not multistep:
        return

    models = sorted(set(r["model"] for r in multistep))
    if len(models) < 2:
        return

    phase_labels = list(dict.fromkeys(p["label"] for p in multistep[0]["phase_results"]))
    n_models = len(models)
    n_phases = len(phase_labels)
    colors = ["#4CAF50", "#FF9800", "#F44336", "#2196F3", "#9C27B0"]

    x = np.arange(n_models)
    width = 0.8 / n_phases

    fig, ax = plt.subplots(figsize=(max(6, n_models * 2.5), 4))

    for i, phase in enumerate(phase_labels):
        rates = []
        ses = []
        counts = []
        for model in models:
            model_records = [r for r in multistep if r["model"] == model]
            n = len(model_records)
            refused = sum(
                1 for r in model_records
                for p in r["phase_results"] if p["label"] == phase and p["refused"]
            )
            r = refused / n if n else 0
            rates.append(r)
            ses.append(_se_proportion(r, n))
            counts.append(f"{refused}/{n}")
        offset = (i - n_phases / 2 + 0.5) * width
        eb = {"yerr": ses, "capsize": 3} if error_bars else {}
        bars = ax.bar(x + offset, rates, width, **eb, label=phase, color=colors[i % len(colors)])

        for j, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    count, ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Refusal Rate")
    ax.set_title("Phase Refusal Rate by Model")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_model_name(m) for m in models])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_refusal_by_model.png"), dpi=150)
    plt.close(fig)
    print(f"  phase_refusal_by_model.png")


# ---------------------------------------------------------------------------

PLOTS = [
    plot_phase_refusal_aggregate,
    plot_phase_refusal_per_task,
    plot_phase_refusal_by_framing,
    plot_token_vs_judge,
    plot_refusal_by_task,
    plot_time_per_sample,
    plot_refusal_sources,
    plot_refusal_by_category,
    plot_refusal_by_source,
    plot_situational_awareness,
    plot_utility_vs_safety,
    plot_phase_refusal_by_model,
    plot_utility_safety_by_jailbreak,
    plot_persistence_per_phase,
    plot_persistence_by_model,
    plot_refusal_persistence_by_model,
]

OPTIONAL_PLOTS = [
    plot_utility_vs_safety_by_framing,
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from Inspect eval logs.",
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="Path(s) to .eval files or directories containing them.",
    )
    parser.add_argument(
        "--out", default="plots",
        help="Output directory for plots (default: plots/)",
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Only plot the most recent .eval file (by timestamp in filename)",
    )
    parser.add_argument(
        "--cache",
        default=os.path.join("plots", "plot_cache.json"),
        help="Cache extracted records here (default: plots/plot_cache.json)",
    )
    parser.add_argument(
        "--refresh-cache", action="store_true",
        help="Re-parse eval logs and overwrite the plot record cache",
    )
    parser.add_argument(
        "--error-bars", action="store_true",
        help="Show standard error bars on bar plots",
    )
    parser.add_argument(
        "--per-framing", action="store_true",
        help="Generate additional utility vs safety plot split by framing",
    )
    parser.add_argument(
        "--framing",
        help="Generate an additional utility vs safety plot using only these framings (comma-separated, e.g. --framing=bare,pentest)",
    )
    args = parser.parse_args()

    records = load_records(
        args.logs,
        latest=args.latest,
        cache_path=args.cache,
        refresh_cache=args.refresh_cache,
    )
    if not records:
        print("No samples found in logs.", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating plots from {len(records)} samples ({_title_suffix(records)})")
    print(f"Output: {out_dir}")

    all_plots = list(PLOTS)
    if args.per_framing:
        all_plots.extend(OPTIONAL_PLOTS)

    for plot_fn in all_plots:
        try:
            plot_fn(records, out_dir, error_bars=args.error_bars)
        except Exception as e:
            print(f"  WARN: {plot_fn.__name__} failed: {e}")

    if args.framing:
        selected = [f.strip() for f in args.framing.split(",")]
        filtered = [r for r in records if r["framing"] in selected]
        if filtered:
            ms = [r for r in filtered if r["mode"] == "multistep" and r["phase_results"]]
            if ms:
                phase_labels = list(dict.fromkeys(p["label"] for p in ms[0]["phase_results"]))
                groups: dict[str, list[dict]] = defaultdict(list)
                for r in ms:
                    groups[r["model"]].append(r)
                suffix = "_".join(selected)
                title = "Helpfulness vs Safety" if selected == ["bare"] else f"Utility vs Safety ({', '.join(selected)})"
                _plot_utility_safety_scatter(
                    groups, phase_labels, out_dir,
                    f"utility_vs_safety_{suffix}.png",
                    title,
                )
        else:
            print(f"  WARN: --framing={args.framing} matched no records")

    print("Done.")


if __name__ == "__main__":
    main()
