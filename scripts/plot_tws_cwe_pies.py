#!/usr/bin/env python3
"""Plot ground-truth CWE composition for each expert TWS score.

Each task contributes total weight 1.0. If a task has multiple numeric
ground-truth CWEs, that weight is split evenly across its CWEs so multi-label
tasks do not inflate the denominator.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "benchmark"
DEFAULT_OUT_DIR = REPO_ROOT / "plots"
DEFAULT_SOURCES = ("vulhub", "exploitdb")
DEFAULT_TOP_K = 10

PALETTE = (
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
)
OTHER_COLOR = "#D0D0D0"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 16,
            "legend.fontsize": 10.5,
            "figure.dpi": 200,
        }
    )


def load_tws_cwe_counts(
    benchmark_root: Path,
    sources: tuple[str, ...],
) -> tuple[dict[int, Counter[str]], dict[int, int], dict[str, int]]:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    task_counts: Counter[int] = Counter()
    stats = Counter()

    for source in sources:
        for metadata_path in sorted((benchmark_root / source).glob("*/metadata.json")):
            stats["metadata_files"] += 1
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            tws = (metadata.get("tws_classification_expert") or {}).get("revealed_score")
            task_cwes = (metadata.get("ground_truth_cwes") or {}).get("task_cwes") or []

            if tws is None:
                stats["missing_tws"] += 1
                continue
            if not task_cwes:
                stats["missing_numeric_cwe"] += 1
                continue

            tws = int(tws)
            task_counts[tws] += 1
            weight = 1.0 / len(task_cwes)
            for cwe in task_cwes:
                counts[tws][str(cwe)] += weight

    return counts, dict(task_counts), dict(stats)


def stable_cwe_colors(counts_by_tws: dict[int, Counter[str]]) -> dict[str, str]:
    global_counts: Counter[str] = Counter()
    for counts in counts_by_tws.values():
        global_counts.update(counts)

    colors: dict[str, str] = {}
    for idx, (cwe, _count) in enumerate(global_counts.most_common()):
        colors[cwe] = PALETTE[idx % len(PALETTE)]
    return colors


def top_k_with_other(counts: Counter[str], top_k: int) -> list[tuple[str, float]]:
    top = counts.most_common(top_k)
    other = sum(counts.values()) - sum(value for _label, value in top)
    if other > 1e-9:
        top.append(("Other", other))
    return top


def format_value(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def display_cwe(label: str) -> str:
    if label.startswith("CWE-"):
        return label.removeprefix("CWE-")
    return label


def plot_tws_pie(
    tws: int,
    counts: Counter[str],
    task_count: int,
    colors_by_cwe: dict[str, str],
    out_path: Path,
    *,
    top_k: int,
) -> None:
    rows = top_k_with_other(counts, top_k)
    total = sum(value for _label, value in rows)
    labels = [label for label, _value in rows]
    values = [value for _label, value in rows]
    colors = [OTHER_COLOR if label == "Other" else colors_by_cwe[label] for label in labels]

    fig, ax = plt.subplots(figsize=(4.6, 3.15))
    wedges, _texts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 0.8},
    )
    ax.text(
        0,
        0.04,
        f"TWS {tws}",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )
    ax.text(
        0,
        -0.14,
        f"n={task_count}",
        ha="center",
        va="center",
        fontsize=12,
        color="#555555",
    )

    legend_labels = [
        f"{display_cwe(label)}  {value / total * 100:.1f}%"
        for label, value in rows
    ]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.90, 0.5),
        frameon=False,
        handlelength=1.0,
        handletextpad=0.5,
        borderaxespad=0.0,
    )
    ax.set(aspect="equal")
    fig.subplots_adjust(left=0.00, right=0.72, top=0.99, bottom=0.01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def write_summary(
    out_dir: Path,
    counts_by_tws: dict[int, Counter[str]],
    task_counts: dict[int, int],
    stats: dict[str, int],
    *,
    top_k: int,
) -> None:
    summary: dict[str, object] = {
        "counting": "fractional task weight split across numeric ground_truth_cwes.task_cwes",
        "top_k": top_k,
        "stats": stats,
        "tws": {},
    }
    for tws in sorted(counts_by_tws):
        rows = top_k_with_other(counts_by_tws[tws], top_k)
        total = sum(value for _label, value in rows)
        summary["tws"][str(tws)] = {
            "tasks": task_counts.get(tws, 0),
            "unique_cwes": len(counts_by_tws[tws]),
            "top_k_share": sum(value for label, value in rows if label != "Other") / total,
            "distribution": [
                {
                    "label": label,
                    "weight": round(value, 6),
                    "percent": round(value / total * 100, 4),
                }
                for label, value in rows
            ],
        }

    out_path = out_dir / "tws_cwe_pies_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        help="Benchmark source directories to include.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    counts_by_tws, task_counts, stats = load_tws_cwe_counts(
        args.benchmark_root,
        tuple(args.sources),
    )
    colors_by_cwe = stable_cwe_colors(counts_by_tws)

    for tws in sorted(counts_by_tws):
        out_path = args.out_dir / f"tws_cwe_pie_tws{tws}.pdf"
        plot_tws_pie(
            tws,
            counts_by_tws[tws],
            task_counts.get(tws, 0),
            colors_by_cwe,
            out_path,
            top_k=args.top_k,
        )
        print(f"Wrote {out_path}")

    write_summary(args.out_dir, counts_by_tws, task_counts, stats, top_k=args.top_k)
    print(f"Wrote {args.out_dir / 'tws_cwe_pies_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
