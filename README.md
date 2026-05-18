# CyberDualEval

CyberDualEval is an evaluation framework and benchmark for measuring how coding agents handle cybersecurity requests. Each benchmark task is presented as a three-phase interaction around a vulnerable program:

1. **Analysis**: identify and explain the vulnerability from source code.
2. **Proof of concept (PoC)**: create and run a minimal demonstration of the vulnerability.
3. **Exploit**: execute an exploit against a target endpoint.

The evaluation measures both over-refusal on benign security work and under-refusal on offensive requests. The PoC phase is scored with a Task Weaponization Score (TWS), because some PoCs are harmless reproducers while others are already direct misuse.

The benchmark payload is hosted on Hugging Face, not in this GitHub repository:

https://huggingface.co/datasets/CyberDualEval/CyberDualEval

This repository contains the evaluation framework, representative configs, plotting scripts, task-review spreadsheet, final figure PDFs, and Croissant/RAI metadata.

## Repository Contents

- `refusal_eval/`: Inspect-based task, solver, scorer, sandbox generation, controller, and validation code.
- `configs/`: representative model and framing configs.
- `scripts/`: benchmark sync, evaluation launch, metadata, plotting, and spreadsheet utilities.
- `docs/`: reviewer-facing documentation and task-review spreadsheet.
- `plots/`: final paper-facing figure PDFs.
- `croissant.json`: Croissant metadata with Responsible AI fields.

The full `benchmark/` directory is intentionally excluded. Download it from Hugging Face before running evaluations.

## Setup

Requirements:

- Python 3.10+
- Docker with Docker Compose
- `uv`
- API credentials for the model provider you evaluate

Install dependencies:

```bash
uv sync
```

Configure credentials:

```bash
cp .env.example .env
# edit .env or export provider variables in your shell
```

Provider variables may include `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, and provider-specific alternatives supported by Inspect.

## Download the Benchmark

Sync the benchmark payload from Hugging Face:

```bash
uv run python scripts/sync_benchmark_from_hf.py \
  --repo-id CyberDualEval/CyberDualEval
```

This creates `benchmark/` locally, extracts bundled source tarballs, and recreates required vendor bundles. If a previous download was interrupted, resume in place:

```bash
uv run python scripts/sync_benchmark_from_hf.py \
  --repo-id CyberDualEval/CyberDualEval \
  --resume
```

If you intentionally want to replace an existing local `benchmark/`, use `--force`; the old directory is moved aside rather than deleted.

## Dataset Summary

CyberDualEval combines four task sources:

- **CyBench**: pwn-style CTF challenges with small vulnerable services.
- **CyberGym**: OSS-Fuzz memory-safety vulnerabilities in open-source projects.
- **Exploit-DB**: verified remote and denial-of-service CVE tasks.
- **Vulhub**: CVE-tagged vulnerable service environments with Docker Compose setups.

The metadata snapshot used for the paper contains 426 task metadata files. The main valid-evaluation set contains 425 tasks, excluding `craftcms_CVE-2025-32432` because the upstream `vulhub/craftcms:5.6.16` image is unavailable.

Each task's Hugging Face metadata includes fields such as `source`, `task_id`, `category`, `cves`, `vuln_class`, `ground_truth_cwes`, `tws_classification`, `tws_classification_expert`, `accuracy`, and `validation`.

## Task Review Spreadsheet

The reviewer-facing task spreadsheet is available at:

- `docs/task_review_spreadsheet.csv`
- `docs/task_review_spreadsheet.xlsx`

It contains one row for each valid task, with source, task ID, vulnerability summary, CVE/CWE labels, expert TWS scores, PoC shape, validation coverage, and reference links.

Regenerate it after downloading `benchmark/` with:

```bash
uv run python scripts/create_task_review_spreadsheet.py
```

## Run an Evaluation

Example main run:

```bash
uv run python scripts/run_models.py configs/config-all-bare-gpt5.5-medium.yaml \
  --max-samples 12 \
  --max-connections 16 \
  --continue-on-fail \
  --retry-on-error=1
```

`scripts/run_models.py` prewarms Vulhub Docker images once for configs that include Vulhub, then runs each model listed in the config through Inspect.

Representative configs are in `configs/`. Scratch, pass-stage, and one-off debugging configs from the internal development repository are intentionally excluded.

## Reproducing Figures

Final paper-facing PDFs are checked into `plots/`. The plotting scripts used to produce them are included under `scripts/`.

Examples:

```bash
uv run python scripts/plot_tws_frontier.py
uv run python scripts/plot_framing_frontier.py
uv run python scripts/plot_tws_cwe_pies.py
```

The exact Inspect logs used for the paper are not distributed in this GitHub repository because they are large and can include raw model transcripts. To regenerate plots from new logs, update the corresponding run inputs and rerun the plotting scripts.

## Responsible Use

CyberDualEval is dual-use. It is intended for controlled benchmark evaluation, safety research, defensive analysis, and authorized testing in isolated environments. Do not use benchmark prompts, reference artifacts, or generated model outputs against systems you do not own or have explicit authorization to test.

Run benchmark tasks in isolated Docker environments, do not expose vulnerable services to public networks, and treat Inspect logs as sensitive because they can contain raw model outputs and tool transcripts.

## Additional Documentation

- `docs/distribution-and-running.md`: setup and running details.
- `docs/poc-rubric.md`: TWS rubric.
- `docs/validation-coverage.md`: validation coverage notes.
- `docs/ground-truth-cwe-report.json`: NVD/CVE CWE coverage report.
