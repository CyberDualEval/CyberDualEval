#!/usr/bin/env python3
"""Publish ``benchmark/`` to a private HuggingFace dataset repo.

Usage:
  uv run python scripts/publish_benchmark_to_hf.py \
      --repo-id CyberDualEval/CyberDualEval \
      --revision-tag v2026.05-passE-vulhub

Requires:
  - HF_TOKEN in environment (or `huggingface-cli login` already run)
  - huggingface_hub installed (declared in pyproject.toml)

Notes:
  - The dataset is created as **private** by default. Flip with --public when
    a license audit has cleared the contents for redistribution.
  - Dev/curation artifacts (.curation/, lockfiles, OS junk) are excluded so
    the published tree contains only what collaborators need to run the eval.
  - After upload, an immutable git tag is created so a collaborator can pin
    a specific revision per campaign.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"

# Patterns excluded from every publish (junk, spoilers, ops bloat).
ALWAYS_IGNORE = [
    # Editor / OS / Python junk
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.gitignore",
    # Vendored git histories accidentally committed inside source/ trees
    # (~150 MB of dead weight; vendor commit history is not needed at eval time).
    "**/.git/**",
    "**/.git",
    # Curation-tooling caches that regenerate on demand (NVD / GitHub / LLM
    # path resolution). Always pure bloat regardless of audience.
    "**/.github_cache.json",
    "**/.nvd_cache.json",
    "**/.nvd_cpe_cache.json",
    "**/.llm_path_cache.json",
    # Stray editor / agent state inside vendor source
    "**/.agent/**",
    # Lockfile for the validation ledger — process artifact, not data.
    "**/_validation_ledger.yaml.lock",
    # HuggingFace reserves `.cache/` for client-side state and rejects any
    # commit operation that touches a path under it. Strip vendor `.cache/`
    # subtrees inside source/ so they don't fail validation at upload time.
    "**/.cache/**",
    # Per-task source/ trees are bundled into source.tar.gz by
    # scripts/bundle_task_sources.py before upload. Without this exclusion the
    # raw cybergym src-vul/ trees alone push the file count past 2M, vs HF's
    # ~100k/repo cap. The sibling source.tar.gz is *not* under source/ so
    # this glob doesn't touch it.
    "**/source/**",
]

# Curation artifacts: per-task .curation/ drafts + gate failure logs and
# per-source worklists. These are spoiler-style (not exposed to the agent
# at eval time, but they leak predicate shape) and primarily useful to
# collaborators *authoring* PoC/exploit fixtures. Included by default since
# they're harmless to eval-runners; pass --exclude-curation to drop them.
CURATION_ARTIFACTS = [
    "**/.curation/**",
    "**/_funnel.yaml",
    "**/_skipped.yaml",
    "**/_validation_ledger.yaml",
    "**/_validation_todo.md",
    "**/candidates.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. CyberDualEval/CyberDualEval")
    parser.add_argument("--revision-tag", help="Tag to create after upload, e.g. v2026.05-passE-vulhub")
    parser.add_argument("--public", action="store_true", help="Create dataset as public (default: private)")
    parser.add_argument("--commit-message", default="publish benchmark snapshot")
    parser.add_argument(
        "--exclude-curation",
        action="store_true",
        help=(
            "Drop .curation/ drafts, gate failure logs, and per-source "
            "worklists from the upload. Default: include them — they're "
            "not loaded into agent context, and they're useful for "
            "collaborators who'll be *authoring* PoC/exploit fixtures."
        ),
    )
    parser.add_argument(
        "--skip-bundle",
        action="store_true",
        help=(
            "Skip the pre-flight bundle step (assumes source.tar.gz files "
            "are already up-to-date). Default: bundle first, since publishing "
            "with a stale tarball ships outdated source to collaborators."
        ),
    )
    args = parser.parse_args()

    ignore_patterns = list(ALWAYS_IGNORE)
    if args.exclude_curation:
        ignore_patterns += CURATION_ARTIFACTS
    print(
        f"Mode: {'lean (no curation)' if args.exclude_curation else 'full (with curation)'} "
        f"({len(ignore_patterns)} ignore patterns)"
    )

    if not BENCHMARK_DIR.is_dir():
        print(f"ERROR: {BENCHMARK_DIR} not found.", file=sys.stderr)
        sys.exit(1)

    # Pre-flight: rebuild stale source.tar.gz bundles so the upload doesn't
    # ship outdated source. This is the half of the bundle/publish handshake
    # the publisher owns; the bundler itself is idempotent and cheap when
    # nothing changed.
    bundler = PROJECT_ROOT / "scripts" / "bundle_task_sources.py"
    if args.skip_bundle:
        print("Skipping pre-flight bundle (--skip-bundle); verifying tarballs are current...")
        check = subprocess.run([sys.executable, str(bundler), "--check"])
        if check.returncode != 0:
            print(
                "ERROR: stale source.tar.gz files detected. Re-run without --skip-bundle "
                "or manually run scripts/bundle_task_sources.py.",
                file=sys.stderr,
            )
            sys.exit(2)
    else:
        print("Pre-flight: bundling task source/ trees -> source.tar.gz ...")
        bundle = subprocess.run([sys.executable, str(bundler)])
        if bundle.returncode != 0:
            print("ERROR: bundle step failed; aborting publish.", file=sys.stderr)
            sys.exit(2)

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )

    print(f"Uploading {BENCHMARK_DIR} -> {args.repo_id} (private={not args.public})")
    # `upload_large_folder` is the right tool for ~50 GB datasets: resumable,
    # multi-threaded, batches commits internally so a transient network blip
    # doesn't restart the whole upload from scratch. It does NOT take
    # `commit_message` (each batched commit gets an auto-generated message)
    # or `revision` (always commits to main); we tag after upload completes.
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(BENCHMARK_DIR),
        ignore_patterns=ignore_patterns,
        private=not args.public,
        print_report=True,
    )

    if args.revision_tag:
        print(f"Creating tag {args.revision_tag}")
        api.create_tag(
            repo_id=args.repo_id,
            repo_type="dataset",
            tag=args.revision_tag,
            tag_message=args.commit_message,
        )

    print(f"Done. Browse at https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
