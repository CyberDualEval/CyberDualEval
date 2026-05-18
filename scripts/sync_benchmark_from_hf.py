#!/usr/bin/env python3
"""Sync ``benchmark/`` from a pinned HuggingFace dataset revision.

Usage:
  uv run python scripts/sync_benchmark_from_hf.py \
      --repo-id CyberDualEval/CyberDualEval \
      --revision v2026.05-passE-vulhub

Requires:
  - HF_TOKEN in environment for private datasets (`huggingface-cli login`)
  - huggingface_hub installed (declared in pyproject.toml)

Behavior:
  - Materializes the dataset tree into ./benchmark/ at the project root.
  - Refuses to overwrite an existing benchmark/ unless --force is passed; this
    avoids silently clobbering uncommitted curation work.
  - On --force the existing tree is renamed to benchmark.bak-<timestamp>/
    rather than deleted, so nothing is lost.
  - Post-download, every ``source.tar.gz`` is extracted in place to recreate
    the ``source/`` tree the eval harness reads from. The publisher tarballs
    these to keep the HF file count under the per-repo limit; the maintainer's
    machine never sees the tarballs because they're built fresh at publish
    time. Pass --keep-tarballs to leave them on disk after extraction
    (default: delete to save ~15GB).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"


def _extract_vendor_bundles(root: Path, keep: bool) -> None:
    """Extract ``benchmark/_vendor/*.tar.gz`` to PROJECT_ROOT.

    These tarballs carry trees that live *outside* benchmark/ on the
    maintainer machine (currently just vendor/cybench/, used by cybench
    compose build contexts). Entry names start with ``vendor/...`` so
    extracting at the project root recreates the original layout.

    Without this, every cybench task fails at ``docker compose build`` on a
    fresh sync because the build context path doesn't exist.
    """
    vendor_dir = root / "_vendor"
    if not vendor_dir.is_dir():
        return
    bundles = sorted(vendor_dir.glob("*.tar.gz"))
    if not bundles:
        return
    print(f"Extracting {len(bundles)} vendor bundle(s) to {PROJECT_ROOT} ...")
    for tb in bundles:
        try:
            with tarfile.open(tb, "r:gz") as tar:
                tar.extractall(path=PROJECT_ROOT, filter="fully_trusted")
            print(f"  ✓ {tb.name}")
        except Exception as e:
            print(f"  ERROR extracting {tb.name}: {e}", file=sys.stderr)
            continue
        if not keep:
            tb.unlink()
    if not keep:
        try:
            vendor_dir.rmdir()  # only succeeds if empty
        except OSError:
            pass


def _extract_tarballs(root: Path, keep: bool) -> None:
    """Walk ``root`` and extract every ``source.tar.gz`` next to itself.

    The tarball entries are named ``source/...`` (see bundle_task_sources.py),
    so extracting at the tarball's parent directory recreates the original
    layout the eval harness reads from. We refuse to overwrite an existing
    ``source/`` next to a tarball — that would silently clobber edits — but
    that should never happen on a fresh sync since the publisher excludes
    ``**/source/**``.

    We use ``filter='fully_trusted'`` because the dataset is one we publish
    ourselves and trust at the source. Several legitimate vendor source trees
    contain absolute-target symlinks (Neo4j data dir, apisix/nginx logs,
    SaltStack ``aclocal``, etc.) — these were captured when source was
    extracted from a live container where the absolute targets were valid
    paths. They land as broken symlinks on disk after extraction, which is
    fine: the eval harness doesn't follow them, and they match the maintainer
    machine's state. The stricter ``filter='data'`` would refuse them outright,
    so 12+ Vulhub tasks would silently fail extraction.
    """
    tarballs = sorted(root.rglob("source.tar.gz"))
    if not tarballs:
        print("No source.tar.gz files found; nothing to extract.")
        return
    print(f"Extracting {len(tarballs)} source.tar.gz bundles ...")
    extracted = 0
    for tb in tarballs:
        target_parent = tb.parent
        source_dir = target_parent / "source"
        if source_dir.exists():
            print(f"  skip {tb.relative_to(root)}: source/ already exists")
            continue
        try:
            with tarfile.open(tb, "r:gz") as tar:
                tar.extractall(path=target_parent, filter="fully_trusted")
            extracted += 1
        except Exception as e:
            print(f"  ERROR extracting {tb.relative_to(root)}: {e}", file=sys.stderr)
            continue
        if not keep:
            tb.unlink()
    print(f"Extracted {extracted}/{len(tarballs)} bundles"
          f"{' (tarballs deleted)' if not keep else ' (tarballs kept)'}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main", help="Branch, tag, or commit hash. Default: main")
    parser.add_argument("--force", action="store_true", help="Move existing benchmark/ aside before sync")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted sync into an existing benchmark/ in place. "
                             "snapshot_download is content-hash-diffed, so already-downloaded "
                             "files are skipped and only the missing/changed ones transfer. "
                             "Use this when a previous sync was killed mid-stream — much faster "
                             "than --force, which restarts the download from scratch.")
    parser.add_argument("--keep-tarballs", action="store_true",
                        help="Leave source.tar.gz on disk after extraction (default: delete)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Don't extract source.tar.gz bundles (for inspection only; eval will not work)")
    args = parser.parse_args()

    if BENCHMARK_DIR.exists():
        if args.force:
            ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
            backup = PROJECT_ROOT / f"benchmark.bak-{ts}"
            BENCHMARK_DIR.rename(backup)
            print(f"Moved existing benchmark/ -> {backup.name}/")
        elif args.resume:
            print(f"Resuming into existing {BENCHMARK_DIR} (already-downloaded files will be skipped)")
        else:
            print(
                f"ERROR: {BENCHMARK_DIR} already exists. Use --resume to continue an "
                f"interrupted sync in place, or --force to back it up and re-download.",
                file=sys.stderr,
            )
            sys.exit(2)

    print(f"Downloading {args.repo_id}@{args.revision} -> {BENCHMARK_DIR}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(BENCHMARK_DIR),
    )

    if not args.skip_extract:
        _extract_tarballs(BENCHMARK_DIR, keep=args.keep_tarballs)
        _extract_vendor_bundles(BENCHMARK_DIR, keep=args.keep_tarballs)

    print("Done. Verify with: ls benchmark/cybench benchmark/vulhub | head")


if __name__ == "__main__":
    main()
