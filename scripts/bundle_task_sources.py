#!/usr/bin/env python3
"""Bundle each task's ``source/`` tree into a deterministic ``source.tar.gz``
sibling, so the HuggingFace publish stays under the per-repo file-count limit.

Why: ``benchmark/`` carries ~2.4M files (cybergym ARVO upstream source dominates
with ~1.95M; vulhub Derby DBs and node_modules add another ~440k). HF caps a
dataset repo at ~100k files. Tarring each ``source/`` reduces the upload to
roughly one file per task (~430 total) without touching the local working tree
the eval harness reads from.

Determinism: tar entries are sorted, uid/gid/uname/gname normalized, mtime
clamped to a fixed epoch. Two consecutive runs over an unchanged ``source/``
produce byte-identical ``source.tar.gz``, so HF revisions only churn when the
underlying source actually changes.

Idempotence: a tarball is rebuilt only when missing or older than the newest
file under its ``source/``. Pass ``--force`` to rebuild unconditionally.

Usage:
  uv run python scripts/bundle_task_sources.py
  uv run python scripts/bundle_task_sources.py --source vulhub
  uv run python scripts/bundle_task_sources.py --force
  uv run python scripts/bundle_task_sources.py --check    # exit 1 if any stale
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import logging
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"

# Cybench compose files use ``build.context: ../../../vendor/cybench/...`` so
# the challenge container can't be built without the vendor tree on disk.
# vendor/ is gitignored and lived only on the maintainer machine until we
# started bundling it here. The tarball lives *inside* benchmark/ so it rides
# along in the HF upload; the sync script extracts it back out to PROJECT_ROOT.
VENDOR_CYBENCH_DIR = PROJECT_ROOT / "vendor" / "cybench"
VENDOR_BUNDLE_PATH = BENCHMARK_DIR / "_vendor" / "cybench.tar.gz"

# Mirror the publisher's ALWAYS_IGNORE patterns that can show up *inside*
# a ``source/`` tree (vendored OS junk, git histories, python caches, HF's
# reserved .cache/ namespace). Other publisher patterns (e.g. .curation/,
# validation lockfiles) live at task level, never under source/, and are
# irrelevant here.
SOURCE_IGNORE = [
    "**/.DS_Store",
    "**/__pycache__",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.git",
    "**/.git/**",
    "**/.cache",
    "**/.cache/**",
    "**/.agent",
    "**/.agent/**",
    "**/.github_cache.json",
    "**/.nvd_cache.json",
    "**/.nvd_cpe_cache.json",
    "**/.llm_path_cache.json",
]

# Fixed epoch: 2020-01-01. Anything stable + before today works; 2020 keeps the
# bundles reproducible without making them look suspiciously like Unix epoch 0.
EPOCH_MTIME = 1577836800

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _matches_any(rel_posix: str, patterns: list[str]) -> bool:
    """fnmatch with `**` semantics. ``rel_posix`` is the path inside source/
    using forward slashes; ``**`` matches any number of components."""
    for pat in patterns:
        # fnmatch handles ``**`` poorly on its own; expand to test against both
        # the raw path and a flattened variant. This covers our patterns which
        # are all of shape ``**/<name>`` or ``**/<name>/**``.
        if fnmatch.fnmatch(rel_posix, pat):
            return True
        # Strip leading ``**/`` and test against any tail
        if pat.startswith("**/"):
            tail = pat[3:]
            # Match if any path suffix matches tail
            parts = rel_posix.split("/")
            for i in range(len(parts)):
                if fnmatch.fnmatch("/".join(parts[i:]), tail):
                    return True
    return False


def _newest_mtime(root: Path) -> float:
    """Return the newest mtime under ``root`` for staleness checks. Includes the
    root dir's own mtime so empty trees still produce a sensible answer."""
    newest = root.stat().st_mtime
    for p in root.rglob("*"):
        try:
            m = p.lstat().st_mtime
        except FileNotFoundError:
            continue
        if m > newest:
            newest = m
    return newest


def _bundle_one(source_dir: Path, force: bool) -> tuple[str, str]:
    """Tar ``source_dir`` -> sibling ``source.tar.gz``. Returns (status, msg).
    status is one of ``written``, ``skipped``, ``error``."""
    tarball = source_dir.with_name("source.tar.gz")
    rel = source_dir.relative_to(BENCHMARK_DIR)

    if tarball.exists() and not force:
        # Stale check: tarball must be at least as new as the newest file in source/
        newest = _newest_mtime(source_dir)
        if tarball.stat().st_mtime >= newest:
            return "skipped", f"{rel} (up-to-date)"

    # Collect + sort entries for deterministic tar layout
    entries: list[Path] = []
    for p in source_dir.rglob("*"):
        rel_in_source = p.relative_to(source_dir).as_posix()
        if _matches_any(rel_in_source, SOURCE_IGNORE):
            continue
        entries.append(p)
    entries.sort(key=lambda p: p.relative_to(source_dir).as_posix())

    tmp_path = tarball.with_suffix(".tar.gz.tmp")
    try:
        # Two layers must be deterministic: the inner tar (entry order, owner,
        # mtimes) and the outer gzip (its own mtime + the embedded original
        # filename). Wrap a manually-constructed GzipFile so we control both
        # gzip header fields; tarfile.open("w:gz") leaks wall-clock time and
        # the temp filename otherwise.
        raw = open(tmp_path, "wb")
        gz = gzip.GzipFile(filename="", fileobj=raw, mode="wb",
                           compresslevel=6, mtime=EPOCH_MTIME)
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for p in entries:
                arcname = p.relative_to(source_dir.parent).as_posix()  # ``source/...``
                ti = tar.gettarinfo(str(p), arcname=arcname)
                if ti is None:
                    continue
                # Normalize for reproducibility
                ti.mtime = EPOCH_MTIME
                ti.uid = 0
                ti.gid = 0
                ti.uname = ""
                ti.gname = ""
                if ti.isreg():
                    with p.open("rb") as f:
                        tar.addfile(ti, f)
                else:
                    tar.addfile(ti)
        gz.close()
        raw.close()
        tmp_path.replace(tarball)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        return "error", f"{rel}: {e}"
    return "written", f"{rel} ({len(entries)} entries -> {tarball.stat().st_size // 1024} KiB)"


def _stale_tarballs(source_dirs: list[Path]) -> list[Path]:
    """Return source dirs whose tarball is missing or older than newest file."""
    stale = []
    for sd in source_dirs:
        tarball = sd.with_name("source.tar.gz")
        if not tarball.exists():
            stale.append(sd)
            continue
        if tarball.stat().st_mtime < _newest_mtime(sd):
            stale.append(sd)
    return stale


def _bundle_vendor_cybench(force: bool) -> tuple[str, str]:
    """Tar ``vendor/cybench/`` -> ``benchmark/_vendor/cybench.tar.gz``.

    Entry names are ``vendor/cybench/...`` so extraction at the project root
    lands the tree where cybench compose files expect it. SOURCE_IGNORE strips
    the same junk (.git, __pycache__, etc.) we strip from per-task source/."""
    if not VENDOR_CYBENCH_DIR.is_dir():
        return "skipped", "vendor/cybench/ not present"

    if VENDOR_BUNDLE_PATH.exists() and not force:
        if VENDOR_BUNDLE_PATH.stat().st_mtime >= _newest_mtime(VENDOR_CYBENCH_DIR):
            return "skipped", "vendor/cybench (up-to-date)"

    VENDOR_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    entries: list[Path] = []
    for p in VENDOR_CYBENCH_DIR.rglob("*"):
        rel_in_vendor = p.relative_to(VENDOR_CYBENCH_DIR).as_posix()
        if _matches_any(rel_in_vendor, SOURCE_IGNORE):
            continue
        entries.append(p)
    entries.sort(key=lambda p: p.relative_to(VENDOR_CYBENCH_DIR).as_posix())

    tmp_path = VENDOR_BUNDLE_PATH.with_suffix(".tar.gz.tmp")
    try:
        raw = open(tmp_path, "wb")
        gz = gzip.GzipFile(filename="", fileobj=raw, mode="wb",
                           compresslevel=6, mtime=EPOCH_MTIME)
        # arcname base is "vendor/cybench" — entries become ``vendor/cybench/...``,
        # so ``tar -xzf cybench.tar.gz -C <project_root>`` recreates the tree
        # at the path cybench composes reference.
        arc_base = Path("vendor") / "cybench"
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for p in entries:
                arcname = (arc_base / p.relative_to(VENDOR_CYBENCH_DIR)).as_posix()
                ti = tar.gettarinfo(str(p), arcname=arcname)
                if ti is None:
                    continue
                ti.mtime = EPOCH_MTIME
                ti.uid = 0
                ti.gid = 0
                ti.uname = ""
                ti.gname = ""
                if ti.isreg():
                    with p.open("rb") as f:
                        tar.addfile(ti, f)
                else:
                    tar.addfile(ti)
        gz.close()
        raw.close()
        tmp_path.replace(VENDOR_BUNDLE_PATH)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        return "error", f"vendor/cybench: {e}"
    size_mib = VENDOR_BUNDLE_PATH.stat().st_size // (1024 * 1024)
    return "written", f"vendor/cybench ({len(entries)} entries -> {size_mib} MiB)"


def _vendor_bundle_stale() -> bool:
    if not VENDOR_CYBENCH_DIR.is_dir():
        return False
    if not VENDOR_BUNDLE_PATH.exists():
        return True
    return VENDOR_BUNDLE_PATH.stat().st_mtime < _newest_mtime(VENDOR_CYBENCH_DIR)


def collect_source_dirs(source_filter: str | None) -> list[Path]:
    sources = ["cybench", "cybergym", "vulhub", "exploitdb"]
    if source_filter:
        sources = [source_filter]
    out: list[Path] = []
    for s in sources:
        root = BENCHMARK_DIR / s
        if not root.is_dir():
            continue
        for task in sorted(root.iterdir()):
            sd = task / "source"
            if sd.is_dir():
                out.append(sd)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["cybench", "cybergym", "vulhub", "exploitdb"],
                        help="Limit to one benchmark source. Default: all four.")
    parser.add_argument("--force", action="store_true", help="Rebuild every tarball even if up-to-date")
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 if any tarball is stale; don't write anything. "
                             "Used by the publisher's pre-flight check.")
    args = parser.parse_args()

    source_dirs = collect_source_dirs(args.source)
    log.info("Found %d source/ trees%s", len(source_dirs),
             f" under {args.source}/" if args.source else "")

    if args.check:
        stale = _stale_tarballs(source_dirs)
        vendor_stale = (args.source is None or args.source == "cybench") and _vendor_bundle_stale()
        if stale or vendor_stale:
            if stale:
                log.error("STALE: %d source.tar.gz missing or older than source/", len(stale))
                for sd in stale[:10]:
                    log.error("  - %s", sd.relative_to(BENCHMARK_DIR))
                if len(stale) > 10:
                    log.error("  ... (%d more)", len(stale) - 10)
            if vendor_stale:
                log.error("STALE: vendor/cybench bundle missing or older than vendor/cybench/")
            log.error("Run: uv run python scripts/bundle_task_sources.py")
            sys.exit(1)
        log.info("All %d tarballs up-to-date.", len(source_dirs))
        return

    written = skipped = errors = 0
    for sd in source_dirs:
        status, msg = _bundle_one(sd, args.force)
        if status == "written":
            written += 1
            log.info("  ✓ %s", msg)
        elif status == "skipped":
            skipped += 1
        else:
            errors += 1
            log.error("  ✗ %s", msg)

    # Vendor bundling is part of the cybench source contract — do it whenever
    # the user is bundling cybench (or all sources), and skip when they
    # explicitly scoped to a different source.
    if args.source in (None, "cybench"):
        status, msg = _bundle_vendor_cybench(args.force)
        if status == "written":
            written += 1
            log.info("  ✓ %s", msg)
        elif status == "skipped":
            skipped += 1
            log.info("  - %s", msg)
        else:
            errors += 1
            log.error("  ✗ %s", msg)

    log.info("Done. written=%d skipped=%d errors=%d", written, skipped, errors)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
