"""Tests for the source-bundle/extract round-trip.

The publisher excludes ``source/`` and ships ``source.tar.gz`` instead. The
sync script extracts the tarball back into ``source/``. This test verifies the
two halves stay consistent: after bundle → extract, the tree is byte-identical
to the original (modulo mtime/uid/gid normalization).

We also cover the determinism guarantee — two consecutive bundles of the same
input must produce byte-identical tarballs — because if it ever breaks, every
publish will churn LFS storage even when no source actually changed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_bundler():
    spec = importlib.util.spec_from_file_location(
        "bundle_task_sources", PROJECT_ROOT / "scripts" / "bundle_task_sources.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bundle_task_sources"] = mod
    spec.loader.exec_module(mod)
    return mod


def _file_hashes(root: Path) -> dict[str, str]:
    """Map relative-path -> SHA256 for every regular file under root."""
    out = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root).as_posix()
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        out[rel] = h
    return out


@pytest.fixture
def fake_task(tmp_path, monkeypatch):
    """Create a minimal benchmark/<source>/<task>/source/ tree, point the
    bundler's BENCHMARK_DIR at tmp_path/benchmark, and yield the source dir."""
    bundler = _load_bundler()
    bench = tmp_path / "benchmark"
    src_dir = bench / "vulhub" / "fake_CVE-2024-1/source"
    src_dir.mkdir(parents=True)
    (src_dir / "README.md").write_text("hello\n")
    (src_dir / "src").mkdir()
    (src_dir / "src" / "main.py").write_text("print('hi')\n")
    (src_dir / "src" / "subdir").mkdir()
    (src_dir / "src" / "subdir" / "deep.txt").write_text("nested\n")
    # Dropped junk that the bundler should ignore
    (src_dir / ".DS_Store").write_text("OSX junk\n")
    (src_dir / ".cache").mkdir()
    (src_dir / ".cache" / "stuff.bin").write_text("cache\n")
    (src_dir / "src" / "__pycache__").mkdir()
    (src_dir / "src" / "__pycache__" / "main.cpython-312.pyc").write_text("bytecode\n")

    monkeypatch.setattr(bundler, "BENCHMARK_DIR", bench)
    return src_dir, bundler


def test_bundle_excludes_ignore_patterns(fake_task):
    src_dir, bundler = fake_task
    status, _ = bundler._bundle_one(src_dir, force=True)
    assert status == "written"

    tarball = src_dir.with_name("source.tar.gz")
    assert tarball.exists()
    with tarfile.open(tarball, "r:gz") as tar:
        names = sorted(m.name for m in tar.getmembers() if m.isreg())
    # Junk excluded
    assert not any(".DS_Store" in n for n in names)
    assert not any(".cache" in n for n in names)
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)
    # Real content present, prefixed with source/
    assert "source/README.md" in names
    assert "source/src/main.py" in names
    assert "source/src/subdir/deep.txt" in names


def test_bundle_is_deterministic(fake_task):
    src_dir, bundler = fake_task
    bundler._bundle_one(src_dir, force=True)
    h1 = hashlib.sha256(src_dir.with_name("source.tar.gz").read_bytes()).hexdigest()
    bundler._bundle_one(src_dir, force=True)
    h2 = hashlib.sha256(src_dir.with_name("source.tar.gz").read_bytes()).hexdigest()
    assert h1 == h2, "tarball not byte-deterministic"


def test_roundtrip_extract_matches_original(fake_task, tmp_path):
    src_dir, bundler = fake_task
    bundler._bundle_one(src_dir, force=True)
    tarball = src_dir.with_name("source.tar.gz")

    # Capture original (minus ignore-listed junk)
    original = _file_hashes(src_dir)
    original = {k: v for k, v in original.items()
                if ".DS_Store" not in k and ".cache" not in k
                and "__pycache__" not in k and not k.endswith(".pyc")}

    # Extract elsewhere and compare
    extract_root = tmp_path / "extracted"
    extract_root.mkdir()
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(extract_root, filter="data")
    extracted = _file_hashes(extract_root / "source")
    assert extracted == original


def test_check_skips_when_uptodate(fake_task):
    src_dir, bundler = fake_task
    status1, _ = bundler._bundle_one(src_dir, force=False)
    assert status1 == "written"
    status2, _ = bundler._bundle_one(src_dir, force=False)
    assert status2 == "skipped"


def test_check_detects_stale_when_source_modified(fake_task):
    src_dir, bundler = fake_task
    bundler._bundle_one(src_dir, force=False)
    tarball = src_dir.with_name("source.tar.gz")
    # Push tarball mtime into the past so source/ becomes "newer"
    import os, time
    past = time.time() - 3600
    os.utime(tarball, (past, past))
    # Touch a file under source/ to bump the dir's newest mtime
    (src_dir / "README.md").write_text("hello v2\n")
    stale = bundler._stale_tarballs([src_dir])
    assert src_dir in stale


def test_check_clean_for_uptodate_tarball(fake_task):
    src_dir, bundler = fake_task
    bundler._bundle_one(src_dir, force=True)
    assert bundler._stale_tarballs([src_dir]) == []


def test_vendor_cybench_bundle_roundtrip(tmp_path, monkeypatch):
    """vendor/cybench/ ships *inside* benchmark/ so it rides along on HF
    upload, but extracts back to PROJECT_ROOT so cybench compose build
    contexts (../../../vendor/cybench/...) resolve. Pin the layout."""
    bundler = _load_bundler()
    bench = tmp_path / "benchmark"
    bench.mkdir()
    vendor = tmp_path / "vendor" / "cybench"
    challenge = vendor / "benchmark" / "hackthebox" / "cyber-apocalypse-2024" / "pwn" / "[Medium] Death Note"
    challenge.mkdir(parents=True)
    (challenge / "Dockerfile").write_text("FROM alpine\n")
    (challenge / "challenge").write_bytes(b"\x7fELF...")
    # Junk that should be excluded
    (vendor / ".git").mkdir()
    (vendor / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    monkeypatch.setattr(bundler, "BENCHMARK_DIR", bench)
    monkeypatch.setattr(bundler, "VENDOR_CYBENCH_DIR", vendor)
    monkeypatch.setattr(bundler, "VENDOR_BUNDLE_PATH", bench / "_vendor" / "cybench.tar.gz")

    status, _ = bundler._bundle_vendor_cybench(force=True)
    assert status == "written"
    bundle = bench / "_vendor" / "cybench.tar.gz"
    assert bundle.is_file()

    # Determinism
    h1 = hashlib.sha256(bundle.read_bytes()).hexdigest()
    bundler._bundle_vendor_cybench(force=True)
    h2 = hashlib.sha256(bundle.read_bytes()).hexdigest()
    assert h1 == h2

    # Entry names must start with vendor/cybench/ — that's what makes
    # extraction at PROJECT_ROOT land things in the right place
    with tarfile.open(bundle, "r:gz") as tar:
        names = sorted(m.name for m in tar.getmembers() if m.isreg())
    assert all(n.startswith("vendor/cybench/") for n in names)
    assert any(".git" not in n for n in names)
    assert not any("/.git/" in n for n in names)
    assert any("Death Note/Dockerfile" in n for n in names)

    # Round-trip: extract at a fresh "project root" and confirm the path the
    # cybench composes reference exists
    extract_root = tmp_path / "fresh_root"
    extract_root.mkdir()
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(extract_root, filter="fully_trusted")
    assert (extract_root / "vendor" / "cybench" / "benchmark" / "hackthebox"
            / "cyber-apocalypse-2024" / "pwn" / "[Medium] Death Note"
            / "Dockerfile").is_file()


def test_vendor_bundle_skipped_when_dir_absent(tmp_path, monkeypatch):
    bundler = _load_bundler()
    monkeypatch.setattr(bundler, "VENDOR_CYBENCH_DIR", tmp_path / "nope")
    monkeypatch.setattr(bundler, "VENDOR_BUNDLE_PATH", tmp_path / "_vendor" / "cybench.tar.gz")
    status, msg = bundler._bundle_vendor_cybench(force=True)
    assert status == "skipped"
    assert "not present" in msg
