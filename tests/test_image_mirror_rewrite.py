"""Tests for the GHCR image-mirror rewriting in generate_compose.

Without these guarantees, setting CYBERAGENTBENCH_IMAGE_MIRROR is a silent
no-op (the failure mode the spec docs accidentally claimed was already wired
up). These tests pin the contract so a future refactor can't quietly break it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from refusal_eval import sandbox


@pytest.fixture(autouse=True)
def _clear_mirror_cache():
    """The mirror map is lru_cached for process lifetime — clear between tests
    so each test sees its own monkeypatched env + manifest."""
    sandbox._load_mirror_map.cache_clear()
    yield
    sandbox._load_mirror_map.cache_clear()


def _write_manifest(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "image-mirror-manifest.json"
    p.write_text(json.dumps({
        "namespace": "ghcr.io/example/mirror",
        "tag": "v1",
        "entries": entries,
        "failures": [],
    }))
    return p


def test_no_env_var_is_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("CYBERAGENTBENCH_IMAGE_MIRROR", raising=False)
    monkeypatch.setattr(sandbox, "MIRROR_MANIFEST_PATH",
                        _write_manifest(tmp_path, [
                            {"upstream": "mongo:4.0",
                             "mirror_digest": "ghcr.io/example/mirror/mongo-4.0@sha256:" + "a" * 64},
                        ]))
    compose = {"services": {"mongo": {"image": "mongo:4.0"}}}
    sandbox._rewrite_for_mirror(compose)
    assert compose["services"]["mongo"]["image"] == "mongo:4.0"  # untouched


def test_env_var_with_matching_entry_rewrites_to_digest(monkeypatch, tmp_path):
    monkeypatch.setenv("CYBERAGENTBENCH_IMAGE_MIRROR", "ghcr.io/example/mirror")
    digest_ref = "ghcr.io/example/mirror/mongo-4.0@sha256:" + "a" * 64
    monkeypatch.setattr(sandbox, "MIRROR_MANIFEST_PATH",
                        _write_manifest(tmp_path, [
                            {"upstream": "mongo:4.0", "mirror_digest": digest_ref,
                             "mirror_tag": "ghcr.io/example/mirror/mongo-4.0:v1"},
                        ]))
    compose = {"services": {"mongo": {"image": "mongo:4.0"}}}
    sandbox._rewrite_for_mirror(compose)
    assert compose["services"]["mongo"]["image"] == digest_ref


def test_falls_back_to_mirror_tag_when_digest_missing(monkeypatch, tmp_path):
    """Some mirror entries have no parsed digest (push stdout didn't surface
    one). The rewriting must still succeed using mirror_tag."""
    monkeypatch.setenv("CYBERAGENTBENCH_IMAGE_MIRROR", "ghcr.io/example/mirror")
    tag_ref = "ghcr.io/example/mirror/mongo-4.0:v1"
    monkeypatch.setattr(sandbox, "MIRROR_MANIFEST_PATH",
                        _write_manifest(tmp_path, [
                            {"upstream": "mongo:4.0", "mirror_digest": None, "mirror_tag": tag_ref},
                        ]))
    compose = {"services": {"mongo": {"image": "mongo:4.0"}}}
    sandbox._rewrite_for_mirror(compose)
    assert compose["services"]["mongo"]["image"] == tag_ref


def test_image_not_in_manifest_passes_through(monkeypatch, tmp_path):
    """An image absent from the manifest is left as-is — docker will pull it
    from upstream. (Common during partial-mirror rollout.)"""
    monkeypatch.setenv("CYBERAGENTBENCH_IMAGE_MIRROR", "ghcr.io/example/mirror")
    monkeypatch.setattr(sandbox, "MIRROR_MANIFEST_PATH",
                        _write_manifest(tmp_path, [
                            {"upstream": "mongo:4.0", "mirror_digest": "ghcr.io/example/mirror/mongo-4.0@sha256:" + "a"*64},
                        ]))
    compose = {"services": {
        "mongo": {"image": "mongo:4.0"},
        "redis": {"image": "redis:7"},   # not mirrored
    }}
    sandbox._rewrite_for_mirror(compose)
    assert compose["services"]["mongo"]["image"].startswith("ghcr.io/example/mirror/")
    assert compose["services"]["redis"]["image"] == "redis:7"


def test_internal_services_never_rewritten(monkeypatch, tmp_path):
    """default (agent), grader, controller use locally-built or hardcoded
    images that should never be redirected even if the upstream name happens
    to match a manifest entry — those entries refer to vendor targets, not
    sidecars."""
    monkeypatch.setenv("CYBERAGENTBENCH_IMAGE_MIRROR", "ghcr.io/example/mirror")
    # Pretend somebody mirrored the agent image too
    monkeypatch.setattr(sandbox, "MIRROR_MANIFEST_PATH",
                        _write_manifest(tmp_path, [
                            {"upstream": sandbox.AGENT_IMAGE,
                             "mirror_digest": "ghcr.io/example/mirror/agent@sha256:" + "b"*64},
                            {"upstream": sandbox.GRADER_IMAGE,
                             "mirror_digest": "ghcr.io/example/mirror/grader@sha256:" + "c"*64},
                        ]))
    compose = {"services": {
        "default": {"image": sandbox.AGENT_IMAGE},
        sandbox.GRADER_SERVICE_NAME: {"image": sandbox.GRADER_IMAGE},
        sandbox.CONTROLLER_SERVICE_NAME: {"image": sandbox.CONTROLLER_IMAGE},
        "vendor_target": {"image": "vendor:1.0"},
    }}
    sandbox._rewrite_for_mirror(compose)
    assert compose["services"]["default"]["image"] == sandbox.AGENT_IMAGE
    assert compose["services"][sandbox.GRADER_SERVICE_NAME]["image"] == sandbox.GRADER_IMAGE
    assert compose["services"][sandbox.CONTROLLER_SERVICE_NAME]["image"] == sandbox.CONTROLLER_IMAGE
    # Untouched (not in manifest) — pass-through
    assert compose["services"]["vendor_target"]["image"] == "vendor:1.0"


def test_missing_manifest_is_noop_with_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("CYBERAGENTBENCH_IMAGE_MIRROR", "ghcr.io/example/mirror")
    monkeypatch.setattr(sandbox, "MIRROR_MANIFEST_PATH", tmp_path / "does-not-exist.json")
    compose = {"services": {"mongo": {"image": "mongo:4.0"}}}
    with caplog.at_level("WARNING"):
        sandbox._rewrite_for_mirror(compose)
    assert compose["services"]["mongo"]["image"] == "mongo:4.0"
    assert any("missing" in rec.message.lower() for rec in caplog.records)
