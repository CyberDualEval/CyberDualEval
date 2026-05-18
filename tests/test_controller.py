"""Unit tests for refusal_eval.controller.controller.

Mocks the docker-compose CLI and grader-healthcheck helpers; exercises
the HTTP surface via aiohttp's in-process TestClient.
"""

from __future__ import annotations

import json
import os

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from refusal_eval.controller import controller as ctrl


@pytest.fixture
def tmp_state_dir(tmp_path, monkeypatch):
    """Redirect the grader_state path to a tmp dir for the duration of a test."""
    state_dir = tmp_path / "grader_state"
    state_dir.mkdir()
    monkeypatch.setattr(ctrl, "GRADER_STATE_DIR", str(state_dir))
    monkeypatch.setattr(ctrl, "GRADER_STATE_FILE", str(state_dir / "token.json"))
    return state_dir


@pytest.fixture
def fake_state(tmp_state_dir):
    """A ControllerState with deterministic token + identifiers."""
    return ctrl.ControllerState(
        token="ctrl-secret",
        project="proj-x",
        target_services=["web"],
        grader_service="grader",
    )


@pytest_asyncio.fixture
async def client(fake_state, monkeypatch):
    """aiohttp TestClient bound to a controller app + fake_state."""
    # Default mocks: compose CLI succeeds; grader becomes healthy.
    async def fake_run_compose(state, args):
        return 0, "", ""

    async def fake_wait_grader_healthy(timeout):
        return True

    monkeypatch.setattr(ctrl, "_run_compose", fake_run_compose)
    monkeypatch.setattr(ctrl, "_wait_grader_healthy", fake_wait_grader_healthy)

    app = ctrl.build_app(fake_state)
    server = TestServer(app)
    async with TestClient(server) as cli:
        yield cli


# ---------- Helper-level tests --------------------------------------------


def test_mint_token_is_hex_64chars():
    token = ctrl._mint_token()
    assert len(token) == 64
    int(token, 16)  # parses as hex


def test_write_grader_state_atomic(tmp_state_dir):
    ctrl._write_grader_state("abc123", phase="poc")
    payload = json.loads((tmp_state_dir / "token.json").read_text())
    assert payload["grader_token"] == "abc123"
    assert payload["phase"] == "poc"
    # No leftover tmpfiles in the directory after a successful write.
    leftovers = [p.name for p in tmp_state_dir.iterdir() if p.name.startswith(".token.")]
    assert leftovers == []


def test_write_grader_state_overwrites(tmp_state_dir):
    ctrl._write_grader_state("v1", phase="initial")
    ctrl._write_grader_state("v2", phase="exploit")
    payload = json.loads((tmp_state_dir / "token.json").read_text())
    assert payload["grader_token"] == "v2"
    assert payload["phase"] == "exploit"


def test_parse_csv_strips_and_filters(monkeypatch):
    monkeypatch.setenv("FOO", "a, b ,, c")
    assert ctrl._parse_csv("FOO") == ["a", "b", "c"]


# ---------- HTTP surface --------------------------------------------------


@pytest.mark.asyncio
async def test_healthz_unauthenticated(client):
    resp = await client.get("/healthz")
    assert resp.status == 200
    body = await resp.json()
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_restart_requires_token(client):
    resp = await client.post(
        "/restart", json={"phase": "exploit", "startup_timeout_seconds": 5}
    )
    assert resp.status == 401


@pytest.mark.asyncio
async def test_restart_rejects_wrong_token(client):
    resp = await client.post(
        "/restart",
        json={"phase": "exploit", "startup_timeout_seconds": 5},
        headers={"X-Controller-Token": "wrong"},
    )
    assert resp.status == 401


@pytest.mark.asyncio
async def test_restart_rotates_token_and_returns_it(client, fake_state, tmp_state_dir):
    resp = await client.post(
        "/restart",
        json={"phase": "exploit", "startup_timeout_seconds": 5},
        headers={"X-Controller-Token": "ctrl-secret"},
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["status"] == "ok"
    assert body["phase"] == "exploit"
    new_token = body["grader_token"]
    assert len(new_token) == 64

    # State file reflects the new token + phase.
    payload = json.loads((tmp_state_dir / "token.json").read_text())
    assert payload["grader_token"] == new_token
    assert payload["phase"] == "exploit"

    # ControllerState tracks the newest token.
    assert fake_state.current_grader_token == new_token


@pytest.mark.asyncio
async def test_restart_each_call_rotates(client, tmp_state_dir):
    headers = {"X-Controller-Token": "ctrl-secret"}
    seen = set()
    for _ in range(3):
        resp = await client.post(
            "/restart", json={"phase": "exploit"}, headers=headers
        )
        assert resp.status == 200
        body = await resp.json()
        seen.add(body["grader_token"])
    assert len(seen) == 3


@pytest.mark.asyncio
async def test_restart_500_on_compose_failure(fake_state, monkeypatch, tmp_state_dir):
    async def boom(state, args):
        return 1, "", "compose blew up"

    async def healthy(timeout):
        return True

    monkeypatch.setattr(ctrl, "_run_compose", boom)
    monkeypatch.setattr(ctrl, "_wait_grader_healthy", healthy)

    app = ctrl.build_app(fake_state)
    server = TestServer(app)
    async with TestClient(server) as cli:
        resp = await cli.post(
            "/restart",
            json={"phase": "exploit"},
            headers={"X-Controller-Token": "ctrl-secret"},
        )
        assert resp.status == 500
        body = await resp.json()
        assert body["status"] == "compose_failed"
        assert "compose blew up" in body["stderr"]


@pytest.mark.asyncio
async def test_restart_504_on_grader_not_ready(fake_state, monkeypatch, tmp_state_dir):
    async def ok_compose(state, args):
        return 0, "", ""

    async def never_healthy(timeout):
        return False

    monkeypatch.setattr(ctrl, "_run_compose", ok_compose)
    monkeypatch.setattr(ctrl, "_wait_grader_healthy", never_healthy)

    app = ctrl.build_app(fake_state)
    server = TestServer(app)
    async with TestClient(server) as cli:
        resp = await cli.post(
            "/restart",
            json={"phase": "exploit", "startup_timeout_seconds": 1},
            headers={"X-Controller-Token": "ctrl-secret"},
        )
        assert resp.status == 504
        body = await resp.json()
        assert body["status"] == "grader_not_ready"


@pytest.mark.asyncio
async def test_shutdown_requires_token(client):
    resp = await client.post("/shutdown")
    assert resp.status == 401


@pytest.mark.asyncio
async def test_shutdown_calls_compose_stop(fake_state, monkeypatch, tmp_state_dir):
    calls = []

    async def record(state, args):
        calls.append(args)
        return 0, "", ""

    monkeypatch.setattr(ctrl, "_run_compose", record)

    app = ctrl.build_app(fake_state)
    server = TestServer(app)
    async with TestClient(server) as cli:
        resp = await cli.post(
            "/shutdown", headers={"X-Controller-Token": "ctrl-secret"}
        )
        assert resp.status == 200
        assert calls == [["stop", "web", "grader"]]
