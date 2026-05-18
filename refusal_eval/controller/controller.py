"""Controller sidecar — owns docker-compose lifecycle for target+grader.

Runs inside its own container on shared_net alongside the agent, the grader,
and the vendor target services. Exposes a small RPC surface that the solver
calls between phases to recreate target + grader (with a fresh HMAC token)
so each validated-subset phase sees clean target state.

Endpoints (gated by X-Controller-Token header except /healthz):
  POST /restart    — mint new grader token, write state file, recreate
                     target + grader services, return new token to caller.
  GET  /healthz    — unauthenticated liveness; returns 200 once startup
                     (initial token mint + state-file write) has completed.
                     Used as the compose healthcheck the grader depends_on.
  POST /shutdown   — stop target + grader. Optional, for dev/curate parity.

Compose project name is discovered at startup via the docker SDK
(`com.docker.compose.project` label off the controller's own container)
— same pattern as TargetFsFileCheckProbe._project_name() in
refusal_eval/grader/probes.py.

Token rotation uses the shared `grader_state` named volume mounted at
/run/grader_state. The state file is `/run/grader_state/token.json`. The
grader reads this file at boot (because we recreate the grader container
on every /restart, this picks up the new token automatically).
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import secrets
import sys
import tempfile

from aiohttp import web

log = logging.getLogger("controller")

DEFAULT_PORT = 9990
DEFAULT_GRADER_HEALTH_TIMEOUT = 30
COMPOSE_PATH = "/run/compose.yaml"
GRADER_STATE_DIR = "/run/grader_state"
GRADER_STATE_FILE = os.path.join(GRADER_STATE_DIR, "token.json")
GRADER_INTERNAL_HOST = "grader"
GRADER_INTERNAL_PORT = 9999


class ControllerState:
    """Process-wide controller state."""

    def __init__(
        self,
        *,
        token: str,
        project: str,
        target_services: list[str],
        grader_service: str,
    ) -> None:
        self.token = token
        self.project = project
        self.target_services = target_services
        self.grader_service = grader_service
        self.lock = asyncio.Lock()
        self.current_grader_token: str | None = None


STATE_KEY: web.AppKey[ControllerState] = web.AppKey("state", ControllerState)


def _discover_project() -> str:
    """Read com.docker.compose.project label off this container."""
    import docker

    hostname = os.environ.get("HOSTNAME", "")
    if not hostname:
        raise RuntimeError("HOSTNAME not set in controller container")
    client = docker.from_env()
    container = client.containers.get(hostname)
    project = container.labels.get("com.docker.compose.project", "")
    if not project:
        raise RuntimeError("compose project label missing on controller container")
    return project


def _mint_token() -> str:
    return secrets.token_hex(32)


def _write_grader_state(token: str, phase: str) -> None:
    """Atomically write the grader state file (tmpfile + rename)."""
    os.makedirs(GRADER_STATE_DIR, exist_ok=True)
    payload = {
        "grader_token": token,
        "minted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": phase,
    }
    fd, tmp_path = tempfile.mkstemp(dir=GRADER_STATE_DIR, prefix=".token.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, GRADER_STATE_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


async def _run_compose(state: ControllerState, args: list[str]) -> tuple[int, str, str]:
    """Run `docker-compose -f <compose> -p <project> <args>`; return (rc, stdout, stderr)."""
    cmd = ["docker-compose", "-f", COMPOSE_PATH, "-p", state.project, *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    return proc.returncode or 0, stdout_b.decode("utf-8", "replace"), stderr_b.decode("utf-8", "replace")


async def _wait_grader_healthy(timeout_seconds: float) -> bool:
    """Poll http://grader:9999/healthz until 200 or timeout."""
    import aiohttp

    deadline = asyncio.get_event_loop().time() + timeout_seconds
    url = f"http://{GRADER_INTERNAL_HOST}:{GRADER_INTERNAL_PORT}/healthz"
    async with aiohttp.ClientSession() as session:
        while asyncio.get_event_loop().time() < deadline:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
    return False


def _check_token(request: web.Request, state: ControllerState) -> web.Response | None:
    supplied = request.headers.get("X-Controller-Token", "")
    if not supplied or supplied != state.token:
        return web.json_response({"error": "unauthorized"}, status=401)
    return None


async def healthz_handler(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def restart_handler(request: web.Request) -> web.Response:
    state: ControllerState = request.app[STATE_KEY]
    log.info("restart: request received from %s", request.remote)
    auth_err = _check_token(request, state)
    if auth_err is not None:
        log.warning("restart: auth failed")
        return auth_err

    try:
        body = await request.json()
    except Exception:
        body = {}
    phase = body.get("phase") or "unknown"
    startup_timeout = float(body.get("startup_timeout_seconds", DEFAULT_GRADER_HEALTH_TIMEOUT))
    log.info("restart: phase=%s timeout=%s", phase, startup_timeout)

    services = list(state.target_services) + [state.grader_service]

    async with state.lock:
        new_token = _mint_token()
        try:
            _write_grader_state(new_token, phase)
        except OSError as e:
            log.error("restart: failed to write grader state: %s", e)
            return web.json_response(
                {"status": "state_write_failed", "error": str(e)},
                status=500,
            )

        # Use `up -d --force-recreate <services>` so target+grader are
        # atomically replaced without touching the project's network or the
        # controller's own container. `--no-deps` prevents compose from
        # touching the controller (which is the grader's depends_on target).
        log.info("restart: recreating services=%s", services)
        rc, stdout, stderr = await _run_compose(
            state, ["up", "-d", "--force-recreate", "--no-deps", *services]
        )
        if rc != 0:
            log.error(
                "restart: compose up --force-recreate failed (rc=%d): %s",
                rc, stderr.strip(),
            )
            return web.json_response(
                {
                    "status": "compose_failed",
                    "action": "up --force-recreate",
                    "exit_code": rc,
                    "stderr": stderr[-2000:],
                },
                status=500,
            )

        log.info("restart: waiting for grader /healthz (timeout=%ss)", startup_timeout)
        ready = await _wait_grader_healthy(startup_timeout)
        if not ready:
            log.error("restart: grader did not become healthy in %ss", startup_timeout)
            return web.json_response(
                {"status": "grader_not_ready", "timeout_seconds": startup_timeout},
                status=504,
            )

        state.current_grader_token = new_token
        log.info("restart: ok, new token minted for phase=%s", phase)

    return web.json_response(
        {
            "status": "ok",
            "grader_token": new_token,
            "phase": phase,
        }
    )


async def shutdown_handler(request: web.Request) -> web.Response:
    state: ControllerState = request.app[STATE_KEY]
    auth_err = _check_token(request, state)
    if auth_err is not None:
        return auth_err
    services = list(state.target_services) + [state.grader_service]
    rc, _, stderr = await _run_compose(state, ["stop", *services])
    if rc != 0:
        return web.json_response({"status": "compose_failed", "stderr": stderr[-2000:]}, status=500)
    return web.json_response({"status": "ok"})


def build_app(state: ControllerState) -> web.Application:
    app = web.Application()
    app[STATE_KEY] = state
    app.router.add_get("/healthz", healthz_handler)
    app.router.add_post("/restart", restart_handler)
    app.router.add_post("/shutdown", shutdown_handler)
    return app


def _parse_csv(env_name: str) -> list[str]:
    raw = os.environ.get(env_name, "")
    return [s.strip() for s in raw.split(",") if s.strip()]


async def _serve(app: web.Application, port: int) -> None:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log.info("controller listening on :%d", port)
    while True:
        await asyncio.sleep(3600)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    token = os.environ.get("CONTROLLER_TOKEN") or ""
    if not token:
        log.error("CONTROLLER_TOKEN not set")
        sys.exit(2)

    target_services = _parse_csv("TARGET_SERVICES")
    if not target_services:
        log.error("TARGET_SERVICES not set")
        sys.exit(2)

    grader_service = os.environ.get("GRADER_SERVICE") or ""
    if not grader_service:
        log.error("GRADER_SERVICE not set")
        sys.exit(2)

    try:
        project = _discover_project()
    except Exception as e:
        log.error("could not discover compose project: %s", e)
        sys.exit(2)

    # Mint T0 + write the state file before the HTTP server comes up.
    # The grader's compose `depends_on: { controller: { condition: service_healthy } }`
    # means it cannot boot until /healthz is 200, which doesn't happen until
    # this `_serve` coroutine starts the listener — so the file is guaranteed
    # populated before the grader reads it.
    initial_token = _mint_token()
    try:
        _write_grader_state(initial_token, phase="initial")
    except OSError as e:
        log.error("failed to write initial grader state: %s", e)
        sys.exit(2)

    state = ControllerState(
        token=token,
        project=project,
        target_services=target_services,
        grader_service=grader_service,
    )
    state.current_grader_token = initial_token

    port = int(os.environ.get("CONTROLLER_PORT", DEFAULT_PORT))
    app = build_app(state)
    asyncio.run(_serve(app, port))


if __name__ == "__main__":
    main()
