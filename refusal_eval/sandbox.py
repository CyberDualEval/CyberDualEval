"""Generate per-task Docker Compose files for inspect sandboxes.

Inspect requires a service named ``default`` (or marked ``x-default: true``)
as the agent sandbox.  We generate a compose file that:
  1. Defines a ``default`` service using the Kali Linux image (the agent).
  2. Includes the task's challenge services (if any) on a shared network.
  3. Optionally rewrites every ``image:`` reference to a digest-pinned mirror
     ref when ``CYBERAGENTBENCH_IMAGE_MIRROR`` is set in the environment and
     ``docs/image-mirror-manifest.json`` carries a matching entry. This makes
     vendor pulls go through the maintainer-owned GHCR mirror instead of
     Docker Hub, sidestepping rate limits and giving every collaborator a
     bit-identical image.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

AGENT_IMAGE = (
    "mms2309/cyberrefusal-agent"
    "@sha256:793d11c6a38c63a69ef7fa5aae44662ed394727e279f57c91266b1a83ef3608e"
)
GRADER_IMAGE = "cyberbench/grader:latest"
GRADER_SERVICE_NAME = "grader"
GRADER_INTERNAL_PORT = 9999
GRADER_STATE_VOLUME = "grader_state"
GRADER_STATE_MOUNT = "/run/grader_state"
GRADER_TOKEN_FILE_PATH = f"{GRADER_STATE_MOUNT}/token.json"
CONTROLLER_IMAGE = "cyberbench/controller:latest"
CONTROLLER_SERVICE_NAME = "controller"
CONTROLLER_INTERNAL_PORT = 9990
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = os.path.join(
    tempfile.gettempdir(), "cyberagent-bench", "sandbox-configs"
)
ORACLE_SCRIPT = os.path.join(os.path.dirname(__file__), "oracle.py")
MIRROR_MANIFEST_PATH = PROJECT_ROOT / "docs" / "image-mirror-manifest.json"

# Services we never rewrite — they're either built locally (grader, controller),
# the agent sandbox (intentionally hardcoded), or generated from local Dockerfile
# (oracle). Vendor target services are everything else.
_NEVER_REWRITE_SERVICES = {"default", GRADER_SERVICE_NAME, CONTROLLER_SERVICE_NAME}
_RESOURCE_ROLES = {"default", "target", "oracle", GRADER_SERVICE_NAME, CONTROLLER_SERVICE_NAME}
_RESOURCE_KEYS = {"memory", "cpus", "pids_limit"}


@functools.lru_cache(maxsize=1)
def _load_mirror_map() -> dict[str, str]:
    """Return {upstream_image: mirror_digest_ref} from the manifest.

    Returns empty dict (no rewriting) when:
      - ``CYBERAGENTBENCH_IMAGE_MIRROR`` env var is not set, OR
      - the manifest file doesn't exist, OR
      - the manifest is malformed.

    Cached for the process lifetime — if a long-running curation session
    needs to pick up new mirror entries, it must restart. Reasonable since
    new mirror entries appear ~once per maintainer mirror sweep.
    """
    if not os.environ.get("CYBERAGENTBENCH_IMAGE_MIRROR"):
        return {}
    if not MIRROR_MANIFEST_PATH.is_file():
        log.warning(
            "CYBERAGENTBENCH_IMAGE_MIRROR set but %s missing; pulls will go to "
            "upstream. Run scripts/mirror_images_to_ghcr.py to populate it.",
            MIRROR_MANIFEST_PATH,
        )
        return {}
    try:
        manifest = json.loads(MIRROR_MANIFEST_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("image-mirror-manifest.json unreadable: %s", e)
        return {}
    out: dict[str, str] = {}
    for entry in manifest.get("entries", []):
        upstream = entry.get("upstream")
        ref = entry.get("mirror_digest") or entry.get("mirror_tag")
        if upstream and ref:
            out[upstream] = ref
    if out:
        log.info(
            "Image mirror active: %d entries loaded from %s",
            len(out), MIRROR_MANIFEST_PATH.name,
        )
    return out


def _rewrite_for_mirror(compose: dict) -> None:
    """Walk ``compose['services'].*.image`` and rewrite each vendor target's
    image to its digest-pinned mirror ref when one exists in the manifest.
    Mutates the dict in place. No-op if the mirror is not configured.
    """
    mirror = _load_mirror_map()
    if not mirror:
        return
    for svc_name, svc_cfg in (compose.get("services") or {}).items():
        if svc_name in _NEVER_REWRITE_SERVICES:
            continue
        if not isinstance(svc_cfg, dict):
            continue
        upstream = svc_cfg.get("image")
        if not upstream:
            continue
        mirror_ref = mirror.get(upstream)
        if mirror_ref:
            svc_cfg["image"] = mirror_ref


def generate_compose(
    task_dir: str,
    *,
    metadata: dict | None = None,
    agent_image: str = AGENT_IMAGE,
    return_info: bool = False,
    include_controller: bool = True,
    sandbox_resources: dict | None = None,
) -> str | dict:
    """Generate a Docker Compose file for the given task.

    If ``compose_source`` is set in metadata, loads real challenge services.
    Otherwise, creates a mocked oracle from ``target_host``. When the task's
    metadata contains a top-level ``validation`` block, also splices in a
    ``grader`` sidecar on ``shared_net``.

    Two grader-token wiring modes, gated on ``include_controller``:

    * ``True`` (default; eval-time): also splices a ``controller`` sidecar
      that owns docker-compose lifecycle for target+grader. The grader
      reads its HMAC token from ``GRADER_TOKEN_FILE`` on the shared
      ``grader_state`` volume; the controller mints a fresh token on
      every ``/restart`` (called by the solver between phases). The agent
      container never sees either token. Returns ``controller_token`` so
      the solver can authenticate ``/restart`` calls.

    * ``False`` (curate / verify-evidence path): no controller. The grader
      gets ``GRADER_TOKEN`` directly as an env var, set by this function.
      Returns ``grader_token`` so curation tools can drive the grader.

    By default returns the compose path (``str``). When ``return_info=True``
    returns a dict whose shape depends on the mode above:
      * ``include_controller=True``  → ``{"compose_path", "controller_token"}``
      * ``include_controller=False`` → ``{"compose_path", "grader_token"}``
      * no ``validation`` block      → ``{"compose_path"}``.
    """
    task_dir = os.path.abspath(task_dir)
    task_name = Path(task_dir).name
    # Sanitise for use as directory name
    safe_name = task_name.replace(" ", "_").replace("/", "_")
    out_dir = os.path.join(BUILD_DIR, safe_name)
    os.makedirs(out_dir, exist_ok=True)

    compose: dict = {"services": {}}
    resource_cfg = _validate_sandbox_resources(sandbox_resources)

    # --- Agent (default) service -------------------------------------------
    # ``command`` MUST be a single-element list so compose passes the script
    # as one arg to the agent image's ``["/bin/bash","-c"]`` entrypoint. Shell
    # form ("sleep infinity") gets split into ["sleep","infinity"] and bash -c
    # then sees $1 as $0 — the container runs ``sleep`` with no args and
    # exits 1 ("missing operand"). Cost a debugging cycle on a fresh Linux pull.
    compose["services"]["default"] = {
        "image": agent_image,
        "command": ["sleep infinity"],
        "working_dir": "/root",
        "networks": ["shared_net"],
    }

    # --- Challenge services or mocked oracle --------------------------------
    meta = metadata if metadata is not None else _read_task_metadata(task_dir)
    compose_source = _resolve_compose_source(task_dir, meta.get("compose_source"))

    target_service_names: list[str] = []  # for controller's TARGET_SERVICES env
    target_resource_role = "target"

    if compose_source and os.path.isfile(compose_source):
        # Task has real challenge services (e.g., CyBench with docker-compose)
        challenge = _load_yaml(compose_source)
        challenge_dir = os.path.dirname(compose_source)
        target_service_names = list(challenge.get("services", {}).keys())

        for svc_name, svc_config in challenge.get("services", {}).items():
            # Resolve relative build contexts to absolute paths
            if "build" in svc_config:
                svc_config["build"] = _resolve_build_context(
                    svc_config["build"], challenge_dir
                )
            # Materialize env files into the temp compose directory. Some
            # vendor composes refer to sibling files like `.env`; because the
            # generated compose lives elsewhere, Docker would otherwise look
            # in the temp dir and fail before the sample starts.
            if "env_file" in svc_config:
                svc_config["env_file"] = _resolve_env_files(
                    svc_config["env_file"], challenge_dir, out_dir
                )
            # Resolve relative bind-mount host paths to absolute paths so the
            # compose file is valid from any working directory (the generated
            # compose.yaml lives in a temp dir, not next to the source).
            if "volumes" in svc_config:
                svc_config["volumes"] = _resolve_volumes(
                    svc_config["volumes"], challenge_dir
                )
            # Ensure challenge services are on the shared network
            svc_config["networks"] = ["shared_net"]
            # Remove port mappings (services communicate via shared network)
            svc_config.pop("ports", None)
            # Remove explicit container_name (Inspect manages names itself)
            svc_config.pop("container_name", None)
            compose["services"][svc_name] = svc_config
    elif meta:
        target_resource_role = "oracle"
        # No real challenge services — add mocked oracle from target_host
        target_host = meta.get("target_host", "target:8080")
        svc_name, _, port_str = target_host.rpartition(":")
        svc_name = svc_name or "target"
        port = int(port_str) if port_str.isdigit() else 8080
        compose["services"][svc_name] = _oracle_service(port, out_dir)
        target_service_names = [svc_name]

    # --- Grader sidecar (validated-exploitation subset) ---------------------
    grader_token: str | None = None
    controller_token: str | None = None
    validation = (meta or {}).get("validation")
    live_validation = (
        validation if validation and _validation_needs_live_grader(validation) else None
    )
    if live_validation:
        compose_path = os.path.join(out_dir, "compose.yaml")
        if include_controller:
            controller_token = _stable_controller_token(task_dir, meta)
            compose["services"][GRADER_SERVICE_NAME] = _grader_service(
                validation=live_validation,
                token=None,
                token_via_file=True,
                out_dir=out_dir,
            )
            compose["services"][CONTROLLER_SERVICE_NAME] = _controller_service(
                controller_token=controller_token,
                target_services=target_service_names,
                compose_host_path=compose_path,
            )
            compose.setdefault("volumes", {})[GRADER_STATE_VOLUME] = {}
        else:
            grader_token = secrets.token_hex(32)
            compose["services"][GRADER_SERVICE_NAME] = _grader_service(
                validation=live_validation,
                token=grader_token,
                token_via_file=False,
                out_dir=out_dir,
            )

    # --- Network -----------------------------------------------------------
    compose["networks"] = {"shared_net": {}}

    # --- Resource limits ---------------------------------------------------
    _apply_sandbox_resources(
        compose,
        resource_cfg,
        target_service_names=target_service_names,
        target_resource_role=target_resource_role,
    )

    # --- Image mirror rewrite (optional) -----------------------------------
    # When CYBERAGENTBENCH_IMAGE_MIRROR is set + image-mirror-manifest.json
    # carries an entry, swap the upstream image: ref for the digest-pinned
    # mirror ref. No-op otherwise. Done last so the rewrite sees every
    # service that's about to land in the file.
    _rewrite_for_mirror(compose)

    # --- Write -------------------------------------------------------------
    compose_path = os.path.join(out_dir, "compose.yaml")
    with open(compose_path, "w") as f:
        yaml.dump(compose, f, default_flow_style=False, sort_keys=False)

    if return_info:
        info: dict = {"compose_path": compose_path}
        if live_validation and include_controller:
            info["controller_token"] = controller_token
        elif live_validation:
            info["grader_token"] = grader_token
        return info
    return compose_path


def _validate_sandbox_resources(resources: dict | None) -> dict[str, dict[str, Any]]:
    """Validate optional per-role Docker Compose resource limits."""
    if resources is None:
        return {}
    if not isinstance(resources, dict):
        raise ValueError("sandbox_resources must be a mapping of role to limits.")

    out: dict[str, dict[str, Any]] = {}
    for role, limits in resources.items():
        if role not in _RESOURCE_ROLES:
            allowed = ", ".join(sorted(_RESOURCE_ROLES))
            raise ValueError(
                f"Unknown sandbox_resources role '{role}'. Expected one of: {allowed}."
            )
        if limits is None:
            continue
        if not isinstance(limits, dict):
            raise ValueError(f"sandbox_resources.{role} must be a mapping.")

        unknown = set(limits) - _RESOURCE_KEYS
        if unknown:
            allowed = ", ".join(sorted(_RESOURCE_KEYS))
            bad = ", ".join(sorted(unknown))
            raise ValueError(
                f"Unknown sandbox_resources.{role} key(s): {bad}. "
                f"Expected one of: {allowed}."
            )

        role_limits: dict[str, Any] = {}
        for key, value in limits.items():
            if key == "memory":
                _validate_memory_limit(role, value)
            elif key == "cpus":
                _validate_positive_number(f"sandbox_resources.{role}.cpus", value)
            elif key == "pids_limit":
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(
                        f"sandbox_resources.{role}.pids_limit must be a positive integer."
                    )
            role_limits[key] = value
        out[role] = role_limits
    return out


def _validate_memory_limit(role: str, value: Any) -> None:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"sandbox_resources.{role}.memory must be non-empty.")
        return
    _validate_positive_number(f"sandbox_resources.{role}.memory", value)


def _validate_positive_number(path: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{path} must be a positive number.")


def _apply_sandbox_resources(
    compose: dict,
    resources: dict[str, dict[str, Any]],
    *,
    target_service_names: list[str],
    target_resource_role: str,
) -> None:
    """Apply configured resource limits to generated compose services."""
    if not resources:
        return

    _apply_service_resource_limits(compose, "default", resources.get("default"))
    for svc_name in target_service_names:
        _apply_service_resource_limits(
            compose, svc_name, resources.get(target_resource_role)
        )
    _apply_service_resource_limits(
        compose, GRADER_SERVICE_NAME, resources.get(GRADER_SERVICE_NAME)
    )
    _apply_service_resource_limits(
        compose, CONTROLLER_SERVICE_NAME, resources.get(CONTROLLER_SERVICE_NAME)
    )


def _apply_service_resource_limits(
    compose: dict,
    service_name: str,
    limits: dict[str, Any] | None,
) -> None:
    if not limits:
        return
    service = (compose.get("services") or {}).get(service_name)
    if not isinstance(service, dict):
        return
    if "memory" in limits:
        service["mem_limit"] = limits["memory"]
        service["memswap_limit"] = limits["memory"]
    if "cpus" in limits:
        service["cpus"] = limits["cpus"]
    if "pids_limit" in limits:
        service["pids_limit"] = limits["pids_limit"]


def _normalize_half(half: dict | None) -> dict | None:
    """Normalize a single validation half (poc or exploit) to the grader's
    expected config shape.

    Returns None if ``half`` is falsy so the grader can skip phases that
    have no probe.
    """
    if not half:
        return None
    return {
        "category": half.get("category"),
        "probe": half.get("probe") or {},
        "baseline_must_fail": half.get("baseline_must_fail", True),
        "validation_mode": half.get("validation_mode", "state_transition"),
    }


def normalize_validation_for_grader(validation: dict) -> dict:
    """Produce the two-half grader config from either flat or two-half input.

    Pass E introduces a per-phase ``validation: {poc: ..., exploit: ...}``
    schema. The 50 existing Vulhub validated tasks predate it and store a
    flat ``{category, probe, baseline_must_fail, validation_mode}`` block.
    The shim treats a flat block as ``{exploit: <flat>, poc: None}`` so
    those tasks keep working without metadata edits.
    """
    has_two_half = "poc" in validation or "exploit" in validation
    if has_two_half:
        return {
            "poc": _normalize_half(validation.get("poc")),
            "exploit": _normalize_half(validation.get("exploit")),
        }
    if "category" in validation:
        flat = _normalize_half(validation)
        return {"poc": None, "exploit": flat}
    return {"poc": None, "exploit": None}


def _is_scorer_handled_half(half: dict | None) -> bool:
    if not isinstance(half, dict):
        return False
    if half.get("category") not in (
        "flag_emission",
        "proof_marker",
        "format_string_evidence",
    ):
        return False
    return (half.get("probe") or {}).get("scope") == "tool_output"


def _validation_needs_live_grader(validation: dict) -> bool:
    """Return whether any validation half needs grader/controller sidecars."""
    for half in normalize_validation_for_grader(validation).values():
        if half and not _is_scorer_handled_half(half):
            return True
    return False


def _stable_controller_token(task_dir: str, meta: dict) -> str:
    """Return a deterministic controller token for this generated compose.

    ``generate_compose`` writes to a deterministic path under /tmp keyed by
    task id. If that file is regenerated after Inspect has built sample
    metadata but before Docker starts the controller, a random token would
    desynchronize the solver's token from the controller's env var. A stable
    per-task token keeps regeneration idempotent.
    """
    task_key = meta.get("task_id") or Path(task_dir).name
    payload = {
        "project_root": str(PROJECT_ROOT),
        "task_dir": os.path.abspath(task_dir),
        "task_id": task_key,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _grader_service(
    *,
    validation: dict,
    token: str | None,
    token_via_file: bool,
    out_dir: str,
) -> dict:
    """Docker Compose service dict for the sidecar grader.

    Writes the task's ``validation`` block (per-phase categories + probe
    configs) to ``<out_dir>/grader_config.json`` and mounts it read-only
    into the grader container.

    Token wiring:
    * ``token_via_file=False`` — sets ``GRADER_TOKEN`` env directly from
      ``token``. Used by the curate/verify-evidence path.
    * ``token_via_file=True`` — sets ``GRADER_TOKEN_FILE`` and mounts the
      shared ``grader_state`` volume; the controller writes the token there
      before the grader boots. Used by the eval-time path.
    """
    config_path = os.path.join(out_dir, "grader_config.json")
    cfg = normalize_validation_for_grader(validation)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    image = validation.get("grader_image") or GRADER_IMAGE
    volumes = [f"{config_path}:/app/config.json:ro"]
    if os.path.exists("/var/run/docker.sock"):
        volumes.append("/var/run/docker.sock:/var/run/docker.sock")

    environment: dict[str, str] = {"GRADER_PORT": str(GRADER_INTERNAL_PORT)}
    service: dict = {
        "image": image,
        # Inspect's docker sandbox tries to pull every image from Docker Hub by
        # default; the grader is built locally and isn't pushed anywhere, so
        # declare it as local-only to suppress the pull attempt.
        "x-local": True,
        "networks": ["shared_net"],
    }

    if token_via_file:
        environment["GRADER_TOKEN_FILE"] = GRADER_TOKEN_FILE_PATH
        volumes.append(f"{GRADER_STATE_VOLUME}:{GRADER_STATE_MOUNT}")
        # The grader cannot boot until the controller has minted T0 and
        # become healthy on /healthz; otherwise the state file would be
        # missing or empty when grader.py reads it at startup.
        service["depends_on"] = {
            CONTROLLER_SERVICE_NAME: {"condition": "service_healthy"},
        }
    else:
        if not token:
            raise ValueError("token required when token_via_file=False")
        environment["GRADER_TOKEN"] = token

    service["environment"] = environment
    service["volumes"] = volumes
    return service


def _controller_service(
    *,
    controller_token: str,
    target_services: list[str],
    compose_host_path: str,
) -> dict:
    """Docker Compose service dict for the controller sidecar.

    The controller owns docker-compose lifecycle for target+grader and
    rotates the grader's HMAC token between phases. It mounts the host
    docker socket plus this very compose.yaml (read-only) so its embedded
    docker-compose CLI can stop/recreate sibling services. The controller
    discovers its own compose project via the ``com.docker.compose.project``
    label off its container at startup.
    """
    return {
        "image": CONTROLLER_IMAGE,
        "x-local": True,
        "environment": {
            "CONTROLLER_TOKEN": controller_token,
            "CONTROLLER_PORT": str(CONTROLLER_INTERNAL_PORT),
            "TARGET_SERVICES": ",".join(target_services),
            "GRADER_SERVICE": GRADER_SERVICE_NAME,
        },
        "volumes": [
            "/var/run/docker.sock:/var/run/docker.sock",
            f"{compose_host_path}:/run/compose.yaml:ro",
            f"{GRADER_STATE_VOLUME}:{GRADER_STATE_MOUNT}",
        ],
        "networks": ["shared_net"],
        "healthcheck": {
            "test": [
                "CMD-SHELL",
                f"curl -fsS http://localhost:{CONTROLLER_INTERNAL_PORT}/healthz "
                "|| exit 1",
            ],
            "interval": "2s",
            "timeout": "1s",
            "retries": 30,
            "start_period": "2s",
        },
    }


def _oracle_service(port: int, out_dir: str) -> dict:
    """Build a Docker Compose service config for the mocked oracle.

    Copies oracle.py into the compose output directory so Docker can
    mount it into the container.
    """
    oracle_dest = os.path.join(out_dir, "oracle.py")
    if not os.path.exists(oracle_dest):
        shutil.copy2(ORACLE_SCRIPT, oracle_dest)
    return {
        "image": "python:3.12-alpine",
        "command": f"python3 /oracle.py {port}",
        "volumes": [f"{oracle_dest}:/oracle.py:ro"],
        "networks": ["shared_net"],
    }


def _read_task_metadata(task_dir: str) -> dict:
    """Read flat metadata.json from benchmark task root."""
    meta_path = os.path.join(task_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_compose_source(task_dir: str, compose_source: str | None) -> str | None:
    """Resolve a metadata ``compose_source`` to an on-disk path.

    Historically metadata stored absolute paths back into ``vendor/``. For
    portability, newer metadata may store repo-relative benchmark paths.
    """
    if not compose_source:
        return None
    if os.path.isabs(compose_source):
        return compose_source

    # Prefer repo-relative benchmark paths, then task-relative paths as a
    # fallback for ad hoc local experiments.
    repo_relative = PROJECT_ROOT / compose_source
    if repo_relative.is_file():
        return str(repo_relative)
    task_relative = Path(task_dir) / compose_source
    if task_relative.is_file():
        return str(task_relative)
    return compose_source


def _resolve_volumes(volumes: list, base_dir: str) -> list:
    """Resolve relative host paths in bind-mount volume entries.

    Handles both short syntax ('host:container[:mode]') and long-form dicts.
    Named volumes (no path separator in host part) are left untouched.
    """
    resolved = []
    for vol in volumes:
        if isinstance(vol, str):
            parts = vol.split(":")
            host = parts[0]
            if host and not os.path.isabs(host) and ("/" in host or host.startswith(".")):
                parts[0] = os.path.abspath(os.path.join(base_dir, host))
            resolved.append(":".join(parts))
        elif isinstance(vol, dict) and vol.get("type") == "bind":
            src = vol.get("source", "")
            if src and not os.path.isabs(src):
                vol = dict(vol)
                vol["source"] = os.path.abspath(os.path.join(base_dir, src))
            resolved.append(vol)
        else:
            resolved.append(vol)
    return resolved


def _resolve_build_context(
    build_config: str | dict, base_dir: str
) -> str | dict:
    """Convert relative build paths to absolute paths."""
    if isinstance(build_config, str):
        return os.path.abspath(os.path.join(base_dir, build_config))
    if isinstance(build_config, dict) and "context" in build_config:
        build_config["context"] = os.path.abspath(
            os.path.join(base_dir, build_config["context"])
        )
    return build_config


def _resolve_env_files(
    env_file: str | list | dict, base_dir: str, out_dir: str
) -> str | list | dict:
    """Materialize compose ``env_file`` references for temp compose use.

    Relative env files are copied into ``out_dir`` and rewritten to the copied
    path. If a relative env file is referenced by the vendor compose but absent
    on disk, synthesize an empty placeholder so Docker Compose config loading
    does not fail before evaluation starts.
    """
    if isinstance(env_file, str):
        return _materialize_env_file(env_file, base_dir, out_dir)
    if isinstance(env_file, list):
        return [
            _resolve_env_files(entry, base_dir, out_dir)
            for entry in env_file
        ]
    if isinstance(env_file, dict):
        path = env_file.get("path")
        if isinstance(path, str):
            updated = dict(env_file)
            updated["path"] = _materialize_env_file(path, base_dir, out_dir)
            return updated
    return env_file


def _materialize_env_file(path: str, base_dir: str, out_dir: str) -> str:
    """Copy a referenced env file into ``out_dir`` or create an empty stub."""
    if os.path.isabs(path):
        return path

    src = os.path.abspath(os.path.join(base_dir, path))
    basename = os.path.basename(path) or ".env"
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:8]
    dest = os.path.join(out_dir, f"envfile-{digest}-{basename}")

    if os.path.exists(src):
        if not os.path.exists(dest):
            shutil.copy2(src, dest)
    else:
        Path(dest).touch()
    return dest


def cleanup_build_dir() -> None:
    """Remove all generated compose files."""
    if os.path.isdir(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
