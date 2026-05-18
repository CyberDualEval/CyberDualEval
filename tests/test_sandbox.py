from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from refusal_eval.sandbox import generate_compose, normalize_validation_for_grader


def _assert_limits(service, *, memory, cpus, pids_limit=None):
    assert service["mem_limit"] == memory
    assert service["memswap_limit"] == memory
    assert service["cpus"] == cpus
    if pids_limit is not None:
        assert service["pids_limit"] == pids_limit


def test_generate_compose_materializes_missing_relative_env_file(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    compose_dir = tmp_path / "vendor"
    compose_dir.mkdir()
    compose_path = compose_dir / "docker-compose.yml"
    compose_path.write_text(
        """
services:
  web:
    image: nginx:alpine
    env_file:
      - .env
""".strip()
    )

    metadata = {
        "task_id": "missing-env-task",
        "target_host": "web:80",
        "compose_source": str(compose_path),
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata))

    generated = generate_compose(str(task_dir), metadata=metadata)
    compose = yaml.safe_load(Path(generated).read_text())

    env_file = compose["services"]["web"]["env_file"]
    assert isinstance(env_file, list)
    assert len(env_file) == 1

    copied_env = Path(env_file[0])
    assert copied_env.is_file()
    assert copied_env.read_text() == ""


def test_generate_compose_resolves_repo_relative_compose_source(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    task_dir = repo_root / "benchmark" / "vulhub" / "demo_task"
    compose_dir = task_dir / "vendor-compose"
    compose_dir.mkdir(parents=True)

    compose_path = compose_dir / "docker-compose.yml"
    compose_path.write_text(
        """
services:
  web:
    image: nginx:alpine
""".strip()
    )

    metadata = {
        "task_id": "demo_task",
        "target_host": "web:80",
        "compose_source": "benchmark/vulhub/demo_task/vendor-compose/docker-compose.yml",
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata))

    monkeypatch.chdir(repo_root)
    generated = generate_compose(str(task_dir), metadata=metadata)
    compose = yaml.safe_load(Path(generated).read_text())
    assert compose["services"]["web"]["image"] == "nginx:alpine"


# ---------- Option α: controller sidecar splicing -------------------------


def _validation_metadata(tmp_path):
    """Helper: build a task dir + metadata with a flat validation block."""
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    compose_dir = tmp_path / "vendor"
    compose_dir.mkdir()
    (compose_dir / "docker-compose.yml").write_text(
        "services:\n  web:\n    image: nginx:alpine\n  db:\n    image: redis:alpine\n"
    )
    metadata = {
        "task_id": "alpha_demo",
        "target_host": "web:80",
        "compose_source": str(compose_dir / "docker-compose.yml"),
        "validation": {
            "category": "outbound_webhook",
            "probe": {"target_host": "web"},
            "baseline_must_fail": True,
        },
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata))
    return task_dir, metadata


def test_generate_compose_eval_path_splices_controller(tmp_path):
    task_dir, metadata = _validation_metadata(tmp_path)

    info = generate_compose(str(task_dir), metadata=metadata, return_info=True)

    assert set(info.keys()) == {"compose_path", "controller_token"}
    assert info["controller_token"] and len(info["controller_token"]) == 64

    compose = yaml.safe_load(Path(info["compose_path"]).read_text())
    services = compose["services"]
    assert "controller" in services
    assert "grader" in services

    grader = services["grader"]
    assert grader["environment"]["GRADER_TOKEN_FILE"] == "/run/grader_state/token.json"
    assert "GRADER_TOKEN" not in grader["environment"]
    assert grader["depends_on"] == {
        "controller": {"condition": "service_healthy"},
    }
    assert any("grader_state:/run/grader_state" in v for v in grader["volumes"])

    controller = services["controller"]
    assert controller["image"] == "cyberbench/controller:latest"
    assert controller["x-local"] is True
    assert controller["environment"]["CONTROLLER_TOKEN"] == info["controller_token"]
    # TARGET_SERVICES enumerates vendor service names (not agent/grader/controller).
    assert sorted(controller["environment"]["TARGET_SERVICES"].split(",")) == ["db", "web"]
    assert controller["environment"]["GRADER_SERVICE"] == "grader"
    # Healthcheck is what the grader's `depends_on: service_healthy` waits on.
    assert "healthcheck" in controller
    assert any("/var/run/docker.sock" in v for v in controller["volumes"])
    assert any("compose.yaml" in v for v in controller["volumes"])

    # The grader_state named volume must be declared at the top level.
    assert compose.get("volumes", {}).get("grader_state") == {}


def test_generate_compose_controller_token_is_stable(tmp_path):
    task_dir, metadata = _validation_metadata(tmp_path)

    first = generate_compose(str(task_dir), metadata=metadata, return_info=True)
    second = generate_compose(str(task_dir), metadata=metadata, return_info=True)

    assert first["controller_token"] == second["controller_token"]
    compose = yaml.safe_load(Path(second["compose_path"]).read_text())
    assert (
        compose["services"]["controller"]["environment"]["CONTROLLER_TOKEN"]
        == first["controller_token"]
    )


def test_generate_compose_curate_path_no_controller(tmp_path):
    task_dir, metadata = _validation_metadata(tmp_path)

    info = generate_compose(
        str(task_dir),
        metadata=metadata,
        return_info=True,
        include_controller=False,
    )

    assert set(info.keys()) == {"compose_path", "grader_token"}
    assert info["grader_token"] and len(info["grader_token"]) == 64

    compose = yaml.safe_load(Path(info["compose_path"]).read_text())
    services = compose["services"]
    assert "controller" not in services
    grader = services["grader"]
    assert grader["environment"]["GRADER_TOKEN"] == info["grader_token"]
    assert "GRADER_TOKEN_FILE" not in grader["environment"]
    assert "depends_on" not in grader
    assert "grader_state" not in compose.get("volumes", {})


def test_generate_compose_no_validation_returns_minimal_shape(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    metadata = {"task_id": "demo", "target_host": "target:8080"}
    (task_dir / "metadata.json").write_text(json.dumps(metadata))

    info = generate_compose(str(task_dir), metadata=metadata, return_info=True)
    assert set(info.keys()) == {"compose_path"}

    compose = yaml.safe_load(Path(info["compose_path"]).read_text())
    assert "controller" not in compose["services"]
    assert "grader" not in compose["services"]
    assert "mem_limit" not in compose["services"]["default"]


def test_generate_compose_applies_resource_limits_by_role(tmp_path):
    task_dir, metadata = _validation_metadata(tmp_path)
    resources = {
        "default": {"memory": "6g", "cpus": 3, "pids_limit": 1024},
        "target": {"memory": "5g", "cpus": 2, "pids_limit": 768},
        "grader": {"memory": "1g", "cpus": 1},
        "controller": {"memory": "512m", "cpus": 0.5},
    }

    info = generate_compose(
        str(task_dir),
        metadata=metadata,
        return_info=True,
        sandbox_resources=resources,
    )

    services = yaml.safe_load(Path(info["compose_path"]).read_text())["services"]
    _assert_limits(services["default"], memory="6g", cpus=3, pids_limit=1024)
    _assert_limits(services["web"], memory="5g", cpus=2, pids_limit=768)
    _assert_limits(services["db"], memory="5g", cpus=2, pids_limit=768)
    _assert_limits(services["grader"], memory="1g", cpus=1)
    _assert_limits(services["controller"], memory="512m", cpus=0.5)


def test_generate_compose_applies_oracle_resource_limits(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    metadata = {"task_id": "demo", "target_host": "target:8080"}

    info = generate_compose(
        str(task_dir),
        metadata=metadata,
        return_info=True,
        sandbox_resources={"oracle": {"memory": "512m", "cpus": 0.5}},
    )

    services = yaml.safe_load(Path(info["compose_path"]).read_text())["services"]
    _assert_limits(services["target"], memory="512m", cpus=0.5)


def test_generate_compose_rejects_invalid_resource_role(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    metadata = {"task_id": "demo", "target_host": "target:8080"}

    with pytest.raises(ValueError, match="Unknown sandbox_resources role"):
        generate_compose(
            str(task_dir),
            metadata=metadata,
            sandbox_resources={"worker": {"memory": "1g"}},
        )


def test_generate_compose_rejects_invalid_resource_key(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    metadata = {"task_id": "demo", "target_host": "target:8080"}

    with pytest.raises(ValueError, match="Unknown sandbox_resources.default"):
        generate_compose(
            str(task_dir),
            metadata=metadata,
            sandbox_resources={"default": {"memory": "1g", "cpu": 1}},
        )


def test_generate_compose_scorer_handled_validation_omits_sidecars(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    metadata = {
        "task_id": "proof-marker-demo",
        "target_host": "target:8080",
        "validation": {
            "poc": {
                "category": "proof_marker",
                "probe": {
                    "marker": "PASS_E_MARKER: demo",
                    "scope": "tool_output",
                },
            },
            "collapse": False,
        },
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata))

    info = generate_compose(str(task_dir), metadata=metadata, return_info=True)

    assert set(info.keys()) == {"compose_path"}
    compose = yaml.safe_load(Path(info["compose_path"]).read_text())
    assert "controller" not in compose["services"]
    assert "grader" not in compose["services"]


def test_generate_compose_format_string_evidence_omits_sidecars(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    metadata = {
        "task_id": "format-string-demo",
        "target_host": "target:8080",
        "validation": {
            "poc": {
                "category": "format_string_evidence",
                "probe": {"scope": "tool_output"},
            },
        },
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata))

    info = generate_compose(str(task_dir), metadata=metadata, return_info=True)

    assert set(info.keys()) == {"compose_path"}
    compose = yaml.safe_load(Path(info["compose_path"]).read_text())
    assert "controller" not in compose["services"]
    assert "grader" not in compose["services"]


def test_generate_compose_mixed_validation_keeps_sidecars(tmp_path):
    task_dir, metadata = _validation_metadata(tmp_path)
    metadata["validation"] = {
        "poc": {
            "category": "proof_marker",
            "probe": {
                "marker": "PASS_E_MARKER: demo",
                "scope": "tool_output",
            },
        },
        "exploit": {
            "category": "outbound_webhook",
            "probe": {"target_host": "web"},
        },
    }

    info = generate_compose(str(task_dir), metadata=metadata, return_info=True)

    assert set(info.keys()) == {"compose_path", "controller_token"}
    compose = yaml.safe_load(Path(info["compose_path"]).read_text())
    assert "controller" in compose["services"]
    assert "grader" in compose["services"]


# ---------- Pass E: two-half validation schema shim -----------------------


def test_normalize_validation_for_grader_flat_treated_as_exploit_only():
    flat = {
        "category": "outbound_webhook",
        "probe": {"target_host": "web"},
        "baseline_must_fail": True,
    }
    out = normalize_validation_for_grader(flat)
    assert out["poc"] is None
    assert out["exploit"]["category"] == "outbound_webhook"
    assert out["exploit"]["probe"] == {"target_host": "web"}
    assert out["exploit"]["baseline_must_fail"] is True
    assert out["exploit"]["validation_mode"] == "state_transition"


def test_normalize_validation_for_grader_two_half_passes_through():
    two_half = {
        "poc": {
            "category": "liveness",
            "probe": {"host": "challenge", "port": 1337},
            "baseline_must_fail": True,
        },
        "exploit": {
            "category": "flag_emission",
            "probe": {"flag": "HTB{x}", "match": "substring", "scope": "tool_output"},
        },
        "collapse": False,
    }
    out = normalize_validation_for_grader(two_half)
    assert out["poc"]["category"] == "liveness"
    assert out["poc"]["probe"] == {"host": "challenge", "port": 1337}
    assert out["exploit"]["category"] == "flag_emission"
    assert out["exploit"]["probe"]["scope"] == "tool_output"


def test_normalize_validation_for_grader_only_poc_half():
    two_half = {
        "poc": {"category": "liveness", "probe": {"host": "x", "port": 1}},
    }
    out = normalize_validation_for_grader(two_half)
    assert out["poc"]["category"] == "liveness"
    assert out["exploit"] is None


def test_grader_config_written_in_two_half_shape(tmp_path):
    """End-to-end: a flat metadata block produces a two-half grader config."""
    task_dir, metadata = _validation_metadata(tmp_path)
    info = generate_compose(str(task_dir), metadata=metadata, return_info=True)

    # The grader_config.json sits next to the generated compose.
    compose_dir = Path(info["compose_path"]).parent
    cfg = json.loads((compose_dir / "grader_config.json").read_text())
    assert set(cfg.keys()) == {"poc", "exploit"}
    assert cfg["poc"] is None
    assert cfg["exploit"]["category"] == "outbound_webhook"
