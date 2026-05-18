from types import SimpleNamespace

import pytest
from inspect_ai.model._model_output import ModelOutput

from refusal_eval.codex_cli_solver import (
    _CODEX_BRIDGE_BASE_PORT,
    _codex_bridge_overrides,
    _codex_phase_prompt,
    _limit_filter_for_phase,
    _reserve_codex_bridge_port,
)
from refusal_eval.prompts import POC_PROMPT


def test_codex_cli_uses_local_bridge_provider() -> None:
    overrides = _codex_bridge_overrides(4321)

    assert overrides["model_provider"] == '"openai-proxy"'
    assert (
        overrides["model_providers.openai-proxy.base_url"]
        == '"http://localhost:4321/v1"'
    )
    assert overrides["model_providers.openai-proxy.env_key"] == '"OPENAI_API_KEY"'
    assert "https://api.openai.com/v1" not in set(overrides.values())
    assert all("direct" not in key for key in overrides)


def test_codex_cli_bridge_port_allocator_advances() -> None:
    port_a = _reserve_codex_bridge_port()
    port_b = _reserve_codex_bridge_port()
    port_c = _reserve_codex_bridge_port()

    assert port_a >= _CODEX_BRIDGE_BASE_PORT
    assert port_b == port_a + 1
    assert port_c == port_b + 1


def test_codex_phase_prompt_removes_submit_tool_instruction() -> None:
    adapted = _codex_phase_prompt(POC_PROMPT)

    assert "call `submit`" not in adapted
    assert "shell command named `submit`" in adapted
    assert "end the Codex session with a concise final answer" in adapted


@pytest.mark.asyncio
async def test_codex_turn_limit_filter_caps_paid_model_calls() -> None:
    state = {"actual_calls": 0, "hit": False}
    filter_fn = _limit_filter_for_phase(limit=2, state=state)
    model = SimpleNamespace(api=SimpleNamespace(model_name="test-model"))

    assert filter_fn is not None
    assert await filter_fn(model, [], [], None, None) is None
    assert await filter_fn(model, [], [], None, None) is None

    output = await filter_fn(model, [], [], None, None)

    assert isinstance(output, ModelOutput)
    assert output.model == "test-model"
    assert output.stop_reason == "stop"
    assert "turn limit reached" in output.completion.lower()
    assert state == {"actual_calls": 2, "hit": True}
