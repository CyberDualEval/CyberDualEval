from __future__ import annotations

from types import SimpleNamespace

import pytest
from inspect_ai.model._chat_message import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput
from inspect_ai.solver import TaskState
from inspect_ai.tool._tool_call import ToolCall

from refusal_eval.solver import multistep_solver


def _model_output_with_tool_call(idx: int) -> ModelOutput:
    message = ChatMessageAssistant(
        content="",
        tool_calls=[
            ToolCall(
                id=f"call_{idx}",
                function="bash",
                arguments={"cmd": "true"},
            )
        ],
    )
    return ModelOutput(
        model="test-model",
        choices=[
            ChatCompletionChoice(message=message, stop_reason="tool_calls")
        ],
    )


@pytest.mark.asyncio
async def test_assistant_turn_limit_ignores_tool_result_messages(monkeypatch):
    calls = {"generate": 0}

    class FakeModel:
        async def generate(self, *, input, tools):
            calls["generate"] += 1
            return _model_output_with_tool_call(calls["generate"])

    async def fake_execute_tools(messages, tools):
        return [
            ChatMessageTool(
                content="tool result",
                tool_call_id=f"call_{calls['generate']}",
                function="bash",
            )
        ], []

    async def fake_reset_workspace(phase_idx, source_files, all_files):
        return None

    monkeypatch.setattr("refusal_eval.solver.get_model", lambda: FakeModel())
    monkeypatch.setattr("refusal_eval.solver.execute_tools", fake_execute_tools)
    monkeypatch.setattr("refusal_eval.solver.reset_workspace", fake_reset_workspace)

    solve = multistep_solver(
        message_limit_per_phase=2,
        assistant_turn_limit_per_phase=3,
    )
    state = TaskState(
        model="test-model",
        sample_id="sample",
        epoch=1,
        input="prompt",
        messages=[ChatMessageUser(content="prompt")],
        metadata={
            "phase_prompts": [{"label": "analysis", "prompt": "Analyze."}],
            "source_files": {},
            "all_files": {},
            "refusal_token": True,
        },
        store={},
    )

    result = await solve(state, SimpleNamespace())

    boundaries = result.store.get("phase_boundaries")
    assert calls["generate"] == 3
    assert boundaries[0]["llm_calls"] == 3
    assert sum(isinstance(m, ChatMessageTool) for m in result.messages) == 3
    assert result.message_limit == 12


@pytest.mark.asyncio
async def test_legacy_message_limit_still_counts_tool_result_messages(monkeypatch):
    calls = {"generate": 0}

    class FakeModel:
        async def generate(self, *, input, tools):
            calls["generate"] += 1
            return _model_output_with_tool_call(calls["generate"])

    async def fake_execute_tools(messages, tools):
        return [
            ChatMessageTool(
                content="tool result",
                tool_call_id=f"call_{calls['generate']}",
                function="bash",
            )
        ], []

    async def fake_reset_workspace(phase_idx, source_files, all_files):
        return None

    monkeypatch.setattr("refusal_eval.solver.get_model", lambda: FakeModel())
    monkeypatch.setattr("refusal_eval.solver.execute_tools", fake_execute_tools)
    monkeypatch.setattr("refusal_eval.solver.reset_workspace", fake_reset_workspace)

    solve = multistep_solver(message_limit_per_phase=2)
    state = TaskState(
        model="test-model",
        sample_id="sample",
        epoch=1,
        input="prompt",
        messages=[ChatMessageUser(content="prompt")],
        metadata={
            "phase_prompts": [{"label": "analysis", "prompt": "Analyze."}],
            "source_files": {},
            "all_files": {},
            "refusal_token": True,
        },
        store={},
    )

    result = await solve(state, SimpleNamespace())

    boundaries = result.store.get("phase_boundaries")
    assert calls["generate"] == 1
    assert boundaries[0]["llm_calls"] == 1
    assert sum(isinstance(m, ChatMessageTool) for m in result.messages) == 1
    assert result.message_limit == 4
