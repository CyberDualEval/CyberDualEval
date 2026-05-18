from types import SimpleNamespace

import pytest
from inspect_ai.model._chat_message import ChatMessageAssistant, ChatMessageUser
from inspect_ai.model._model_output import ModelOutput

from refusal_eval.claude_code_solver import (
    _CLAUDE_CODE_DOWNLOAD_BASE_URL,
    _adapt_claude_code_messages,
    _claude_code_phase_prompt,
    _limit_filter_for_phase,
    _patch_claude_code_download_base,
)
from refusal_eval.prompts import POC_PROMPT


@pytest.mark.asyncio
async def test_claude_code_download_base_patch_uses_current_download_host() -> None:
    _patch_claude_code_download_base()

    from inspect_swe._claude_code import agentbinary

    assert await agentbinary._claude_code_gcs_bucket() == _CLAUDE_CODE_DOWNLOAD_BASE_URL


def test_claude_code_phase_prompt_removes_submit_tool_instruction() -> None:
    adapted = _claude_code_phase_prompt(POC_PROMPT)

    assert "call `submit`" not in adapted
    assert "shell command named `submit`" in adapted
    assert "end the Claude Code session with a concise final answer" in adapted


def test_claude_code_message_adapter_only_rewrites_user_text() -> None:
    messages = [
        ChatMessageUser(content=POC_PROMPT),
        ChatMessageAssistant(content="existing assistant text"),
    ]

    adapted = _adapt_claude_code_messages(messages)

    assert "call `submit`" not in adapted[0].content
    assert adapted[1].content == "existing assistant text"


@pytest.mark.asyncio
async def test_claude_code_turn_limit_filter_caps_paid_model_calls() -> None:
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
