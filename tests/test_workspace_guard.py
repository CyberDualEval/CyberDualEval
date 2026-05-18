from __future__ import annotations

from types import SimpleNamespace

import pytest

from refusal_eval.solver import _assert_no_exposed_task_mount


class FakeSandbox:
    def __init__(self, result):
        self.result = result
        self.commands = []

    async def exec(self, cmd):
        self.commands.append(cmd)
        return self.result


@pytest.mark.asyncio
async def test_workspace_guard_passes_when_root_task_absent():
    sb = FakeSandbox(SimpleNamespace(returncode=0, stdout="", stderr=""))

    await _assert_no_exposed_task_mount(sb)

    assert sb.commands


@pytest.mark.asyncio
async def test_workspace_guard_fails_when_root_task_visible():
    sb = FakeSandbox(
        SimpleNamespace(
            returncode=42,
            stdout="/root/task/metadata.json\n/root/task/.curation/draft.json\n",
            stderr="",
        )
    )

    with pytest.raises(RuntimeError) as exc:
        await _assert_no_exposed_task_mount(sb)

    msg = str(exc.value)
    assert "/root/task is visible" in msg
    assert "/root/task/metadata.json" in msg
