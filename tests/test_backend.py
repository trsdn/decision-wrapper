import asyncio
import json
from pathlib import Path

import pytest

from decision_wrapper.backend import BackendError, CopilotCliBackend, nano_aiu_to_credits


class FakeProcess:
    def __init__(self, *, block=False):
        self.returncode = None
        self.block = block
        self.started = asyncio.Event()
        self.killed = False
        self.waited = False

    async def communicate(self):
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        self.returncode = 0
        events = [
            {"type": "assistant.message", "data": {"content": "A"}},
            {"type": "result", "usage": {"premiumRequests": 0.33, "totalNanoAiu": 125000000}},
        ]
        return "\n".join(json.dumps(event) for event in events).encode(), b""

    def kill(self):
        self.killed = True
        self.returncode = -1

    async def wait(self):
        self.waited = True
        return self.returncode


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["none", "high"])
async def test_default_calls_keep_model_and_isolate_sessions(monkeypatch, effort):
    calls = []

    async def spawn(*args, **kwargs):
        calls.append((args, kwargs))
        assert Path(kwargs["cwd"]).is_dir()
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    backend = CopilotCliBackend(binary="copilot-test")
    for _ in range(2):
        result = await backend.complete("pick A", model="gpt-6-luna", reasoning_effort=effort)
        assert result.text == "A"
        assert result.premium_requests == pytest.approx(0.33)
        assert result.ai_credits == pytest.approx(0.125)

    for args, kwargs in calls:
        assert args[0] == "copilot-test"
        assert "--disable-builtin-mcps" in args
        assert "--no-custom-instructions" in args
        assert args[-1] == "--available-tools="
        assert sum(arg.startswith("--available-tools") for arg in args) == 1
        assert args[args.index("--model") + 1] == "gpt-6-luna"
        assert args[args.index("--reasoning-effort") + 1] == effort
        assert not Path(kwargs["cwd"]).exists()
    assert calls[0][1]["cwd"] != calls[1][1]["cwd"]
    assert calls[0][0][calls[0][0].index("--session-id") + 1] != calls[1][0][
        calls[1][0].index("--session-id") + 1
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_args", [[], ["--no-custom-instructions"]])
async def test_explicit_extra_args_override_defaults(monkeypatch, extra_args):
    calls = []

    async def spawn(*args, **kwargs):
        calls.append(args)
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    backend = CopilotCliBackend(extra_args=extra_args)
    await backend.complete("pick A", model="gpt-5.4-mini", reasoning_effort="low")
    args = calls[0]
    assert list(args[args.index("--session-id") + 2:]) == extra_args + ["--available-tools="]


@pytest.mark.parametrize("extra_args", [
    ["--available-tools=shell"], ["--available-tools", "shell"], ["--"],
])
def test_extra_args_cannot_enable_tools_or_stop_option_parsing(extra_args):
    with pytest.raises(BackendError, match="must not enable tools"):
        CopilotCliBackend(extra_args=extra_args)


@pytest.mark.asyncio
async def test_mutated_extra_args_are_rejected_before_spawning(monkeypatch):
    async def forbidden_spawn(*args, **kwargs):
        pytest.fail("tool-enabling arguments must not reach the CLI")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_spawn)
    backend = CopilotCliBackend(extra_args=[])
    backend.extra_args.append("--available-tools=shell")
    with pytest.raises(BackendError, match="must not enable tools"):
        await backend.complete("pick A", model="gpt-6-luna", reasoning_effort="none")


@pytest.mark.asyncio
async def test_explicit_empty_tool_lists_are_deduplicated(monkeypatch):
    async def spawn(*args, **kwargs):
        assert args.count("--available-tools=") == 1
        assert args[-1] == "--available-tools="
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    await CopilotCliBackend(extra_args=["--available-tools="]).complete(
        "pick A", model="gpt-6-luna", reasoning_effort="none")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["", " ", "--available-tools=shell", "--"])
async def test_model_cannot_inject_cli_options(monkeypatch, model):
    async def forbidden_spawn(*args, **kwargs):
        pytest.fail("a model ID must not inject CLI options")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_spawn)
    with pytest.raises(BackendError, match="model identifier"):
        await CopilotCliBackend().complete("pick A", model=model, reasoning_effort="none")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_interrupted_calls_kill_and_reap_process(monkeypatch, cancel):
    process = FakeProcess(block=True)
    scratch_dirs = []

    async def spawn(*args, **kwargs):
        scratch_dirs.append(Path(kwargs["cwd"]))
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    backend = CopilotCliBackend(timeout_s=0.01 if not cancel else 90)
    task = asyncio.create_task(
        backend.complete("pick A", model="gpt-5.4-mini", reasoning_effort="low")
    )
    await process.started.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(BackendError, match="timed out"):
            await task
    assert process.killed
    assert process.waited
    assert not scratch_dirs[0].exists()

    process.block = False
    result = await backend.complete("pick A", model="gpt-5.4-mini", reasoning_effort="low")
    assert result.text == "A"


@pytest.mark.parametrize("nano, credits", [(0, 0), (125000000, 0.125), (1000000000, 1)])
def test_nano_aiu_conversion(nano, credits):
    assert nano_aiu_to_credits(nano) == pytest.approx(credits)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1000000000"])
def test_invalid_credit_accounting_is_rejected(value):
    with pytest.raises(BackendError, match="invalid AI-credit accounting"):
        nano_aiu_to_credits(value)


@pytest.mark.asyncio
async def test_usage_file_credits_override_legacy_result_event(monkeypatch):
    async def spawn(*args, **kwargs):
        path = Path(args[args.index("--usage-output-file") + 1])
        path.write_text(json.dumps({"totalNanoAiu": 250000000}), encoding="utf-8")
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    result = await CopilotCliBackend().complete(
        "pick A", model="gpt-5.4-mini", reasoning_effort="none"
    )
    assert result.ai_credits == pytest.approx(0.25)
    assert result.premium_requests == pytest.approx(0.33)


def test_legacy_counter_is_not_converted_to_credits():
    raw = "\n".join(json.dumps(event) for event in [
        {"type": "assistant.message", "data": {"content": "A"}},
        {"type": "result", "usage": {"premiumRequests": 0.33}},
    ])
    result = CopilotCliBackend._parse_jsonl(raw, 1)
    assert result.premium_requests == pytest.approx(0.33)
    assert result.ai_credits is None
