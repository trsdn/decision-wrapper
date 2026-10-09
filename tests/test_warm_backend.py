import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

pytest.importorskip("copilot")

from copilot.session_events import (
    AssistantMessageData,
    AssistantUsageData,
    AssistantUsageCopilotUsage,
    SessionErrorData,
    SessionEvent,
    SessionEventType,
    SessionIdleData,
)

from decision_wrapper import warm_backend
from decision_wrapper.backend import BackendError, create_backend
from decision_wrapper.engine import decide
from decision_wrapper.schema import DecisionRequest, NoulQuestion
from decision_wrapper.warm_backend import WarmCopilotCliBackend


def event(kind, data, *, agent_id=None):
    return SessionEvent(
        type=SessionEventType(kind),
        data=data,
        id=uuid.uuid4(),
        timestamp=datetime.now(timezone.utc),
        agent_id=agent_id,
    )


class FakeSession:
    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.session_id = config["session_id"]
        self.disconnected = False
        self.aborted = False
        self.model_changes = []
        self.rpc = SimpleNamespace(
            model=SimpleNamespace(set_reasoning_effort=self.set_reasoning_effort)
        )

    async def set_reasoning_effort(self, params):
        model = self.config["model"]
        reasoning_effort = params.reasoning_effort
        self.model_changes.append((model, reasoning_effort))
        self.config["reasoning_effort"] = reasoning_effort
        return SimpleNamespace(reasoning_effort=reasoning_effort)

    async def send(self, prompt, *, agent_mode):
        assert agent_mode == "interactive"
        self.client.prompts.append(prompt)
        self.client.started.set()
        if self.client.block:
            await asyncio.Event().wait()
        emit = self.config["on_event"]
        if self.client.error:
            emit(event("session.error", SessionErrorData(
                error_type="provider", message="model unavailable"
            )))
            return
        emit(event("assistant.message", AssistantMessageData(
            content="N", message_id="child"
        ), agent_id="child-agent"))
        emit(event("session.idle", SessionIdleData(), agent_id="child-agent"))
        emit(event("assistant.message", AssistantMessageData(
            content=("E" if "E) " in prompt else "A") if "A) " in prompt else "Y",
            message_id="root",
        )))
        if self.client.accounting:
            emit(event("assistant.usage", AssistantUsageData(
                model=self.config["model"],
                cost=self.client.legacy_cost,
                copilot_usage=(
                    AssistantUsageCopilotUsage(total_nano_aiu=self.client.nano_aiu)
                    if self.client.nano_aiu is not None else None
                ),
                input_tokens=123,
                duration=timedelta(milliseconds=17),
            )))
        emit(event("session.idle", SessionIdleData()))

    async def abort(self):
        self.aborted = True

    async def disconnect(self):
        self.disconnected = True


class FakeClient:
    instances = []

    def __init__(self, **kwargs):
        self.options = kwargs
        self.sessions = []
        self.prompts = []
        self.deleted = []
        self.starts = 0
        self.stops = 0
        self.block = False
        self.error = False
        self.accounting = True
        self.legacy_cost = 0.33
        self.nano_aiu = 125000000
        self.started = asyncio.Event()
        self.instances.append(self)

    async def start(self):
        self.starts += 1
        await asyncio.sleep(0)

    async def stop(self):
        self.stops += 1

    async def list_models(self):
        return [
            SimpleNamespace(
                id=model,
                supported_reasoning_efforts=["none", "minimal", "low", "high"],
            )
            for model in ["gpt-5.4-mini", "gpt-6-luna", "mai-code-1.1-flash"]
        ]

    async def create_session(self, **config):
        session = FakeSession(self, config)
        self.sessions.append(session)
        return session

    async def delete_session(self, session_id):
        self.deleted.append(session_id)


@pytest.fixture
def fake_sdk(monkeypatch):
    FakeClient.instances = []
    monkeypatch.setattr(warm_backend, "CopilotClient", FakeClient)
    return FakeClient


@pytest.mark.asyncio
async def test_warm_samples_share_process_but_not_sessions(fake_sdk):
    backend = create_backend(warm=True)
    assert isinstance(backend, WarmCopilotCliBackend)
    async with backend:
        client = fake_sdk.instances[0]
        args = client.options["connection"].args
        assert "--available-tools=" in args
        assert "--no-ask-user" in args
        assert "--no-eager-powershell-resolution" in args
        assert "--disable-builtin-mcps" in args
        scratch = Path(client.options["working_directory"])
        request = DecisionRequest(
            model="gpt-6-luna",
            reasoning_effort="high",
            state="apples",
            questions={
                "a": NoulQuestion(instructions="First?"),
                "b": NoulQuestion(instructions="Second?"),
            },
        )
        response = await decide(request, backend, samples=3)
        assert all(answer.value is True for answer in response.answers.values())
        assert response.usage.samples == 6
        assert response.usage.premium_requests == pytest.approx(1.98)
        assert response.usage.ai_credits == pytest.approx(0.75)
        assert response.usage.prompt_tokens == 6 * 123
        assert response.usage.api_duration_ms == 6 * 17
        assert len(fake_sdk.instances) == 1
        assert client.starts == 1
        assert client.stops == 0
        assert len({session.session_id for session in client.sessions}) == 6
        for session in client.sessions:
            assert session.config["model"] == "gpt-6-luna"
            assert session.config["reasoning_effort"] == "high"
            assert session.config["available_tools"] == []
            assert session.config["skip_custom_instructions"] is True
            assert session.config["memory"] == {"enabled": False}
            assert session.disconnected
            assert session.session_id in client.deleted
        assert scratch.is_dir()
    assert client.stops == 1
    assert not scratch.exists()
    await backend.close()
    assert client.stops == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["none", "minimal", "low", "high"])
async def test_warm_reasoning_levels_reach_session_before_send(fake_sdk, effort):
    async with WarmCopilotCliBackend() as backend:
        result = await backend.complete(
            "pick Y", model="gpt-6-luna", reasoning_effort=effort
        )
        session = fake_sdk.instances[0].sessions[0]
        assert session.config["reasoning_effort"] == effort
        if effort in ("none", "minimal"):
            assert session.model_changes == [("gpt-6-luna", effort)]
        else:
            assert session.model_changes == []
        assert result.text == "Y"


@pytest.mark.asyncio
async def test_unsupported_reasoning_is_not_silently_downgraded(fake_sdk, monkeypatch):
    async def list_models(self):
        return [SimpleNamespace(
            id="mai-code-1.1-flash",
            supported_reasoning_efforts=["low", "medium", "high"],
        )]

    monkeypatch.setattr(fake_sdk, "list_models", list_models)
    async with WarmCopilotCliBackend() as backend:
        with pytest.raises(BackendError, match="'none' is not supported"):
            await backend.complete(
                "pick Y", model="mai-code-1.1-flash", reasoning_effort="none"
            )
        client = fake_sdk.instances[0]
        assert not client.prompts
        assert not client.sessions[0].model_changes
        assert client.sessions[0].disconnected
        assert client.sessions[0].session_id in client.deleted


@pytest.mark.asyncio
async def test_wrong_applied_reasoning_is_reported(fake_sdk, monkeypatch):
    async def wrong_effort(self, params):
        return SimpleNamespace(reasoning_effort="low")

    monkeypatch.setattr(FakeSession, "set_reasoning_effort", wrong_effort)
    async with WarmCopilotCliBackend() as backend:
        with pytest.raises(BackendError, match="applied reasoning effort 'low'"):
            await backend.complete(
                "pick Y", model="gpt-5.4-mini", reasoning_effort="none"
            )
        assert not fake_sdk.instances[0].prompts


@pytest.mark.asyncio
async def test_concurrent_lazy_start_only_starts_one_process(fake_sdk):
    backend = WarmCopilotCliBackend()
    try:
        results = await asyncio.gather(*(
            backend.complete("pick Y", model=model, reasoning_effort="low")
            for model in ["gpt-5.4-mini", "mai-code-1.1-flash", "gpt-6-luna"]
        ))
        assert [result.text for result in results] == ["Y"] * 3
        assert len(fake_sdk.instances) == 1
        assert fake_sdk.instances[0].starts == 1
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_warm_interruption_aborts_and_clears_sample(fake_sdk, cancel):
    backend = WarmCopilotCliBackend(
        timeout_s=0.02 if not cancel else 90, max_concurrency=1
    )
    async with backend:
        client = fake_sdk.instances[0]
        client.block = True
        task = asyncio.create_task(
            backend.complete("pick Y", model="gpt-5.4-mini", reasoning_effort="low")
        )
        await client.started.wait()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(BackendError, match="timed out"):
                await task
        session = client.sessions[0]
        assert session.aborted
        assert session.disconnected
        assert session.session_id in client.deleted

        client.block = False
        result = await backend.complete("next", model="gpt-6-luna", reasoning_effort="high")
        assert result.text == "Y"
        assert client.starts == 1
        assert len(client.sessions) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "accounting"])
async def test_warm_failures_are_explicit_and_sessions_are_cleared(fake_sdk, failure):
    async with WarmCopilotCliBackend() as backend:
        client = fake_sdk.instances[0]
        if failure == "error":
            client.error = True
            message = "model unavailable"
        else:
            client.accounting = False
            message = "usage accounting"
        with pytest.raises(BackendError, match=message):
            await backend.complete("pick Y", model="gpt-5.4-mini", reasoning_effort="low")
        assert client.sessions[0].disconnected
        assert client.sessions[0].session_id in client.deleted


@pytest.mark.asyncio
async def test_startup_failure_cleans_process_and_scratch(fake_sdk, monkeypatch):
    async def fail_start(self):
        raise RuntimeError("startup failed")

    monkeypatch.setattr(fake_sdk, "start", fail_start)
    backend = WarmCopilotCliBackend()
    with pytest.raises(BackendError, match="startup failed"):
        await backend.complete("pick Y", model="gpt-5.4-mini", reasoning_effort="low")
    client = fake_sdk.instances[0]
    assert client.stops == 1
    assert not Path(client.options["working_directory"]).exists()


@pytest.mark.asyncio
async def test_cancellation_during_session_creation_clears_late_session(fake_sdk, monkeypatch):
    creating = asyncio.Event()
    finish_creation = asyncio.Event()
    original = fake_sdk.create_session

    async def slow_create(self, **config):
        creating.set()
        await finish_creation.wait()
        return await original(self, **config)

    monkeypatch.setattr(fake_sdk, "create_session", slow_create)
    async with WarmCopilotCliBackend() as backend:
        task = asyncio.create_task(
            backend.complete("pick Y", model="gpt-5.4-mini", reasoning_effort="low")
        )
        await creating.wait()
        task.cancel()
        await asyncio.sleep(0)
        finish_creation.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        client = fake_sdk.instances[0]
        assert client.sessions[0].disconnected
        assert client.sessions[0].session_id in client.deleted
        assert not client.prompts


@pytest.mark.asyncio
async def test_stalled_session_creation_stops_runtime(fake_sdk, monkeypatch):
    async def stalled_create(self, **config):
        await asyncio.Event().wait()

    monkeypatch.setattr(fake_sdk, "create_session", stalled_create)
    backend = WarmCopilotCliBackend(timeout_s=0.02)
    with pytest.raises(BackendError, match="session creation stalled"):
        await backend.complete("pick Y", model="gpt-5.4-mini", reasoning_effort="low")
    client = fake_sdk.instances[0]
    assert client.stops == 1
    assert not Path(client.options["working_directory"]).exists()


@pytest.mark.parametrize("command", ["decide", "benchmark"])
def test_cli_warm_flag_owns_process_lifecycle(fake_sdk, command):
    from typer.testing import CliRunner
    from decision_wrapper.cli import app

    args = [command]
    if command == "decide":
        args += [str(Path("examples") / "request.json"), "--samples", "1"]
    else:
        args += [
            str(Path("datasets") / "sample_benchmark.jsonl"),
            "--models", "gpt-5.4-mini,gpt-6-luna",
            "--reasoning-efforts", "high", "--samples", "1",
        ]
    result = CliRunner().invoke(app, args + ["--warm"])
    assert result.exit_code == 0, result.output
    assert len(fake_sdk.instances) == 1
    client = fake_sdk.instances[0]
    assert client.starts == 1
    assert client.stops == 1
    assert len(client.sessions) == (2 if command == "decide" else 6)
    assert len(client.deleted) == len(client.sessions)
    assert not Path(client.options["working_directory"]).exists()


def test_api_warm_process_survives_requests_and_stops_at_shutdown(fake_sdk, monkeypatch):
    from fastapi.testclient import TestClient
    from decision_wrapper import app as app_module

    monkeypatch.setattr(app_module, "_backend", create_backend(warm=True))
    monkeypatch.setattr(app_module, "_default_samples", 1)
    payload = {
        "model": "gpt-6-luna", "reasoning_effort": "high", "state": "apples",
        "questions": {"q": {"type": "noul", "instructions": "True?"}},
    }
    with TestClient(app_module.app) as http:
        client = fake_sdk.instances[0]
        assert client.starts == 1
        for _ in range(2):
            response = http.post("/v1/decisions", json=payload)
            assert response.status_code == 200
            assert response.json()["answers"]["q"]["value"] is True
            assert response.json()["usage"]["prompt_tokens"] == 123
            assert response.json()["usage"]["ai_credits"] == pytest.approx(0.125)
        client.error = True
        response = http.post("/v1/decisions", json=payload)
        assert response.status_code == 502
        assert "model unavailable" in response.json()["detail"]
        assert len(fake_sdk.instances) == 1
        assert client.stops == 0
        assert len(client.sessions) == 3
        assert len(client.deleted) == 3
        assert http.get("/healthz").status_code == 200
    assert client.stops == 1


def test_serve_warm_flag_sets_server_mode(monkeypatch):
    import uvicorn
    from typer.testing import CliRunner
    from decision_wrapper.cli import app

    calls = []
    monkeypatch.delenv("DECISION_WRAPPER_WARM", raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = CliRunner().invoke(app, ["serve", "--warm"])
    assert result.exit_code == 0, result.output
    import os
    assert os.environ["DECISION_WRAPPER_WARM"] == "1"
    assert calls[0][0] == ("decision_wrapper.app:app",)


@pytest.mark.asyncio
async def test_credit_accounting_does_not_require_legacy_cost(fake_sdk):
    async with WarmCopilotCliBackend() as backend:
        fake_sdk.instances[0].legacy_cost = None
        result = await backend.complete("pick Y", model="gpt-5.4-mini", reasoning_effort="low")
        assert result.ai_credits == pytest.approx(0.125)


@pytest.mark.asyncio
async def test_missing_sdk_credits_are_unknown_not_legacy_multiplier(fake_sdk):
    async with WarmCopilotCliBackend() as backend:
        fake_sdk.instances[0].nano_aiu = None
        result = await backend.complete("pick Y", model="gpt-5.4-mini", reasoning_effort="low")
        assert result.premium_requests == pytest.approx(0.33)
        assert result.ai_credits is None
