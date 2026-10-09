"""Keep the Copilot CLI process warm, with a fresh session for every sample."""

from __future__ import annotations

import asyncio
import tempfile
import time
import uuid

from copilot import CopilotClient, RuntimeConnection
from copilot._jsonrpc import JsonRpcError, ProcessExitedError
from copilot.generated.rpc import ModelSetReasoningEffortRequest
from copilot.session import CopilotSession, ReasoningEffort as SdkReasoningEffort
from copilot.session_events import (
    AssistantMessageData,
    AssistantUsageData,
    SessionErrorData,
    SessionEvent,
    SessionIdleData,
)

from .backend import (
    COPILOT_BIN,
    BackendError,
    CallResult,
    CopilotCliBackend,
    nano_aiu_to_credits,
)
from .schema import ReasoningEffort

SDK_ERRORS = (OSError, RuntimeError, JsonRpcError, ProcessExitedError)


class WarmCopilotCliBackend(CopilotCliBackend):
    def __init__(
        self,
        *,
        binary: str = COPILOT_BIN,
        timeout_s: float = 90.0,
        max_concurrency: int = 4,
        extra_args: list[str] | None = None,
    ) -> None:
        super().__init__(
            binary=binary,
            timeout_s=timeout_s,
            max_concurrency=max_concurrency,
            extra_args=extra_args,
        )
        if extra_args is None:
            self.extra_args += [
                "--no-ask-user",
                "--no-eager-powershell-resolution",
            ]
        self._client: CopilotClient | None = None
        self._scratch: tempfile.TemporaryDirectory | None = None
        self._lifecycle_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._client is not None:
                return
            scratch = tempfile.TemporaryDirectory(prefix="decision-wrapper-warm-")
            client: CopilotClient | None = None
            try:
                client = CopilotClient(
                    connection=RuntimeConnection.for_stdio(
                        path=self.binary, args=self._tool_free_args()
                    ),
                    working_directory=scratch.name,
                    log_level="none",
                )
                await asyncio.wait_for(client.start(), timeout=self.timeout_s)
            except BaseException:
                try:
                    if client is not None:
                        await client.stop()
                finally:
                    scratch.cleanup()
                raise
            self._client = client
            self._scratch = scratch

    async def close(self) -> None:
        async with self._lifecycle_lock:
            client, scratch = self._client, self._scratch
            self._client = None
            self._scratch = None
            try:
                if client is not None:
                    await client.stop()
            finally:
                if scratch is not None:
                    scratch.cleanup()

    async def _run(
        self,
        prompt: str,
        *,
        model: str,
        reasoning_effort: ReasoningEffort,
    ) -> CallResult:
        start = time.perf_counter()
        try:
            await self.start()
        except asyncio.TimeoutError as exc:
            raise BackendError(
                f"warm copilot CLI startup timed out after {self.timeout_s}s"
            ) from exc
        except SDK_ERRORS as exc:
            raise BackendError(f"warm copilot CLI startup failed: {exc}") from exc
        client = self._client
        scratch = self._scratch
        if client is None or scratch is None:
            raise BackendError("warm copilot CLI backend is closed")

        session: CopilotSession | None = None
        creation: asyncio.Task[CopilotSession] | None = None
        done = asyncio.Event()
        content: str | None = None
        error: BackendError | None = None
        usage: list[AssistantUsageData] = []
        events: list[dict] = []

        def on_event(event: SessionEvent) -> None:
            nonlocal content, error
            if event.agent_id:
                return
            events.append(event.to_dict())
            data = event.data
            if isinstance(data, AssistantMessageData):
                content = data.content
            elif isinstance(data, AssistantUsageData):
                usage.append(data)
            elif isinstance(data, SessionErrorData):
                error = BackendError(f"warm copilot CLI session failed: {data.message}")
                done.set()
            elif isinstance(data, SessionIdleData):
                done.set()

        async def classify() -> None:
            nonlocal session, creation
            sdk_effort: SdkReasoningEffort | None
            if reasoning_effort == "none" or reasoning_effort == "minimal":
                sdk_effort = None
            else:
                sdk_effort = reasoning_effort
            creation = asyncio.create_task(
                client.create_session(
                    session_id=str(uuid.uuid4()),
                    model=model,
                    reasoning_effort=sdk_effort,
                    working_directory=scratch.name,
                    available_tools=[],
                    skip_custom_instructions=True,
                    memory={"enabled": False},
                    streaming=False,
                    on_event=on_event,
                )
            )
            # Keep the creation reply available for cleanup if this call is cancelled.
            session = await asyncio.shield(creation)
            if sdk_effort is None:
                # The SDK create-session type omits these CLI-supported levels.
                supported = {
                    effort
                    for info in await client.list_models()
                    if info.id == model
                    for effort in info.supported_reasoning_efforts or []
                }
                if reasoning_effort not in supported:
                    raise BackendError(
                        f"reasoning effort {reasoning_effort!r} is not supported "
                        f"for model {model!r}; supported={sorted(supported)}"
                    )
                applied = await session.rpc.model.set_reasoning_effort(
                    ModelSetReasoningEffortRequest(reasoning_effort=reasoning_effort)
                )
                if applied.reasoning_effort != reasoning_effort:
                    raise BackendError(
                        f"warm copilot CLI applied reasoning effort "
                        f"{applied.reasoning_effort!r}, requested {reasoning_effort!r}"
                    )
            await session.send(prompt, agent_mode="interactive")
            await done.wait()
            if error is not None:
                raise error

        try:
            try:
                await asyncio.wait_for(classify(), timeout=self.timeout_s)
            except asyncio.TimeoutError as exc:
                if session is not None:
                    await session.abort()
                raise BackendError(
                    f"warm copilot CLI timed out after {self.timeout_s}s"
                ) from exc
            except asyncio.CancelledError:
                if session is not None:
                    await session.abort()
                raise
            finally:
                if session is None and creation is not None:
                    try:
                        session = await asyncio.wait_for(
                            asyncio.shield(creation), timeout=self.timeout_s
                        )
                    except asyncio.TimeoutError as exc:
                        creation.cancel()
                        try:
                            await self.close()
                        finally:
                            await asyncio.gather(creation, return_exceptions=True)
                        raise BackendError(
                            "warm copilot CLI session creation stalled; runtime stopped"
                        ) from exc
                if session is not None:
                    try:
                        await session.disconnect()
                    finally:
                        await client.delete_session(session.session_id)
        except BackendError:
            raise
        except SDK_ERRORS as exc:
            raise BackendError(f"warm copilot CLI call failed: {exc}") from exc

        if not content:
            raise BackendError("warm copilot CLI produced no assistant.message event")
        if not usage:
            raise BackendError("warm copilot CLI produced no usage accounting")
        credits = [
            nano_aiu_to_credits(item.copilot_usage.total_nano_aiu)
            if item.copilot_usage is not None else None
            for item in usage
        ]
        return CallResult(
            text=content,
            premium_requests=sum(item.cost for item in usage if item.cost is not None),
            prompt_tokens=sum(item.input_tokens or 0 for item in usage),
            api_duration_ms=sum(
                item.duration.total_seconds() * 1000
                for item in usage
                if item.duration is not None
            ),
            wall_time_ms=(time.perf_counter() - start) * 1000,
            raw_events=events,
            ai_credits=(
                sum(value for value in credits if value is not None)
                if all(value is not None for value in credits) else None
            ),
        )
