"""Backend that answers single-letter classification prompts by shelling out
to the GitHub Copilot CLI (`copilot -p ...`), instead of calling a raw
completions API.

Rationale: models like ``gpt-6-luna`` and ``gpt-5.4-mini`` (as named in this
environment) are only reachable through the Copilot CLI's model routing, not
through a public OpenAI-compatible endpoint. The CLI has no logprobs/JSON
schema mode, so calibration here is done by Monte-Carlo voting: each question
is sent as ``samples`` independent, concurrent CLI calls, and the answer
distribution is the empirical vote share.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .schema import ReasoningEffort

COPILOT_BIN = shutil.which("copilot") or "copilot"


class BackendError(RuntimeError):
    """Raised when the Copilot CLI fails or returns no usable answer."""


@dataclass
class CallResult:
    """Result of a single `copilot -p ...` invocation."""

    text: Optional[str]
    premium_requests: float = 0.0
    prompt_tokens: int = 0
    api_duration_ms: float = 0.0
    wall_time_ms: float = 0.0
    raw_events: list = field(default_factory=list)


class CopilotCliBackend:
    """Wraps non-interactive `copilot -p` invocations as a decision backend."""

    def __init__(
        self,
        *,
        binary: str = COPILOT_BIN,
        timeout_s: float = 90.0,
        max_concurrency: int = 4,
        extra_args: Optional[list[str]] = None,
    ) -> None:
        self.binary = binary
        self.timeout_s = timeout_s
        self._semaphore = asyncio.Semaphore(max_concurrency)
        # Flags that keep calls fast, deterministic, and side-effect free:
        # no repo instructions, no built-in MCP servers, no interactive
        # approval prompts (required for non-interactive mode).
        self.extra_args = extra_args or [
            "--allow-all-tools",
            "--disable-builtin-mcps",
            "--no-custom-instructions",
        ]

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        reasoning_effort: ReasoningEffort,
    ) -> CallResult:
        async with self._semaphore:
            return await self._run(prompt, model=model, reasoning_effort=reasoning_effort)

    async def _run(
        self,
        prompt: str,
        *,
        model: str,
        reasoning_effort: ReasoningEffort,
    ) -> CallResult:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="decision-wrapper-") as scratch:
            args = [
                self.binary,
                "-p",
                prompt,
                "--model",
                model,
                "--reasoning-effort",
                reasoning_effort,
                "--silent",
                "--output-format",
                "json",
                "--log-level",
                "none",
                "-C",
                scratch,
                "--session-id",
                str(uuid.uuid4()),
                *self.extra_args,
            ]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=scratch,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_s
                )
            except asyncio.TimeoutError as exc:
                raise BackendError(
                    f"copilot CLI timed out after {self.timeout_s}s"
                ) from exc

            wall_time_ms = (time.perf_counter() - start) * 1000

            if proc.returncode != 0 and not stdout:
                raise BackendError(
                    f"copilot CLI exited {proc.returncode}: {stderr.decode(errors='replace')[:2000]}"
                )

            return self._parse_jsonl(stdout.decode(errors="replace"), wall_time_ms)

    @staticmethod
    def _parse_jsonl(raw: str, wall_time_ms: float) -> CallResult:
        content: Optional[str] = None
        premium_requests = 0.0
        prompt_tokens = 0
        api_duration_ms = 0.0
        events = []

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
            etype = event.get("type")
            if etype == "assistant.message":
                data = event.get("data", {})
                if data.get("content"):
                    content = data["content"]
            elif etype == "result":
                usage = event.get("data", event).get("usage", {}) if isinstance(
                    event.get("data"), dict
                ) else event.get("usage", {})
                premium_requests = float(usage.get("premiumRequests", 0.0) or 0.0)
                api_duration_ms = float(usage.get("totalApiDurationMs", 0.0) or 0.0)

        if content is None:
            raise BackendError(
                "copilot CLI produced no assistant.message event; "
                f"got {len(events)} events"
            )

        return CallResult(
            text=content,
            premium_requests=premium_requests,
            prompt_tokens=prompt_tokens,
            api_duration_ms=api_duration_ms,
            wall_time_ms=wall_time_ms,
            raw_events=events,
        )


_ISOLATED_LETTER_RE = re.compile(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])")
_ANY_UPPER_RE = re.compile(r"[A-Z]")


def extract_letter(text: str, valid_letters: set[str]) -> Optional[str]:
    """Pull the first candidate letter out of a (hopefully one-character)
    CLI response, tolerating stray punctuation/whitespace/markdown.

    Prefers letters that stand alone (not inside a word, e.g. the "B" in
    "The answer is B." but not the "A" in "answer"), falling back to a bare
    uppercase-letter scan if no isolated token matches.
    """

    for match in _ISOLATED_LETTER_RE.finditer(text):
        upper = match.group(0).upper()
        if upper in valid_letters:
            return upper
    for match in _ANY_UPPER_RE.finditer(text):
        if match.group(0) in valid_letters:
            return match.group(0)
    return None
