"""Native OpenAI Decisions API, separate from Copilot's voting backend."""

from __future__ import annotations

import math
import os

import httpx

from .backend import BackendError
from .engine import _build_options
from .schema import DecisionRequest

ENDPOINT = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"
BASE_INPUT_USD_PER_MILLION = 0.10
PRICING_SOURCE = "https://developers.openai.com/api/docs/guides/decisions"
SAFE_ERROR_CODES = {
    "invalid_api_key", "insufficient_quota", "rate_limit_exceeded", "model_not_found",
    "permission_denied", "invalid_request_error", "context_length_exceeded",
    "billing_hard_limit_reached", "account_deactivated", "server_error",
}


class OpenAIDecisionsError(BackendError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class DecisionRefusal(OpenAIDecisionsError):
    """A delivered model refusal, not an infrastructure outage."""


def build_payload(request: DecisionRequest) -> dict:
    if request.model != MODEL:
        raise OpenAIDecisionsError("OpenAI Decisions currently supports only gpt-6-luna")
    if request.reasoning_effort != "none":
        raise OpenAIDecisionsError("Decisions has no reasoning-effort parameter; use the adapter's none placeholder")
    questions = []
    for name, question in request.questions.items():
        if not name.strip():
            raise OpenAIDecisionsError("OpenAI decision question names must not be blank")
        item = {"name": name, "instructions": question.instructions, "type": question.type}
        options = _build_options(question)
        if question.type == "noul":
            item["type"] = "predicate"
            if question.criteria:
                item["instructions"] += "\n" + "\n".join(
                    f"{label}: {description}" for _, label, description in options)
        elif question.type == "choice":
            item["choices"] = [{"value": label, "description": description}
                               for _, label, description in options]
        else:
            item["levels"] = [{"label": label, "description": description}
                              for _, label, description in options]
        questions.append(item)
    return {"model": request.model, "input": request.state, "questions": questions}


def _number(value: object, low: float, high: float) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise OpenAIDecisionsError("OpenAI returned an invalid numeric decision field")
    return float(value)


def normalize_response(request: DecisionRequest, raw: dict) -> dict:
    """Preserve native posteriors/confidence; map wire arrays to Jev-style keys."""
    if raw.get("model") != request.model:
        raise OpenAIDecisionsError("OpenAI returned a different model identity")
    answers = raw.get("answers")
    if not isinstance(answers, list) or len(answers) != len(request.questions):
        raise OpenAIDecisionsError("OpenAI answer count does not match the request")
    normalized = {}
    for answer in answers:
        if not isinstance(answer, dict):
            raise OpenAIDecisionsError("OpenAI returned an invalid answer object")
        name = answer.get("name")
        if not isinstance(name, str) or name not in request.questions or name in normalized:
            raise OpenAIDecisionsError("OpenAI returned duplicate, missing or unknown question names")
        question = request.questions[name]
        if answer.get("type") == "refusal":
            raise DecisionRefusal("OpenAI refused a decision question")
        kind = "predicate" if question.type == "noul" else question.type
        if answer.get("type") != kind:
            raise OpenAIDecisionsError("OpenAI answer type does not match the question")
        if question.type == "noul":
            normalized[name] = {"type": "noul", "noul": _number(answer.get("probability"), 0, 1)}
            continue
        labels = [label for _, label, _ in _build_options(question)]
        entries = answer.get("probabilities")
        if not isinstance(entries, list) or len(entries) != len(labels):
            raise OpenAIDecisionsError("OpenAI returned an incomplete probability distribution")
        probabilities = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise OpenAIDecisionsError("OpenAI returned an invalid probability entry")
            value = entry.get("value")
            if question.type == "score":
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(labels):
                    raise OpenAIDecisionsError("OpenAI returned an invalid score level index")
                if entry.get("label") != labels[value]:
                    raise OpenAIDecisionsError("OpenAI score labels do not match the ordered rubric")
                value = str(value)
            if not isinstance(value, str) or value in probabilities:
                raise OpenAIDecisionsError("OpenAI returned an invalid or duplicate probability label")
            probabilities[value] = _number(entry.get("probability"), 0, 1)
        expected = set(labels) if question.type == "choice" else {str(i) for i in range(len(labels))}
        if set(probabilities) != expected:
            raise OpenAIDecisionsError("OpenAI probability labels do not match the request")
        result = {"type": question.type, "probabilities": probabilities,
                  "confidence": _number(answer.get("confidence"), 0, 1)}
        if question.type == "choice":
            if answer.get("choice") not in labels:
                raise OpenAIDecisionsError("OpenAI selected a value outside the allowed choices")
            result["choice"] = answer["choice"]
        else:
            result["score"] = _number(answer.get("score"), 0, len(labels) - 1)
            result["legend"] = {str(i): label for i, label in enumerate(labels)}
        normalized[name] = result
    return {"model": raw["model"], "answers": normalized, "usage": raw.get("usage")}


def input_billing(raw: dict) -> dict:
    """Base-rate estimate from reported tokens, never an invoice or AI credits."""
    usage = raw.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise OpenAIDecisionsError("OpenAI returned invalid usage accounting")
    tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    if tokens is not None and (isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0):
        raise OpenAIDecisionsError("OpenAI returned invalid input-token accounting")
    return {
        "input_tokens": tokens,
        "base_input_cost_usd_estimate": (
            tokens * BASE_INPUT_USD_PER_MILLION / 1_000_000 if tokens is not None else None),
    }


def _redact_value(value: object, secret: str) -> object:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, list):
        return [_redact_value(item, secret) for item in value]
    if isinstance(value, dict):
        return {key.replace(secret, "[REDACTED]"): _redact_value(item, secret)
                for key, item in value.items()}
    return value


class OpenAIDecisionsBackend:
    def __init__(self, *, api_key: str | None = None, timeout_s: float = 90):
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not key or not key.strip():
            raise OpenAIDecisionsError("OPENAI_API_KEY is required for the native Decisions API")
        self._key = key.strip()
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=self._timeout_s, follow_redirects=False)

    async def close(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            await client.aclose()

    async def predict(self, request: DecisionRequest) -> dict:
        payload = build_payload(request)
        if self._client is None:
            raise OpenAIDecisionsError("OpenAI Decisions backend has not been started")
        try:
            response = await self._client.post(ENDPOINT, json=payload)
        except httpx.RequestError as exc:
            raise OpenAIDecisionsError(
                f"OpenAI Decisions connection failed ({type(exc).__name__})") from None
        if not response.is_success:
            code = ""
            try:
                error = response.json().get("error", {})
                candidate = error.get("code") if isinstance(error, dict) else None
                if isinstance(candidate, str) and candidate in SAFE_ERROR_CODES:
                    code = f" ({candidate})"
            except (ValueError, AttributeError):
                pass
            # Never expose provider messages or HTTP objects: they can echo credentials.
            raise OpenAIDecisionsError(
                f"OpenAI Decisions HTTP {response.status_code}{code}",
                status_code=response.status_code)
        try:
            data = response.json()
        except ValueError:
            raise OpenAIDecisionsError("OpenAI Decisions returned invalid JSON") from None
        if not isinstance(data, dict):
            raise OpenAIDecisionsError("OpenAI Decisions returned an invalid response object")
        # Persist only decision/usage fields, redacting any unexpected credential echo.
        return {field: _redact_value(data.get(field), self._key)
                for field in ("model", "answers", "usage")}
