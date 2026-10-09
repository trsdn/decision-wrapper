import copy
import json
from pathlib import Path

import httpx
import pytest
import respx

from decision_wrapper.benchmark.local import prediction_values, run
from decision_wrapper.openai_backend import (
    ENDPOINT, MODEL, DecisionRefusal, OpenAIDecisionsBackend, OpenAIDecisionsError,
    build_payload, input_billing, normalize_response,
)
from decision_wrapper.schema import DecisionRequest


def request():
    return DecisionRequest(
        model=MODEL, reasoning_effort="none", state="Das Paket ist angekommen.",
        questions={
            "arrived": {"type": "noul", "instructions": "Ist das Paket angekommen?",
                        "criteria": {"true": "angekommen", "false": "nicht angekommen"}},
            "department": {"type": "choice", "instructions": "Which department?",
                           "criteria": {"shipping": "Parcels", "billing": "Invoices"}},
            "severity": {"type": "score", "instructions": "How severe?",
                         "criteria": {"10": "High", "1": "Low", "2": "Middle"}},
        })


def reply():
    return {"model": MODEL, "answers": [
        {"name": "severity", "type": "score", "score": 0.97, "confidence": 0.42,
         "probabilities": [
             {"value": 0, "label": "1", "probability": 0.49},
             {"value": 1, "label": "2", "probability": 0.05},
             {"value": 2, "label": "10", "probability": 0.46},
         ]},
        {"name": "department", "type": "choice", "choice": "shipping", "confidence": 0.77,
         "probabilities": [{"value": "shipping", "probability": 0.9},
                           {"value": "billing", "probability": 0.1}]},
        {"name": "arrived", "type": "predicate", "probability": 0.98},
    ], "usage": {"input_tokens": 200, "output_tokens": 0, "total_tokens": 200}}


def test_native_request_preserves_state_descriptions_and_numeric_rubric_order():
    source = request()
    payload = build_payload(source)
    assert payload["input"] == source.state
    assert payload["model"] == MODEL
    assert "reasoning_effort" not in payload
    assert "expected" not in payload
    assert payload["questions"][0] == {
        "name": "arrived", "type": "predicate",
        "instructions": "Ist das Paket angekommen?\ntrue: angekommen\nfalse: nicht angekommen"}
    assert payload["questions"][1]["choices"] == [
        {"value": "shipping", "description": "Parcels"},
        {"value": "billing", "description": "Invoices"}]
    assert payload["questions"][2]["levels"] == [
        {"label": "1", "description": "Low"}, {"label": "2", "description": "Middle"},
        {"label": "10", "description": "High"}]


def test_native_response_preserves_confidence_and_uses_mode_not_rounded_mean():
    raw = reply()
    original = copy.deepcopy(raw)
    normalized = normalize_response(request(), raw)
    assert normalized["answers"]["department"]["confidence"] == 0.77
    assert normalized["answers"]["department"]["probabilities"] == {"shipping": 0.9, "billing": 0.1}
    assert normalized["answers"]["severity"]["score"] == 0.97
    assert normalized["answers"]["arrived"]["noul"] == 0.98
    assert prediction_values("openai", request(), raw) == {
        "arrived": True, "department": "shipping", "severity": "1"}
    assert raw == original


@pytest.mark.parametrize("mutation", [
    lambda raw: raw.update(model="different-model"),
    lambda raw: raw["answers"].pop(),
    lambda raw: raw["answers"][0].update(name="arrived"),
    lambda raw: raw["answers"][2].update(probability=float("nan")),
    lambda raw: raw["answers"][2].update(probability=True),
    lambda raw: raw["answers"][1].update(type="score"),
    lambda raw: raw["answers"][1].update(choice="not-allowed"),
    lambda raw: raw["answers"][1]["probabilities"][1].update(value="shipping"),
    lambda raw: raw["answers"][0]["probabilities"][0].update(value=True),
    lambda raw: raw["answers"][0]["probabilities"][0].update(label="10"),
    lambda raw: raw["answers"][1].update(confidence=2),
])
def test_native_protocol_rejects_invalid_answers(mutation):
    raw = reply()
    mutation(raw)
    with pytest.raises(OpenAIDecisionsError):
        normalize_response(request(), raw)


def test_refusal_is_explicit_and_billing_is_still_available():
    raw = reply()
    raw["answers"][0] = {"name": "severity", "type": "refusal"}
    with pytest.raises(DecisionRefusal):
        normalize_response(request(), raw)
    assert input_billing(raw) == {"input_tokens": 200, "base_input_cost_usd_estimate": 0.00002}
    assert input_billing({}) == {"input_tokens": None, "base_input_cost_usd_estimate": None}


@pytest.mark.parametrize("tokens", [True, -1, "200", 1.5])
def test_invalid_billing_is_not_silently_reported_as_free(tokens):
    with pytest.raises(OpenAIDecisionsError):
        input_billing({"usage": {"input_tokens": tokens}})


def test_malformed_usage_is_not_treated_as_missing_accounting():
    with pytest.raises(OpenAIDecisionsError, match="invalid usage accounting"):
        input_billing({"usage": "invalid"})


def test_missing_key_is_explicit(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(OpenAIDecisionsError, match="OPENAI_API_KEY is required"):
        OpenAIDecisionsBackend()


@pytest.mark.asyncio
async def test_http_transport_uses_official_endpoint_and_closes():
    with respx.mock() as mock:
        route = mock.post(ENDPOINT).respond(200, json=reply())
        backend = OpenAIDecisionsBackend(api_key="unit-test-key")
        await backend.start()
        client = backend._client
        try:
            assert await backend.predict(request()) == reply()
            sent = route.calls[0].request
            assert sent.headers["Authorization"] == "Bearer unit-test-key"
            assert json.loads(sent.content) == build_payload(request())
        finally:
            await backend.close()
        assert client.is_closed
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 429, 500, 302])
async def test_http_errors_never_expose_provider_messages_or_follow_redirects(status):
    with respx.mock() as mock:
        mock.post(ENDPOINT).respond(status, json={
            "error": {"message": "Invalid credential unit-test-key", "code": "invalid_api_key"}},
            headers={"Location": "https://example.com/collect"})
        backend = OpenAIDecisionsBackend(api_key="unit-test-key")
        await backend.start()
        try:
            with pytest.raises(OpenAIDecisionsError) as caught:
                await backend.predict(request())
            assert caught.value.status_code == status
            assert "unit-test-key" not in str(caught.value)
            assert str(status) in str(caught.value)
        finally:
            await backend.close()


@pytest.mark.asyncio
async def test_timeout_and_invalid_json_are_sanitized():
    backend = OpenAIDecisionsBackend(api_key="unit-test-key")
    await backend.start()
    try:
        with respx.mock() as mock:
            route = mock.post(ENDPOINT)
            route.side_effect = httpx.ReadTimeout("unit-test-key")
            with pytest.raises(OpenAIDecisionsError, match="ReadTimeout") as caught:
                await backend.predict(request())
            assert "unit-test-key" not in str(caught.value)
            route.respond(200, text="unit-test-key malformed JSON")
            with pytest.raises(OpenAIDecisionsError, match="invalid JSON") as caught:
                await backend.predict(request())
            assert "unit-test-key" not in str(caught.value)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_arbitrary_error_codes_cannot_echo_credentials():
    with respx.mock() as mock:
        mock.post(ENDPOINT).respond(401, json={"error": {"code": "unit_test_secret"}})
        backend = OpenAIDecisionsBackend(api_key="unit_test_secret")
        await backend.start()
        try:
            with pytest.raises(OpenAIDecisionsError) as caught:
                await backend.predict(request())
            assert str(caught.value) == "OpenAI Decisions HTTP 401"
        finally:
            await backend.close()


@pytest.mark.asyncio
async def test_successful_but_malformed_provider_echo_is_redacted_before_persistence():
    with respx.mock() as mock:
        body = reply()
        body["unexpected_authorization"] = "Bearer unit-test-key"
        body["answers"][0]["unexpected"] = {
            "echo": ["unit-test-key"], "unit-test-key": "Bearer unit-test-key"}
        mock.post(ENDPOINT).respond(200, json=body)
        backend = OpenAIDecisionsBackend(api_key="unit-test-key")
        await backend.start()
        try:
            raw = await backend.predict(request())
            assert "unit-test-key" not in json.dumps(raw)
            assert "unexpected_authorization" not in raw
            assert raw["usage"]["input_tokens"] == 200
        finally:
            await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", [{"reasoning_effort": "low"}, {"model": "gpt-5.4-mini"}])
async def test_unsupported_settings_fail_before_reading_data_or_billing(tmp_path, settings):
    with pytest.raises(ValueError, match="without a reasoning-effort"):
        await run("openai", tmp_path / "missing.jsonl", tmp_path, 4, 42, **settings)


@pytest.mark.asyncio
async def test_runner_preserves_billed_invalid_reply_and_unknown_cost_on_http_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    dataset = Path(__file__).resolve().parents[1] / "datasets" / "bilingual_benchmark.jsonl"
    responses = iter([
        httpx.Response(200, json={"model": MODEL, "answers": [
            {"name": "warmup", "type": "predicate", "probability": 1}],
            "usage": {"input_tokens": 100}}),
        httpx.Response(200, json={"model": MODEL, "answers": [
            {"name": "warmup", "type": "predicate", "probability": 1}],
            "usage": {"input_tokens": 100}}),
        httpx.Response(200, json={"model": MODEL, "answers": [],
                                  "usage": {"input_tokens": 200}}),
        httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}}),
    ])
    with respx.mock() as mock:
        route = mock.post(ENDPOINT)
        route.side_effect = lambda _: next(responses)
        report = await run("openai", dataset, tmp_path, 4, 42)
        assert route.call_count == 4
    items = [json.loads(line) for line in (tmp_path / "openai-items.jsonl").read_text().splitlines()]
    assert items[0]["raw"]["answers"] == []
    assert items[0]["input_tokens"] == 200
    assert items[0]["base_input_cost_usd_estimate"] == 0.00002
    assert items[1]["base_input_cost_usd_estimate"] is None
    assert report["openai_billing"]["base_input_cost_usd_estimate"] is None
    assert report["total_ai_credits_including_warmup"] is None
    assert report["complete"] is False
    assert report["attempted"] == 2
    assert report["planned"] == 40


@pytest.mark.asyncio
async def test_existing_openai_output_is_rejected_before_key_access(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "openai-items.jsonl").write_text("existing results", encoding="utf-8")
    dataset = Path(__file__).resolve().parents[1] / "datasets" / "bilingual_benchmark.jsonl"
    with pytest.raises(FileExistsError, match="fresh output"):
        await run("openai", dataset, tmp_path, 4, 42)
