import hashlib
import json
from types import SimpleNamespace

import pytest

jevbench = pytest.importorskip("jevbench")
from jevbench.tasks import Task, dataset_hash

from decision_wrapper.benchmark.jevbench import JevBenchSuite


def suite_for(task):
    suite = JevBenchSuite.__new__(JevBenchSuite)
    suite.tasks = [task]
    suite.by_id = {task.id: task}
    suite.tiers = {task.id: "original"}
    return suite


def task(kind="score"):
    question = {"type": kind, "instructions": "Read the state."}
    if kind == "score":
        question["criteria"] = ["low", "middle", "high"]
        labels, expected = ["0", "1", "2"], 0
    elif kind == "choice":
        question["criteria"] = {"A": "first", "B": "second"}
        labels, expected = ["A", "B"], "A"
    else:
        labels, expected = ["no", "yes"], "yes"
    return Task("test", "ordinal", {"text": "sample"}, question, labels, expected, "public")


def test_adapter_keeps_reference_answer_out_of_model_input():
    original = task()
    row = suite_for(original).rows()[0]
    assert row["questions"] == {"decision": {
        "type": "score", "instructions": "Read the state.",
        "criteria": {"0": "low", "1": "middle", "2": "high"},
    }}
    assert row["state"] == '{"text": "sample"}'
    assert original.question["criteria"] == ["low", "middle", "high"]


def test_native_ordinal_uses_original_scorer_argmax_and_separate_expected_value():
    raw = {"answers": {"decision": {
        "type": "score", "score": 0.97,
        "probabilities": {"0": 0.49, "1": 0.05, "2": 0.46},
    }}}
    record = suite_for(task()).score("test", "strands", raw, None)
    assert record["correct"] is True
    assert record["predicted"] == "0"
    assert record["ordinal_ev"] == pytest.approx(0.97)


@pytest.mark.parametrize("engine,answer", [
    ("laya", {"type": "noul", "noul": 0.8}),
    ("openai", {"type": "noul", "noul": 0.8}),
    ("copilot", {"type": "noul", "probability": 0.8}),
])
def test_boolean_semantics_map_to_official_yes_no_labels(engine, answer):
    record = suite_for(task("noul")).score(
        "test", engine, {"answers": {"decision": answer}}, None)
    assert record["probs"] == {"yes": 0.8, "no": pytest.approx(0.2)}
    assert record["correct"] is True


def test_gliner_is_label_only_not_an_invented_probability_distribution():
    suite = suite_for(task("choice"))
    record = suite.score("test", "gliner", {"decision": {"label": "A", "confidence": 0.7}}, None)
    assert record["correct"] is True
    assert record["probs"] is None
    summary = suite.summary([{"jevbench": record, "latency_ms": 100}])
    assert summary["accuracy"] == 1
    assert summary["calibration_n"] == 0
    assert summary["brier_mean"] is None
    assert summary["ordinal_mae"] is None


def test_failed_tasks_stay_in_official_accuracy_denominator():
    suite = suite_for(task("choice"))
    record = suite.score("test", "laya", None, "context window exceeded")
    summary = suite.summary([{"jevbench": record, "latency_ms": 10}])
    assert summary["complete"] is True
    assert summary["n_attempted"] == summary["n_scorable"] == 1
    assert summary["accuracy"] == 0
    assert summary["operational_success"] == 0


def test_original_scorer_handles_rounded_distributions_without_adapter_repair():
    raw = {"answers": {"decision": {
        "type": "choice", "choice": "A", "probabilities": {"A": 0.604, "B": 0.4},
    }}}
    record = suite_for(task("choice")).score("test", "laya", raw, None)
    assert record["valid"] and record["renormalized"]
    assert record["probs_as_returned"] == {"A": 0.604, "B": 0.4}
    raw["answers"]["decision"]["probabilities"] = {"A": 0.6, "B": 0.6}
    record = suite_for(task("choice")).score("test", "laya", raw, None)
    assert not record["valid"] and not record["correct"]


def test_corpus_accepts_only_line_ending_changes_and_rejects_content_edits(tmp_path, monkeypatch):
    original = task()
    payload = (original.to_json() + "\n").encode("utf-8")
    public = tmp_path / "datasets" / "public"
    public.mkdir(parents=True)
    path = public / "original.jsonl"
    path.write_bytes(payload.replace(b"\n", b"\r\n"))
    manifest = {"protocol": "test", "splits": [{
        "name": "original", "n": 1, "sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_sha256": dataset_hash([original]),
    }]}
    (tmp_path / "datasets" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(jevbench, "__file__", str(tmp_path / "jevbench" / "__init__.py"))
    monkeypatch.setattr("decision_wrapper.benchmark.jevbench.subprocess.check_output",
                        lambda *args, **kwargs: "pinned-commit\n")
    assert len(JevBenchSuite(tmp_path).tasks) == 1
    path.write_bytes(payload.replace(b"sample", b"changed"))
    with pytest.raises(ValueError, match="official manifest"):
        JevBenchSuite(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("model,effort,expected_model", [
    (None, "none", "gpt-5.4-mini"),
    ("gpt-6-luna", "none", "gpt-6-luna"),
    ("gpt-6-luna", "low", "gpt-6-luna"),
])
async def test_runner_preserves_billed_raw_outputs_and_closes_backend(
        tmp_path, monkeypatch, invalid, model, effort, expected_model):
    from decision_wrapper.benchmark.local import run

    suite = suite_for(task("noul"))
    suite.root = tmp_path
    suite.metadata = {"canonical_dataset_sha256": "fixture"}
    monkeypatch.setattr("decision_wrapper.benchmark.jevbench.JevBenchSuite", lambda root: suite)
    monkeypatch.setattr("decision_wrapper.benchmark.local.importlib.metadata.version", lambda name: "test")

    class Backend:
        closed = False

        async def start(self):
            pass

        async def close(self):
            self.closed = True

    backend = Backend()
    monkeypatch.setattr("decision_wrapper.backend.create_backend", lambda **kwargs: backend)
    calls = 0

    async def decide(request, **kwargs):
        nonlocal calls
        calls += 1
        assert request.model == expected_model
        assert request.reasoning_effort == effort
        key = next(iter(request.questions))
        raw = {"answers": {key: {
            "type": "noul", "value": True,
            "probability": True if invalid and calls > 2 else 0.8,
        }}}
        return SimpleNamespace(model_dump=lambda: raw, usage=SimpleNamespace(ai_credits=0.2))

    monkeypatch.setattr("decision_wrapper.benchmark.local.decide", decide)
    report = await run("copilot", tmp_path / "unused.jsonl", tmp_path / "output", 4, 42, tmp_path,
                       model=model, reasoning_effort=effort)
    record = json.loads((tmp_path / "output" / "copilot-items.jsonl").read_text(encoding="utf-8"))
    assert backend.closed and calls == 3
    assert report["total_ai_credits_including_warmup"] == pytest.approx(0.6)
    assert record["raw"]["answers"]["decision"]["probability"] == (True if invalid else 0.8)
    assert record["ai_credits"] == 0.2
    assert record["values"]["decision"] is (None if invalid else True)
    assert report["jevbench"]["n_correct"] == (0 if invalid else 1)
    assert report["metadata"]["checkpoint"] == record["model"] == expected_model
    assert report["metadata"]["reasoning_effort"] == record["reasoning_effort"] == effort
    assert report["jevbench"]["model_identities"] == [expected_model]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["noul", "choice", "score"])
async def test_openai_runner_uses_native_probabilities_with_original_scorer(tmp_path, monkeypatch, kind):
    import httpx
    import respx

    from decision_wrapper.benchmark.local import run
    from decision_wrapper.openai_backend import ENDPOINT, MODEL

    suite = suite_for(task(kind))
    suite.root = tmp_path
    suite.metadata = {"canonical_dataset_sha256": "fixture"}
    monkeypatch.setattr("decision_wrapper.benchmark.jevbench.JevBenchSuite", lambda root: suite)
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")

    def reply(sent):
        payload = json.loads(sent.content)
        assert "expected" not in payload
        assert "reasoning_effort" not in payload
        question = payload["questions"][0]
        answer = {"name": question["name"], "type": question["type"]}
        if answer["type"] == "predicate":
            answer["probability"] = 0.8
        elif answer["type"] == "choice":
            answer.update(choice="A", confidence=0.6, probabilities=[
                {"value": "A", "probability": 0.9}, {"value": "B", "probability": 0.1}])
        else:
            answer.update(score=0.97, confidence=0.4, probabilities=[
                {"value": 0, "label": "0", "probability": 0.49},
                {"value": 1, "label": "1", "probability": 0.05},
                {"value": 2, "label": "2", "probability": 0.46}])
        return httpx.Response(200, json={"model": MODEL, "answers": [answer],
                                         "usage": {"input_tokens": 100}})

    with respx.mock() as mock:
        route = mock.post(ENDPOINT)
        route.side_effect = reply
        report = await run("openai", tmp_path / "unused.jsonl", tmp_path / "output",
                           4, 42, tmp_path)
        assert route.call_count == 3
    record = json.loads((tmp_path / "output" / "openai-items.jsonl").read_text(encoding="utf-8"))
    assert report["complete"] and report["jevbench"]["n_correct"] == 1
    assert report["metadata"]["reasoning_effort"] is None
    assert record["jevbench"]["probs_source"] == "native"
    assert record["jevbench"]["cost_usd"] is None
    assert "openai" in record["jevbench"]["cost_basis"]
    assert report["openai_billing"]["input_tokens"] == 300
    assert report["openai_billing"]["base_input_cost_usd_estimate"] == pytest.approx(0.00003)
    if kind == "score":
        assert record["jevbench"]["predicted"] == "0"
        assert record["jevbench"]["ordinal_ev"] == pytest.approx(0.97)
