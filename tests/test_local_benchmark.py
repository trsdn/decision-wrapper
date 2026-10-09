import copy
import json
from pathlib import Path

import pytest

from decision_wrapper.benchmark.local import (
    dataset_reference, gliner_tasks, gliner_values, load_pairs, native_questions, native_values, run, summarize,
)
from decision_wrapper.schema import DecisionRequest


def score_request():
    return DecisionRequest(state="test", questions={"answer": {
        "type": "score", "instructions": "Severity?",
        "criteria": {"10": "high", "1": "low", "2": "middle"},
    }})


def test_dataset_metadata_uses_relative_paths_without_user_profiles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert dataset_reference(tmp_path / "datasets" / "public") == str(Path("datasets") / "public")
    assert dataset_reference(tmp_path.parent / "private" / "input.jsonl") == "input.jsonl"


def test_native_score_maps_ordinal_indices_to_original_numeric_keys():
    request = score_request()
    assert native_questions(request)["answer"]["criteria"] == ["low", "middle", "high"]
    raw = {"answers": {"answer": {"type": "score", "score": 1.1,
                                  "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}}}}
    original = copy.deepcopy(raw)
    assert native_values(request, raw) == {"answer": "2"}
    assert raw == original


def test_native_score_uses_mode_not_rounded_expectation():
    raw = {"answers": {"answer": {
        "type": "score", "score": 0.97,
        "probabilities": {"0": 0.49, "1": 0.05, "2": 0.46},
    }}}
    assert native_values(score_request(), raw) == {"answer": "1"}


@pytest.mark.parametrize("probabilities", [
    {"0": 0.1, "1": 0.9},
    {"0": 0.1, "1": 0.2, "2": 0.3},
    {"0": -0.1, "1": 0.7, "2": 0.4},
    {"0": float("nan"), "1": 0.7, "2": 0.3},
    {"0": True, "1": 0.0, "2": 0.0},
])
def test_native_rejects_invalid_or_incomplete_distribution(probabilities):
    with pytest.raises(ValueError):
        native_values(score_request(), {"answers": {"answer": {
            "type": "score", "probabilities": probabilities}}})


def test_gliner_boolean_preserves_native_confidence_without_fabricating_distribution():
    request = DecisionRequest(state="test", questions={"answer": {
        "type": "noul", "instructions": "Correct?",
        "criteria": {"true": "correct", "false": "incorrect"},
    }})
    tasks = gliner_tasks(request)
    assert tasks["answer"] == {"labels": {"true": "correct", "false": "incorrect"},
                               "prompt": "Correct?"}
    raw = {"answer": {"label": "false", "confidence": 0.61}}
    assert gliner_values(request, raw) == {"answer": False}
    assert raw["answer"] == {"label": "false", "confidence": 0.61}
    with pytest.raises(ValueError, match="invalid GLiNER label"):
        gliner_values(request, {"answer": {"label": "maybe", "confidence": 0.9}})


def test_bilingual_dataset_contains_twenty_matched_pairs():
    path = Path(__file__).resolve().parents[1] / "datasets" / "bilingual_benchmark.jsonl"
    rows = load_pairs(path)
    assert len(rows) == 40
    assert sum(row["meta"]["language"] == "de" for row in rows) == 20
    assert {row["questions"]["answer"]["type"] for row in rows} == {"choice", "noul", "score"}


@pytest.mark.parametrize("mutation", ["missing-language", "different-expected", "duplicate-id"])
def test_loader_rejects_invalid_pairs(tmp_path, mutation):
    source = Path(__file__).resolve().parents[1] / "datasets" / "bilingual_benchmark.jsonl"
    rows = load_pairs(source)[:2]
    if mutation == "missing-language":
        rows.pop()
    elif mutation == "different-expected":
        rows[1]["expected"]["answer"] = "technical"
    else:
        rows[1]["id"] = rows[0]["id"]
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        load_pairs(path)


def test_summary_counts_failures_and_does_not_report_unknown_credits_as_free():
    rows = [
        {"language": "de", "correct": 1, "expected": {"answer": True},
         "error": None, "latency_ms": 100, "ai_credits": 0},
        {"language": "de", "correct": 0, "expected": {"answer": False},
         "error": "timeout", "latency_ms": 120000, "ai_credits": None},
    ]
    assert summarize(rows) == [{"language": "de", "correct": 1, "total": 2,
                               "errors": 1, "p50_ms": 100, "p95_ms": 100,
                               "ai_credits": None}]


@pytest.mark.asyncio
async def test_existing_output_is_rejected_before_starting_paid_backend(tmp_path, monkeypatch):
    dataset = Path(__file__).resolve().parents[1] / "datasets" / "bilingual_benchmark.jsonl"
    (tmp_path / "copilot-items.jsonl").write_text("existing results", encoding="utf-8")

    def forbidden_backend(**kwargs):
        pytest.fail("paid backend must not start for an existing output")

    monkeypatch.setattr("decision_wrapper.backend.create_backend", forbidden_backend)
    with pytest.raises(FileExistsError, match="fresh output"):
        await run("copilot", dataset, tmp_path, 4, 42)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine,settings", [
    ("laya", {"model": "gpt-6-luna"}),
    ("gliner", {"reasoning_effort": "low"}),
    ("copilot", {"reasoning_effort": "invalid"}),
    ("copilot", {"model": ""}),
])
async def test_invalid_runtime_settings_fail_before_loading_or_billing(tmp_path, engine, settings):
    with pytest.raises(ValueError):
        await run(engine, tmp_path / "nonexistent.jsonl", tmp_path, 4, 42, **settings)
