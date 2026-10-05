from decision_wrapper.backend import extract_letter
from decision_wrapper.benchmark.report import summarize, to_markdown
from decision_wrapper.benchmark.runner import ItemResult


def test_extract_letter_tolerates_noise():
    assert extract_letter("A", {"A", "B"}) == "A"
    assert extract_letter("The answer is B.", {"A", "B"}) == "B"
    assert extract_letter("`A`\n", {"A", "B"}) == "A"
    assert extract_letter("z", {"A", "B"}) is None


def test_summarize_and_markdown():
    results = [
        ItemResult("i1", "gpt-5.4-mini", "low", correct=1, total=1, latency_ms=100.0, premium_requests=0.33, samples=3),
        ItemResult("i2", "gpt-5.4-mini", "low", correct=0, total=1, latency_ms=200.0, premium_requests=0.33, samples=3),
    ]
    rows = summarize(results)
    assert len(rows) == 1
    assert rows[0]["model"] == "gpt-5.4-mini"
    assert rows[0]["accuracy"] == 0.5
    md = to_markdown(rows)
    assert "gpt-5.4-mini" in md
    assert "accuracy" in md
