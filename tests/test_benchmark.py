import pytest

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
        ItemResult("i1", "gpt-5.4-mini", "low", correct=1, total=1, latency_ms=100.0, premium_requests=0.33, samples=3, ai_credits=0.125),
        ItemResult("i2", "gpt-5.4-mini", "low", correct=0, total=1, latency_ms=200.0, premium_requests=0.33, samples=3, ai_credits=0.25),
    ]
    rows = summarize(results)
    assert len(rows) == 1
    assert rows[0]["model"] == "gpt-5.4-mini"
    assert rows[0]["accuracy"] == 0.5
    assert rows[0]["total_ai_credits"] == pytest.approx(0.375)
    assert rows[0]["avg_ai_credits"] == pytest.approx(0.1875)
    assert "total_premium_requests" not in rows[0]
    md = to_markdown(rows)
    assert "gpt-5.4-mini" in md
    assert "accuracy" in md


@pytest.mark.parametrize("credits", [[0, 0], [0.125, None], [None, None]])
def test_report_distinguishes_zero_and_unknown_credits(credits):
    results = [
        ItemResult(str(index), "gpt-5.4-mini", "none", correct=1, total=1,
                   latency_ms=1, premium_requests=100, samples=1, ai_credits=value)
        for index, value in enumerate(credits)
    ]
    row = summarize(results)[0]
    if any(value is None for value in credits):
        assert row["avg_ai_credits"] is None
        assert row["total_ai_credits"] is None
        assert "unknown" in to_markdown([row])
    else:
        assert row["avg_ai_credits"] == 0
        assert row["total_ai_credits"] == 0
