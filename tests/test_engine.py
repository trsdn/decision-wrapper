import pytest

from decision_wrapper.backend import CallResult, CopilotCliBackend
from decision_wrapper.engine import decide
from decision_wrapper.schema import ChoiceQuestion, DecisionRequest, NoulQuestion, ScoreQuestion


class FakeBackend(CopilotCliBackend):
    """Backend stand-in that returns scripted letters instead of shelling
    out to the real `copilot` CLI, so tests are fast and deterministic."""

    def __init__(self, scripted_letters):
        super().__init__(max_concurrency=8)
        self._scripted = list(scripted_letters)

    async def complete(self, prompt, *, model, reasoning_effort):
        letter = self._scripted.pop(0)
        return CallResult(text=letter, premium_requests=0.33, api_duration_ms=10.0, wall_time_ms=10.0)


@pytest.mark.asyncio
async def test_noul_question_votes_majority():
    backend = FakeBackend(["Y", "Y", "N"])
    request = DecisionRequest(
        model="gpt-5.4-mini",
        state="Ripe apples are usually red.",
        questions={"is_true": NoulQuestion(instructions="Is this true?")},
    )
    response = await decide(request, backend, samples=3)
    answer = response.answers["is_true"]
    assert answer.value is True
    assert answer.probability == pytest.approx(2 / 3)
    assert answer.confidence == pytest.approx(2 / 3)
    assert response.usage.samples == 3
    assert response.usage.premium_requests == pytest.approx(0.99)


@pytest.mark.asyncio
async def test_choice_question_maps_letters_to_labels():
    backend = FakeBackend(["A", "A", "A"])
    request = DecisionRequest(
        model="gpt-6-luna",
        state="Charged twice for order A-4471.",
        questions={
            "department": ChoiceQuestion(
                instructions="Which team should handle this?",
                criteria={"billing": "Payments and refunds", "technical": "Bugs"},
            )
        },
    )
    response = await decide(request, backend, samples=3)
    answer = response.answers["department"]
    assert answer.choice == "billing"
    assert answer.probabilities["billing"] == pytest.approx(1.0)
    assert answer.confidence == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_score_question_orders_rubric_numerically():
    backend = FakeBackend(["E"])  # 5th option after numeric sort 1..5
    request = DecisionRequest(
        model="gpt-5.4-mini",
        state="Payouts failing for 3 days.",
        questions={
            "urgency": ScoreQuestion(
                instructions="How urgent?",
                criteria={"3": "moderate", "1": "none", "5": "critical", "2": "low", "4": "high"},
            )
        },
    )
    response = await decide(request, backend, samples=1)
    assert response.answers["urgency"].value == "5"


@pytest.mark.asyncio
async def test_unparseable_responses_are_skipped_not_fatal():
    backend = FakeBackend(["A", "???", "A"])
    request = DecisionRequest(
        model="gpt-5.4-mini",
        state="x",
        questions={
            "q": ChoiceQuestion(instructions="pick one", criteria={"a": "desc a", "b": "desc b"})
        },
    )
    response = await decide(request, backend, samples=3)
    assert response.answers["q"].choice == "a"
    assert response.usage.samples == 2  # one sample was unparseable and skipped
