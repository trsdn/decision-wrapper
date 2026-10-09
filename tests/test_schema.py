import pytest
from pydantic import ValidationError

from decision_wrapper.schema import ChoiceQuestion, DecisionRequest, NoulQuestion


def test_decision_request_requires_at_least_one_question():
    with pytest.raises(ValidationError):
        DecisionRequest(model="gpt-5.4-mini", state="x", questions={})


def test_choice_question_requires_two_options():
    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="pick", criteria={"only": "one"})


def test_defaults():
    req = DecisionRequest(
        state="x", questions={"q": NoulQuestion(instructions="is x?")}
    )
    assert req.model == "gpt-5.4-mini"
    assert req.reasoning_effort == "low"


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh"])
def test_reasoning_effort_levels_round_trip(effort):
    req = DecisionRequest(
        state="x",
        questions={"q": NoulQuestion(instructions="is x?")},
        reasoning_effort=effort,
    )
    assert req.model_dump()["reasoning_effort"] == effort


def test_unknown_reasoning_effort_is_rejected():
    with pytest.raises(ValidationError):
        DecisionRequest(
            state="x",
            questions={"q": NoulQuestion(instructions="is x?")},
            reasoning_effort="invalid",
        )
