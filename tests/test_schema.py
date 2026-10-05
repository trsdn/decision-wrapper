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
