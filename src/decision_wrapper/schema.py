"""Request/response contract for the decision API.

This mirrors the Jev (TypeSafe AI) "System One" decision contract: a single
``state`` plus one or more typed ``questions`` (noul / choice / score),
answered with calibrated probabilities and a confidence score instead of
free-form text.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Reasoning-effort ("thinking mode") levels. These are passed straight through
# to the underlying chat model when it supports a `reasoning_effort`-style
# parameter, and otherwise degrade to a prompt-engineered instruction.
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]


class QuestionType(str, Enum):
    noul = "noul"
    choice = "choice"
    score = "score"


class NoulQuestion(BaseModel):
    """A yes/no question. Returns calibrated P(true)."""

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: Optional[Dict[Literal["true", "false"], str]] = None


class ChoiceQuestion(BaseModel):
    """Pick exactly one option from a fixed set."""

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: Dict[str, str] = Field(..., min_length=2)

    @field_validator("criteria")
    @classmethod
    def _max_26_options(cls, v: Dict[str, str]) -> Dict[str, str]:
        if len(v) > 26:
            raise ValueError("choice questions support at most 26 options")
        return v


class ScoreQuestion(BaseModel):
    """Grade the state against an ordered rubric."""

    type: Literal["score"] = "score"
    instructions: str
    # Ordered rubric, lowest to highest, e.g. {"1": "no issue", ..., "5": "critical"}
    criteria: Dict[str, str] = Field(..., min_length=2)


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


class DecisionRequest(BaseModel):
    model: str = "gpt-5.4-mini"
    state: str
    questions: Dict[str, Question]
    reasoning_effort: ReasoningEffort = "low"

    @model_validator(mode="after")
    def _non_empty_questions(self) -> "DecisionRequest":
        if not self.questions:
            raise ValueError("at least one question is required")
        return self


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    value: bool
    probability: float  # P(true)
    confidence: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: Dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    value: str
    probabilities: Dict[str, float]
    confidence: float


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


class Usage(BaseModel):
    """Cost/latency accounting.

    The Copilot CLI backend bills in "premium requests" (its own credit
    unit), not raw tokens, and each invocation is a full agent turn (large
    fixed system-prompt overhead) rather than a bare completions call. We
    surface both the raw CLI accounting and an optional USD estimate.
    """

    samples: int = 1
    premium_requests: float = 0.0
    prompt_tokens: int = 0
    api_duration_ms: float = 0.0
    cost_usd: Optional[float] = None


class DecisionResponse(BaseModel):
    model: str
    reasoning_effort: ReasoningEffort
    answers: Dict[str, Answer]
    usage: Usage
    latency_ms: float
