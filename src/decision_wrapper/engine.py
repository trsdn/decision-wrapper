"""Evaluates a DecisionRequest by turning each question into a single-letter
classification prompt, sampling the Copilot CLI backend N times in parallel,
and reporting the empirical vote distribution as calibrated probabilities.
"""

from __future__ import annotations

import asyncio
import string
import time
from collections import Counter
from typing import Dict, List, Tuple

from .backend import BackendError, CopilotCliBackend, extract_letter
from .schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    Question,
    ReasoningEffort,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
)

LETTERS = string.ascii_uppercase


def _build_options(question: Question) -> List[Tuple[str, str, str]]:
    """Return [(letter, label, description), ...] in a stable, deterministic order."""

    if isinstance(question, NoulQuestion):
        criteria = question.criteria or {}
        return [
            ("Y", "true", criteria.get("true", "Yes / true")),
            ("N", "false", criteria.get("false", "No / false")),
        ]
    if isinstance(question, ChoiceQuestion):
        return [
            (LETTERS[i], label, desc)
            for i, (label, desc) in enumerate(question.criteria.items())
        ]
    if isinstance(question, ScoreQuestion):
        # Sort rubric keys numerically when possible, else lexically, so the
        # letter order follows the rubric's low -> high ordering.
        def sort_key(item: Tuple[str, str]):
            label = item[0]
            try:
                return (0, float(label))
            except ValueError:
                return (1, label)

        ordered = sorted(question.criteria.items(), key=sort_key)
        return [(LETTERS[i], label, desc) for i, (label, desc) in enumerate(ordered)]
    raise TypeError(f"unsupported question type: {question!r}")


def _build_prompt(state: str, question: Question, options: List[Tuple[str, str, str]]) -> str:
    option_lines = "\n".join(f"{letter}) {desc}" for letter, _label, desc in options)
    valid = "/".join(letter for letter, _, _ in options)
    return (
        "You are a structured decision classifier. Read the STATE and answer the "
        "QUESTION by choosing exactly one option.\n\n"
        f"STATE:\n{state}\n\n"
        f"QUESTION: {question.instructions}\n\n"
        f"OPTIONS:\n{option_lines}\n\n"
        f"Respond with ONLY a single character ({valid}) and nothing else: "
        "no punctuation, no explanation, no markdown."
    )


async def evaluate_question(
    backend: CopilotCliBackend,
    state: str,
    question: Question,
    *,
    model: str,
    reasoning_effort: ReasoningEffort,
    samples: int,
) -> Tuple[Answer, Usage]:
    options = _build_options(question)
    valid_letters = {letter for letter, _, _ in options}
    prompt = _build_prompt(state, question, options)

    results = await asyncio.gather(
        *(
            backend.complete(prompt, model=model, reasoning_effort=reasoning_effort)
            for _ in range(samples)
        ),
        return_exceptions=True,
    )

    votes: Counter[str] = Counter()
    premium_requests = 0.0
    ai_credits: float | None = 0.0
    api_duration_ms = 0.0
    prompt_tokens = 0
    ok = 0
    errors: List[str] = []
    for r in results:
        if isinstance(r, BaseException):
            errors.append(str(r))
            ai_credits = None
            continue
        premium_requests += r.premium_requests
        api_duration_ms += r.api_duration_ms
        prompt_tokens += r.prompt_tokens
        ai_credits = (
            ai_credits + r.ai_credits
            if ai_credits is not None and r.ai_credits is not None else None
        )
        letter = extract_letter(r.text or "", valid_letters)
        if letter is None:
            errors.append(f"unparseable response: {r.text!r}")
            continue
        votes[letter] += 1
        ok += 1

    if ok == 0:
        raise BackendError(
            f"all {samples} samples failed for question; errors={errors[:3]}"
        )

    label_by_letter = {letter: label for letter, label, _ in options}
    probabilities = {label_by_letter[letter]: count / ok for letter, count in votes.items()}
    for letter, label, _ in options:
        probabilities.setdefault(label, 0.0)

    winner_letter, winner_count = votes.most_common(1)[0]
    winner_label = label_by_letter[winner_letter]
    confidence = winner_count / ok

    usage = Usage(
        samples=ok,
        premium_requests=premium_requests,
        api_duration_ms=api_duration_ms,
        prompt_tokens=prompt_tokens,
        ai_credits=ai_credits,
    )

    if isinstance(question, NoulQuestion):
        answer: Answer = NoulAnswer(
            value=winner_label == "true",
            probability=probabilities.get("true", 0.0),
            confidence=confidence,
        )
    elif isinstance(question, ChoiceQuestion):
        answer = ChoiceAnswer(
            choice=winner_label, probabilities=probabilities, confidence=confidence
        )
    else:
        answer = ScoreAnswer(
            value=winner_label, probabilities=probabilities, confidence=confidence
        )
    return answer, usage


async def decide(
    request: DecisionRequest,
    backend: CopilotCliBackend,
    *,
    samples: int = 3,
) -> DecisionResponse:
    start = time.perf_counter()

    names = list(request.questions.keys())
    tasks = [
        asyncio.create_task(
            evaluate_question(
                backend,
                request.state,
                request.questions[name],
                model=request.model,
                reasoning_effort=request.reasoning_effort,
                samples=samples,
            )
        )
        for name in names
    ]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    answers: Dict[str, Answer] = {}
    total_usage = Usage(samples=0, premium_requests=0.0, api_duration_ms=0.0, ai_credits=0.0)
    for name, (answer, usage) in zip(names, results):
        answers[name] = answer
        total_usage.samples += usage.samples
        total_usage.premium_requests += usage.premium_requests
        total_usage.api_duration_ms += usage.api_duration_ms
        total_usage.prompt_tokens += usage.prompt_tokens
        total_usage.ai_credits = (
            total_usage.ai_credits + usage.ai_credits
            if total_usage.ai_credits is not None and usage.ai_credits is not None else None
        )

    latency_ms = (time.perf_counter() - start) * 1000
    return DecisionResponse(
        model=request.model,
        reasoning_effort=request.reasoning_effort,
        answers=answers,
        usage=total_usage,
        latency_ms=latency_ms,
    )
