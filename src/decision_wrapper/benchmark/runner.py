"""Benchmark harness: run a labeled dataset across a grid of
(model x reasoning_effort) combinations and measure latency, cost
(AI credits), and accuracy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..backend import CopilotCliBackend, create_backend
from ..engine import decide
from ..schema import DecisionRequest, ReasoningEffort


@dataclass
class DatasetItem:
    state: str
    questions: Dict[str, Any]
    expected: Dict[str, Any]
    id: Optional[str] = None


def load_dataset(path: Path) -> List[DatasetItem]:
    items: List[DatasetItem] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            items.append(
                DatasetItem(
                    state=row["state"],
                    questions=row["questions"],
                    expected=row["expected"],
                    id=row.get("id", str(i)),
                )
            )
    return items


@dataclass
class ItemResult:
    item_id: str
    model: str
    reasoning_effort: str
    correct: int
    total: int
    latency_ms: float
    premium_requests: float
    samples: int
    mismatches: List[str] = field(default_factory=list)
    ai_credits: Optional[float] = None


@dataclass
class RunConfig:
    models: List[str]
    reasoning_efforts: List[ReasoningEffort]
    samples: int = 3
    max_concurrency: int = 4
    timeout_s: float = 90.0
    warm: bool = False


def _answer_value(answer) -> Any:
    if hasattr(answer, "value") and hasattr(answer, "probability"):
        return answer.value
    if hasattr(answer, "choice"):
        return answer.choice
    return answer.value


async def run_benchmark(
    items: Iterable[DatasetItem], config: RunConfig
) -> List[ItemResult]:
    async with create_backend(
        warm=config.warm,
        max_concurrency=config.max_concurrency,
        timeout_s=config.timeout_s,
    ) as backend:
        return await _run_items(items, config, backend)


async def _run_items(
    items: Iterable[DatasetItem], config: RunConfig, backend: CopilotCliBackend
) -> List[ItemResult]:
    results: List[ItemResult] = []
    items = list(items)

    for model in config.models:
        for effort in config.reasoning_efforts:
            for item in items:
                request = DecisionRequest(
                    model=model,
                    state=item.state,
                    questions=item.questions,
                    reasoning_effort=effort,
                )
                start = time.perf_counter()
                response = await decide(request, backend, samples=config.samples)
                wall_ms = (time.perf_counter() - start) * 1000

                correct = 0
                mismatches = []
                for qname, expected_value in item.expected.items():
                    actual = _answer_value(response.answers[qname])
                    if actual == expected_value:
                        correct += 1
                    else:
                        mismatches.append(
                            f"{qname}: expected={expected_value!r} actual={actual!r}"
                        )

                results.append(
                    ItemResult(
                        item_id=item.id or "?",
                        model=model,
                        reasoning_effort=effort,
                        correct=correct,
                        total=len(item.expected),
                        latency_ms=wall_ms,
                        premium_requests=response.usage.premium_requests,
                        samples=response.usage.samples,
                        mismatches=mismatches,
                        ai_credits=response.usage.ai_credits,
                    )
                )
    return results
