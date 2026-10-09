"""Aggregate ItemResult rows into a per-(model, reasoning_effort) markdown
and CSV report: accuracy, p50/p95 latency, and actual AI-credit usage.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .runner import ItemResult


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(results: List[ItemResult]) -> List[Dict]:
    groups: Dict[Tuple[str, str], List[ItemResult]] = defaultdict(list)
    for r in results:
        groups[(r.model, r.reasoning_effort)].append(r)

    rows = []
    for (model, effort), rs in sorted(groups.items()):
        total_correct = sum(r.correct for r in rs)
        total_qs = sum(r.total for r in rs)
        latencies = [r.latency_ms for r in rs]
        credits = [r.ai_credits for r in rs if r.ai_credits is not None]
        complete_accounting = len(credits) == len(rs)
        rows.append(
            {
                "model": model,
                "reasoning_effort": effort,
                "items": len(rs),
                "accuracy": total_correct / total_qs if total_qs else 0.0,
                "p50_latency_ms": _percentile(latencies, 0.5),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "avg_ai_credits": statistics.fmean(credits) if complete_accounting else None,
                "total_ai_credits": sum(credits) if complete_accounting else None,
            }
        )
    return rows


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_markdown(rows: List[Dict]) -> str:
    if not rows:
        return "_no results_\n"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        cells = []
        for h in headers:
            v = row[h]
            cells.append("unknown" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
