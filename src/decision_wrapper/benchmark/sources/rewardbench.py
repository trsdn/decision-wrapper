"""Converts RewardBench (allenai/reward-bench) preference pairs into this
repo's decision-wrapper dataset format, so model/reasoning-effort combos can
be scored against one of the actual benchmarks used in the published Jev
evaluation (CMU study, arXiv 2609.26550), not just a toy dataset.

RewardBench's "filtered" split is 2,985 (prompt, chosen, rejected) triples
across 23 subsets (alpacaeval-*, llmbar-*, mt-bench-*, hep-*, math-prm,
refusals-*, xstest-*, donotanswer). Each row becomes a two-option `choice`
question ("which response is better?"), with the chosen/rejected position
randomized per item to avoid positional bias, and response text truncated to
keep prompts (and CLI call cost/latency) bounded.
"""

from __future__ import annotations

import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

PARQUET_URL = (
    "https://huggingface.co/datasets/allenai/reward-bench/resolve/"
    "refs%2Fconvert%2Fparquet/default/filtered/0000.parquet"
)


def download_parquet(dest: Path) -> Path:
    if not dest.exists():
        urllib.request.urlretrieve(PARQUET_URL, dest)  # noqa: S310 (trusted HF URL)
    return dest


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " [...truncated]"


def stratified_sample(
    rows: List[dict], n: int, *, seed: int = 42
) -> List[dict]:
    """Pick `n` rows spread as evenly as possible across RewardBench's
    `subset` categories, so a small sample still covers preference, safety,
    code, math, and adversarial items rather than whatever happens to sort
    first."""

    rng = random.Random(seed)
    by_subset: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_subset[row["subset"]].append(row)
    for bucket in by_subset.values():
        rng.shuffle(bucket)

    subsets = sorted(by_subset.keys())
    picked: List[dict] = []
    i = 0
    while len(picked) < n and any(by_subset[s] for s in subsets):
        subset = subsets[i % len(subsets)]
        if by_subset[subset]:
            picked.append(by_subset[subset].pop())
        i += 1
    return picked[:n]


def to_dataset_items(
    rows: List[dict], *, seed: int = 42, max_response_chars: int = 1500
) -> List[dict]:
    rng = random.Random(seed)
    items = []
    for row in rows:
        chosen = _truncate(row["chosen"], max_response_chars)
        rejected = _truncate(row["rejected"], max_response_chars)
        if rng.random() < 0.5:
            response_a, response_b, expected = chosen, rejected, "response_a"
        else:
            response_a, response_b, expected = rejected, chosen, "response_b"

        items.append(
            {
                "id": f"rewardbench-{row['subset']}-{row['id']}",
                "state": f"PROMPT:\n{row['prompt']}",
                "questions": {
                    "preferred": {
                        "type": "choice",
                        "instructions": (
                            "Which response better answers the prompt: more "
                            "helpful, accurate, relevant, and appropriate?"
                        ),
                        "criteria": {
                            "response_a": response_a,
                            "response_b": response_b,
                        },
                    }
                },
                "expected": {"preferred": expected},
                "meta": {"subset": row["subset"], "source": "rewardbench"},
            }
        )
    return items


def write_jsonl(items: List[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")


def build_sample(
    out_path: Path,
    *,
    n: int = 30,
    seed: int = 42,
    cache_parquet: Optional[Path] = None,
    max_response_chars: int = 1500,
) -> List[dict]:
    import pyarrow.parquet as pq  # local import: optional dependency

    cache_parquet = cache_parquet or Path("/tmp/rewardbench.parquet")
    download_parquet(cache_parquet)
    rows = pq.read_table(cache_parquet).to_pylist()
    sampled = stratified_sample(rows, n, seed=seed)
    items = to_dataset_items(sampled, seed=seed, max_response_chars=max_response_chars)
    write_jsonl(items, out_path)
    return items
