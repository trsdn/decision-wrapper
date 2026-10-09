"""Adapt published JevBench tasks; delegate every metric to its pinned scorer."""

from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

from ..schema import DecisionRequest


class JevBenchSuite:
    def __init__(self, root: Path):
        import jevbench
        from jevbench.tasks import dataset_hash, load_jsonl, sha256_file

        self.root = root.resolve()
        if Path(jevbench.__file__).resolve().parent.parent != self.root:
            raise ValueError("installed JevBench scorer does not match --jevbench-root")
        manifest = json.loads((self.root / "datasets" / "manifest.json").read_text(encoding="utf-8"))
        splits = {split["name"]: split for split in manifest["splits"]}
        self.tasks = []
        self.tiers = {}
        for path in sorted((self.root / "datasets" / "public").glob("*.jsonl")):
            split = splits[path.stem]
            tasks = load_jsonl(str(path))
            lf_hash = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            if (len(tasks) != split["n"] or
                    (sha256_file(str(path)) != split["sha256"] and lf_hash != split["sha256"])
                    or dataset_hash(tasks) != split["canonical_sha256"]):
                raise ValueError(f"public corpus does not match official manifest: {path.name}")
            if any(task.split != "public" for task in tasks):
                raise ValueError("non-public task found in public corpus")
            self.tasks.extend(tasks)
            self.tiers.update({task.id: path.stem for task in tasks})
        if not self.tasks or len({task.id for task in self.tasks}) != len(self.tasks):
            raise ValueError("empty corpus or duplicate official task IDs")
        self.by_id = {task.id: task for task in self.tasks}
        commit = subprocess.check_output(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True).strip()
        self.metadata = {
            "protocol": manifest["protocol"], "source_commit": commit,
            "canonical_dataset_sha256": dataset_hash(self.tasks),
            "n_public_tasks": len(self.tasks), "scorer": "jevbench.scoring + jevbench.summarize",
            "scope": "published public corpus only; not the current sealed-inclusive leaderboard",
            "language": "not annotated by the published corpus; no translations added",
            "structured_state": "serialized as Unicode JSON with sorted keys for every adapter",
            "calibration": "GLiNER is label-only; Copilot probabilities are one-sample vote shares, not native likelihoods",
        }

    def rows(self) -> list[dict]:
        rows = []
        for task in self.tasks:
            question = dict(task.question)
            if question["type"] == "score":
                question["criteria"] = {str(i): text for i, text in enumerate(question["criteria"])}
            state = (task.state if isinstance(task.state, str)
                     else json.dumps(task.state, ensure_ascii=False, sort_keys=True))
            request = DecisionRequest(state=state, questions={"decision": question})
            expected = task.expected == "yes" if question["type"] == "noul" else str(task.expected)
            rows.append({
                "id": task.id, "state": state, "questions": request.model_dump()["questions"],
                "expected": {"decision": expected},
                "meta": {"case": task.id, "language": "unspecified", "category": task.family,
                         "source": "JevBench official public corpus"},
            })
        return rows

    def score(self, task_id: str, engine: str, raw: dict | None, error: str | None) -> dict:
        from jevbench.scoring import score_label, score_task

        task = self.by_id[task_id]
        source = ("label_only_no_calibrated_distribution" if engine == "gliner"
                  else "empirical_vote_share_samples_1" if engine == "copilot" else "native")
        probabilities = None
        label = None
        schema_error = None
        if error is None:
            try:
                if engine == "gliner":
                    label = raw["decision"]["label"]
                    if task.question["type"] == "noul":
                        if label not in {"true", "false"}:
                            raise ValueError("GLiNER noul label must be true or false")
                        label = "yes" if label == "true" else "no"
                else:
                    answer = raw["answers"]["decision"]
                    if answer["type"] != task.question["type"]:
                        raise ValueError("native answer type differs from official task")
                    if answer["type"] == "noul":
                        p = answer["probability"] if engine == "copilot" else answer["noul"]
                        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
                            raise ValueError("invalid native noul probability")
                        probabilities = {"yes": p, "no": 1 - p}
                    else:
                        if answer["type"] == "choice" and answer["choice"] not in task.labels:
                            raise ValueError("native choice is outside official label set")
                        probabilities = answer["probabilities"]
            except (KeyError, TypeError, ValueError) as exc:
                schema_error = f"{type(exc).__name__}: {exc}"
        if error is not None or schema_error is not None:
            scored = {"valid": False, "strict_valid": False, "renormalized": False,
                      "correct": False, "predicted": None, "probs": None}
        elif engine == "gliner":
            scored = score_label(label, task)
        else:
            scored = score_task(probabilities, task)
        return {
            "task_id": task.id, "family": task.family, "split": task.split, "group": task.group,
            "tier": self.tiers[task.id], "ok": error is None,
            "model": engine, "probs_source": source, "probs_as_returned": probabilities,
            "cost_usd": None,
            "cost_basis": ("actual_ai_credits_reported_separately" if engine == "copilot"
                           else "openai_input_tokens_and_base_usd_estimate_reported_separately" if engine == "openai"
                           else "local_cpu_no_provider_tariff"),
            "error": error, "schema_error": schema_error or scored.get("error"), **scored,
        }

    def summary(self, records: list[dict]) -> dict:
        from jevbench.summarize import metric, summarize

        scored = [{**record["jevbench"], "latency_s": record["latency_ms"] / 1000}
                  for record in records]
        report = summarize(self.tasks, scored)
        report["per_type"] = {
            kind: metric([task for task in self.tasks if task.question["type"] == kind], scored)
            for kind in ("noul", "choice", "score")
        }
        report["per_tier"] = {
            tier: metric([task for task in self.tasks if self.tiers[task.id] == tier], scored)
            for tier in sorted(set(self.tiers.values()))
        }
        return report
