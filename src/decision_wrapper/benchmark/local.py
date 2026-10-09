"""Bilingual and published JevBench comparisons across native and Copilot backends."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import get_args

from ..engine import _build_options, decide
from ..schema import DecisionRequest, ReasoningEffort, ScoreQuestion

CHECKPOINTS = {
    "laya": "convaiinnovations/laya-multilingual",
    "gliner": "fastino/GLiNER2.5-multi-Decide",
    "strands": "StrandsAgents/strands-decider-2B-hobson-v21",
    "copilot": "gpt-5.4-mini",
    "openai": "gpt-6-luna",
}


def dataset_reference(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def load_pairs(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("empty bilingual dataset")
    pairs: dict[str, dict] = {}
    ids = set()
    for row in rows:
        request = DecisionRequest.model_validate(row)
        if row["id"] in ids:
            raise ValueError(f"duplicate item id: {row['id']}")
        ids.add(row["id"])
        if set(row["expected"]) != set(request.questions):
            raise ValueError(f"expected answers do not match questions: {row['id']}")
        language = row["meta"]["language"]
        if language not in {"en", "de"}:
            raise ValueError(f"unsupported language: {language}")
        pair = pairs.setdefault(row["meta"]["case"], {})
        if language in pair:
            raise ValueError(f"duplicate language in pair: {row['meta']['case']}")
        pair[language] = row
    for case, pair in pairs.items():
        if set(pair) != {"en", "de"}:
            raise ValueError(f"incomplete bilingual pair: {case}")
        if pair["en"]["expected"] != pair["de"]["expected"]:
            raise ValueError(f"conflicting bilingual answers: {case}")
        for key, question in pair["en"]["questions"].items():
            translated = pair["de"]["questions"][key]
            if (question["type"] != translated["type"]
                    or set(question.get("criteria") or {}) !=
                    set(translated.get("criteria") or {})):
                raise ValueError(f"incompatible paired question: {case}/{key}")
    return rows


def native_questions(request: DecisionRequest) -> dict:
    questions = {}
    for key, question in request.questions.items():
        payload = question.model_dump(exclude_none=True)
        if isinstance(question, ScoreQuestion):
            payload["criteria"] = [description for _, _, description
                                   in _build_options(question)]
        questions[key] = payload
    return questions


def _probability(value: object) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"invalid native probability: {value!r}")
    return float(value)


def native_values(request: DecisionRequest, raw: dict) -> dict:
    """Use posterior argmax for ordinal accuracy, not rounded expected scores."""
    values = {}
    if set(raw["answers"]) != set(request.questions):
        raise ValueError("native answer fields do not match the request")
    for key, question in request.questions.items():
        answer = raw["answers"][key]
        if answer["type"] != question.type:
            raise ValueError(f"native answer type mismatch: {key}")
        if question.type == "noul":
            values[key] = _probability(answer["noul"]) >= 0.5
            continue
        labels = [label for _, label, _ in _build_options(question)]
        native_labels = ([str(i) for i in range(len(labels))]
                         if question.type == "score" else labels)
        probabilities = answer["probabilities"]
        if set(probabilities) != set(native_labels):
            raise ValueError(f"incomplete native distribution: {key}")
        p = [_probability(probabilities[label]) for label in native_labels]
        if not math.isclose(sum(p), 1.0, abs_tol=0.005):
            raise ValueError(f"native distribution does not sum to one: {key}")
        index = max(range(len(p)), key=p.__getitem__)
        values[key] = labels[index]
        if question.type == "choice" and answer["choice"] != values[key]:
            raise ValueError(f"native choice disagrees with distribution: {key}")
    return values


def gliner_tasks(request: DecisionRequest) -> dict:
    return {key: {"labels": {label: description for _, label, description
                             in _build_options(question)},
                  "prompt": question.instructions}
            for key, question in request.questions.items()}


def gliner_values(request: DecisionRequest, raw: dict) -> dict:
    values = {}
    if set(raw) != set(request.questions):
        raise ValueError("GLiNER answer fields do not match the request")
    for key, question in request.questions.items():
        answer = raw[key]
        labels = {label for _, label, _ in _build_options(question)}
        label = answer["label"]
        if label not in labels:
            raise ValueError(f"invalid GLiNER label: {label!r}")
        _probability(answer["confidence"])
        values[key] = label == "true" if question.type == "noul" else label
    return values


def prediction_values(engine: str, request: DecisionRequest, raw: dict) -> dict:
    if engine == "openai":
        from ..openai_backend import normalize_response

        return native_values(request, normalize_response(request, raw))
    if engine == "copilot":
        return {key: raw["answers"][key]["choice" if question.type == "choice" else "value"]
                for key, question in request.questions.items()}
    if engine == "gliner":
        return gliner_values(request, raw)
    return native_values(request, raw)


def _credits(rows: list[dict]) -> float | None:
    values = [row["ai_credits"] for row in rows]
    return None if any(value is None for value in values) else sum(values)


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(row["language"], []).append(row)
    summaries = []
    for language, items in sorted(groups.items()):
        latency = sorted(row["latency_ms"] for row in items if row["error"] is None)
        summaries.append({
            "language": language,
            "correct": sum(row["correct"] for row in items),
            "total": sum(len(row["expected"]) for row in items),
            "errors": sum(row["error"] is not None for row in items),
            "p50_ms": statistics.median(latency) if latency else None,
            "p95_ms": latency[math.ceil(0.95 * len(latency)) - 1] if latency else None,
            "ai_credits": _credits(items),
        })
    return summaries


async def run(engine: str, dataset: Path, output: Path, threads: int, seed: int,
              jevbench_root: Path | None = None, *, model: str | None = None,
              reasoning_effort: ReasoningEffort = "none") -> dict:
    if engine not in {"copilot", "openai"} and (model is not None or reasoning_effort != "none"):
        raise ValueError("model and reasoning-effort overrides require the copilot engine")
    if engine == "openai" and (reasoning_effort != "none" or model not in {None, CHECKPOINTS["openai"]}):
        raise ValueError("OpenAI Decisions supports gpt-6-luna without a reasoning-effort parameter")
    if reasoning_effort not in get_args(ReasoningEffort):
        raise ValueError(f"invalid reasoning effort: {reasoning_effort}")
    checkpoint = CHECKPOINTS[engine] if model is None else model
    if not checkpoint.strip():
        raise ValueError("model must not be blank")
    suite = None
    if jevbench_root is not None:
        from .jevbench import JevBenchSuite

        suite = JevBenchSuite(jevbench_root)
    rows = suite.rows() if suite else load_pairs(dataset)
    random.Random(seed).shuffle(rows)
    output.mkdir(parents=True, exist_ok=True)
    if (output / f"{engine}-items.jsonl").exists():
        raise FileExistsError(f"{engine} results already exist; choose a fresh output directory")
    metadata = {
        "engine": engine, "checkpoint": checkpoint, "device": "remote" if engine in {"copilot", "openai"} else "cpu",
        "threads": threads if engine not in {"copilot", "openai"} else None,
        "python": sys.version, "platform": platform.platform(), "order_seed": seed,
        "dataset": dataset_reference(suite.root / "datasets" / "public" if suite else dataset),
        "dataset_sha256": (suite.metadata["canonical_dataset_sha256"] if suite
                           else hashlib.sha256(dataset.read_bytes()).hexdigest()),
        "reasoning_effort": reasoning_effort if engine == "copilot" else None,
        "samples": 1, "packages": {},
        "confidence_note": "Native scores are retained; GLiNER has no full distribution and Copilot's one-sample vote shares are not native likelihoods.",
    }
    if suite:
        metadata["jevbench"] = suite.metadata
    packages = (["httpx"] if engine == "openai" else ["github-copilot-sdk"] if engine == "copilot"
                else ["torch", "transformers", "peft",
                      {"laya": "laya", "gliner": "gliner2", "strands": "strands-decider"}[engine]])
    for package in packages:
        metadata["packages"][package] = importlib.metadata.version(package)
    started = time.perf_counter()
    backend = None
    if engine == "openai":
        from ..openai_backend import OpenAIDecisionsBackend

        backend = OpenAIDecisionsBackend()
        await backend.start()
        metadata["context_policy"] = "complete text and typed questions; no client-side truncation"
        metadata["confidence_note"] = "Native OpenAI distributions and separate confidence are retained; score accuracy uses posterior argmax, not rounded mean."

        async def predict(request, language):
            return await backend.predict(request), None
    elif engine == "copilot":
        from ..backend import create_backend

        backend = create_backend(warm=True, timeout_s=120)
        await backend.start()

        async def predict(request, language):
            response = await decide(request, backend=backend, samples=1)
            raw = response.model_dump()
            return raw, response.usage.ai_credits
    else:
        import torch
        from huggingface_hub import HfApi, snapshot_download

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        revision = HfApi().model_info(checkpoint).sha
        metadata["revision"] = revision
        local_path = snapshot_download(checkpoint, revision=revision)
        if engine == "laya":
            import laya

            model = laya.load(local_path, device="cpu")
            metadata["context_policy"] = "8192 tokens; native option/instruction head budgets and state truncation"

            async def predict(request, language):
                raw = model.predict(request.state, native_questions(request), lang=language, max_len=8192)
                return raw, 0.0
        elif engine == "gliner":
            from gliner2 import AutoExtractor

            model = AutoExtractor.from_pretrained(local_path).to("cpu").eval()
            metadata["context_policy"] = "native GLiNER tokenizer/collator window and truncation"

            async def predict(request, language):
                with torch.inference_mode():
                    raw = model.classify_text(request.state, gliner_tasks(request),
                                              include_confidence=True)
                return raw, 0.0
        else:
            from strands_decider.infer import load_engine
            from strands_decider.schema import SystemOneRequest

            model = load_engine(local_path, device="cpu")
            model.cfg.strict_window = True
            metadata["context_policy"] = f"strict input window: {model.model.config.max_length} tokens; refusals count wrong"

            async def predict(request, language):
                native = SystemOneRequest(state=request.state, questions=native_questions(request))
                raw = model.evaluate(native).model_dump()
                return raw, 0.0
    metadata["load_ms"] = (time.perf_counter() - started) * 1000
    print(f"{engine}: loaded in {metadata['load_ms'] / 1000:.1f}s", flush=True)
    results = []
    warmups = []
    try:
        for language, state, instruction in [
            ("en", "The parcel has arrived.", "Has the parcel arrived?"),
            ("de", "Das Paket ist angekommen.", "Ist das Paket angekommen?"),
        ]:
            request = DecisionRequest(state=state, model=checkpoint, reasoning_effort=reasoning_effort,
                                      questions={"warmup": {"type": "noul", "instructions": instruction}})
            start = time.perf_counter()
            raw, credits = await predict(request, language)
            prediction_values(engine, request, raw)
            warmup = {"language": language, "latency_ms": (time.perf_counter() - start) * 1000,
                      "ai_credits": credits}
            if engine == "openai":
                from ..openai_backend import input_billing

                warmup.update(input_billing(raw), raw=raw)
            warmups.append(warmup)
        consecutive_errors = 0
        with (output / f"{engine}-items.jsonl").open("x", encoding="utf-8") as stream:
            for index, row in enumerate(rows, 1):
                request = DecisionRequest.model_validate(
                    {**row, "model": checkpoint, "reasoning_effort": reasoning_effort})
                start = time.perf_counter()
                record = {"id": row["id"], "case": row["meta"]["case"],
                          "language": row["meta"]["language"], "category": row["meta"]["category"],
                          "model": checkpoint, "reasoning_effort": metadata["reasoning_effort"],
                          "expected": row["expected"], "values": {}, "raw": None,
                          "correct": 0, "error": None,
                          "ai_credits": None if engine in {"copilot", "openai"} else 0.0}
                if engine == "openai":
                    record.update(input_tokens=None, base_input_cost_usd_estimate=None,
                                  normalized=None, refused=False, http_status=None)
                try:
                    raw, credits = await predict(request, record["language"])
                    record.update(raw=raw, ai_credits=credits)
                    if engine == "openai":
                        from ..openai_backend import input_billing, normalize_response

                        record.update(input_billing(raw))
                        record["normalized"] = normalize_response(request, raw)
                    if suite is None:
                        values = prediction_values(engine, request, raw)
                        record.update(values=values,
                                      correct=sum(values[key] == expected
                                                  for key, expected in row["expected"].items()))
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                    if engine == "openai":
                        from ..openai_backend import DecisionRefusal, OpenAIDecisionsError

                        record["refused"] = isinstance(error, DecisionRefusal)
                        if isinstance(error, OpenAIDecisionsError):
                            record["http_status"] = error.status_code
                    print(f"{engine}/{row['id']}: {record['error']}", file=sys.stderr, flush=True)
                record["latency_ms"] = (time.perf_counter() - start) * 1000
                if suite:
                    scored = suite.score(row["id"], engine,
                                         record["normalized"] if engine == "openai" else record["raw"],
                                         record["error"])
                    scored["model"] = checkpoint
                    record["jevbench"] = scored
                    record["correct"] = int(bool(scored["correct"]))
                    predicted = scored["predicted"]
                    if request.questions["decision"].type == "noul" and predicted is not None:
                        predicted = predicted == "yes"
                    record["values"] = {"decision": predicted}
                    if scored["schema_error"]:
                        print(f"{engine}/{row['id']}: schema error: {scored['schema_error']}",
                              file=sys.stderr, flush=True)
                results.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                if suite and index % 25 == 0:
                    print(f"{engine}: {index}/{len(rows)} attempted", flush=True)
                if engine == "openai" and record["http_status"] in {401, 403, 402, 429}:
                    print("openai: stopping after authorization, quota or rate-limit error; run incomplete",
                          file=sys.stderr, flush=True)
                    break
                if suite:
                    refusal = record.get("refused") or (record["error"] and "context window" in record["error"])
                    consecutive_errors = consecutive_errors + 1 if record["error"] and not refusal else 0
                    if consecutive_errors >= 3:
                        print(f"{engine}: stopping after three consecutive runtime errors; run incomplete",
                              file=sys.stderr, flush=True)
                        break
    finally:
        if backend is not None:
            await backend.close()
    report = {"metadata": metadata, "warmups": warmups, "summary": summarize(results),
              "total_ai_credits_including_warmup": _credits(warmups + results),
              "complete": len(results) == len(rows), "attempted": len(results), "planned": len(rows)}
    if engine == "openai":
        from ..openai_backend import BASE_INPUT_USD_PER_MILLION, PRICING_SOURCE

        billed = warmups + results
        report["openai_billing"] = {
            "input_tokens": (sum(row["input_tokens"] for row in billed)
                             if all(row["input_tokens"] is not None for row in billed) else None),
            "base_input_cost_usd_estimate": (
                sum(row["base_input_cost_usd_estimate"] for row in billed)
                if all(row["base_input_cost_usd_estimate"] is not None for row in billed) else None),
            "base_input_usd_per_million": BASE_INPUT_USD_PER_MILLION,
            "pricing_source": PRICING_SOURCE,
            "note": "Base-rate estimate on reported total input tokens, including warmups; not an invoice. Regional premiums, long-context multipliers and account-specific adjustments are excluded. Copilot AI credits are not applicable.",
        }
    if suite:
        report["jevbench"] = suite.summary(results)
    (output / f"{engine}-run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=CHECKPOINTS, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("datasets") / "bilingual_benchmark.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", help="Copilot model override, or gpt-6-luna for OpenAI Decisions")
    parser.add_argument("--reasoning-effort", choices=get_args(ReasoningEffort), default="none",
                        help="Copilot reasoning effort; default: none")
    parser.add_argument("--jevbench-root", type=Path,
                        help="Use this installed official JevBench checkout instead of the paired dataset")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.engine not in {"copilot", "openai"} and (args.model is not None or args.reasoning_effort != "none"):
        parser.error("--model and --reasoning-effort overrides require --engine copilot")
    if args.engine == "openai" and (args.reasoning_effort != "none" or args.model not in {None, CHECKPOINTS["openai"]}):
        parser.error("OpenAI Decisions supports only gpt-6-luna; no reasoning-effort override")
    asyncio.run(run(args.engine, args.dataset, args.output, args.threads, args.seed, args.jevbench_root,
                    model=args.model, reasoning_effort=args.reasoning_effort))


if __name__ == "__main__":
    main()
