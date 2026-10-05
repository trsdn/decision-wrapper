# decision-wrapper

A **Jev-style "System One" decision API**, implemented as a wrapper around
the GitHub Copilot CLI's model routing (`gpt-6-luna`, `gpt-5.4-mini`, ...),
with a built-in benchmark harness for comparing models and reasoning-effort
("thinking mode") settings on latency, cost, and accuracy.

## Background: what is a "Jev-style" decision API?

[Jev](https://jevapi.dev) (TypeSafe AI, announced Sept 2026) popularized the
"decision-only model" category: instead of generating free-form text, you
send a `state` (text to evaluate) plus one or more typed `questions`, and get
back typed, calibrated answers:

- **noul** — boolean, returns `P(true)`
- **choice** — pick one of N options, returns a probability per option
- **score** — grade against an ordered rubric, returns a probability per level

Every answer includes a calibrated **confidence** score so calling code can
threshold "trust this" vs. "escalate to a human / a bigger model." OpenAI
shipped a similar "Decisions API" (real-time classify/route, fixed answer
sets) at DevDay 2026, and Laya is an open-source alternative — the three
form the current "decision-only model" category
([background](https://flowtivity.ai/blog/decisions-api-vs-jev-vs-laya/)).

This repo reproduces that **request/response contract** (see
[`schema.py`](src/decision_wrapper/schema.py)), but instead of a purpose-built
decision model, it answers questions by calling chat models already available
through the Copilot CLI's model routing.

## Why wrap the Copilot CLI instead of a raw completions API?

Models named `gpt-6-luna`, `gpt-5.4-mini`, etc. in this environment are only
reachable through `copilot`'s model routing — there is no public
OpenAI-compatible endpoint for them. So the backend
([`backend.py`](src/decision_wrapper/backend.py)) shells out to:

```
copilot -p "<prompt>" --model <model> --reasoning-effort <level> \
  --silent --allow-all-tools --disable-builtin-mcps --no-custom-instructions \
  --output-format json --log-level none -C <scratch-dir>
```

and parses the JSONL event stream for the final `assistant.message` and the
`result` event's usage block (`premiumRequests`, `totalApiDurationMs`).

**Important, measured caveat:** the CLI is a full coding-agent harness, not a
bare completions endpoint. Every call pays a large fixed system-prompt
overhead (~15-20k tokens, several seconds of wall time, a flat
`premiumRequests` charge per call) *regardless of how trivial the question
is*. This is, empirically, exactly the "chat model vs. decision model" cost
gap the Jev/Laya/OpenAI comparisons describe — see the benchmark section
below to quantify it for your own use case.

### Calibration without logprobs

The Copilot CLI has no logprobs or JSON-schema/function-calling mode, so this
wrapper can't read a token probability distribution directly (which is how
Jev/Laya calibrate). Instead, each question is turned into a single-letter
classification prompt (options are relabeled `A`/`B`/`C`... or `Y`/`N`), and
sent as `N` independent, concurrent CLI calls (`--samples`, default 3). The
**empirical vote share** across those samples is reported as the probability
distribution, and `confidence` is the winning option's vote share. This is a
standard self-consistency / Monte-Carlo calibration technique, and it
composes naturally with the benchmark harness: more samples trade latency and
cost for better-calibrated confidence.

## Request/response contract

```json
POST /v1/decisions
{
  "model": "gpt-5.4-mini",
  "state": "I was charged twice for order A-4471. Please refund the duplicate payment.",
  "reasoning_effort": "low",
  "questions": {
    "is_urgent": { "type": "noul", "instructions": "Does this message convey urgency?" },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    }
  }
}
```

```json
{
  "model": "gpt-5.4-mini",
  "reasoning_effort": "low",
  "answers": {
    "is_urgent": { "type": "noul", "value": false, "probability": 0.33, "confidence": 0.67 },
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 1.0, "technical": 0.0, "sales": 0.0 },
      "confidence": 1.0
    }
  },
  "usage": { "samples": 6, "premium_requests": 1.98, "api_duration_ms": 9400.0 },
  "latency_ms": 18230.4
}
```

`reasoning_effort` ("thinking mode") accepts `minimal | low | medium | high |
xhigh`, passed straight through to `copilot --reasoning-effort`.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires the `copilot` CLI to be installed, authenticated, and on `PATH`.

## Usage

**Run the server:**

```bash
decision-wrapper serve --port 8099
curl -s localhost:8099/v1/decisions -d @examples/request.json -H 'content-type: application/json'
```

**One-shot CLI:**

```bash
decision-wrapper decide examples/request.json --samples 3
```

**Benchmark across models x reasoning-effort modes:**

```bash
decision-wrapper benchmark datasets/sample_benchmark.jsonl \
  --models gpt-5.4-mini,gpt-6-luna,mai-code-1.1-flash \
  --reasoning-efforts low,medium,high \
  --samples 3 --out-dir benchmark-results/
```

This runs every dataset item through every `(model, reasoning_effort)`
combination and reports, per combination: accuracy (vs. the dataset's
`expected` answers), p50/p95 latency, and average/total `premium_requests`
("cost"). Results are printed as a table and written to
`benchmark-results/report.{csv,md}`.

Dataset format (`JSONL`, one object per line):

```json
{"state": "...", "questions": {"name": {"type": "choice", "instructions": "...", "criteria": {...}}}, "expected": {"name": "expected_label"}}
```

## Benchmark provenance: this harness vs. the published Jev benchmarks

This repo currently ships only a **3-item toy dataset** (`datasets/sample_benchmark.jsonl`) to exercise the harness end-to-end. It does **not** reproduce the actual benchmarks used in TypeSafe AI's and the CMU study's evaluation of Jev:

| Benchmark | What it tests | Jev score | GPT-6 score |
| --- | --- | --- | --- |
| RewardBench | preference judging | 92.2% | 93.5% |
| HaluEval | factuality | 87.5% | 86.7% |
| Answer adjudication | custom | 94.0% | 96.7% |
| JudgeBench | derivation checking | 78.6% | 93.1% |
| RM-Bench (hard pairs) | style-adversarial | 74.8% | 94.6% |

(Carnegie Mellon, arXiv 2609.26550 — JEV cost $0.044 / 1k judgments at 0.152s vs. GPT-6's $12.182 / 1k at 1.885s.)

Reproducing these against this CLI-backed wrapper is possible (dataset loaders for RewardBench/HaluEval/JudgeBench/RM-Bench would convert each into this repo's `{state, questions, expected}` JSONL format), but note the cost/time tradeoff: these datasets run from ~2,000 to ~18,000 items, and each item here costs one Copilot-CLI agent turn (~0.33-1+ premium requests, ~15-25s wall time with `samples=3`). A full RewardBench sweep would take many hours and a large premium-request budget — a stratified sample is the practical starting point.

### Measured per-model cost note

`premium_requests` per CLI call varies notably by model, independent of question count:

| Model | premium_requests / call |
| --- | --- |
| gpt-5.4-mini | ~0.33 |
| gpt-6-luna | ~0.33 |
| mai-code-1.1-flash | ~1.0 |

## Limitations

- Latency is dominated by Copilot CLI agent-turn overhead (seconds per
  call), not model inference time — this wrapper is for evaluating
  *decision quality and relative cost/latency across models and reasoning
  modes*, not for low-latency production routing. For that, a true
  decision-only model (Jev, Laya, or a raw completions API with logprobs)
  will be orders of magnitude faster and cheaper.
- `premium_requests` reflects Copilot billing; convert to USD using your own
  plan's per-premium-request rate if you need a dollar figure.
- Monte-Carlo calibration (vote share) is an approximation of true calibrated
  probability; it's bounded by `1/samples` resolution and the model's actual
  consistency across resamples.

## Tests

```bash
pytest
```
