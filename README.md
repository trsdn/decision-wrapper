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
threshold "trust this" vs. "escalate to a human / a bigger model."
OpenAI's [Decisions API](https://developers.openai.com/api/docs/guides/decisions)
provides native predicate, choice and score answers through a dedicated public
beta endpoint. It is also available as a separate benchmark engine below.

This repo reproduces that **request/response contract** (see
[`schema.py`](src/decision_wrapper/schema.py)), but instead of a purpose-built
decision model, it answers questions by calling chat models already available
through the Copilot CLI's model routing.

## Why wrap the Copilot CLI instead of a raw completions API?

The core backend uses Copilot's authenticated model routing without requiring
separate provider API keys. Copilot model IDs and reasoning controls are not
assumed to be interchangeable with public provider endpoints. The backend
([`backend.py`](src/decision_wrapper/backend.py)) shells out to:

```
copilot -p "<prompt>" --model <model> --reasoning-effort <level> \
  --silent --allow-all-tools --available-tools= --disable-builtin-mcps --no-custom-instructions \
  --output-format json --log-level none --usage-output-file <usage-json-path> \
  -C <scratch-dir>
```

and parses the JSONL event stream for the final `assistant.message` and the
`result` event's latency block (`totalApiDurationMs`). Actual AI-credit usage
comes from `totalNanoAiu` in the CLI's usage-output file; warm mode reads
`assistant.usage.data.copilotUsage.totalNanoAiu` through the SDK.
One AI credit is 1,000,000,000 nano-AIU. Legacy `premiumRequests` and SDK `cost`
fields are **not** AI credits and are never converted into them.

Timed-out or cancelled calls terminate and reap their subprocess before
releasing the concurrency slot, so abandoned calls do not slow later requests.
Python callers can replace the default flags via `CopilotCliBackend(extra_args=...)`,
including an empty list to opt out of the MCP/custom-instruction defaults.
**Tools remain unavailable in both cold and warm modes**, including with explicit
extra arguments: `--available-tools=` is enforced, and attempts to enable tools
or terminate option parsing through `extra_args` fail before starting the CLI.
The cold CLI requires `--allow-all-tools` for non-interactive turns; this grants
permission but does not make any tools available. A scratch directory is not a
sandbox. The HTTP server has no built-in authentication: keep its default
loopback binding, or put authentication/access controls in front of it before
allowing network clients. Untrusted requests can still consume model credits.

**Important, measured caveat:** the CLI is a full coding-agent harness, not a
bare completions endpoint. In one-shot mode, every call pays a large fixed system-prompt
overhead (~15-20k tokens and several seconds of wall time) *regardless of how
trivial the question is*. Actual credit usage depends on the model's token
prices, output length, and cache hits, not a flat request multiplier.
This is, empirically, exactly the "chat model vs. decision model" cost
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
  "usage": { "samples": 6, "ai_credits": null, "api_duration_ms": 9400.0 },
  "latency_ms": 18230.4
}
```

`reasoning_effort` ("thinking mode") accepts `none | minimal | low | medium | high |
xhigh`, passed straight through to `copilot --reasoning-effort`.
The default remains `low`; support for each level depends on the selected model.
For `none`/`minimal` in warm mode, the wrapper checks the runtime's advertised
levels and applies the setting through the SDK's reasoning-only RPC before sending
the prompt. Unsupported levels fail explicitly rather than falling back.

`usage.ai_credits` reports actual metered usage. It is `null` when accounting
is unavailable or incomplete, **not zero**. Completed calls that return an
unparseable answer still contribute their consumed credits. The old
`usage.premium_requests` field is retained for legacy-client compatibility only;
it is not the current billing metric.

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

### Warm mode (optional)

Warm mode keeps one Copilot CLI process running and communicates with it through
the official Python SDK. It still uses Copilot's model routing, not a raw
completions API. Every sample opens a **new, tool-free session** with memory and
custom instructions disabled, then disconnects and deletes it. This provides the
fresh-conversation behavior of `/clear` without restarting the process or sharing
conversation history between votes.

Warm calls also disable interactive questions and eager PowerShell preparation.
Requires Python 3.11+, a CLI that supports these flags, and the optional SDK extra:

```bash
pip install -e ".[dev,warm]"
decision-wrapper serve --port 8099 --warm
decision-wrapper decide examples/request.json --samples 3 --warm
decision-wrapper benchmark datasets/sample_benchmark.jsonl \
  --models gpt-5.4-mini,gpt-6-luna,mai-code-1.1-flash \
  --reasoning-efforts high --samples 1 --warm
```

For the server, `DECISION_WRAPPER_WARM=1` also enables warm mode. The process starts
at server startup and stops at shutdown; the benchmark shares it across all items
and models. A one-shot `decide --warm` shares it across that request's questions
and samples only. Startup is outside the reported decision/benchmark-item latency;
the first item can still be slower while runtime/provider caches warm up.

Timeouts and cancellations abort the affected sample and clear its session.
If session creation itself stalls beyond its timeout and a bounded cleanup wait,
the warm runtime is stopped rather than leaving an orphan session.
Model, reasoning effort, sample count, vote aggregation, and response shape are
unchanged. Actual credit usage is reported in `usage.ai_credits`; warm mode also
reports SDK input-token counts in `usage.prompt_tokens`. Cache hits and smaller
prompts can reduce credit usage, but there is no flat per-call credit charge.
Without `--warm`, the existing one-shot backend is used.

Illustrative local measurements on Windows (2026-10-05), using the three bundled
toy items, reasoning `high`, and one sample per question:

| Model | One-shot median | Warm median |
| --- | --- | --- |
| mai-code-1.1-flash | 13.54 s | 6.21 s |
| gpt-5.4-mini | 11.37 s | 3.96 s |
| gpt-6-luna | 11.25 s | 4.03 s |

These are separate small runs, not a controlled statistical comparison.
All nine warm classifications matched the toy labels. Warm process startup
(roughly 1-3 s in local probes) is excluded from the item medians; inference and
fresh-session setup still cost seconds. CLI flags alone did not show a consistent
latency improvement in a separate paired probe.

### Local decision models: paired English/German comparison

`datasets/bilingual_benchmark.jsonl` contains 20 authored cases, each in English
and German: routing, boolean checks (including negation and numeric thresholds),
and ordinal severity. This is a paired **smoke benchmark**, not translated
RewardBench or a standardized multilingual leaderboard.

The comparison uses the same multilingual checkpoint for both languages:
`convaiinnovations/laya-multilingual`, `fastino/GLiNER2.5-multi-Decide`, and
`StrandsAgents/strands-decider-2B-hobson-v21`. Mini with reasoning `none` is the
warm Copilot reference. Local models use their native heads, not the Copilot
letter/vote engine. Native distributions and confidence are retained verbatim;
GLiNER's selected-label confidence is not presented as a complete calibrated
distribution. Native score probabilities are mapped from ordinal indices to
the request's numerically sorted rubric keys, using argmax for discrete accuracy.

Install Laya and GLiNER in the project environment:

```powershell
$env:PIP_INDEX_URL = "https://packagefeedproxy.microsoft.io/pypi/simple"
$env:PIP_EXTRA_INDEX_URL = ""
.\.venv\Scripts\python.exe -m pip install -e ".[dev,warm,local]"
.\.venv\Scripts\python.exe -m decision_wrapper.benchmark.local --engine laya --output benchmark-results\bilingual
.\.venv\Scripts\python.exe -m decision_wrapper.benchmark.local --engine gliner --output benchmark-results\bilingual
.\.venv\Scripts\python.exe -m decision_wrapper.benchmark.local --engine copilot --output benchmark-results\bilingual
```

Strands uses newer Transformers/PEFT versions and needs a **separate environment**.
When its distribution is absent from the approved feed, build the official
[source repository](https://github.com/strands-labs/strands-decider) locally;
all package dependencies still resolve through that feed:

```powershell
New-Item -ItemType Directory -Path benchmark-results\runtimes -Force
git clone https://github.com/strands-labs/strands-decider.git benchmark-results\runtimes\strands-decider
.\.venv\Scripts\python.exe -m venv benchmark-results\runtimes\strands-env
benchmark-results\runtimes\strands-env\Scripts\python.exe -m pip install -e benchmark-results\runtimes\strands-decider -e .
benchmark-results\runtimes\strands-env\Scripts\python.exe -m decision_wrapper.benchmark.local --engine strands --output benchmark-results\bilingual
```

Record the source commit with `git -C benchmark-results\runtimes\strands-decider
rev-parse HEAD`. Model artifacts download from Hugging Face; model checkpoint
revisions, package versions, dataset hash and randomized item order are recorded
in each run. The default is CPU with four Torch threads. Each model loads once,
then receives two unscored warmups; loading/download and warmup time are excluded
from item p50/p95. Run local models sequentially to avoid mutual CPU contention.
Reports include every raw answer, per-language accuracy, errors and nearest-rank
p95. Failures count against accuracy and are excluded from successful-call
latency percentiles. Copilot AI credits are metered, with warmup costs recorded
separately; local inference consumes zero Copilot credits, not zero hardware or
electricity costs. Twenty cases per language cannot establish calibration or
a reliable general quality ranking. These are benchmark adapters only; the
existing Copilot server contract and default backend are unchanged.

### Native OpenAI Decisions comparison

`--engine openai` calls the actual **`POST https://api.openai.com/v1/decisions`**
endpoint using the existing `httpx` dependency, not Responses structured outputs
or the Copilot voting engine. The currently supported model is `gpt-6-luna`.
The adapter maps `noul` to `predicate`, preserves choice values/descriptions,
and maps numerically sorted score rubrics to ordered levels. Native probabilities
and the provider's separate confidence field are retained. Discrete score accuracy
uses posterior argmax, not the rounded probability-weighted mean.

Supply `OPENAI_API_KEY` only in the invoking process. For an authorized key in
the clipboard, this example never prints it or writes it to a file:

```powershell
$env:OPENAI_API_KEY = (Get-Clipboard -Raw).Trim()
try {
    .\.venv\Scripts\python.exe -m decision_wrapper.benchmark.local --engine openai --output benchmark-results\openai-bilingual
    if ($LASTEXITCODE -ne 0) { throw "OpenAI bilingual comparison failed" }
    .\.venv\Scripts\python.exe -m decision_wrapper.benchmark.local --engine openai --jevbench-root benchmark-results\runtimes\jevbench --output benchmark-results\openai-jevbench
} finally {
    Remove-Item Env:\OPENAI_API_KEY
}
```

These commands send the benchmark states, instructions and allowed answers to
OpenAI; gold answers, repository instructions and source files are not sent.
HTTP errors expose status and safe error codes, never provider messages or
authorization headers. Redirects are not followed. Refusals/invalid outputs
count as failures, and their delivered raw responses and reported usage remain
recorded. Authorization, quota and rate-limit failures stop the run without
automatic retries; incomplete results are marked as such.

The endpoint has **no documented reasoning-effort parameter**. Do not label this
configuration "Luna none" or "Luna low": the runner's default `none` is only an
internal placeholder and is not sent to OpenAI. Other reasoning settings and
model overrides are rejected before a paid request.

OpenAI billing is separate from Copilot AI credits. Reports preserve token usage
and show a **base-rate USD estimate** using the published Decisions input price
of $0.10 per million tokens, including warmups. The endpoint has no output-token
charges. This is not an invoice: regional premiums, long-context multipliers and
account-specific adjustments are excluded. Missing accounting stays unknown;
`ai_credits` is not applicable. The existing `serve`, `decide` and grid benchmark
commands continue using Copilot; this integration is benchmark-only.

### Published JevBench comparison

The same runtime adapter can evaluate the official
[JevBench source repository](https://github.com/fstandhartinger/jevbench).
The pinned checkout used here publishes **231 tasks** in `easy`, `original` and
`hard`, covering 18 families and all three question types. Its manifest identifies
the corpus as `jevbench::v1.2`. The live v1.6 leaderboard advertises a different
300-item public set plus sealed items; those are not the published corpus in this
checkout. This run is a **public-corpus comparison**, not a reproduction of the
current sealed-inclusive leaderboard or its composite score.

Install the official scorer in both existing runtime environments, using the
same approved-feed configuration as above:

```powershell
git clone https://github.com/fstandhartinger/jevbench.git benchmark-results\runtimes\jevbench
git -C benchmark-results\runtimes\jevbench checkout --detach bb05a335bc809e61b20c0f745d25499a82b326fc
.\.venv\Scripts\python.exe -m pip install -e benchmark-results\runtimes\jevbench
benchmark-results\runtimes\strands-env\Scripts\python.exe -m pip install -e benchmark-results\runtimes\jevbench
.\.venv\Scripts\python.exe -m decision_wrapper.benchmark.local --engine laya --jevbench-root benchmark-results\runtimes\jevbench --output benchmark-results\jevbench-run
.\.venv\Scripts\python.exe -m decision_wrapper.benchmark.local --engine gliner --jevbench-root benchmark-results\runtimes\jevbench --output benchmark-results\jevbench-run
benchmark-results\runtimes\strands-env\Scripts\python.exe -m decision_wrapper.benchmark.local --engine strands --jevbench-root benchmark-results\runtimes\jevbench --output benchmark-results\jevbench-run
.\.venv\Scripts\python.exe -m decision_wrapper.benchmark.local --engine copilot --jevbench-root benchmark-results\runtimes\jevbench --output benchmark-results\jevbench-run
```

For other Copilot models or reasoning levels, use explicit overrides. Give each
configuration its own output directory; the defaults remain Mini with `none`:

```powershell
.\.venv\Scripts\python.exe -m decision_wrapper.benchmark.local --engine copilot --model gpt-6-luna --reasoning-effort none --jevbench-root benchmark-results\runtimes\jevbench --output benchmark-results\jevbench-luna-none
.\.venv\Scripts\python.exe -m decision_wrapper.benchmark.local --engine copilot --model gpt-6-luna --reasoning-effort low --jevbench-root benchmark-results\runtimes\jevbench --output benchmark-results\jevbench-luna-low
```

Overrides apply to both warmups and timed calls, are recorded per item and per
run, and are rejected for native local engines. Unsupported runtime reasoning
levels fail explicitly; they never fall back to another setting.

Use a fresh output directory for each comparison; per-item files are created
exclusively, flushed and synced before the next call. All models use the same
seeded task order, one sample and two unscored warmups. Local models should run
sequentially to avoid mutual CPU contention. The adapter verifies the source
checkout, task counts, file hashes (allowing Windows CRLF) and official canonical
hashes before inference. Structured states are rendered as the same Unicode JSON
for every model. No reference answer is sent to the model; no translations are
added, and the published corpus has no per-task language annotations.

Accuracy, ordinal expected-value MAE, Brier, ECE, family breakdowns and paraphrase
consistency come directly from `jevbench.scoring` and `jevbench.summarize`.
Ordinal accuracy uses argmax, while MAE uses the native expected level.
GLiNER is scored label-only and receives no invented distribution or calibration
metrics. Copilot's one-sample vote shares are not native likelihoods, so its
calibration metrics are not like-for-like with native local distributions.
Official latency percentiles include all attempts and interpolate; the adjacent
runtime summary reports successful-call latency separately.

Context behavior is recorded per model: Laya uses an 8192-token state window with
its native instruction/option budgets, GLiNER retains native window/truncation
behavior, and Strands refuses inputs beyond its native window. Refusals and invalid
answers count against accuracy. Three consecutive runtime errors stop the model's
run; unattempted tasks stay unattempted and the official summary marks it
incomplete. Metered Copilot AI credits, including separate warmup usage, accompany
the official summary; unknown accounting stays unknown. USD costs are not inferred
from a local inference result.

**Benchmark across models x reasoning-effort modes:**

```bash
decision-wrapper benchmark datasets/sample_benchmark.jsonl \
  --models gpt-5.4-mini,gpt-6-luna,mai-code-1.1-flash \
  --reasoning-efforts low,medium,high \
  --samples 3 --out-dir benchmark-results/
```

This runs every dataset item through every `(model, reasoning_effort)`
combination and reports, per combination: accuracy (vs. the dataset's
`expected` answers), p50/p95 latency, and average/total `ai_credits`
("cost"). Results are printed as a table and written to
`benchmark-results/report.{csv,md}`.

Dataset format (`JSONL`, one object per line):

```json
{"state": "...", "questions": {"name": {"type": "choice", "instructions": "...", "criteria": {...}}}, "expected": {"name": "expected_label"}}
```

## Benchmark provenance: this harness vs. the published Jev benchmarks

This repo ships a **3-item toy dataset**, a **40-row authored EN/DE smoke dataset**
and a **30-row RewardBench-derived sample**. It also records comparisons on the
231 public tasks of the pinned JevBench checkout. These are not reproductions of
the full benchmarks used in TypeSafe AI's and the CMU study's evaluation of Jev:

| Benchmark | What it tests | Jev score | GPT-6 score |
| --- | --- | --- | --- |
| RewardBench | preference judging | 92.2% | 93.5% |
| HaluEval | factuality | 87.5% | 86.7% |
| Answer adjudication | custom | 94.0% | 96.7% |
| JudgeBench | derivation checking | 78.6% | 93.1% |
| RM-Bench (hard pairs) | style-adversarial | 74.8% | 94.6% |

(Carnegie Mellon, arXiv 2609.26550 — JEV cost $0.044 / 1k judgments at 0.152s vs. GPT-6's $12.182 / 1k at 1.885s.)

Reproducing these against this CLI-backed wrapper is possible (dataset loaders for RewardBench/HaluEval/JudgeBench/RM-Bench would convert each into this repo's `{state, questions, expected}` JSONL format), but note the cost/time tradeoff: these datasets run from ~2,000 to ~18,000 items. Each item requires Copilot-CLI agent turns whose AI-credit usage depends on metered tokens and whose latency depends on warm/one-shot mode and sampling. A full RewardBench sweep can take many hours and a substantial AI-credit budget; a stratified sample is the practical starting point.

### AI-credit accounting

AI credits are based on model-specific input, output, and cached-token prices.
The wrapper uses the runtime's actual nano-AIU totals rather than a pricing table
or legacy request multipliers. Reports show `avg_ai_credits` and
`total_ai_credits`; if any item has incomplete accounting, the combination's
credit totals are unknown (blank in CSV), not zero or a partial sum.

Historical reports that recorded only premium-request counters cannot be
converted into AI-credit costs. Their accuracy and latency measurements remain
usable, but any billing comparison needs actual nano-AIU data.

See [GitHub's usage-based billing documentation](https://docs.github.com/en/copilot/concepts/billing-and-usage/individuals/billing).

## Published result artifacts and licenses

For the full English results report, methodology and interpretation, see
[Benchmark Results Report](BENCHMARK-REPORT.md).

The following measured runs are explicitly allowlisted in `.gitignore`; new
runs remain local until separately reviewed. Only the named JSON, JSONL and CSV
files are included, not the rest of the working directory:

| Run | Scope |
|---|---|
| [`bilingual-20261005`](benchmark-results/bilingual-20261005) | 20 matched EN/DE cases; Laya, GLiNER, Strands and Copilot Mini |
| [`jevbench-20261005-2234`](benchmark-results/jevbench-20261005-2234) | 231 public JevBench tasks; local models and Copilot Mini |
| [`jevbench-luna-20261005-2351`](benchmark-results/jevbench-luna-20261005-2351) | Same tasks with Copilot Luna `none` and `low` |
| [`openai-decisions-20261009-105643`](benchmark-results/openai-decisions-20261009-105643) | Native OpenAI Decisions on the bilingual and JevBench sets; audited comparison JSON/CSV |
| [`rewardbench-20261005-174839`](benchmark-results/rewardbench-20261005-174839) | Stratified 30-row preference smoke comparison; historical credit accounting remains unknown |

For the published 231-task corpus, official-scoring results are:

| Configuration | Correct | Accuracy | p50 | p95 |
|---|---:|---:|---:|---:|
| Laya multilingual, CPU | 108/231 | 46.8% | 0.167 s | 7.029 s |
| GLiNER multilingual, CPU | 126/231 | 54.5% | 0.221 s | 17.987 s |
| Strands 2B v21, CPU | 175/231 | 75.8% | 2.091 s | 36.270 s |
| Copilot Mini `none` | 190/231 | 82.3% | 2.194 s | 2.559 s |
| Copilot Luna `none` | 206/231 | 89.2% | 2.499 s | 3.509 s |
| OpenAI Decisions, Luna | 208/231 | 90.0% | 0.180 s | 0.255 s |
| Copilot Luna `low` | 223/231 | 96.5% | 2.723 s | 4.970 s |

These runs used the same tasks and scorer, but occurred on different days and
use unpinned provider model aliases. Native distributions and Copilot's
one-sample vote shares are not like-for-like calibration measurements. OpenAI
has no documented reasoning-effort setting. See the run metadata for revisions,
context policies, costs, threads and warmups; this is not the current
sealed-inclusive leaderboard.

The only publication edits to historical measurements are replacing absolute
user-profile dataset paths with repository-relative references. Latencies,
answers, probabilities, dataset hashes and billing measurements are unchanged.
Future native benchmark metadata uses relative paths (or just the filename for
external datasets); this does not sanitize arbitrary private dataset contents.

Environments, runtime/source checkouts, caches, `.env` files, Parquet downloads
and model weights are excluded. Do not force-add `benchmark-results/runtimes`
or distribute a ZIP of the entire working directory.

Project code and authored examples use [MIT](LICENSE). Imported data retains
its original terms; the RewardBench sample is **ODC-By, not MIT**.
See [THIRD-PARTY.md](THIRD-PARTY.md) and [datasets/LICENSE](datasets/LICENSE) for
attribution, transformations and limitations. These notices do not certify
individual upstream content rights or private provider/employer agreements.

## Limitations

- Latency is dominated by Copilot CLI agent-turn overhead (seconds per
  call), not model inference time — this wrapper is for evaluating
  *decision quality and relative cost/latency across models and reasoning
  modes*, not for low-latency production routing. For that, a true
  decision-only model (Jev, Laya, or a raw completions API with logprobs)
  will be orders of magnitude faster and cheaper.
- `ai_credits` reflects metered usage, not a flat per-call charge. Credits may
  differ between repeated calls due to token counts and cache reuse.
- Monte-Carlo calibration (vote share) is an approximation of true calibrated
  probability; it's bounded by `1/samples` resolution and the model's actual
  consistency across resamples.

## Tests

```bash
pytest
```
