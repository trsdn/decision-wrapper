# decision-wrapper

A Jev-style "System One" decision API (`state` + typed `noul`/`choice`/`score`
questions -> calibrated answers with confidence), implemented by shelling out
to the **GitHub Copilot CLI** rather than a raw completions API, because the
models this wraps (`gpt-6-luna`, `gpt-5.4-mini`, `mai-code-1.1-flash`, ...)
are only reachable through the CLI's model routing. See `README.md` for the
full design rationale (why CLI-wrapping instead of an API, why Monte-Carlo
calibration instead of logprobs) before changing the core contract.

## Setup / install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,datasets]"   # datasets extra needed for fetch-rewardbench (pyarrow)
```

Requires the `copilot` CLI installed, authenticated, and on `PATH` — the real
backend (`CopilotCliBackend`) shells out to it; there is no mock/stub mode
for the server or `decide`/`benchmark` CLI commands.

## Test / build commands

```bash
pytest -q                                   # full suite (fast, ~0.1s — all backend calls are mocked)
pytest tests/test_engine.py -q              # one file
pytest tests/test_engine.py::test_choice_question_maps_letters_to_labels -q  # one test
```

Unit tests never invoke the real `copilot` binary — they subclass
`CopilotCliBackend` and override `complete()` (see `FakeBackend` in
`tests/test_engine.py`) to return scripted `CallResult`s. There is no CI
config and no linter configured in this repo yet.

### Manual end-to-end verification (slow, costs Copilot premium requests)

There's no integration-test harness for the real CLI path; verify manually:

```bash
decision-wrapper decide examples/request.json --samples 2   # one DecisionRequest, ~20-30s
decision-wrapper serve --port 8099 &                         # then curl /v1/decisions
decision-wrapper benchmark datasets/sample_benchmark.jsonl --models gpt-5.4-mini --reasoning-efforts low --samples 1
```

Each `copilot` subprocess call takes ~15-25s wall time (it's a full agent
turn, not a bare completions call) — expect multi-item benchmark runs to
take minutes, not seconds. Keep `--concurrency` <= ~4-6 and
`CopilotCliBackend(timeout_s=...)` generous (90s+) for anything with
longer prompts (e.g. RewardBench responses); higher concurrency has caused
CLI timeouts in practice.

## Architecture

Request flow: `schema.DecisionRequest` -> `engine.decide()` -> fan-out one
`evaluate_question()` per question (all questions in a request run
concurrently) -> each question samples `backend.CopilotCliBackend.complete()`
N times in parallel (Monte-Carlo calibration) -> vote tally -> `schema.Answer`.

- **`schema.py`** — the Jev-compatible wire contract. `NoulQuestion` /
  `ChoiceQuestion` / `ScoreQuestion` in, `NoulAnswer` / `ChoiceAnswer` /
  `ScoreAnswer` out. `reasoning_effort` (`minimal|low|medium|high|xhigh`) is a
  per-request field, passed straight through to `copilot --reasoning-effort`.
- **`backend.py`** (`CopilotCliBackend`) — the only place that invokes
  `copilot -p ...`. Builds the CLI args (`--allow-all-tools
  --disable-builtin-mcps --no-custom-instructions --output-format json`, a
  fresh scratch dir + `--session-id` per call), parses the JSONL event
  stream, and pulls the answer out of the **last `assistant.message`
  event's `data.content`** plus cost/latency out of the **`result` event's
  `usage` block** (`premiumRequests`, `totalApiDurationMs`). If the CLI's
  JSONL event names change, this is the only file that needs updating.
  `extract_letter()` here is deliberately noise-tolerant (prefers an
  isolated single-letter token over a letter embedded in a word) because
  the CLI is a chat agent, not a constrained-decoding API — it doesn't
  always return exactly one character even when asked to.
- **`engine.py`** — turns each `Question` into a single-letter classification
  prompt (`_build_options`/`_build_prompt`): choice options get letters
  `A, B, C...` in dict-insertion order; noul is always `Y`/`N`; score rubrics
  are sorted numerically-then-lexically by key before assigning letters, so
  rubric order (low->high) is preserved regardless of dict insertion order.
  Runs `samples` (default 3, set by caller) independent CLI calls per
  question and reports the **vote share** as the probability distribution —
  there's no logprobs path; this *is* the calibration mechanism. A sample
  that fails or returns an unparseable letter is dropped, not fatal, as long
  as at least one sample succeeds.
- **`app.py`** — thin FastAPI wrapper (`POST /v1/decisions`) around
  `engine.decide()`; one process-wide `CopilotCliBackend` instance, sample
  count fixed at process start via `DECISION_WRAPPER_SAMPLES` env var (not
  configurable per-request).
- **`cli.py`** — `typer` app: `serve` (uvicorn), `decide` (one-shot JSON
  file), `benchmark` (grid sweep), `fetch-rewardbench` (dataset converter).
- **`benchmark/runner.py` + `benchmark/report.py`** — sweeps a
  `(model x reasoning_effort)` grid over a JSONL dataset
  (`{state, questions, expected}` per line) and aggregates per-combination
  accuracy / p50 / p95 latency / premium-request cost.
  **Known gap:** `run_benchmark()` currently does not catch per-item
  exceptions — one `BackendError` (e.g. a CLI timeout) aborts the entire
  sweep rather than being recorded as a failure and continuing.
- **`benchmark/sources/rewardbench.py`** — downloads `allenai/reward-bench`
  (HF datasets-server parquet, no auth/`datasets` package needed beyond
  `pyarrow`), stratified-samples across its 23 `subset` categories, and
  **randomizes chosen/rejected into `response_a`/`response_b` per item**
  (seeded by the same `seed` passed to `build_sample`) to avoid positional
  bias — don't remove the randomization when regenerating samples.

## Conventions

- Dataset JSONL rows (both `datasets/sample_benchmark.jsonl` and
  RewardBench-derived files) always have `state`, `questions` (same shape as
  `DecisionRequest.questions`), `expected` (answer-field-name -> expected
  value, compared via `runner._answer_value()` against the response's
  `.value`/`.choice`), and optionally `id`/`meta`.
- `premium_requests` (Copilot's own billing unit) is the cost metric
  throughout — not USD. It varies by model independent of question size
  (observed: ~0.33/call for `gpt-5.4-mini`/`gpt-6-luna`, ~1.0/call for
  `mai-code-1.1-flash`); convert to dollars yourself if needed, there's no
  built-in pricing table.
- When adding a new question type or backend, keep the Jev wire contract
  (`schema.py`) compatible — the point of this repo is matching that
  contract while swapping the backend, not inventing a new one.
