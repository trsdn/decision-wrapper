# decision-wrapper

A Jev-style "System One" decision API (`state` + typed `noul`/`choice`/`score`
questions -> calibrated answers with confidence), implemented by shelling out
to the **GitHub Copilot CLI** for its authenticated model routing
(`gpt-6-luna`, `gpt-5.4-mini`, `mai-code-1.1-flash`, ...).
Native local models and OpenAI's dedicated Decisions API are separate benchmark
adapters, not replacements for the default server. See `README.md` and
`BENCHMARK-REPORT.md` before changing the core contract or comparing interfaces.

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
pytest -q                                   # full suite; provider calls are mocked
pytest tests/test_engine.py -q              # one file
pytest tests/test_engine.py::test_choice_question_maps_letters_to_labels -q  # one test
```

Unit tests never invoke the real `copilot` binary — they subclass
`CopilotCliBackend` and override `complete()` (see `FakeBackend` in
`tests/test_engine.py`) to return scripted `CallResult`s. There is no CI
config and no linter configured in this repo yet.

### Manual end-to-end verification (slow, consumes Copilot AI credits)

There's no integration-test harness for the real CLI path; verify manually:

```bash
decision-wrapper decide examples/request.json --samples 2   # one DecisionRequest, ~20-30s
decision-wrapper serve --port 8099 &                         # then curl /v1/decisions
decision-wrapper benchmark datasets/sample_benchmark.jsonl --models gpt-5.4-mini --reasoning-efforts low --samples 1
```

Cold subprocess calls have substantial startup/agent-turn overhead. Use
`--warm` to reuse the process, while retaining a fresh tool-free session per
sample. Multi-item benchmark runs can take minutes. Keep `--concurrency` <= ~4-6 and
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
  `ScoreAnswer` out. `reasoning_effort` (`none|minimal|low|medium|high|xhigh`)
  is a per-request field. Copilot model capabilities are validated; unsupported
  efforts are rejected rather than silently replaced.
- **`backend.py`** (`CopilotCliBackend`) — the only place that invokes
  `copilot -p ...`. Builds the CLI args (`--allow-all-tools --available-tools=
  --disable-builtin-mcps --no-custom-instructions --output-format json`, a
  fresh scratch dir + `--session-id` per call), parses the JSONL event
  stream, and pulls the answer out of the **last `assistant.message`
  event's `data.content`** and latency from the **`result` event's `usage`
  block** (`totalApiDurationMs`). AI credits come from actual session usage
  (`totalNanoAiu / 1e9`), not the legacy premium-request counter.
  `_tool_free_args()` must enforce empty tool availability, including extra
  arguments. Auto-approval permission flags do not expose unavailable tools.
  `extract_letter()` here is deliberately noise-tolerant (prefers an
  isolated single-letter token over a letter embedded in a word) because
  the CLI is a chat agent, not a constrained-decoding API — it doesn't
  always return exactly one character even when asked to.
- **`warm_backend.py`** — optional official SDK runtime (`warm` extra),
  reusing the CLI process with a fresh session per sample. Both startup
  arguments and session `available_tools=[]` enforce tool-free execution.
- **`openai_backend.py`** — native `POST /v1/decisions` adapter used only by
  benchmarks. Preserves provider distributions and token accounting; no
  documented reasoning-effort parameter is sent to this API.
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
  `engine.decide()`; one process-wide cold or warm Copilot backend, sample
  count fixed at process start via `DECISION_WRAPPER_SAMPLES` env var (not
  configurable per-request).
- **`cli.py`** — `typer` app: `serve` (uvicorn), `decide` (one-shot JSON
  file), `benchmark` (grid sweep), and `fetch-rewardbench` (dataset converter).
  Bilingual and JevBench comparisons run through
  `python -m decision_wrapper.benchmark.local`; JevBench uses
  `--jevbench-root` to select the pinned public corpus.
- **`benchmark/runner.py` + `benchmark/report.py`** — sweeps a
  `(model x reasoning_effort)` grid over a JSONL dataset
  (`{state, questions, expected}` per line) and aggregates per-combination
  accuracy / p50 / p95 latency / AI-credit usage.
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
- Copilot usage is measured in **AI credits** (`totalNanoAiu / 1e9`).
  Legacy `premium_requests` fields are retained for compatibility and must
  never be interpreted as credits. OpenAI token-based USD estimates are
  separate; unavailable accounting remains unknown, not zero.
- Only explicitly allowlisted, audited benchmark artifacts belong in Git.
  Do not publish runtime checkouts, environments, model weights, credentials,
  or private paths. Keep dataset licensing separate from the root MIT license.
- When adding a new question type or backend, keep the Jev wire contract
  (`schema.py`) compatible — the point of this repo is matching that
  contract while swapping the backend, not inventing a new one.
