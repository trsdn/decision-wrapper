# Decision-wrapper benchmark results

**Report date:** 2026-10-09

**Scope:** Recorded JevBench, bilingual smoke-test and RewardBench comparisons.
Results come from the saved runs, not new inference calls.

## Executive summary

**Copilot Luna with `low` reasoning delivered the highest accuracy on the tested
JevBench corpus: 223/231 correct, or 96.5%.** Native OpenAI Decisions delivered
90.0% accuracy with a median latency of 180 ms, approximately 14 times faster
than Copilot Luna `none` in the recorded runs.

**Strands 2B was the strongest tested local model**, reaching 75.8% on JevBench
and 39/40 on the authored English/German smoke test. Its CPU tail latency was
substantial. Laya and GLiNER had fast medians but much lower JevBench accuracy.

These findings describe these inputs, adapters and execution settings. They
are not a general model ranking, a production SLA or a reproduction of the
current sealed-inclusive JevBench leaderboard.

## 1. Evaluation setup

| Evaluation | Inputs | Purpose |
|---|---|---|
| Published JevBench corpus | 231 public tasks: 48 easy, 72 original, 111 hard | Broader decision-quality and latency comparison |
| Bilingual smoke test | 20 matched cases in English and German, 40 rows total | Basic behavior across languages and all three question types |
| RewardBench sample | 30 stratified preference pairs covering 23 subsets | Small preference-judging comparison |

The JevBench checkout is pinned at
`bb05a335bc809e61b20c0f745d25499a82b326fc`, with protocol
`jevbench::v1.2`. Its canonical public-dataset SHA-256 is:

```text
dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51
```

The corpus covers 139 choice, 74 boolean (`noul`) and 18 score questions.
Every JevBench configuration used the same task order, seeded with 42, and the
original `jevbench.scoring` and `jevbench.summarize` implementations. Structured
states were rendered as Unicode JSON with sorted keys. Gold answers were not
sent to the models. The published corpus does not annotate task languages;
no translations were added, so JevBench is not the paired EN/DE evaluation.

Copilot ran through a persistent CLI process, with a fresh session for each
sample and one sample per question. Local models loaded once and ran on CPU
with four Torch threads; they ran sequentially to avoid mutual CPU contention.
OpenAI used the dedicated `/v1/decisions` endpoint, not Responses structured
outputs. Each configuration received two unscored warmups.

Loading, initialization and warmup time are excluded from reported item
latencies. JevBench p50/p95 use the official scorer's interpolated percentiles
over all attempts. Bilingual p95 uses the runtime summary's nearest-rank
percentile. All completed JevBench runs had zero runtime and schema errors.

## 2. JevBench: accuracy and latency

**All rows below evaluate the same 231 public tasks.** Latencies are seconds;
p50 is the median and p95 describes the slower tail.

| Configuration | Correct | Accuracy | p50 | p95 |
|---|---:|---:|---:|---:|
| **Copilot Luna `low`** | **223/231** | **96.5%** | 2.723 | 4.970 |
| **OpenAI Decisions, Luna** | **208/231** | **90.0%** | **0.180** | **0.255** |
| Copilot Luna `none` | 206/231 | 89.2% | 2.499 | 3.509 |
| Copilot Mini `none` | 190/231 | 82.3% | 2.194 | 2.559 |
| Strands 2B v21, CPU | 175/231 | 75.8% | 2.091 | 36.270 |
| GLiNER2.5 multilingual, CPU | 126/231 | 54.5% | 0.221 | 17.987 |
| Laya multilingual, CPU | 108/231 | 46.8% | 0.167 | 7.029 |

### What the differences mean

Luna `low` produced **17 more correct answers than Luna `none`**, a gain of
7.4 percentage points. Its median latency was approximately 9% higher.
The gain was concentrated in harder tasks:

| Copilot Luna setting | Easy, out of 48 | Original, out of 72 | Hard, out of 111 |
|---|---:|---:|---:|
| `none` | 48 | 70 | 88 |
| `low` | 48 | 72 | 103 |

Native OpenAI Decisions was approximately **13.9 times faster at p50** than
Copilot Luna `none`, with two additional correct answers. That two-answer
difference is too small to establish a robust general quality advantage.
Compared with Luna `low`, the native API was approximately 15.1 times faster
at p50 but answered 15 fewer tasks correctly.

Among local models, Strands had the highest accuracy but a 36.270-second p95.
The low medians of Laya and GLiNER did not translate into uniformly low tail
latency or strong accuracy on this corpus. Median speed alone is insufficient
for choosing a production decision backend.

## 3. JevBench: provider usage and cost

Totals include the two warmups. **Copilot AI credits and OpenAI USD are
different units and are not converted into one another.**

| Configuration | Copilot AI credits, metered | OpenAI base-rate USD estimate |
|---|---:|---:|
| Copilot Mini `none` | 33.97710 | Not applicable |
| Copilot Luna `none` | 10.94604 | Not applicable |
| Copilot Luna `low` | 11.31483 | Not applicable |
| OpenAI Decisions, Luna | Not applicable | $0.0153873 |
| Local Laya / GLiNER / Strands | 0 | Not applicable |

Luna `low` consumed approximately **3.4% more AI credits** than Luna `none`
in these runs. Both Luna configurations used fewer credits than Mini despite
their higher accuracy. These are observed totals, including cache effects,
not universal per-model prices.

The native OpenAI run reported **153,873 input tokens**, including warmups.
Its USD figure uses the documented Decisions base input rate of **$0.10 per
million tokens**. Reported cached, cache-write and output-token counts were
zero. The estimate is not an invoice and excludes regional premiums,
long-context multipliers and account-specific adjustments.

Local inference consumed no Copilot credits. Hardware, electricity, memory,
installation and operational costs were not measured; local inference is
therefore not described as cost-free.

## 4. English/German smoke test

Each language has the same 20 underlying cases and expected labels. The cases
are authored examples, not an independent multilingual leaderboard.

| Configuration | English correct | German correct | English p50 | German p50 |
|---|---:|---:|---:|---:|
| OpenAI Decisions, Luna | 20/20 | 20/20 | 0.198 s | 0.172 s |
| Copilot Mini `none` | 20/20 | 20/20 | 2.304 s | 2.211 s |
| Strands 2B v21, CPU | 19/20 | 20/20 | 1.507 s | 1.674 s |
| GLiNER2.5 multilingual, CPU | 15/20 | 16/20 | 0.162 s | 0.167 s |
| Laya multilingual, CPU | 10/20 | 11/20 | 0.097 s | 0.108 s |

The native API and Mini answered all 40 examples correctly. Strands missed
one English example. Twenty cases per language cannot establish a reliable
language-specific quality ranking or production confidence thresholds.
**Luna `none` and `low` were not evaluated on this paired set**, so their
German performance must not be inferred from the JevBench results.

Mini's run consumed **4.62954 AI credits**, including warmups. The native API
reported **7,383 input tokens** and a **$0.0007383 base-rate estimate**, also
including warmups. Local hardware costs remain unmeasured.

## 5. RewardBench preference smoke test

This comparison used the same 30 sampled pairs, warm Copilot execution and one
sample per question. Configurations were interleaved by item. Chosen/rejected
responses were placed into seeded `response_a`/`response_b` positions to avoid
a fixed positional advantage.

| Configuration | Correct | Accuracy | p50 | p95 |
|---|---:|---:|---:|---:|
| Copilot Mini `none` | 29/30 | 96.7% | 2.013 s | 3.293 s |
| Copilot Luna `none` | 28/30 | 93.3% | 2.260 s | 2.607 s |
| Copilot Luna `low` | 28/30 | 93.3% | 2.517 s | 6.648 s |

There were no backend failures. Mini's lead is only one example and does not
establish a general preference-judging advantage. This is an unweighted
sample comparison, not the full official RewardBench score.

**AI-credit costs are unknown for these historical runs.** Their legacy
premium-request counters cannot be converted into AI credits. Accuracy and
latency remain usable; the missing billing data is not reported as zero.

## 6. Limits and practical interpretation

The runs were separate executions saved under the `20261005` and `20261009`
artifact groups. Provider model aliases were not pinned to immutable model
revisions. Server load, caching, routing and model updates can affect latency
and quality; this is not a synchronized controlled comparison.

The interfaces also differ. Copilot answers a letter-classification prompt
inside an agent harness. OpenAI Decisions returns native typed answers and
has **no documented reasoning-effort parameter**; it is not a "Luna none"
or "Luna low" configuration. Matching model names do not establish identical
inference behavior across these interfaces.

Copilot's one-sample probability distributions are vote-share proxies, not
native likelihoods. Laya, Strands and OpenAI retain native distributions;
GLiNER provides a selected label and confidence, not a complete distribution.
No missing distribution was synthesized. Score accuracy uses posterior
argmax where a native distribution is available, and vote-share argmax for
Copilot; GLiNER is scored label-only. Official ordinal MAE uses the distribution's
expected level. Retaining native probabilities is not proof of calibration
for a production domain; thresholds need separate validation.

Context handling is backend-specific: Laya uses an 8192-token state window
with native instruction/option budgets; GLiNER retains its native window and
truncation; Strands uses a strict 4096-token window. These settings are recorded
in the run metadata. Public benchmark tasks may overlap with training data;
no sealed holdout or production-domain validation was performed.

**Selection guidance for the measured workloads:**

- **Highest observed accuracy:** Copilot Luna `low`, if multi-second response
  time and hosted inference are acceptable.
- **Low-latency hosted decisions:** Native OpenAI Decisions is the strongest
  candidate in this comparison, with 90.0% accuracy and a 255 ms p95.
- **Local-only inference:** Strands is the quality-leading local candidate,
  but its CPU tail latency needs workload-specific validation.
- **Laya or GLiNER:** Consider them for narrowly scoped, validated
  classifications rather than assuming their fast medians imply broad
  decision quality.

These are candidates for application-specific evaluation, not blanket
deployment recommendations. The native OpenAI integration is currently a
benchmark adapter; the default server still uses Copilot.

## 7. Evidence and attribution

| Artifact group | Contents |
|---|---|
| [Local models and Mini on JevBench](benchmark-results/jevbench-20261005-2234/) | Per-item records, run metadata, comparison JSON/CSV |
| [Luna `none` and `low` on JevBench](benchmark-results/jevbench-luna-20261005-2351/) | Both configurations and combined comparison |
| [Native OpenAI Decisions](benchmark-results/openai-decisions-20261009-105643/) | JevBench and bilingual records; audited comparison against saved Copilot runs |
| [Original bilingual comparison](benchmark-results/bilingual-20261005/) | Laya, GLiNER, Strands and Mini records |
| [RewardBench sample comparison](benchmark-results/rewardbench-20261005-174839/) | Per-item results and aggregate report |

The saved native-API/Copilot JevBench comparison was replayed through the
original scorer: all 924 referenced outcomes matched, with identical task
ordering. The 40 native-API bilingual predictions also matched on replay.
Publication sanitization changed dataset-path metadata, not measured answers,
latencies, distributions, hashes or billing totals.

For dataset/model sources, licenses and redistribution limitations, see
[THIRD-PARTY.md](THIRD-PARTY.md) and [datasets/LICENSE](datasets/LICENSE).
The RewardBench-derived sample and measurements contain information from
[RewardBench](https://huggingface.co/datasets/allenai/reward-bench), made
available under [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/);
they are not relicensed under this project's MIT license. This report does
not certify additional upstream content rights or private provider/employer
agreements.
