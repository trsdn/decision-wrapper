# Third-party data, models and services

The root [MIT license](LICENSE) covers this project's code, documentation and
authored examples, including `datasets/sample_benchmark.jsonl` and
`datasets/bilingual_benchmark.jsonl`. It does not relicense third-party datasets,
model weights, external source checkouts or provider services.

## RewardBench

Contains information from the [RewardBench evaluation dataset by Allen Institute
for AI and its contributors](https://huggingface.co/datasets/allenai/reward-bench),
which is made available under the
[Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/).

This notice applies to `datasets/rewardbench_sample.jsonl` and the derived
measurements under `benchmark-results/rewardbench-20261005-174839/`.
The dataset card identifies the database license as `odc-by`. The license URI
above provides the license terms; this sample is not licensed under this
project's MIT license.

The converter selects a stratified sample from the `filtered` split, rearranges
the chosen/rejected responses into seeded `response_a`/`response_b` positions,
and can truncate responses with an explicit ` [...truncated]` marker. It converts
preference labels into typed choice questions without claiming authorship of the
source prompts or answers. Source subset names and IDs are retained in each row.
See `src/decision_wrapper/benchmark/sources/rewardbench.py` for the transformation.

The source dataset card links the component datasets and their authors:
[dataset details and subset summary](https://huggingface.co/datasets/allenai/reward-bench#dataset-details).
ODC-By covers database rights, not automatically every individual content,
privacy or contractual right (sections 2.3 and 2.4). Those original rights remain
with their holders. The source includes safety/adversarial examples and is
evaluation data, not advice or endorsement.

Reference: [RewardBench: Evaluating Reward Models for Language Modeling,
Lambert et al., 2024](https://arxiv.org/abs/2403.13787).

## JevBench

The comparisons use the [JevBench source repository by fstandhartinger and
contributors](https://github.com/fstandhartinger/jevbench), pinned at
`bb05a335bc809e61b20c0f745d25499a82b326fc`.
Its manifest identifies the evaluated public corpus as `jevbench::v1.2`:
48 easy, 72 original and 111 hard tasks, 231 in total.

The upstream [third-party notice](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/THIRD-PARTY.md)
describes its harness/original decisions as MIT and retains separate rights for
imported datasets. The upstream
[hard-tier documentation](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/datasets/HARD-TIER.md)
identifies the public hard tier as MIT. Do not treat the entire checkout or every
imported task as covered by a single license.

The published result files retain task IDs, expected labels, model answers,
probabilities and measurements, not the original task states/instructions or
the unpublished held-out corpus. The scorer and task texts are obtained from
the original repository for reproduction, not vendored here. These results are
not a reproduction of the current sealed-inclusive leaderboard.

## Local models and external code

| Evaluated checkpoint | Author / source | Declared model-card license |
|---|---|---|
| `convaiinnovations/laya-multilingual` | [Laya](https://huggingface.co/convaiinnovations/laya-multilingual) | Apache-2.0 |
| `fastino/GLiNER2.5-multi-Decide` | [Fastino](https://huggingface.co/fastino/GLiNER2.5-multi-Decide) | Apache-2.0 |
| `StrandsAgents/strands-decider-2B-hobson-v21` | [Strands Agents](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21) | Apache-2.0 |

The Strands runtime used the [official source](https://github.com/strands-labs/strands-decider)
at `3e94e9d84c620ed5a95f1a3310c3decb971e261c`. Actual model revisions and package
versions are recorded in each run. The model-card declarations are not a blanket
license for base models or training data. We publish measurements, not model
weights, installed dependencies, environments or external source checkouts.
Those retain their own notices and terms when obtained separately.

## Provider services and measurement scope

GitHub Copilot and OpenAI retain their own service terms and trademarks. This
project is not affiliated with or endorsed by those providers or dataset/model
authors. No provider credentials are distributed. Copilot AI credits are metered
billing units; OpenAI USD values are published-base-rate estimates, not invoices.
Neither is a license grant for underlying provider materials.

Runs use model aliases that can change over time and took place on different
days. The 40-row bilingual dataset is an authored smoke test, not a validated
multilingual leaderboard. Publication of these artifacts does not certify any
private employer/provider agreement or additional rights in upstream contents.
