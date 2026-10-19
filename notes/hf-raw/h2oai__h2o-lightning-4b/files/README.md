---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- decision-model
- decision-making
- calibration
- jevbench
- multimodal
- jev
- typed-decisions
---

![H2O Lightning: Decision Models](assets/banner.png)

# H2O-Lightning-4B

H2O-Lightning-4B is a 4B-parameter decision model from H2O.ai, built on
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). It answers typed decision questions about a record (a
document, a ticket, a policy, a conversation) and about images that come with the record (photos,
screenshots, scanned documents, charts). It returns a probability for every option:

- **choice**: picks one of a set of named options;
- **yes/no** (`noul`): gives the probability that a statement is true;
- **score**: gives an ordinal level on a stated scale, with its probability distribution.

It runs on **unmodified vLLM 0.30.0** with a small standard-library shim in front (`h2o_lightning_shim.py`). Each
decision is one forward pass and one output token, so the cost is input tokens only. Since v1.3 the weights read a
prompt in either of two ways, **two-way** (the recommended default) or **one-way**; see
[Two-way reading](#two-way-reading-recommended).

**Try it in your browser:** [live demo on Hugging Face Spaces](https://huggingface.co/spaces/h2oai/h2o-lightning-4b) (text and images; built by the Hugging Face team and running on ZeroGPU).

## #1 on JevBench, October 7–19, 2026

H2O-Lightning-4B v1.1 has been **#1 on the official JevBench Composite Score** of open-weight systems from **October 7
through October 19, 2026**, across **JevBench versions v1.6.1 through v1.7.46**, including every model added in that
time. It **outscores Jev itself: 72.5 to 71.5**, and is ahead of every open 12B, 26B and 31B model on the board.
([leaderboard](https://benchmarkheaven.com/jev-models), read 2026-10-07 and 2026-10-19)

![JevBench v1.6.1 Composite Score, open-weights board, 2026-10-07: H2O-Lightning-4B v1.1 at #1](assets/jevbench_composite_2026-10-07.png)

| | Composite | Intelligence | Calibration | Speed | Cost | $ per 1,000 decisions |
|---|---|---|---|---|---|---|
| **H2O-Lightning-4B v1.1** (4B) | **72.5** | 60.0 | 90.0 | 92.6 | 60.3 | $0.021 (est.) |
| Jev 1.13.0 (reference, hosted) | 71.5 | 63.6 | 90.6 | 91.5 | 54.7 | $0.032 |
| Quyet-1.0-Large (31B) | 71.4 | 73.4 | 90.0 | 86.9 | 50.5 | $0.045 (est.) |

The composite is the equal-weight harmonic mean of the four axes. The 31B models lead on raw intelligence, while this
4B matches their calibration at a fraction of the cost and latency.

| Four axes, against a 31B | Capability against cost |
|---|---|
| ![The four JevBench axes, H2O-Lightning-4B v1.1 (solid) against Quyet-1.0-Large 31B (dashed)](assets/jevbench_four_axes_2026-10-07.png) | ![Capability against cost per 1,000 decisions on the open-weights board, H2O-Lightning-4B highlighted](assets/jevbench_capability_vs_cost_2026-10-07.png) |

### Accuracy, latency and JevBench scores, side by side

Microsoft's [Microsoft-Decision-1 announcement](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/)
compares accuracy, latency and calibration side by side. The chart below keeps Microsoft's accuracy numbers unchanged,
corrects the latency and puts it on one H100 basis, and shows both JevBench v1.6.1 scores. Each model has composite above capability; the animation sorts by composite, capability, latency and accuracy. Capability averages Intelligence and Calibration, excluding speed and cost.

![Accuracy (Microsoft's 36-benchmark comparison), median latency on an H100 basis and JevBench v1.6.1 composite and capability for H2O-Lightning-4B v1.1, Jev 1.13.0, Quyet-1.0-Large, deck-31B, Surogate Rune 26B-A4B, OpenAI Decisions, Microsoft-Decision-1 and Strands-Decider 2B, ordered by composite](assets/accuracy_latency_composite_2026-10-10.gif)

![Vertical comparison of JevBench composite and capability scores, with median decision latency](assets/jevbench_scores_latency_2026-10-10.png)

[Vector SVG](assets/jevbench_scores_latency_2026-10-10.svg) · [PDF](assets/jevbench_scores_latency_2026-10-10.pdf)

*Footnote: we corrected the latency from Microsoft's page. For self-hosted models it used the leaderboard's adjusted
figure (measured time x 2 + 0.15 s, which the board itself labels "assumption, not measured"), so this model appeared
at 210 ms instead of its measured 29 ms. The chart uses the board's measured median (`latency.p50_s_raw` in the
[JevBench v1.6.1 data](https://benchmarkheaven.com/api/jevbench/latest)) on one H100 basis: Quyet-1.0-Large (116 ms)
and deck-31B (127 ms) were timed on an H100 and are unchanged; Surogate Rune was timed on an RTX PRO 6000, which has
about half an H100's memory bandwidth and compute, so its 116 ms is halved to 58 ms (the most favourable conversion for
it); this model was timed by the board on an RTX 5090 at 29 ms, and our own serial H100 runs give 28-32 ms, so 29 ms is
kept. Microsoft-Decision-1 now uses JevBench’s measured API median **460 ms** (replacing Microsoft’s announcement’s 85 ms), with composite **69.1 / API #6**. OpenAI Decisions uses **298 ms** (about 0.30 s), with composite **62.5 / API #8**. Both and Jev include network time and are not rescaled. The accuracy numbers are
Microsoft's own, on benchmarks they chose, so they are likely to favour their model; the JevBench composite is scored on
held-out items that no submitter, us included, has seen. Composite and ranks: [open-weights](https://benchmarkheaven.com/jev-models)
and [API](https://benchmarkheaven.com/jev-models/api) boards, read 2026-10-10.*

**How we got there:**
- **A held-out evaluation set:** independently written evaluation items that are never trained on, used to compare
  candidate models before a release.
- **Rules written down first:** ship / no-ship criteria are fixed before results are seen.
- **No benchmark test items in training:** all training data is deduplicated against every evaluation set, exact
  and near-duplicate.
- **Gap hunting:** comparing against other systems by topic and question type, running adversarial sets, and filling
  the gaps with new, verified data.
- **Data quality first:** real, licence-clean data, with labels from human annotation or agreement among several
  independent strong models.
- **Calibration as a first-class goal:** one forward pass, one output token, and probabilities that mean what they say.

## See it in action

Three live runs on one NVIDIA H100 (vLLM 0.30.0). The operations inbox and the web agent are shown in **real time**:
nothing is sped up or slowed down. DOOM, where the game waits for each answer, is shown on the model's **decision
clock**, which advances only by the model's own time per call. Companies, people and amounts are fictional.

<video src="https://huggingface.co/h2oai/h2o-lightning-4b/resolve/main/assets/demos/demos_reel.mp4" poster="https://huggingface.co/h2oai/h2o-lightning-4b/resolve/main/assets/demos/demos_reel_poster.jpg" controls autoplay muted loop playsinline width="100%"></video>

One panel, three demos in turn: **operations inbox** (1,000 claims against a written policy, 32 at a time on one GPU,
all decided in 63 s), **web agent** (8 real browsers, 192 vendors in 145 s, every decision correct against answers
written beforehand) and **DOOM** (a Freedoom level played to the exit, four decisions per call). Each demo is followed by what was checked afterwards. The clips are excerpts; the full runs follow.

**Full-length runs (uncut), on YouTube.** The same three demos, each run shown complete, with chapters. In DOOM, E1M1 is one
of the two maps the thin controller (aiming, routing) was tuned on; the model itself was never trained on Doom.

[![H2O-Lightning-4B: full-length demo runs on YouTube](https://img.youtube.com/vi/2Qp04Wu0A14/maxresdefault.jpg)](https://youtu.be/2Qp04Wu0A14)

## Measured

**Which weights.** `main` carries the **v1.3.1** weights (pin `--revision v1.3.1`): the v1.3 weights continued for
one more pass over the training mix at a lower learning rate, trained for both attention patterns (see
[Two-way reading](#two-way-reading-recommended)). Earlier revisions stay pinnable: `v1.3.0.5` (the v1.3 weights with
every fix of this card's tooling; tag `v1.3` marks the first v1.3 commit, which fails to load and must not be
pinned), `v1.2.3` (one-way only; the measured configuration from "Measured on one NVIDIA RTX PRO 4500" onward) and
`v1.1` (the revision submitted to the JevBench board; the #1 claim above). The v1.3.1 and v1.3 weights' own results
on the same public items, through the same CLI and shim on one NVIDIA L40S (48 GB), vLLM 0.30.0, through `serve.sh`
(no flags for two-way, `TWO_WAY=0` for one-way; all 231 answered and valid in both modes; no yes/no answer between
0.2 and 0.8 in either mode; client latency p50 38 ms two-way and 39 ms one-way, serial, for v1.3.1):

| JevBench's 231 public items | correct | by tier: easy / standard / hard |
|---|---|---|
| **v1.3.1** (`main`), two-way reading (the default; `bash serve.sh`) | **203 / 231 (87.9 %)** | 48/48 · 70/72 · 85/111 |
| v1.3.1 (`main`), one-way reading (`TWO_WAY=0`; stock generate-mode vLLM) | **198 / 231 (85.7 %)** | 48/48 · 70/72 · 80/111 |
| v1.3 (`--revision v1.3.0.5`), two-way | 202 / 231 (87.4 %) | 48/48 · 70/72 · 84/111 |
| v1.3 (`--revision v1.3.0.5`), one-way | 198 / 231 (85.7 %) | 48/48 · 70/72 · 80/111 |
| v1.2.3 (`--revision v1.2.3`), one-way | 205 / 231 (88.7 %) | 48/48 · 71/72 · 86/111 |

On JevBench's public items v1.3.1 is one item above v1.3 two-way and equal one-way (paired against the v1.3 runs:
two-way gains 1 and loses 0, one-way gains 1 and loses 1, exact McNemar p = 1.0 for both: no difference; the two
revisions give the same top answer on 229 of 231 items two-way and 228 of 231 one-way), and 2 (two-way) and 7
(one-way) items behind v1.2.3 (paired against the v1.2.3 run: two-way gains 4 and loses 6, p = 0.75; one-way gains 4
and loses 11, p = 0.12: neither difference is significant). Two-way is ahead of one-way on them. Pin `--revision v1.2.3` for the measured configuration below (`--revision v1.1` for the board's weights); use
`main` for two-way reading.

**Decision Index 0.3 public suite**, our own runs with the official scoring kit, at the shipped temperatures and
yes/no floor, with `max_state_tokens` raised to 1000000 for these runs (shipped: 32000) so that no record was trimmed:
v1.3.1 text **57.86 two-way / 56.72 one-way** (140,178 scored rows); vision **67.13 two-way / 67.51 one-way** (public
vision suite, 12,210 rows). Every number below was read through the package's own `serve.sh` (two-way rows: the
shipped shim on the pooling engine; one-way rows: `TWO_WAY=0`), each release by its own package (v1.2.3 and v1.1
one-way only; these are our runs of the public Decision Index 0.3 kit with the official scorer, not a leaderboard
result). v1.3.1 is ahead of v1.3 on text in both modes and behind it on vision two-way by 0.73, the one measured
regression of this revision:

| Decision Index 0.3 public suite, our runs | text (140,178 rows) | vision (12,210 rows) |
|---|---|---|
| **v1.3.1** (`main`), two-way | **57.86** | **67.13** |
| v1.3.1 (`main`), one-way | 56.72 | 67.51 |
| v1.3 (`--revision v1.3.0.5`), two-way | 57.06 | 67.86 |
| v1.3 (`--revision v1.3.0.5`), one-way | 55.88 | 67.57 |
| v1.2.3 (`--revision v1.2.3`), one-way | 51.41 | 63.09 |
| v1.1 (`--revision v1.1`, the JevBench board's weights), one-way | 51.35 | no image input in that package |
| `image` tag (the ImageJevBench entry; weights 10c16498, run with the current tag's package), one-way | 52.25 | 64.27 |

The score is the kit's DI index (0-100),
not percent correct; the leaderboard's private tests were not run. The v1.3 card's two-way cells (57.10 and 68.00)
and v1.3.1's first reads (57.90 and 67.28) came from a separate pooling client with the same readout; the `serve.sh`
re-reads above differ from them by at most 0.15. Both modes were scored with the same temperatures
and yes/no floor, the released one-way values (choice 0.75, yes/no 0.8, score 0.65, floor 0.801, images 0.65):
nothing was fitted separately for two-way, so `serve_config.json`'s `two_way` block carries those values as an
interim calibration, and a two-way fit may follow in a later revision. A parity check of the pooled readout against
vLLM's log-probability readout with the same (one-way) attention, measured on the v1.3 weights (the readout code is
unchanged in v1.3.1), agreed on 99.32 % of 293 vision questions (median |dp| 0.0005) and at the 99.5-99.8 % bar on
text, so the two-way numbers are a property of the attention mode, not of the readout.

Measured on one **NVIDIA RTX PRO 4500 Blackwell (32 GB)** with `serve.sh` as shipped (vLLM 0.30.0; text temperature
0.8, image temperature 0.65, images up to 1,048,576 pixels), **v1.2.3 weights**:

| JevBench | public items | result |
|---|---|---|
| JevBench (text) | the 231 public items, JevBench's own CLI | **205 / 231 (88.7 %)**: easy 48/48 · standard 71/72 · hard 86/111 |
| ImageJevBench | the 8 published examples (the only public items released with images) | **8 / 8** |
| image validation sets | 1,877 questions over 16 public licence-clean sets (table below) | **87.4 %** |

Details follow.

### Text: JevBench's public set

From **JevBench's own CLI**, with `--adapter typesafe`, on the 231 public items (`datasets/public/easy.jsonl`,
`original.jsonl` and `hard.jsonl`; dataset hash `dc3995d8…`) at JevBench commit `bb05a33`, with the commands below.

The measured H100 was an **80 GB SXM5**, with a **700 W configured power limit** (board692-2G520-0200-000). These are not H100 PCIe measurements; the timing values below are unchanged.

| | RTX PRO 4500 Blackwell 32 GB | 1x H100 80 GB SXM5 |
|---|---|---|
| correct | 205 / 231 (88.7 %) | 205 / 231 (88.7 %) |
| by tier: easy / standard / hard | 48/48 · 71/72 · 86/111 | 48/48 · 71/72 · 86/111 |
| answered and valid | 231 / 231 | 231 / 231 |
| yes/no answers with P(yes) strictly between 0.2 and 0.8 | 0 of 74 | 0 of 74 |
| mean input tokens per decision | 654 | 654 |
| client latency p50 / p95, standard 72 items, serial | 29.6 ms / 29.8 ms | 28.8 ms / 29.3 ms |
| client latency p50 / p95, all 231 items, serial | 30.2 ms / 225.6 ms | 29.8 ms / 69.9 ms |
| server-reported latency p50 / p95, all 231 items, serial | 30.0 ms / 225.3 ms | 29.5 ms / 69.5 ms |
| raw LoRA: correct, separate run | 205 / 231 (88.7 %) | 204 / 231 (88.3 %) |
| raw LoRA: client latency p50 / p95, standard 72 items, serial | 57.3 ms / 58.0 ms | 69.2 ms / 72.1 ms |
| raw LoRA: client latency p50 / p95, all 231 items, serial | 58.0 ms / 272.6 ms | 70.7 ms / 148.1 ms |
| raw LoRA: server-reported latency p50 / p95, all 231 items, serial | 57.9 ms / 272.4 ms | 70.4 ms / 147.7 ms |

**Latency rechecked October 10, 2026, using the exact v1.2.3 release**
(`acaf0d4ea251e54de928c75ef4352670d33192d3`). Each GPU processes the same
**231 distinct public requests once**, one request at a time, over a persistent
localhost HTTP connection. **Server startup, compilation, and 10 separate warmup
requests are excluded from both client and server timing statistics.** No
cached repeat passes are included; the measured pass had **zero prefix-cache
hits** on both GPUs. Client latency covers the complete shim request and
response; server latency is the shim's reported processing time. Both use
BF16, 40,960-token context, default CUDA graphs and prefix caching, with matched
vLLM 0.30.0 / Torch 2.13.0 cu129, Transformers 5.19.0 and tokenizers 0.23.3.

**Separate raw LoRA measurement:** the additional rows use the released adapter from
that same pinned v1.2.3 revision on `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with its standard alpha/rank scaling
(128/64), deployment multiplier 1.0, BF16 and `max_num_seqs=64`.
The preceding rows measure the merged checkpoint with its shipped server defaults.
These are two deployment configurations and separate weight representations.
The LoRA run uses the same pinned runtime, calibration, 40,960-token context,
CUDA graphs, prefix caching, 231 distinct serial requests, and ten separate
warmups excluded from both timing statistics. Native loading and activation of
all 496 adapter tensors across 248 decoder projections were checked; separate
base→adapter→base controls verified an adapter effect and restored base routing.
The H100 LoRA run scored 204/231, compared with 205/231 on RTX, using the same pinned adapter and calibration.

For the merged-model rows, the wider p95 is across different prompt lengths, not parallel traffic: the
standard prompts contain 110–163 input tokens, while the hardest include up to
3,952. That longest request took 316.5 ms on RTX and 85.9 ms on H100. These are single-request timings, not throughput or concurrent-load
measurements; latency depends on the GPU and input length.
JevBench's own measurement of v1.1: median 29 ms (board data, latency.p50_s_raw).

**Identity check for the merged-model evaluator:** a correct setup reproduces 205/231 with the tier split above. Five hard items
sit on near-ties (top two within 0.03) that GPU numerics can flip. Four are wrong here:
`hard-opus-a-temporal_numeric-07`, `hard-opus-a-temporal_numeric-12`, `hard-opus-c-temporal_numeric-04` and
`hard-sol-b-temporal_numeric-03`. One is right: `hard-sol-c-judge_hard-13`.

### Images

**(a) The published ImageJevBench examples: 8 / 8.** ImageJevBench's own harness is not public, and only 8 of its items
are published with their images (on the benchmark's page: two everyday photos, four screenshots with five labelled
click markers, one geometry diagram, one financial table). We read all 8 through this API: 8 correct. These 8 items
are not a benchmark score.

**(b) Public validation sets with human or source ground truth.** Multiple-choice questions we built from each
dataset's own annotations (human labels, boxes, receipt fields, table cells, a chart's own data); none come from a
model. None of these images or questions was used for training. Read through this API, one image per request, as a
data URI in `state`.

| set (licence) | what is asked | n | correct | ECE |
|---|---|---:|---:|---:|
| Open Images V7 validation (CC BY 4.0 annotations) | is there a ‹class› in the photo (human-verified labels) | 150 | 88.0 % | 0.034 |
| Open Images V7 validation | how many ‹class› in the photo (human boxes, 2-8) | 150 | 63.3 % | 0.121 |
| VizWiz validation (CC BY 4.0) | can the question be answered from this photo | 120 | 84.2 % | 0.057 |
| VizWiz validation | the answer (6+ of 10 annotators) | 120 | 100.0 % | 0.015 |
| TextVQA validation (CC BY 4.0) | text in the photo (6+ of 10 annotators) | 120 | 99.2 % | 0.024 |
| CORD v2 test + validation (CC BY 4.0) | receipt total | 95 | 99.0 % | 0.014 |
| CORD v2 test + validation | one item's price | 29 | 96.5 % | 0.043 |
| CORD v2 test + validation | number of item lines | 113 | 85.0 % | 0.050 |
| TAT-QA dev (CC BY 4.0) | financial-report table questions (the table shown as an image) | 120 | 95.8 % | 0.037 |
| DocLayNet test (CDLA-Permissive-1.0) | which section heading is on the page | 120 | 100.0 % | 0.011 |
| DocLayNet test | how many tables are on the page (0-3) | 120 | 75.8 % | 0.057 |
| DocLayNet test | does the page contain a picture | 120 | 90.0 % | 0.039 |
| Our World in Data charts (CC BY 4.0) | which country is highest or lowest (the chart's own data) | 100 | 98.0 % | 0.058 |
| Our World in Data charts | did a value rise between two years | 100 | 95.0 % | 0.069 |
| Our World in Data charts | roughly what is a value | 100 | 98.0 % | 0.066 |
| Rico app screens (CC BY 4.0) | which of 5 marked elements reaches a goal (human widget captions) | 200 | 65.0 % | 0.094 |
| **all** | | **1,877** | **87.4 %** | **0.017** |

ECE: 10-bin expected calibration error of the top answer's probability.

## Routing thresholds

When a decision is uncertain you may want to hand it to a stronger model or a person. This table shows that trade-off:
at a confidence threshold, how many questions get **escalated** (top probability below the threshold) and how many
**slip through confidently wrong** (top probability at or above it, but wrong), as a share of all questions.

Measured (v1.2.3 weights, one-way) on 3,474 held-out questions from two public third-party decision sets, with no overlap with our training
data; text temperatures as shipped (choice 0.75, yes/no 0.8, score 0.65). The thresholds were fixed before the
analysis was run.

| question type (n) | threshold | escalated | accuracy on the rest | confidently wrong |
|---|---|---|---|---|
| choice (1,854) | 0.8 | 34.1 % | 95.5 % | 3.0 % |
| choice | 0.9 | 47.5 % | 98.2 % | 1.0 % |
| yes/no (1,367) | 0.8 | 40.8 % | 96.0 % | 2.3 % |
| yes/no | 0.9 | 59.5 % | 99.1 % | 0.4 % |
| score (253) | 0.8 | 46.2 % | 96.3 % | 2.0 % |
| score | 0.9 | 59.7 % | 98.0 % | 0.8 % |
| all (3,474) | 0.8 | 37.7 % | 95.8 % | 2.6 % |
| all (3,474) | 0.9 | 53.1 % | 98.5 % | 0.7 % |

Without a threshold, accuracy is 82.7 % (choice), 84.7 % (yes/no) and 80.2 % (score).

The per-question predictions behind this table, the converted questions (with their sources and licences) and a
script that recomputes it at any threshold, temperature or escalation budget are in
[`routing_reproducibility/`](routing_reproducibility/). A re-read moves the small score cells (253 questions) by about
one question: at threshold 0.9 the re-read in that folder gives 99.0 % accuracy on the rest instead of 98.0 %, because
one wrong score question sits right at the threshold (0.896).

These numbers use [routing mode](#optional-routing-mode) (the yes/no floor off), for deployments behind a fallback. As
shipped (the evaluated configuration), yes/no answers are reported at no less than 0.801, so none would be escalated below
0.8.

## Run it

Hardware measured: one RTX PRO 4500 Blackwell (32 GB); the weights (`model.safetensors`) are 9.3 GB. Install vLLM 0.30.0 in a fresh virtual environment
(for example `python3 -m venv .venv && . .venv/bin/activate`):

```bash
pip install "vllm==0.30.0+cu129" "torchcodec==0.16.0+cu129" --extra-index-url https://wheels.vllm.ai/0.30.0/cu129 --extra-index-url https://download.pytorch.org/whl/cu129
```

The shim needs Python 3.8 or later and nothing else.

```bash
hf download h2oai/h2o-lightning-4b --local-dir h2o-lightning-4b
```

If a folder from an earlier v1.3 download contains `label_rows.safetensors`, delete that file (or download into a
fresh folder): vLLM reads it as a weight shard and fails to start (details under Two-way reading).

That is `main`: the v1.3 weights, both reading modes, the Box question and image references. For the exact
configuration behind the measurements below, pin the earlier weights instead:
`hf download h2oai/h2o-lightning-4b --revision v1.2.3 --local-dir h2o-lightning-4b` (one-way only; the board's
weights are `--revision v1.1`).

**Checking this card against the files.** `test_card.py` (with `card_common.py`, shipped here) runs what the card
says and compares it with what the card says: `CARD_REV=<commit> python3 -m pytest test_card.py -m static` checks
files, pins, tags, layout, links and the registered claims in seconds; `CARD_LIVE=1` adds the launch commands and
example requests above on a GPU (the hybrid-mode adapter check included); `-m slow` with `CARD_BENCH231_DIR` recomputes
the benchmark numbers above. Every release is gated on it.

**Recommended: two-way reading, one command** (one GPU, the same memory as one-way; stock vLLM 0.30.0):

```bash
bash h2o-lightning-4b/serve.sh
```

That starts vLLM's pooling runner on the weights with two-way attention and the shim in front of it, warms up, and
prints `ready` with `http://127.0.0.1:8741/v1/systemone`. `TWO_WAY=0 bash h2o-lightning-4b/serve.sh` (or
`ATTENTION=one-way`) is the one-way alternative: one stock generate-mode vLLM, exactly the v1.2.3 setup. What each
mode is, when to use which, and what two-way needs: [Two-way reading](#two-way-reading-recommended).
`ONE_ENGINE=1 bash h2o-lightning-4b/serve.sh` serves both modes and the Box question from **one** generate-mode
engine, each request choosing its attention mode; it needs our vLLM build (per-request attention), see
[One engine, both modes](#one-engine-both-modes). The two steps behind the command, for a manual start:

**1. Serve the model.** Keep `config.json` as shipped: its `"head_dtype": "float32"` makes vLLM compute the logits in
fp32. Inputs over the context limit get HTTP 422. The `--limit-mm-per-prompt` / `--mm-processor-kwargs` options set
the image limits (up to 4 images per request, each scaled to at most 1,048,576 pixels inside vLLM, the cap the shim
sends with every request).

Two-way (the pooling runner; the override at both config levels, prefix caching and chunked prefill off):

```bash
vllm serve ./h2o-lightning-4b --served-model-name h2oai/h2o-lightning-4b --host 127.0.0.1 --port 8000 \
  --max-model-len 40960 --gpu-memory-utilization 0.90 \
  --runner pooling --convert embed --pooler-config '{"task": "embed", "seq_pooling_type": "LAST", "use_activation": false}' \
  --hf-overrides '{"is_causal": false, "text_config": {"is_causal": false}}' \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' --mm-processor-kwargs '{"max_pixels": 1048576}'
```

One-way (a stock generate-mode server):

```bash
vllm serve ./h2o-lightning-4b --served-model-name h2oai/h2o-lightning-4b --host 127.0.0.1 --port 8000 \
  --max-model-len 40960 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' --mm-processor-kwargs '{"max_pixels": 1048576}'
```

If vLLM stops at startup because FlashInfer cannot compile its sampling kernels (an older system CUDA toolkit), start it
with `VLLM_USE_FLASHINFER_SAMPLER=0` in the environment: the decisions do not use the sampler.

**2. Start the shim** on the same machine, telling it which kind of server it talks to:

```bash
# two-way (the pooling runner above)
python3 h2o-lightning-4b/h2o_lightning_shim.py --config h2o-lightning-4b/serve_config.json --vllm-two-way http://127.0.0.1:8000 --port 8741
# one-way (the generate-mode server)
python3 h2o-lightning-4b/h2o_lightning_shim.py --config h2o-lightning-4b/serve_config.json --vllm http://127.0.0.1:8000 --port 8741
```

Run the shim as shipped for evaluations and benchmarks (including JevBench): `serve_config.json` is the evaluated
configuration. For production routing, see [Optional: routing mode](#optional-routing-mode).

Before answering, the shim checks that vLLM serves `h2oai/h2o-lightning-4b` and that every label is one token at the
answer slot; a two-way deployment also checks the shipped label rows against the served weights and the engine's
answers to three fixed probes against the release's recorded answers; if any check fails it answers 503, so a
misconfigured server stops a run instead of scoring it. Before the first image request it also checks that vLLM's
chat template renders the same prompt as the text path.

**3. Warm up** until this returns HTTP 200:

```bash
curl -s http://127.0.0.1:8741/v1/systemone -H 'Content-Type: application/json' -d '{"state": "warm-up",
  "questions": {"decision": {"type": "noul", "instructions": "Is this a warm-up?",
  "criteria": {"true": "yes", "false": "no"}}}}'
```

**Example requests.** Against the server above (`bash h2o-lightning-4b/serve.sh`, or steps 1 and 2). It serves the
model as `h2oai/h2o-lightning-4b` (`GET /v1/models`); the OpenAI route also accepts `h2o-lightning-4b`.
The OpenAI route and the score confidences shown in this card come with the shim on main; the v1.2.3
revision pinned above serves `/v1/systemone` only and reports a score's confidence as the normalized maximum, so
download main (`hf download h2oai/h2o-lightning-4b --local-dir h2o-lightning-4b`) to use them.

The `/v1/systemone` API, one yes/no (`noul`), one choice and one score question about one record:

```bash
curl -s http://127.0.0.1:8741/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "Checkout returns HTTP 500 for every customer since 09:12.", "plan": "enterprise"},
  "questions": {
    "outage":   {"type": "noul", "instructions": "Is this a full outage?",
                 "criteria": {"true": "yes", "false": "no"}},
    "team":     {"type": "choice", "instructions": "Which team should take it?",
                 "criteria": {"payments": "Payments", "web": "Web frontend", "support": "Customer support"}},
    "urgency":  {"type": "score", "instructions": "How urgent is it?", "criteria": ["Low", "Medium", "High"]}}}'
```

The same from Python (`pip install requests`):

```python
import requests

r = requests.post("http://127.0.0.1:8741/v1/systemone", json={
    "state": {"ticket": "Checkout returns HTTP 500 for every customer since 09:12.", "plan": "enterprise"},
    "questions": {
        "outage": {"type": "noul", "instructions": "Is this a full outage?",
                   "criteria": {"true": "yes", "false": "no"}},
        "team": {"type": "choice", "instructions": "Which team should take it?",
                 "criteria": {"payments": "Payments", "web": "Web frontend", "support": "Customer support"}},
        "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["Low", "Medium", "High"]}}})
answers = r.json()["answers"]
print(answers["outage"]["noul"], answers["team"]["choice"], answers["urgency"]["score"])
# 0.801 payments 1.72...  (team: payments 0.860, web 0.102, support 0.038; urgency: High 0.753)
```

OpenAI's Decisions API (`POST /v1/decisions`) with OpenAI's Python client (`pip install "openai>=3.26"`), pointed at
the local server: a predicate, a choice and a score question.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8741/v1", api_key="unused")  # the local shim ignores the key
decision = client.decisions.create(
    model="h2o-lightning-4b",
    input="Ticket: checkout returns HTTP 500 for every customer since 09:12. Plan: enterprise.",
    questions=[
        {"type": "predicate", "name": "outage", "instructions": "Is this a full outage?"},
        {"type": "choice", "name": "team", "instructions": "Which team should take it?",
         "choices": [{"value": "payments", "description": "Payments"}, {"value": "web", "description": "Web frontend"},
                     {"value": "support", "description": "Customer support"}]},
        {"type": "score", "name": "urgency", "instructions": "How urgent is it?",
         "levels": [{"label": "Low"}, {"label": "Medium"}, {"label": "High"}]},
    ],
)
print(decision.model_dump_json(indent=1))
```

Each answer comes back in OpenAI's shape (rounded): `{"type": "predicate", "name": "outage", "probability": 0.801}`,
`{"type": "choice", "name": "team", "choice": "payments", "confidence": 0.839, "probabilities": [{"value": "payments",
"probability": 0.892}, ...]}`, `{"type": "score", "name": "urgency", "score": 1.752, "confidence": 0.775,
"probabilities": [{"label": "Low", "value": 0, "probability": 0.022}, ...]}`. The OpenAI route asks the model through the same
decision as `/v1/systemone`, in the same option format: a choice's value is the option and its description the
option's text, a score level is its label, and a predicate is a yes/no question with the default wording ("the
statement holds" / "it does not"). Its numbers differ from the call above because the input is a sentence rather than
the same record and the predicate has no yes/no descriptions.
The outage probability is 0.801 on both routes because yes/no answers are reported at no
less than 0.801 on their chosen side. A choice's `confidence` is the normalized maximum (0 for
a uniform answer, 1 for a certain one); a score's is the probability of its most likely level. An image goes in an
`input` message as `{"type": "input_image", "image_url": "data:image/png;base64,..."}` next to
`{"type": "input_text", "text": ...}`.

**4. Run JevBench's harness**:

```bash
cd <JEVBENCH> && python3 -m jevbench.cli run \
  --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8741 --key-env '' --model h2oai/h2o-lightning-4b \
  --cost-basis self_hosted_gpu --reserve-usd 0 --results <OUT>/results.jsonl --raw-dir <OUT>/raw \
  --ledger <OUT>/ledger.jsonl --manifest <OUT>/manifest.json --run-label h2o-lightning-4b --delay-s 0
```

**The API.** Send requests to `POST /v1/systemone`. One request can carry several questions about the same state:

```json
{"state": "Ticket 4471: since 09:10 the checkout page returns HTTP 500 for every customer; no orders have gone through in 40 minutes. Severity guide: P1 = a revenue path fully down; P2 = degraded but working; P3 = cosmetic.",
 "questions": {
   "priority":  {"type": "choice", "instructions": "Which priority does the severity guide assign?",
                 "criteria": {"p1": "Revenue path fully down", "p2": "Degraded but working", "p3": "Cosmetic"}},
   "all_users": {"type": "noul", "instructions": "The problem affects every customer.",
                 "criteria": {"true": "Affects everyone", "false": "Affects only some"}},
   "urgency":   {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["Low", "Medium", "High"]}}}
```

The response (rounded here):

```json
{"model": "h2oai/h2o-lightning-4b",
 "answers": {
   "all_users": {"type": "noul", "noul": 0.931},
   "priority": {"type": "choice", "choice": "p1", "confidence": 0.922, "probabilities": {"p1": 0.948, "p2": 0.029, "p3": 0.023}},
   "urgency": {"type": "score", "legend": {"0": "Low", "1": "Medium", "2": "High"}, "score": 1.894, "confidence": 0.905, "probabilities": {"0": 0.011, "1": 0.084, "2": 0.905}}},
 "probability_source": "native", "effort_used": {"passes_per_decision": 1, "tier": "fast"},
 "usage": {"input_tokens": 231, "latency_ms": 74.9, "output_tokens": 3, "processed_input_tokens": 437}}
```

`usage.input_tokens` counts the prompt head that the questions share once; `usage.output_tokens` is one label
token read per answered question. A record over the state limit (`max_state_tokens`) is cut for reading, but
`usage.input_tokens` still counts it whole; `usage.processed_input_tokens` is what vLLM read.

**Images.** Put each image in the request as a `data:image/...;base64,...` URI. It can go anywhere inside `state` (as a
string, an object value or a list item), or in a top-level `images` list next to `state` and `questions`. The `images`
list also takes bare base64 strings and `https` image links (the shim fetches a link itself, from public addresses
only and with size and time limits, and does not store it; the server needs network access for those, and a link it
cannot fetch gets HTTP 422). Chat-style parts (`{"type": "image_url", "image_url": {"url": ...}}`) and multipart/form-data (the JSON in a
`request` field, each image as a file part) work too.
- Each image taken out of `state` leaves `[image N]` in its place, so the record can refer to it. Images in the
  top-level list are numbered first.
- Up to 4 images per request and 20 MB per image.
- Image links: `https` on port 443 (plain `http` needs `LIGHTNING_IMAGE_URL_ALLOW_HTTP=true`), an `image/*` answer with HTTP 200 after at most 3 redirects, 5 s and 20 MB per link (`LIGHTNING_IMAGE_FETCH_SECONDS`, `LIGHTNING_IMAGE_FETCH_MAX_BYTES`); `LIGHTNING_IMAGE_URLS=off` refuses links.
- Each image is scaled to at most 1,048,576 pixels, and image answers use their own temperature, 0.65 (`serve_config.json`,
  `"image"`); text answers use per-type temperatures (`serve_config.json`).
- A request without an image takes exactly the text path above.

This example uses `example_receipt.png` from this repository:

```python
import base64, json, urllib.request
img = base64.b64encode(open("h2o-lightning-4b/example_receipt.png", "rb").read()).decode()
req = {"state": {"receipt": "data:image/png;base64," + img,
                 "claim": {"employee": "J. Ortiz", "category": "office supplies", "amount": "444.26"}},
       "questions": {
         "matches":  {"type": "noul", "instructions": "The receipt total equals claim.amount.",
                      "criteria": {"true": "Same amount", "false": "Different amount"}},
         "items":    {"type": "choice", "instructions": "How many item lines are on the receipt?",
                      "criteria": {"three": "3", "four": "4", "five": "5", "six": "6"}},
         "legible":  {"type": "score", "instructions": "How legible is the receipt?",
                      "criteria": ["Unreadable", "Partly readable", "Fully readable"]}}}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8741/v1/systemone", json.dumps(req).encode(),
                                                  {"Content-Type": "application/json"}))
print(json.dumps(json.load(r), indent=1))
```

The response (rounded here):

```json
{"model": "h2oai/h2o-lightning-4b",
 "answers": {
   "items": {"type": "choice", "choice": "six", "confidence": 0.969, "probabilities": {"five": 0.015, "four": 0.005, "six": 0.977, "three": 0.003}},
   "legible": {"type": "score", "legend": {"0": "Unreadable", "1": "Partly readable", "2": "Fully readable"}, "score": 1.962, "confidence": 0.972, "probabilities": {"0": 0.009, "1": 0.019, "2": 0.972}},
   "matches": {"type": "noul", "noul": 0.967}},
 "probability_source": "native", "effort_used": {"passes_per_decision": 1, "tier": "fast"},
 "usage": {"images": 1, "input_tokens": 585, "latency_ms": 140.7, "output_tokens": 3, "processed_input_tokens": 1489}}
```

`usage.input_tokens` includes the image tokens; `usage.images` counts the images read.

**Status codes.** An input the shim cannot take gets HTTP 422, which the runner scores as one wrong answer. That covers:
- over the context limit;
- more than 255 options;
- an unknown question type or a malformed question;
- more than 4 images, or an image that is not valid base64 image data.

An empty request gets 400. vLLM's 401, 403 and 429 pass through. Any other vLLM failure is a 502, which counts toward the
runner's three-consecutive-failures stop.

**Tests** (no GPU): `python3 h2o-lightning-4b/test_shim.py` and `python3 h2o-lightning-4b/test_security.py`.

## Two-way reading (recommended)

**What it is.** A decoder reads a prompt one way: each token attends only to what comes before it, and the answer is
read at the last position. Since v1.3 the weights are trained to answer under two patterns: **one-way** (causal, as
every text generator and stock vLLM read a prompt) and **two-way**: bidirectional attention on the model's
full-attention layers, so the answer position sees the record, the question and the options in both directions
before the readout (the Gated DeltaNet layers, the model's linear-attention layers, stay causal). It is one forward
pass and one readout in both modes, and the same shim, request format and calibration.

**Why both.** Two-way reads better on text: on the Decision Index 0.3 public suite +1.1 (57.86 against 56.72 for
v1.3.1; +1.2 for v1.3), and 203 against 198 on JevBench's 231 public items. On vision it does not: v1.3.1's
two-way is 0.38 below its one-way (67.13 against 67.51) and 0.73 below v1.3's two-way (67.86); on v1.3 two-way was
0.29 above one-way. All measured above. One-way needs nothing beyond stock vLLM's generate mode, works with prefix caching, and serves Box questions
(a generation) from the same engine; it is the mode of every earlier release. The weights were trained to handle
both, so a deployment can pick either, and a client can ask per request when both engines run (the Box pair,
`LIGHTNING_BOX_ENABLED=1`); the default single engine answers two-way and gives HTTP 400 to a one-way request.

**When to use which.** Two-way (the default) for decisions, text or images, on one GPU: the recommended choice. One-way
when you need Box questions without a second engine, when many questions share one long record and you want the
prefix cache, when you want the exact v1.2.3 behaviour, or on a server you cannot start in pooling mode.

**How two-way is served.** vLLM's pooling runner, started with the override
`{"is_causal": false, "text_config": {"is_causal": false}}` (the nested entry is the one the layers read; the outer
one turns chunked prefill and prefix caching off, which two-way prompts must not have; `serve.sh` sets both and the
two flags explicitly), returns the model's final hidden state at the answer slot instead of generating. The shim
multiplies that state with the lm_head rows of the 255 option labels shipped in `label_rows.lightning` (the
safetensors format under a name vLLM's weight loader does not pick up; v1.3's first commit shipped it as
`label_rows.safetensors`, which vLLM read as a weight shard and failed to start on, with "There is no module or
parameter named 'W'": if your folder has that file, delete `label_rows.safetensors` from it (a re-download into the
same folder does not remove it) and download `main` again, or download into a fresh folder; `serve.sh` refuses to
start while the stale file is present)
(`pooled_readout.py`, standard library) and reads every option exactly as the one-way path reads it from vLLM's
`logprob_token_ids`. Requirements: vLLM 0.30.0 (stock; no patch), the model folder from `main`
(`label_rows.lightning` beside `serve_config.json`), one GPU with the memory one-way needs (measured on a 48 GB L40S
at `--gpu-memory-utilization 0.90`; 9.3 GB of weights), `--max-model-len 40960` as shipped. The first two-way
request also hashes `model.safetensors` once (the identity check below; about a minute on a slow disk; `serve.sh`'s
warm-up waits for it). The pooling runner has no
prefix cache, so a multi-question decision computes each question's whole prompt: `usage.processed_input_tokens`
reports what was read; `usage.input_tokens` is the same count as one-way (the shared record once). The two-way
calibration (`serve_config.json` `two_way.temperature`, `two_way.noul_floor`, `two_way.image_temperature`) ships
with the one-way values, the ones every number above was measured with; a missing value falls back to the one-way one
with a warning when the engine is first checked.

**Launching each mode.**

```bash
bash h2o-lightning-4b/serve.sh                              # two-way (default): one pooling-runner vLLM + the shim
TWO_WAY=0 bash h2o-lightning-4b/serve.sh                    # one-way: one stock generate-mode vLLM + the shim
LIGHTNING_BOX_ENABLED=1 bash h2o-lightning-4b/serve.sh      # two-way + Box: adds a generate-mode vLLM for Box
ONE_ENGINE=1 bash h2o-lightning-4b/serve.sh                 # one engine, both modes per request + Box (our vLLM build: h2oai/vllm h2o/per-request-attention @236568a780)
```

With `TWO_WAY=1` the shim is started with `--vllm-two-way`; every request is read two-way; a request asking for
`"attention_mode": "causal"` gets HTTP 400 (there is no one-way engine), never a silent two-way answer, and every
response states the mode it was read in (`"attention_mode"`). Box questions are a generation, which the pooling
runner cannot do: with `LIGHTNING_BOX_ENABLED=1`, `serve.sh` starts a second, generate-mode vLLM on port 8001 after
the two-way engine is up, if the GPU has `BOX_ENGINE_MIB` (default 14,000 MiB) free; with `LIGHTNING_BOX_ENABLED=1`
both engines are started at `--gpu-memory-utilization` 0.45 (two copies of the weights and two caches; measured on a
48 GB L40S; set `GPU_UTIL` to change it), Box questions run on the
generate engine, and a request may choose either mode (`"attention_mode": "causal"` or `"bidirectional"`, also
`"one-way"` / `"two-way"`; without it, two-way). If there is no room, or that engine fails to start, decisions keep
serving two-way and a box question gets a clear HTTP 400 (the alternative is `TWO_WAY=0`, one-way with Box on one
engine). The `/v1/decisions` route takes the same `attention_mode` field.

```bash
curl -s http://127.0.0.1:8741/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "Checkout returns HTTP 500 for every customer since 09:12.", "plan": "enterprise"},
  "questions": {"outage": {"type": "noul", "instructions": "Is this a full outage?",
                           "criteria": {"true": "yes", "false": "no"}}}}'
```

**The start-up guard.** Before its first two-way answer the shim checks that the engine serves the model, that
`label_rows.lightning` belongs to the served `model.safetensors` (its sha256, hashed once and kept in
`.model_sha256.json` beside the weights) and holds this model's label tokens, that the pooled state is not normalised,
and that the engine's answers to three fixed probes are within 0.05 of the answers recorded on the release weights
(`serve_config.json` `two_way.probe_reference`): an engine on other weights or another dtype, or started without the
override (it would read one-way and answer plausibly; in our check such an engine differed from the recorded two-way
answers by 0.11 and from the one-way engine by 0.004), is refused with HTTP 503 and the reason, checked again after
10 s (`SHIM_TWO_WAY_RETRY_SECONDS`). With the Box pair up, the two engines must also differ on the probes.

### One engine, both modes

Since v1.3.1 the same shim can serve two-way, one-way and the Box question from **one** generate-mode vLLM, each
request choosing its mode: `ONE_ENGINE=1 bash h2o-lightning-4b/serve.sh`. It needs our vLLM build,
[h2oai/vllm, branch h2o/per-request-attention](https://github.com/h2oai/vllm/tree/h2o/per-request-attention)
(commit `236568a780`, on vLLM 0.30.0): a request carrying `vllm_xargs {"attention_mode": "bidirectional"}` is
prefilled with the full-attention layers reading both ways and the Gated DeltaNet layers causal, exactly as the
pooling runner reads a two-way prompt, and the answer comes back through the ordinary `logprob_token_ids` path.
The shim sends that flag for every two-way read (`SHIM_TWO_WAY_FLAG=1`, set by `serve.sh`), nothing for a one-way
read or a Box generation. What the build does for a flagged request: it schedules the whole prompt in one step
(no chunked prefill, no block-aligned split of the linear-attention state; a prompt longer than
`--max-num-batched-tokens` is refused for that request, so `serve.sh` sets it to `--max-model-len`), reads nothing
from and writes nothing to the prefix cache (a two-way prefix is not reusable; unflagged requests keep the cache),
and allows one output token (`max_tokens` 1: the readout). Unflagged requests are untouched: the engine is stock
vLLM for them. The build runs the Triton attention backend (`--attention-backend TRITON_ATTN`; the FlashAttention
backend needs FA4 for a per-sequence causal flag); `serve.sh ONE_ENGINE=1` sets it.

**Parity (text).** On JevBench's 231 public text items (no image parity figure yet) the flagged read of the v1.3.1 weights agrees with the package's
two-way read (`bash serve.sh`, the pooling engine) on 230 of 231 argmax answers (per-item max |dp| median 0.0004,
p99 0.016, max 0.020) and scores 204 / 231 against the two-way read's 203; the unflagged read matches the stock
one-way engine on 231 / 231 (max |dp| 0.022; 198 / 231). The three fixed probes of the start-up guard differ from
the recorded two-way answers by at most 0.004, and the flagged and unflagged reads of the probes differ by up to
0.09. (On the v1.3 weights the same comparison gave 230 of 231, max |dp| 0.029.)

**The start-up guard, one engine.** Before its first answer the shim reads the three probes flagged and unflagged
on the same engine: the flagged answers must be within 0.05 of the answers recorded on the release weights
(`serve_config.json` `two_way.probe_reference`) and must differ from the unflagged ones (an engine without the
build ignores the flag and answers one-way: it is refused with HTTP 503 and the reason, re-checked after 10 s).
With one engine every response still states its `"attention_mode"`; a request without the field is read two-way.

**Building it.** `pip install vllm==0.30.0` into a fresh environment, then overlay the branch's changed runtime
files (`git diff --name-only v0.30.0 236568a780 -- vllm/` lists them, 14 files under `vllm/`) over the installed
`vllm` package, or install the branch from source. Start with `ONE_ENGINE=1 bash h2o-lightning-4b/serve.sh`, or
by hand: `vllm serve ./h2o-lightning-4b ... --attention-backend TRITON_ATTN --max-num-batched-tokens 40960` and
`SHIM_TWO_WAY_FLAG=1 python3 h2o-lightning-4b/h2o_lightning_shim.py --config h2o-lightning-4b/serve_config.json --vllm http://127.0.0.1:8000 --port 8741`.
Stock vLLM keeps serving the two-engine layouts above; the one-engine layout is the recommended one where the
build is available (one copy of the weights and one cache, the prefix cache for one-way and Box, and every mode
per request).

**Reference implementation.** For Qwen3.5 the nested override on stock vLLM 0.30.0 is the whole mechanism. Our vLLM
branch, [h2oai/vllm, branch h2o/gemma4-noncausal-full-attention](https://github.com/h2oai/vllm/tree/h2o/gemma4-noncausal-full-attention)
(commit `88cd4a0d82`), is the reference implementation of the same two-way reading: it adds a `noncausal_full_attention`
config declaration that reaches the attention layers of both Qwen3.5 and Gemma-4 (stock vLLM ignores a flat override
on a multimodal folder, which is why the entry above is nested), encoder-only attention for Gemma-4's full-attention
layers (including the E4B's KV-shared layers), a guard that refuses an engine whose declared layers are still causal,
and parity tests against the transformers reference. The branch
[h2o/two-way-reference](https://github.com/h2oai/vllm/tree/h2o/two-way-reference) (commit `79c479a1be`) is the
same change on v0.30.0 without the tests that need our data. Use stock vLLM for this model's two-engine layouts; use
the branches to read how it works, to serve a Gemma-4-based model two-way, or (h2o/per-request-attention) for the
one-engine layout.

### Image references and the Box question

Images can be sent by reference: put each image once under `"assets"`
(`{"<ref>": {"type": "image", "format": "png", "data_base64": "..."}}`, `"jpg"` for JPEG) and write `<ref>` wherever
the image belongs in `state`; the shim replaces the reference with the image. References match as plain text, so
every occurrence counts, even inside a longer word (`photo` also matches "a photo of"); each distinct reference is
sent to the model once, and every occurrence points at that image. Pick references that cannot occur in your text,
such as `IMG_1`. PNG and JPEG only, 16 MiB per file,
every asset referenced at least once, no URL fetching; a malformed asset map is a 400.

A `box` question, ``{"type": "box", "instructions": "the coffee mug in `photo`"}``, answers with a bounding box
`{"left", "top", "right", "bottom"}` in fractions of the image, `confidence` (calibrated from the box generation's own token
probabilities), `present` (whether the named object is visible) and `found` (whether a box was
located; if not, the box is the whole image with confidence 0). `present` answers whether the named object is visible; it is not a score of the box. `confidence` is the score of the box: in our measurement on 300 public COCO, document and UI images held out from the calibration fit, it ranks correct boxes above wrong ones with AUC 0.77, with expected calibration error 0.056. Box is off by default: a box question gets 400 until
`serve_config.json` sets `"box": {"enabled": true, ...}` (or `LIGHTNING_BOX_ENABLED=1`). `box.mode` chooses how the box is found: `"generate"` (the default: one constrained generation of the box),
`"edges"` (each edge read as a distribution over positions, the coordinate its expected value) or `"hybrid"` (both,
the tighter box answered). `edges` and `hybrid` need a model trained on edge readouts; keep `generate` with this
release.

## Serving at scale

Measured with the one-way (generate-mode) engine on the v1.2.3 weights; the pooling runner behind the two-way
default has no prefix cache and was not measured at scale. `VLLM_ARGS` reaches both engines with `serve.sh`.

**Why a stock server stalls.** Qwen3.5 interleaves 24 Gated DeltaNet (linear-attention) layers with 8 full-attention
layers. With prefix caching on, vLLM 0.30.0 keeps their recurrent state only at 528-token block boundaries (272 with
the bfloat16 state below), and its scheduler defers a request whose cached prefix ends on a state block written in the
same step, then stops scanning the waiting queue. The shim sends a decision's questions together over one shared record, so under load the engine runs
about one decision per scheduler step, however many are waiting. Zeroing the state of new sequences also synchronises
the GPU in each linear-attention layer in every step that starts new sequences.

**Two patches fix it**, in [`vllm_patches/`](vllm_patches/) (sha256 in `vllm_patches/SHA256SUMS`; vLLM 0.30.0 only):
`vllm030_scheduler_skip_same_step_mamba.patch` skips the deferred request for that step instead of ending the scan
(scheduling only), and `vllm030_gdn_initial_state_no_sync.patch` zeroes those states with `torch.where` (no GPU sync,
identical values). In our test, both patches together gave the same outputs as stock vLLM 0.30.0 (687
development-set decisions with known answers identical; largest change in the raw label probabilities of 231 questions:
0.000). Patch A changes only which requests share a step, so under load results can differ by ordinary BF16 batching
noise, as they do on stock vLLM. If you pinned `--revision v1.2.3`, download `main` for `vllm_patches/` and the current serve.sh. `VLLM_PATCHES=1 bash h2o-lightning-4b/serve.sh` applies them to the vLLM
in the active environment before starting it; it edits that install, so use a dedicated virtual environment. Without
`VLLM_PATCHES=1`, serve.sh runs stock vLLM.

**Recommended flags** for many decisions in flight: `--max-cudagraph-capture-size 2048` (measured below). We also set
`--max-num-batched-tokens 8192`; it was not set in the runs below. Optional: `--mamba-ssm-cache-dtype bfloat16` (about
3 % more throughput; no answer changed on the 687 decisions, largest probability change 0.022) and `--quantization fp8` (below). With serve.sh, pass them in `VLLM_ARGS`, for example
`VLLM_PATCHES=1 VLLM_ARGS="--max-cudagraph-capture-size 2048 --max-num-batched-tokens 8192" bash h2o-lightning-4b/serve.sh`.

One H100 80GB, one vLLM engine (`--gpu-memory-utilization 0.85`), each decision three questions sent concurrently over a
fresh record (each question prompt about 740 tokens), the shim's request pattern, but sent straight to vLLM rather than
through the shim (decisions per second, median latency). Through the shim, raise `SHIM_QUESTION_THREADS` (default 64)
to at least three times the decisions in flight.

| vLLM 0.30.0 | 1 in flight | 16 in flight | 64 in flight |
|---|---|---|---|
| stock | 13.3/s, 74 ms | 29.0/s, 551 ms | 29.0/s, 2,252 ms |
| patched, BF16, `--max-cudagraph-capture-size 2048` | 16.4/s, 61 ms | 79.0/s, 203 ms | 81.6/s, 766 ms |
| patched, FP8, `--max-cudagraph-capture-size 2048` | 18.6/s, 53 ms | 91.3/s, 176 ms | 99.2/s, 631 ms |

**Answers.** The patches changed no answer in our test. The larger CUDA-graph capture size only adds BF16 batching
noise (1 of the 687 answers changed, a near-tie; largest probability change 0.019). FP8 changes about 3 % of answers
(21 of the 687), almost all near-ties (19 where the top two options were within 0.1 of each other); the accuracy
difference is not significant (FP8 got 3 fewer right, McNemar p = 0.63). The calibration temperatures in
`serve_config.json` were fitted on BF16 output, not re-fitted for FP8: FP8 runs use them as they are for now.

SGLang 0.5.21's multi-item scoring API (`/v1/score`) was faster than patched vLLM only at saturation (90.7 vs 81.6
decisions/s at 64 in flight, both BF16), was slower at 1-16 in flight, and shifts answers (97.8 % of raw label argmaxes
agree with vLLM).

## Optional: routing mode

**Not for benchmark or leaderboard runs.** Those use the shipped `serve_config.json` unchanged.

As shipped, the shim reports each yes/no answer at no less than 0.801 on its chosen side (`"noul_floor": 0.801` in
`serve_config.json`). That is the evaluated configuration, and it is why no shipped yes/no answer falls between 0.2 and
0.8. If you put the model in front of a fallback, and escalate uncertain answers to a larger model or a person, you
usually want the raw yes/no probability instead, so uncertain yes/no answers can be flagged. In that deployment only,
start the shim with the floor off:

```bash
SHIM_NOUL_FLOOR=0 python3 h2o-lightning-4b/h2o_lightning_shim.py --config h2o-lightning-4b/serve_config.json --vllm http://127.0.0.1:8000 --port 8741
```

The same `SHIM_NOUL_FLOOR` / `SHIM_TEMPERATURE` overrides apply to a two-way deployment (`--vllm-two-way`, or
`SHIM_NOUL_FLOOR=0 bash h2o-lightning-4b/serve.sh`): they set both modes' values.

The floor never changes which answer is chosen, and choice and score answers are unaffected. See
[Routing thresholds](#routing-thresholds) for what each threshold escalates with the floor off.

## Hybrid mode: decisions and text from one server

`adapter/` holds H2O-Lightning-4B's decision adapter as a LoRA for the base model
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). Serve the base model with the adapter on vLLM, and one
server answers both kinds of request:
- **decisions** with `model: "h2oai/h2o-lightning-4b"` (the adapter), through the same shim and readout as the
  merged model;
- **free text** with `model: "Qwen/Qwen3.5-4B"` (the base model, no adapter).

On JevBench's 231 public items (v1.2.3 weights and their adapter), the adapter scored the same as the merged model
(205/231) and gave the same top answer on 230 of 231 items. `adapter/` on `main` is the v1.3.1 adapter, the one the
v1.3.1 weights were merged from; its hybrid-mode agreement with the merged v1.3.1 weights on the same 231 items is 230 of 231
(one-way reading on both sides, the adapter at 197 / 231 against the merged weights' 198 / 231, median per-item max |dp| 0.0009,
largest 0.045; measured on one L40S at v1.3.1; the v1.3 pair measured the same way agreed on 230 of 231 too, and the
mis-named adapter file of v1.3 to v1.3.0.3 agreed on only 178 of 231, scoring 172 / 231, the base model's answers). Hybrid mode reads one-way only (the LoRA engine is vLLM's
generate mode; there is no two-way reading through an adapter). The merged weights are exactly the base plus this
adapter, and serving the base with the adapter reproduces the merged weights' answers to within run-to-run noise:
on these 231 items 230 of 231 agree, and on our own wider check of 7,369 questions (the v1.3 weights and their
adapter, both sides served with the same vLLM settings) 99.7 % of the answers were identical with no measurable
difference in score. Use the merged weights (`bash serve.sh`, the recommended default) when the answer must match
the published numbers exactly; use hybrid mode when one server must also produce text. Pin `--revision v1.2.3` on the download below for the earlier measured pair. The adapter file on `main` was
re-exported at v1.3.0.4 with vLLM's tensor names (see the must-rename note below): the v1.3 to v1.3.0.3 commits carry
the same tensors under names vLLM 0.30 does not match, so hybrid mode on those commits served the base model under
the adapter's name without any error. If your `adapter/` came from one of them, download `main` again (the file is
replaced) and run the adapter-active check below. Measured on one RTX A6000: 94 ms per decision (p50, versus 60 ms for the merged model);
about 1.3 s for a 90-token reply from the base model.

**0. Get the adapter** into the same folder as the download above (shim and `serve_config.json` included):

```bash
hf download h2oai/h2o-lightning-4b --include "adapter/*" --local-dir h2o-lightning-4b && cd h2o-lightning-4b
```

**1. Serve the base model with the adapter** (vLLM 0.30.0, from the folder that contains `adapter/`):

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --host 127.0.0.1 --port 8000 \
  --max-model-len 40960 --gpu-memory-utilization 0.90 --max-num-seqs 64 \
  --enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules h2oai/h2o-lightning-4b=./adapter \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' --mm-processor-kwargs '{"max_pixels": 1048576}'
```

- `--max-num-seqs`: Qwen3.5's linear-attention layers need one state block per running sequence, so keep this at or
  below what fits. vLLM stops at startup with a clear message if it's too high.
- `VLLM_USE_FLASHINFER_SAMPLER=0` is needed only where FlashInfer can't compile its sampler (an older CUDA toolkit).
  The decisions don't use the sampler.

**2. Put the shim in front for decisions.** Use the same shim and config as the merged model. The LoRA's name is the
served model name the shim expects:

```bash
python3 h2o_lightning_shim.py --config serve_config.json --vllm http://127.0.0.1:8000 --port 8741
```

Decisions then go to `POST http://127.0.0.1:8741/v1/systemone` exactly as described above.

**3. Text from the base model on the same server:**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "Qwen/Qwen3.5-4B",
  "messages": [{"role": "user", "content": "Write a two-sentence status update for customers about a checkout outage that is now fixed."}],
  "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": false}}'
```

**Check that the adapter is active.** The two lines must differ; if they are identical, vLLM is serving the base model
under the adapter's name:

```bash
for m in Qwen/Qwen3.5-4B h2oai/h2o-lightning-4b; do curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d "{\"model\": \"$m\", \"prompt\": \"Ticket: checkout is down for every customer. Priority (P1, P2 or P3):\", \"max_tokens\": 1, \"temperature\": 0, \"logprobs\": 5}" | python3 -c "import json,sys; print(sys.argv[1], json.load(sys.stdin)['choices'][0]['logprobs']['top_logprobs'][0])" "$m"; done
```

> **Must-rename note for converted or re-exported adapters.** vLLM loads Qwen3.5-4B as
> `Qwen3_5ForConditionalGeneration`, so the adapter's tensor names must be
> `base_model.model.model.language_model.layers.N.…`. The file on `main` uses those names since v1.3.0.4 (the
> `test_shim.py` packaging test checks every key of `adapter/adapter_model.safetensors`). An adapter saved from a
> text-only load of the model (`base_model.model.model.layers.N.…`, which is what v1.3 to v1.3.0.3 shipped) is
> accepted by vLLM without any warning and changes nothing: the output equals the base model. Rename the keys (insert
> `language_model.` after `model.model.`; the safetensors header is the only thing that changes) and run the check
> above.

The adapter's own scaling is the standard `lora_alpha / r`. Use it as is, with no extra scale.

## What it was trained on

H2O-Lightning-4B was fine-tuned from Qwen3.5-4B in three stages. Every training example is a record (a document,
ticket, policy, conversation, game state...) with one typed question about it and a graded answer. The first stage is
broad, the second concentrates on reading long documents and on numbers, dates and probabilities, and the last stage
mixes everything back in with more support, everyday-language, safety and coding cases.

![Share of training examples by use case in each of the three fine-tuning stages, and overall: documents, policies and rules 37%, numbers, dates and probability 22%, games and simulations 12%, everyday language and knowledge 10%, support and operations 10%, safety and guardrails 7%, coding and tool calls 2%, finance and commerce under 1%](assets/datamix_stages.png)

| Use case | Stage 1: broad | Stage 2: focused | Stage 3: final | All stages |
|---|---:|---:|---:|---:|
| Documents, policies & rules (contracts, policies, multi-step reading, judging answers) | 34% | 49% | 36% | **37%** |
| Numbers, dates & probability | 24% | 41% | 15% | **22%** |
| Games & simulations (state tracking, planning, control) | 23% | 2% | 6% | **12%** |
| Everyday language & knowledge | 8% | 1% | 15% | **10%** |
| Support & operations (tickets, routing, workflows) | 6% | 4% | 16% | **10%** |
| Safety & guardrails | 6% | 3% | 9% | **7%** |
| Coding & tool calls | - | - | 3% | **2%** |
| Finance & commerce | - | - | 1% | **0.5%** |

Use cases are assigned per data family, so the split is approximate.

**Question types.** About half of the examples are choice questions, with yes/no and score questions making up the
rest. Yes/no and score questions get a bigger share in the final stage.

![Share of training examples by question type in each stage: overall choice 54%, yes/no 29%, score 16%](assets/qtype_stages.png)

| Question type | Stage 1 | Stage 2 | Stage 3 | All stages |
|---|---:|---:|---:|---:|
| Choice | 62% | 62% | 46% | **54%** |
| Yes / no | 23% | 29% | 34% | **29%** |
| Score | 14% | 9% | 20% | **16%** |

**Text and images.** All fine-tuning examples were text, and almost all (over 99%) were in English. The model reads
images with Qwen3.5-4B's vision encoder, which fine-tuning did not change.

## Other runtimes

The weights also run outside vLLM. The builds and measurements in this section are the **v1.2.x weights, one-way**
(the v1.3 weights and two-way reading are not yet built for these runtimes):

- **GGUF for llama.cpp:** [h2oai/h2o-lightning-4b-gguf](https://huggingface.co/h2oai/h2o-lightning-4b-gguf) has the GGUF
  files and the vision projector. Its default files serve the decision API (`/v1/systemone`) straight from `llama-server`
  with the H2O-Lightning decision support proposed in llama.cpp [pull request #30361](https://github.com/ggml-org/llama.cpp/pull/30361);
  its card explains how to build that until the pull request is merged. For standard llama.cpp, the same repo has
  `stock` files and a shim that serves the same API, with a one-command launcher.
- **MLX for Apple Silicon (community):** [netprojects/h2o-lightning-4b-mlx-q8](https://huggingface.co/netprojects/h2o-lightning-4b-mlx-q8)
  (8-bit) and [netprojects/h2o-lightning-4b-mlx-q4](https://huggingface.co/netprojects/h2o-lightning-4b-mlx-q4) (4-bit),
  with code and benchmark scripts at [internet-projects/h2o-lightning-mlx](https://github.com/internet-projects/h2o-lightning-mlx).
  They run this repo's shim unchanged, with MLX in place of vLLM. NetProjects reports 205 / 231 on JevBench's public
  set for the 8-bit build, the same as above. These builds are made and maintained by NetProjects, not by H2O.ai.

### Our own MLX measurements on Apple Silicon

We converted these weights with mlx-vlm at four precisions and ran them on the GPU of an Apple M4 (24 GB), reading
answers with the same readout as this repo's shim. On 793 held-out decision questions, compared with the same model
served by vLLM:

| Runtime | Size (GB) | Same answer as vLLM | Accuracy change (95% CI) | Calibration error (ECE) | Brier score |
|---|---:|---:|---:|---:|---:|
| vLLM, BF16 (reference) | | | | 0.033 | 0.242 |
| MLX, BF16 | 9.10 | 99.5 % | +0.1 pts (−0.3 to +0.5) | 0.033 | 0.242 |
| MLX, 8-bit | 5.16 | 99.0 % | −0.1 pts (−0.8 to +0.5) | 0.034 | 0.242 |
| MLX, 4-bit mixed | 3.37 | 92.3 % | −0.4 pts (−2.1 to +1.4) | 0.029 | 0.252 |
| MLX, 4-bit | 3.06 | 91.3 % | −0.4 pts (−2.3 to +1.4) | 0.041 | 0.254 |

BF16 and 8-bit agree with vLLM on 99 % or more of answers, with the same calibration. The 4-bit builds show no
accuracy change we can detect on this sample, but they change about 1 answer in 12 and their probabilities are a
little less accurate (Brier about 0.01 higher). The mixed build keeps the token embedding (shared with the output
head) and some projections at 6 or 8 bits. ECE and Brier (lower is better) are measured on the 679 of these questions
from public third-party sets. Sizes include the vision encoder, kept at BF16. We have not published this MLX
conversion; the community builds above are separate files and were not measured here.

## Limitations

- Evaluated mostly in English. Decides best with up to about 16 options; up to 255 are accepted.
- The probabilities are calibrated for decision questions of this kind, not for open-ended text.
- Not a chat model: it answers one decision per question and does not generate explanations.
- Images: no video input; counting many small objects and reading values off line charts are the weakest image
  skills we measured.

## Disclosures

- No JevBench or ImageJevBench data, public or otherwise, was used for training.
- No rules keyed to any benchmark item's wording, id or answer; no network calls at serving time (except fetching an
  image URL a request supplies).
- Price basis for an evaluator: Qwen3.5-4B at bf16, one output token per decision; 654 mean input tokens on the
  public text set. Image tokens are counted in `usage.input_tokens`.

## License

Apache-2.0. Built on [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team (Alibaba Cloud),
Apache-2.0; served with unmodified [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0).
