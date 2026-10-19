---
license: other
license_name: nvidia-nemotron-open-model-license
license_link: LICENSE
base_model: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
library_name: transformers
pipeline_tag: text-generation
tags:
- decision-making
- calibration
- jevbench
- nemotron
- jev
- typed-decisions
- decision-model
---

![H2O Lightning: Decision Models](assets/banner.png)

# H2O-Lightning-4B-Nemotron-Nano

H2O-Lightning-4B-Nemotron-Nano is a 4B-parameter decision model from H2O.ai, built on
[nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16). It answers
typed decision questions about a record (a document, a ticket, a policy, a conversation). It returns a probability for
every option:

- **choice**: picks one of a set of named options;
- **yes/no** (`noul`): gives the probability that a statement is true;
- **score**: gives an ordinal level on a stated scale, with its probability distribution.

It reads **text only**. It runs on **unmodified vLLM 0.30.0** with a small shim in front (`lightning_nemo4b_shim.py`).
Each decision is one forward pass and one output token, so the cost is input tokens only.

Sibling model: [H2O-Lightning-4B](https://huggingface.co/h2oai/h2o-lightning-4b) is built on Qwen3.5-4B and has its own
card. Every measurement on this card is this model's own.

## Measured

Measured with `serve.sh` as shipped (vLLM 0.30.0; one temperature, 0.75, for every
question type), on one **NVIDIA L40S**.

### Text: JevBench's public set

From **JevBench's own CLI**, with `--adapter typesafe`, on the 231 public items (`datasets/public/easy.jsonl`,
`original.jsonl` and `hard.jsonl`; dataset hash `dc3995d8…`) at JevBench commit `bb05a33`, with the commands below.

| | 1x L40S |
|---|---|
| correct | 192 / 231 (83.1 %) |
| by tier: easy / standard / hard | 48/48 · 71/72 · 73/111 |
| answered and valid | 231 / 231 |
| yes/no answers with P(yes) strictly between 0.2 and 0.8 | 0 of 74 |
| mean input tokens per decision | 699 |
| latency p50 / p95, standard tier, serial | 0.027 s / 0.027 s |
| latency p50 / p95, all 231 items, serial | 0.028 s / 0.124 s |

Latency depends on the GPU.

**Memory.** vLLM loads the weights into 7.47 GiB of GPU memory. `serve.sh` passes vLLM's `--gpu-memory-utilization 0.90`
(`GPU_UTIL`), and most of the rest of that share becomes the KV cache: on one L40S, 31.64 GiB, 1,611,093 tokens, or
39.33 requests of the full 40,960 tokens at once. Lower `GPU_UTIL` to share the GPU with other processes.

### Decision Index 0.3 public suite

Our own runs of the public Decision Index 0.3 kit (`decision-index` at commit `f8552a1`, edition 0.3, its official
scorer) on the public text suite (140,178 scored rows; the model reads no images, so the vision half does not
apply), served with this release's `serve.sh`:

| Decision Index 0.3 public suite, our runs | DI index | raw index |
|---|---|---|
| as shipped (`serve.sh`: 40,960-token context, 12,000-token state budget) | **53.20** | 65.46 |
| no state trimming (`MAX_LEN=131072`, `SHIM_MAX_STATE_TOKENS=1000000`) | 53.22 | 65.47 |

The DI index is the kit's chance-corrected index (0-100), not percent correct. These are our runs, not a leaderboard
result; the leaderboard's private tests were not run.
No row ended in an error in either run.

## Routing thresholds

When a decision is uncertain you may want to hand it to a stronger model or a person. This table shows that trade-off:
at a confidence threshold, how many questions get **escalated** (top probability below the threshold) and how many
**slip through confidently wrong** (top probability at or above it, but wrong), as a share of all questions.

Measured on 3,474 held-out questions from two public third-party decision sets, with no overlap with our training
data; text temperature as shipped (0.75). The thresholds were fixed before the analysis was run.

| question type (n) | threshold | escalated | accuracy on the rest | confidently wrong |
|---|---|---|---|---|
| choice (1,854) | 0.8 | 43.6 % | 94.3 % | 3.2 % |
| choice | 0.9 | 56.9 % | 96.9 % | 1.3 % |
| yes/no (1,367) | 0.8 | 47.3 % | 93.5 % | 3.4 % |
| yes/no | 0.9 | 64.3 % | 98.2 % | 0.7 % |
| score (253) | 0.8 | 54.9 % | 89.5 % | 4.7 % |
| score | 0.9 | 77.5 % | 96.5 % | 0.8 % |
| all (3,474) | 0.8 | 45.9 % | 93.7 % | 3.4 % |
| all (3,474) | 0.9 | 61.3 % | 97.3 % | 1.0 % |

Without a threshold, accuracy is 75.4 % (choice), 79.4 % (yes/no) and 68.4 % (score).

These numbers use [routing mode](#optional-routing-mode) (the yes/no floor off), for deployments behind a fallback. As
shipped (the evaluated configuration), yes/no answers are reported at no less than 0.801, so none would be escalated below
0.8.

## Run it

Hardware measured: one L40S; the weights are 7.95 GB. Install vLLM 0.30.0 in a fresh virtual environment
(for example `python3 -m venv .venv && . .venv/bin/activate`):

```bash
pip install "vllm==0.30.0+cu129" --extra-index-url https://wheels.vllm.ai/0.30.0/cu129 --extra-index-url https://download.pytorch.org/whl/cu129
```

The shim runs in the same environment: it is standard-library Python apart from the `tokenizers` package, which vLLM
installs.

```bash
hf download h2oai/h2o-lightning-4b-nemotron-nano --local-dir h2o-lightning-4b-nemotron-nano
```

**1. Serve the model.** Keep `config.json` as shipped: its `"head_dtype": "float32"` makes vLLM compute the logits in
fp32. Keep vLLM's default `--chat-template-content-format`. A long record is never refused for its length: a state over
12,000 tokens keeps its head and its tail around a `...` line.

```bash
vllm serve ./h2o-lightning-4b-nemotron-nano --served-model-name h2oai/h2o-lightning-4b-nemotron-nano --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization 0.90
```

**2. Start the shim** on the same machine:

```bash
python3 h2o-lightning-4b-nemotron-nano/lightning_nemo4b_shim.py --config h2o-lightning-4b-nemotron-nano/serve_config.json --vllm http://127.0.0.1:8000 --port 8741 --tokenizer h2o-lightning-4b-nemotron-nano/tokenizer.json
```

Run the shim as shipped for evaluations and benchmarks (including JevBench): `serve_config.json` is the evaluated
configuration. For production routing, see [Optional: routing mode](#optional-routing-mode).

`bash h2o-lightning-4b-nemotron-nano/serve.sh` runs steps 1 and 2 together; the measurements above used it. Before
answering, the shim checks that vLLM serves `h2oai/h2o-lightning-4b-nemotron-nano`, that vLLM's chat rendering of a
probe question is token for token the prompt the model expects, that the tokenizer it counts with is vLLM's, and that
the letter mask is applied; if not, it answers 503, so a misconfigured server stops a run instead of scoring it.

**3. Warm up** until this returns HTTP 200:

```bash
curl -s http://127.0.0.1:8741/v1/systemone -H 'Content-Type: application/json' -d '{"state": "warm-up",
  "questions": {"decision": {"type": "noul", "instructions": "Is this a warm-up?",
  "criteria": {"true": "yes", "false": "no"}}}}'
```

**4. Run JevBench's harness**:

```bash
cd <JEVBENCH> && python3 -m jevbench.cli run \
  --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8741 --key-env '' --model h2oai/h2o-lightning-4b-nemotron-nano \
  --cost-basis self_hosted_gpu --reserve-usd 0 --results <OUT>/results.jsonl --raw-dir <OUT>/raw \
  --ledger <OUT>/ledger.jsonl --manifest <OUT>/manifest.json --run-label h2o-lightning-4b-nemotron-nano --delay-s 0
```

**The API.** Send requests to `POST /v1/systemone` (`/api/alpha/decisions` is an alias). One request can carry several
questions about the same state:

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
{"model": "h2oai/h2o-lightning-4b-nemotron-nano",
 "answers": {"priority": {"type": "choice", "choice": "p1",
                         "probabilities": {"p1": 0.931, "p2": 0.046, "p3": 0.024}, "confidence": 0.896},
             "all_users": {"type": "noul", "noul": 0.949},
             "urgency": {"type": "score", "score": 1.855, "legend": {"0": "Low", "1": "Medium", "2": "High"},
                        "probabilities": {"0": 0.02, "1": 0.105, "2": 0.875}, "confidence": 0.875}},
 "probability_source": "native", "effort_used": {"tier": "fast", "passes_per_decision": 1},
 "usage": {"input_tokens": 268, "processed_input_tokens": 268, "output_tokens": 3, "latency_ms": 58.3}}
```

`usage.input_tokens` counts the prompt head that the questions share once; `usage.output_tokens` is one label
token read per answered question. A record cut or selected to fit the state budget is still counted whole
in `usage.input_tokens`; `usage.processed_input_tokens` is what vLLM read.

**Text only.** The model has no image input. Send records as text; nothing in `state` is read as an image.

**Status codes.** An input the shim cannot take gets HTTP 422, which the runner scores as one wrong answer. That covers:
- instructions and options alone over the context limit;
- more than 255 options (26 lettered, then the trained wide labels);
- an unknown question type or a malformed question.

A request with no questions gets 400; a body that is not an object with `state` and `questions` gets 422. vLLM's 401, 403 and 429 pass through. Any other vLLM failure is a 502, and a misconfigured
vLLM is a 503; both count toward the runner's three-consecutive-failures stop.

**OpenAI's Decisions API.** The shim also serves `POST /v1/decisions` for OpenAI's Python client (`pip install
"openai>=3.26"`, then `OpenAI(base_url="http://127.0.0.1:8741/v1", api_key="unused").decisions.create(model="h2o-lightning-4b-nemotron-nano",
input=..., questions=[...])`; `h2oai/h2o-lightning-4b-nemotron-nano` works too). Each request is asked through the same decision as
`/v1/systemone`, in the option format the model was trained on (a choice's description is the option's text, its
value when it has none; a score level is its label; a predicate is a yes/no question with the trained texts); the
answers and `usage` come back in OpenAI's shape. This model reads text only: a request with an input_image is refused (HTTP 422).

**Tests** (no GPU): `python3 h2o-lightning-4b-nemotron-nano/test_shim.py` and `python3 h2o-lightning-4b-nemotron-nano/test_security.py`.

### Image references and the Box question

This model reads text only: a request with image `assets` or a `box` question gets 400.

### Log-probabilities the shim needs

Every answer is a probability distribution over all of the question's options. Up to 20 options are read in one
request from vLLM's top 20 log-probabilities under the option-letter mask (`--max-logprobs 20`, vLLM's default, which
`serve.sh` sets explicitly; do not lower it). From this release on, a question with 21 to 26 options is read by the
option letters' own token ids (`logprob_token_ids`, one request), so every option gets its probability; earlier
releases read those questions from the top 20 too, and a letter outside them read the 1e-12 floor.
`"letter_read": "masked"` in `serve_config.json` (or `LIGHTNING_LETTER_READ=masked`) restores the earlier read. Questions with more than 26 options use the wide labels and are always read by token id. A yes/no question without a `"true"` option is asked as true / false, as the model was trained. At start-up the shim checks both readouts on a 26-option probe and
answers 503 with the reason when the server cannot give them (it needs vLLM 0.30 with structured outputs).

## Optional: routing mode

**Not for benchmark or leaderboard runs.** Those use the shipped `serve_config.json` unchanged.

As shipped, the shim reports each yes/no answer at no less than 0.801 on its chosen side (`"noul_floor": 0.801` in
`serve_config.json`). That is the evaluated configuration, and it is why no shipped yes/no answer falls between 0.2 and
0.8. If you put the model in front of a fallback, and escalate uncertain answers to a larger model or a person, you
usually want the raw yes/no probability instead, so uncertain yes/no answers can be flagged. In that deployment only,
start the shim with the floor off:

```bash
SHIM_NOUL_FLOOR=0 python3 h2o-lightning-4b-nemotron-nano/lightning_nemo4b_shim.py --config h2o-lightning-4b-nemotron-nano/serve_config.json --vllm http://127.0.0.1:8000 --port 8741 --tokenizer h2o-lightning-4b-nemotron-nano/tokenizer.json
```

The floor never changes which answer is chosen, and choice and score answers are unaffected. See
[Routing thresholds](#routing-thresholds) for what each threshold escalates with the floor off.

## Weights

`model.safetensors` is the fine-tuned model merged into one bf16 file (sha256 `94c7be97a7622a1c66da630f289ee67a506d03319af7bd636a5c2b0305ebc8f6`), built on
[nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) at revision `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`. No adapter is
shipped. This release replaces the weights of the previous one with a model trained longer, on more data, in three
stages (below); the serving files (`serve.sh`, the shim, `serve_config.json`, `config.json` and the tokenizer) are
unchanged, so a deployment of the previous release serves these weights with the same commands and memory.

## What it was trained on

H2O-Lightning-4B-Nemotron-Nano was fine-tuned from NVIDIA-Nemotron-3-Nano-4B in three stages, from easier to
harder, with one mixed distribution in every stage; the final stage also mixes in additional multi-question decision
data. Every training example is a record (a document, ticket, policy, conversation, game state...) with one or more
typed questions about it and graded answers: 1,121,833 examples in all. Documents, policies and rules make up the largest share.

![Share of H2O-Lightning-4B-Nemotron-Nano's training examples by use case: documents, policies and rules 27%, everyday language and knowledge 15%, numbers, dates and probability 14%, support and operations 13%, coding and tool calls 12%, games, simulations and music 10%, safety and guardrails 7%, finance and commerce 1%](assets/datamix.png)

| Use case | Share of examples |
|---|---:|
| Documents, policies & rules (contracts, policies, multi-step reading, judging answers) | **27%** |
| Everyday language & knowledge | **15%** |
| Numbers, dates & probability | **14%** |
| Support & operations (tickets, routing, workflows) | **13%** |
| Coding & tool calls | **12%** |
| Games, simulations & music (state tracking, planning, control, music) | **10%** |
| Safety & guardrails | **7%** |
| Finance & commerce | **1%** |

Use cases are assigned per data family, so the split is approximate.

**Question types.** About two thirds of the examples are choice questions, a little under a quarter are yes/no and the rest are score questions.

![Share of H2O-Lightning-4B-Nemotron-Nano's training examples by question type: choice 67%, yes/no 23%, score 10%](assets/qtype.png)

| Question type | Share of examples |
|---|---:|
| Choice | **67%** |
| Yes / no | **23%** |
| Score | **10%** |

**Text only.** All fine-tuning examples were text. About 98% were in English; the rest are across 22 other languages.

**Mamba-2 output projections.** The fine-tune adds LoRA adapters to the Mamba-2, attention and MLP projections. In the first two stages and the earlier part of the third, the fused Mamba-2 training kernel applied the mixer's output projection (`out_proj`) inside itself, so the adapters on it received no gradient and stayed at zero. The last part of the third stage applied them outside the kernel, so they were trained there only. In the released weights every `out_proj` differs from the base, by at most 0.0056 per weight, a smaller change than in the input projections (`in_proj`, at most 0.077). All the measurements on this card are of these released weights.

## Limitations

- Evaluated mostly in English. At most 255 options (A-Z, then the trained wide labels); a request with more gets 422.
- Text only: no image or video input.
- The probabilities are calibrated for decision questions of this kind, not for open-ended text.
- Not a chat model: it answers one decision per question and does not generate explanations.

## Disclosures

- No JevBench or ImageJevBench data, public or otherwise, was used for training.
- No rules keyed to any benchmark item's wording, id or answer; no network calls other than the local vLLM server.
- Price basis for an evaluator: NVIDIA-Nemotron-3-Nano-4B at bf16, one output token per decision; 699 mean input
  tokens on the public text set.

## License

The weights are a derivative of NVIDIA-Nemotron-3-Nano-4B and are distributed under the
**NVIDIA Nemotron Open Model License** (`LICENSE`; v. December 15, 2025,
[nvidia.com](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/)).
Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License. Built on
[nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) (revision
`dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`) by NVIDIA.

The shim (`lightning_nemo4b_shim.py`, with `state_cut.py` and `selector_v2.py`), `test_shim.py`, `serve.sh`,
`serve_config.json` and this documentation are Apache-2.0 (`LICENSE-Apache-2.0`). The prompt layout and the one-token
option-letter readout the shim implements come from
[Cygnet](https://github.com/blockbrain-ai/cygnet-recipe) (commit `74f9555`, MIT; `LICENSE-cygnet`). Served with
unmodified [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0). See `NOTICE`.
