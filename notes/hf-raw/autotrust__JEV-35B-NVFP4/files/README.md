---
license: apache-2.0
base_model: autotrust/JEV-35B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-classification
language:
  - en
tags:
  - system-one
  - typed-decisions
  - decision-model
  - calibrated-probabilities
  - jev
  - noul
  - choice
  - score
  - lora
  - qwen3_5_moe
  - mixture-of-experts
  - vllm
  - nvfp4
  - modelopt
  - fp4
model-index:
  - name: autotrust/JEV-35B-NVFP4
    results:
      - task:
          type: text-classification
          name: Jev Decision Index 0.3, public suite, System 1 (own scoring with the kit, not a board entry)
        dataset:
          type: decision-index
          name: Decision Index 0.3 public suite
        metrics:
          - type: decision_index
            name: Decision Index 0.3 public index
            value: 59.18
---

# autotrust/JEV-35B-NVFP4


[![AutoTrust/GuruSearch recommendation & search live demo: news.guru.so](https://huggingface.co/autotrust/JEV-27B/resolve/main/assets/autotrust_gurusearch_demo.png)](https://news.guru.so/ui)

## 🔴 Live: AutoTrust/GuruSearch recommendation & search demo → [news.guru.so](https://news.guru.so/ui)

> [!IMPORTANT]
> **Try AutoTrust/GuruSearch live (new, 11 October 2026).** A recommendation and search demo in which every ranking is a
> calibrated System 1 decision, made in about half a second:
> * **Recommendation:** the latest headlines from 6 news feeds, ranked by importance.
> * **Search:** results from Google News, Bing News and Yahoo News re-ranked by relevance, side by side with the
>   search engines' own order, plus a short answer with citations.

### [autotrust/JEV-35B](https://huggingface.co/autotrust/JEV-35B) with NVFP4 routed experts: the same model in 23 GiB of GPU memory instead of 66 GiB

## NVFP4 version

This repository is **autotrust/JEV-35B with its 10,240 routed expert MLPs (40 layers × 256 experts) quantized to
NVFP4**. Everything else is the bf16 model unchanged: attention, linear attention, the shared expert, the routers, the
vision tower, the MTP layer, the lm_head, the System 1 adapter, the decision head and the temperatures. Below the NVFP4
comparison, the rest of this card is the JEV-35B card; its tables are the bf16 results unless marked otherwise.

| | JEV-35B (bf16) | **JEV-35B-NVFP4** |
|---|---:|---:|
| download (weights) | 71.9 GB | **25.6 GB** |
| GPU memory for the weights (vLLM) | 66.5 GiB | **23.3 GiB** (−65 %) |
| Decision Index 0.3, public suite (our scoring with the kit) | 59.75 | 59.18 |
| same top-1 answer as bf16 (308,559 Decision Index answers) | — | 96.6 % |
| approximate public vision score (our rebuild, 13,361 rows) | 72.08 | 71.23 |
| same answer as bf16 (vision rows) | — | 95.2 % |
| median single-request latency, text / image (one B200) | 215 / 378 ms | 227 / 371 ms |
| computer use, numbered boxes + element text (60 tasks) | 95 % | 95 % |
| computer use, numbered boxes only (60 tasks) | 38 % | 37 % |
| robot arm, pick and place (20 scenes) | 75 % | 65 % |

* **Text: −0.57 on the Decision Index.** All five areas within 1.2 skill points of bf16 (Knowledge 0.422, Language 0.628,
  Retrieval 0.706, Tools 0.753, Arts 0.413); the largest per-benchmark changes are Home appliance simulator −5.7,
  CRUXEval −4.7 and GPQA +5.4.
* **Images: −0.85 on the vision rebuild**, a small but real loss (paired bootstrap 95 % interval −1.5 to −0.2), mostly on
  CharXiv charts (−5.3).
* **Robot arm: 13 of 20 scenes against 15 of 20.** Four scenes changed outcome (3 lost, 1 gained), within noise (sign
  test p = 0.63); every grasped cube still reached the tray (13/13).
* **Speed:** the same latency as bf16 on one B200 for single requests; the gain is memory (one 32–48 GB GPU is enough).
* **Games** (seed 0, one run each, noisy): 2048 largest tile 64 (bf16 32), Snake 5 food (bf16 11), Quick, Draw! 85.9 %
  on finished sketches (bf16 86.3 %), chess mate-in-one 45 % (bf16 47 %), Connect Four 1 win of 6 (bf16 0).

The two videos below were recorded with this NVFP4 checkpoint; per-episode NVFP4 results are in
[`reports/demos-nvfp4/`](reports/demos-nvfp4), the bf16 ones in [`reports/demos/`](reports/demos).

**What is quantized and how.** NVFP4 is NVIDIA's 4-bit floating-point format: FP4 E2M1 values in groups of 16 with an
FP8 E4M3 scale per group and an FP32 scale per tensor. Per expert, the gate, up and down projections are quantized (gate
and up share one tensor scale, as vLLM fuses them); the expert activations use static scales calibrated on 3,072 System 1
prompts. The checkpoint uses the ModelOpt NVFP4 layout (`quant_method: modelopt`), so vLLM loads it without any flag.
The System 1 adapter has no LoRA on the routed experts, so `adapter_vllm/` is the same as for the bf16 model.

### Deploy the NVFP4 version (vLLM)

```bash
hf download autotrust/JEV-35B-NVFP4 --local-dir JEV-35B-NVFP4
bash JEV-35B-NVFP4/serve.sh     # vLLM on :8000
```

`serve.sh` and the API are exactly those of JEV-35B (see [Quick start](#quick-start-vllm)); vLLM reads the
quantization from `config.json` / `hf_quant_config.json`.

| GPU | how NVFP4 runs |
|---|---|
| B200 / B300 / GB200 (SM100) | native W4A4: FlashInfer TRT-LLM NVFP4 MoE kernels (tested, B200) |
| RTX PRO 6000, RTX 5090, DGX Spark (SM12x) | native W4A4: CUTLASS / FlashInfer CUTLASS FP4 MoE kernels (not tested here) |
| H100 / H200 / A100 (SM80–90) | Marlin W4A16: weights in FP4, activations bf16 (not tested here) |

---

## The JEV-35B card (bf16 results)

### A System 1 decision model on Qwen3.5-35B-A3B (35 B parameters, about 3 B active per token), up to 256 options in one pass

**Decision Index 0.3, public suite: 59.75** (our scoring with the kit), against 53.64 for
[autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) on the board. Approximate public vision score
**72.08** (our rebuild of the vision benchmarks; JEV-27B-VL 71.67 on the same rebuild, a tie within noise). Median single-request latency
241 ms on one B200 (JEV-27B-VL 271 ms).

> **Two models, two organisations.** **TypeSafe Jev 1.13** is the hosted, closed model made by TypeSafe AI.
> **autotrust/JEV-35B-NVFP4** is an independent open-weights model built by AutoTrust AI; it is not affiliated with,
> endorsed by, or a product of TypeSafe AI.

## Decision Index 0.3 (public suite)

| | public index |
|---|---:|
| **autotrust/JEV-35B, System 1** (our run with the kit) | **59.75** |
| autotrust/JEV-27B-VL, System 1 (board, public part) | 53.64 |

| area (skill) | Knowledge & Reasoning | Language | Retrieval & Classification | Tools & Automation | Arts & Taste |
|---|---:|---:|---:|---:|---:|
| **JEV-35B** | **0.426** | **0.632** | **0.710** | **0.765** | **0.422** |
| JEV-27B-VL | 0.414 | 0.564 | 0.537 | 0.735 | 0.416 |

JEV-35B: all 140,178 scoreable requests of the 0.3 public suite answered (0 errors), System 1 only (thinking off),
every choice read in one pass (up to 256 options), scored with the kit's `score --edition 0.3`. Our scoring, not a board
entry: the board's Full score also counts private tests (80 %), which only its maintainers run. JEV-27B-VL: the board's
public numbers for JEV-27B, whose text decisions JEV-27B-VL reproduces (see the JEV-27B-VL card).

JEV-35B is ahead in all five areas and on 24 of 37 benchmarks. Largest gains (skill points): PhishNChips +42.3, HoVer
+28.3, Habermas Machine +28.1, VAST +22.8, WinoGrande +18.9, BANKING77 +18.3, iSarcasmEval +17.2, When2Call +14.9, GPQA
+9.5. JEV-27B-VL is ahead on POP909-CL (+25.7), GSM8K (+18.3), ANLI (+11.4), BPoMP (+10.8), NLI4CT (+6.7) and BBH (+4.8).

### Every benchmark

Skill rescales the benchmark's own metric so that chance is 0 (below chance counts as 0); ★ = gold benchmark (weight 1.2);
the higher score is in bold.

**Knowledge & Reasoning** (area skill 0.426; JEV-27B-VL 0.414)

| benchmark | JEV-35B | JEV-27B-VL |
|---|---:|---:|
| GPQA Diamond ★ | **0.360** | 0.265 |
| GSM8K (0.3 rebuild) | 0.359 | **0.542** |
| ChessBench | **0.127** | 0.091 |
| MuSR | 0.361 | **0.402** |
| SATA-Bench | 0.284 | **0.328** |
| CRUXEval | 0.546 | **0.583** |
| CLadder | **0.440** | 0.410 |
| HLE ★ | 0.000 | 0.000 |
| MMLU-Pro ★ | **0.642** | 0.567 |
| BBH ★ | 0.647 | **0.695** |
| WinoGrande ★ | **0.841** | 0.651 |

**Language Understanding** (area skill 0.632; JEV-27B-VL 0.564)

| benchmark | JEV-35B | JEV-27B-VL |
|---|---:|---:|
| ContractNLI | **0.732** | 0.639 |
| ANLI ★ | 0.545 | **0.658** |
| HellaSwag ★ | **0.962** | 0.900 |
| ACOS | **0.280** | 0.169 |
| FinEntity | **0.856** | 0.755 |
| iSarcasmEval | **0.426** | 0.254 |
| VAST | **0.550** | 0.322 |
| NLI4CT | 0.627 | **0.694** |
| RAGTruth | **0.666** | 0.604 |

**Retrieval & Classification** (area skill 0.710; JEV-27B-VL 0.537)

| benchmark | JEV-35B | JEV-27B-VL |
|---|---:|---:|
| BANKING77 ★ | **0.922** | 0.739 |
| CLINC150+OOS ★ | **0.943** | 0.845 |
| BRIGHT ★ | **0.421** | 0.417 |
| Amazon ESCI | **0.519** | 0.430 |
| PhishNChips | **0.661** | 0.238 |
| HoVer | **0.761** | 0.478 |

**Tools & Automation** (area skill 0.765; JEV-27B-VL 0.735)

| benchmark | JEV-35B | JEV-27B-VL |
|---|---:|---:|
| BFCL ★ | **0.950** | 0.946 |
| ToolRet | 0.600 | **0.604** |
| API-Bank ★ | **0.856** | 0.825 |
| Home appliance simulator | 0.500 | **0.523** |
| When2Call | **0.863** | 0.714 |

**Arts & Human Taste** (area skill 0.422; JEV-27B-VL 0.416)

| benchmark | JEV-35B | JEV-27B-VL |
|---|---:|---:|
| BPoMP | 0.750 | **0.859** |
| Humicroedit | **0.232** | 0.223 |
| POP909-CL | 0.132 | **0.390** |
| cfcolor | 0.256 | **0.283** |
| Habermas Machine | **0.416** | 0.135 |
| New Yorker | **0.744** | 0.607 |

## Images (approximate public vision score)

The vision benchmarks of the Decision Index are not published; this is our rebuild of their public datasets
(CV-Bench, BLINK, RealWorldQA, CharXiv, InfographicVQA, Mind2Web, CORD + FUNSD, Hateful Memes, R-Bench-M, MMMU-Pro
vision; Winoground not included), scored with the board's chance correction and weights. The same rebuild gives
JEV-27B-VL 71.67 against its board score of 71.53 on the same benchmarks.

| | approximate public vision score | accuracy | ECE |
|---|---:|---:|---:|
| **JEV-35B** | **72.08** | **78.6 %** | 0.050 |
| JEV-27B-VL (same rebuild) | 71.67 | 78.0 % | 0.044 |

| benchmark | JEV-35B skill | JEV-27B-VL skill |
|---|---:|---:|
| CV-Bench | 75.7 | **76.9** |
| BLINK | 55.6 | **56.7** |
| RealWorldQA | 68.9 | **72.2** |
| CharXiv | 80.7 | 80.7 |
| InfographicVQA | 95.0 | **95.9** |
| Mind2Web | **83.6** | 83.2 |
| KIE (CORD+FUNSD) | **98.7** | 98.6 |
| Moderation (Hateful Memes) | **47.7** | 38.0 |
| R-Bench-M | 28.1 | **30.3** |
| MMMU-Pro vision | **43.5** | 39.2 |

Overall the two models are level: the difference (+0.4) is inside the noise (paired bootstrap 95% interval −0.7 to
+1.6). Only two per-benchmark differences are statistically significant (paired McNemar test, p < 0.01), both in favour of
JEV-35B: moderation (+9.6) and MMMU-Pro (+4.3). The small deficits on RealWorldQA, CV-Bench, BLINK, InfographicVQA and
R-Bench-M (−1 to −3) are not significant. The vision tower is Qwen3.5-35B-A3B's, unchanged; image decisions are
zero-shot.

## Speed

One sequential client, the same 587 rows (387 text rows sampled across the Decision Index, 200 image rows), POST
/v1/decide on `serve.sh` (vLLM, System 1 as a LoRA), one B200:

| median (p95) | text | image | all |
|---|---:|---:|---:|
| **JEV-35B** | 215 ms (415) | 378 ms (784) | 241 ms (682) |
| JEV-27B-VL (same benchmark) | 207 ms (617) | 457 ms (910) | 271 ms (746) |

## Computer use, robot arm and games

Same demo code, seeds, scenes and opponents as the [JEV-27B-VL card](https://huggingface.co/autotrust/JEV-27B-VL); every
step is one System 1 decision (`POST /v1/decide`, thinking off), one model on one B200.

**Computer use: screenshot → which element to click.** A real browser (headless Chromium). Every clickable element gets a
numbered box; System 1 picks the next click (or "the task is complete"), the browser clicks it, and the loop repeats.

<video src="https://huggingface.co/autotrust/JEV-35B-NVFP4/resolve/main/videos/computer_use_shop.mp4" controls autoplay loop muted playsinline width="100%"></video>

**Robot arm: pick and place from a camera image** (MuJoCo). At every step System 1 answers two questions from the top
camera: is the target left or right of the gripper, and above or below it? The arm halves its step whenever an answer flips.

<video src="https://huggingface.co/autotrust/JEV-35B-NVFP4/resolve/main/videos/robot_arm_pick_place.mp4" controls autoplay loop muted playsinline width="100%"></video>

| | **JEV-35B** | JEV-27B-VL |
|---|---:|---:|
| Computer use, numbered boxes + element text (60 tasks: shop, settings, mail) | 95% | 95% |
| Computer use, numbered boxes only (60 tasks) | **38%** | 10% |
| ms per click decision, median (6 browsers in parallel) | **397** | 720 |
| Robot arm, binary-decision servo (20 scenes): pick-and-place success | 75% | 75% |
| Robot arm: median distance from the cube centre when grasping | 2.5 cm | 2.7 cm |
| Robot arm: placed in the tray, once grasped | 15/15 | 15/15 |
| Robot arm: ms per decision, median | 239 | 239 |
| Robot arm, direct choice among 8 motor actions (10 scenes) | 0% | 0% |

* **Computer use with element text: 95%, the same three failures as JEV-27B-VL** (shop seeds 12, 15, 20: the colour
  swatch carries no text and is skipped).
* **Numbered boxes only (every element read from pixels): 38% against 10%.** JEV-35B completes 90% of the settings tasks
  but only 15% of mail and 10% of shop; most failures still declare the task complete too early (24 of 37).
* **Robot arm: same success rate, different scenes.** Each model misses 5 of 20 grasps (both miss scenes 4 and 19), every
  miss 3 cm or more off the cube centre; once grasped, every cube reaches the tray.
* Choosing directly among 8 motor commands fails for both models; decompose control into simple visual questions.

**Games** (seed 0 for both models; board as image + text):

| game | **JEV-35B** | JEV-27B-VL |
|---|---:|---:|
| 2048: score / largest tile | 336 / 32 | **2,080 / 128** |
| Connect Four against a rule-based opponent (6 games) | 0 wins, 6 losses | 0 wins, 6 losses |
| Flappy Bird: pipes passed | 0 | **3** |
| Snake: food eaten | 11 | **21** |
| Quick, Draw! top-1 among 16 (320 sketches; 30% / 60% / 100% of strokes) | 35.0 / 57.5 / 86.3% | **41.6 / 62.2 / 87.8%** |
| Chess mate in one, choice among 16 moves (200 Lichess puzzles) | 47.0% | **52.0%** |

**For game play, use JEV-27B-VL.** Over five seeds JEV-35B's largest 2048 tile averages 64 (random play: 102), it passes
no Flappy Bird pipe and eats 3–18 pieces of food in Snake (mean 10). Per-episode results: [`reports/demos/`](reports/demos).

## Quick start (vLLM)

```bash
hf download autotrust/JEV-35B-NVFP4 --local-dir JEV-35B-NVFP4
bash JEV-35B-NVFP4/serve.sh    # vLLM on :8000; one GPU with 32 GB or more
```

`serve.sh` runs `serve_decide.py`: the standard vLLM OpenAI server (same flags as `vllm serve`) with a `POST /v1/decide`
route. Plain requests are System 2; requests for the LoRA module `jev-decision` (`adapter_vllm/`: backbone LoRA + the
decision head as an `lm_head` LoRA) are System 1. Tested with a vLLM development build from September 2026.

The adapter has no LoRA on the routed experts. `serve.sh` sets `JEV_DECIDE_NO_MOE_LORA=1` and `--lora-target-modules`
so that the routed experts run on vLLM's normal MoE kernels (the decision head needs `--max-lora-rank 320`; vLLM's
MoE-LoRA kernel supports at most rank 128).

### System 1: `POST /v1/decide`

```bash
curl localhost:8000/v1/decide -H 'Content-Type: application/json' -d '{
  "kind": "choice",
  "state": "Customer message: my card was charged twice for the same order.",
  "question": "Which team should handle this ticket?",
  "options": ["billing", "shipping", "technical support", "account security"]}'
```

| field | value |
|---|---|
| `kind` | `noul`: yes/no, probabilities for `["false", "true"]` · `score`: 0–5 · `choice`: your `options` |
| `state` | what the decision is about: a string, a JSON object, or a list mixing text and images |
| `question` | one question about the state |
| `options` | `choice` only: 2–256 strings |
| `thinking` | `"off"` (default) |

The response has `options`, `probabilities`, `choice`, `choice_index` and `usage`. `GET /v1/decide/info` lists the
defaults.

### System 2

```python
import requests
requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "autotrust/JEV-35B-NVFP4",
    "messages": [{"role": "user", "content": "In one sentence, what is safety stock?"}],
    "max_tokens": 200})
```

If you call `/v1/completions` for System 1 yourself, pass `top_k: 0` and `top_p: 1.0` so that the returned
probabilities are not truncated.

## Engine and read-out

System 1 reads the hidden state at the last token of a bare-text prompt:

```text
[kind] choice
[state] ...
[question] ...
[options]
A) ...
B) ...
[decision]:
```

A 264-slot linear head (fp32) gives one logit per slot; the active slots of the question's kind are soft-maxed with a
per-kind temperature (`calibration.json`). In vLLM the head is expressed as an `lm_head` LoRA (`adapter_vllm/`, rank
320) and the head bias is added client-side (`adapter_vllm/decision_head.json`).

## Limitations

* System 1 only on this card: thinking (adaptive System 2) has not been evaluated for this model.
* Knowledge & Reasoning is the weak area (GPQA 0.360, MMLU-Pro 0.642, HLE below chance); JEV-27B-VL is better on GSM8K,
  ANLI and BBH.
* Weaker than JEV-27B-VL at sequential game play (2048, Flappy Bird, Snake).
* The vision score is our rebuild, not the board's; image decisions are zero-shot.
* NVFP4: Decision Index −0.57 and vision −0.85 against bf16; only the B200 (SM100) kernels were tested.
* English-centric; not for high-stakes decisions without confidence gating.

## Files

```
model-*.safetensors       Qwen/Qwen3.5-35B-A3B with NVFP4 routed experts (ModelOpt layout), everything else bf16
config.json · hf_quant_config.json · nvfp4_activation_amax.json
                          quantization config and calibrated expert activation ranges
tokenizer* · chat_template.jinja · generation_config.json · *_config.json
adapter/                  System 1 LoRA (peft), rank 32
head.safetensors          264-slot decision head (fp32)
judge_config.json         slot layout, verbalizer ids, read-out
calibration.json          per-kind temperatures
adapter_vllm/             System 1 for vLLM: backbone LoRA + the head as an lm_head LoRA (rank 320), plus decision_head.json
serve_decide.py · serve.sh
                          vLLM server with POST /v1/decide next to the OpenAI endpoints
videos/                   computer-use and robot-arm episodes
reports/demos/            per-episode computer-use, robot-arm and game results (bf16)
reports/demos-nvfp4/      the same for this NVFP4 checkpoint
```

## License

Apache-2.0. This repository contains the weights of [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
(Apache-2.0), with the routed experts quantized to NVFP4, plus the System 1 adapter, decision head and calibration of
[autotrust/JEV-35B](https://huggingface.co/autotrust/JEV-35B).
