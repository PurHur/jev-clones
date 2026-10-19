---
base_model: mistralai/Ministral-3-8B-Instruct-2512-BF16
base_model_relation: adapter
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
language:
  - en
  - ja
  - zh
  - es
  - fr
  - de
  - pt
  - ru
  - ko
tags:
  - mistral3
  - decision-model
  - typed-decisions
  - jev
  - jevbench
  - calibration
  - decode-free
  - multilingual
  - vision-language
  - lora
  - peft
---

# Standard One 8B (LoRA adapter)

> **Updated weights (v2.2, 2026-10-04).** If you downloaded this adapter before, download it again or pin
> `revision="v2.2"`. Earlier versions stay available under the tags `v1`, `v1.1` and `v2`.

**Version:** v2.2

Standard One scores a bounded set of answers for a supplied scenario and returns probabilities through
`POST /v1/systemone`. It does not generate free-form response text. This repository contains the 8B
LoRA adapter and a merge recipe; serving requires a merged checkpoint and the server code.

> For ready-to-serve merged BF16 weights and server code, use
> [StandardOne-8B](https://huggingface.co/StandardThinking/StandardOne-8B).

| If you need | Repository |
|---|---|
| 8B adapter weights and merge recipe | [StandardOne-8B-LoRA](https://huggingface.co/StandardThinking/StandardOne-8B-LoRA) (this repository) |
| Merged 8B checkpoint and server code | [StandardOne-8B](https://huggingface.co/StandardThinking/StandardOne-8B) |
| Smaller adapter weights and merge recipe | [StandardOne-3B-LoRA](https://huggingface.co/StandardThinking/StandardOne-3B-LoRA) |
| Smaller merged checkpoint | [StandardOne-3B](https://huggingface.co/StandardThinking/StandardOne-3B) |

In the reported served evaluations, 8B scores higher than 3B on the public standard and hard tiers;
3B has a lower median latency on the measured short-request profile. See [Benchmarks](#benchmarks)
for the measurement conditions and limitations.

![Standard One benchmark card: JevBench public tiers, held-out suites, stated-distribution probability, hard-tier calibration, latency and throughput for Standard One 8B, Standard One 3B and Jev 1.13.](docs/assets/00-benchmark-card.png)

*Historical v2 chart: the figure combines results from different measurement paths, measured 24–26 September 2026. See [Changes in v2.2](#changes-in-v22) for this version.*

## At a glance

- Send a state and a bounded rubric to receive probabilities for the supplied labels:
  `choice` selects among labeled options, `noul` is yes/no, and `score` uses an ordinal scale.
  The endpoint scores the labels in one forward pass without decoding answer text after this adapter
  is merged and served.
- In the same-run **offline** comparison with the untuned base (measured on v2), 8B improves on four of six suites, and declines on public easy and public hard. These are not served-endpoint results.
- Probabilities are temperature-scaled; calibration (hard-tier ECE, distribution total-variation) was
  checked on v2 — see Benchmarks below.
- Multilingual: English plus Japanese, Chinese, Spanish, French, German, Portuguese and Russian, with a
  smaller Korean share. The merged checkpoint retains the Pixtral vision tower and accepts image
  data URLs; this card does not report a separate image-input benchmark.
- Apache-2.0 throughout: base model, adapter, merged weights and server code.

## Merge the adapter

```python
import torch
from transformers import Mistral3ForConditionalGeneration
from peft import PeftModel

base = Mistral3ForConditionalGeneration.from_pretrained(
    "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    revision="f6fae9795746f63c9be8344932f01275f3c63734",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base, ".").merge_and_unload()
model.save_pretrained("./StandardOne-8B-merged", safe_serialization=True)
# then copy the base snapshot's tokenizer / chat template / preprocessor / generation config
# files into ./StandardOne-8B-merged alongside the merged weights.
```

Then serve `./StandardOne-8B-merged` with the `sglang.launch_server` and `jev-adapter` setup in
[StandardOne-8B's QUICKSTART.md](https://huggingface.co/StandardThinking/StandardOne-8B/blob/main/QUICKSTART.md#standard-one-8b),
substituting the merged directory for `--model-path` (served model name `standard-one-8b`, no system
prompt, `--prompt-wording served --label-scheme upper --default-temperature 1.65`). That guide also has the virtual-environment installation,
request format and curl example.

## Prompt wording

`jev-adapter` can phrase a request in two ways. `served` is the adapter's default and the wording used for the v2.2
measurements (`--default-temperature 1.65`, labels `A`–`Z`, then `AA`, `AB`, …); `native` accepts at most 26 options per question.
The table below is a v2 measurement on the same served endpoint (no system prompt), with temperatures fitted on v2.

| `--prompt-wording` | What the prompt looks like | Temperatures (default; choice / noul / score) | Mean accuracy, 10 suites |
|---|---|---|---:|
| `native` | `State:` / `Question:` / `Options:` headers, options as `A. name: description` | 0.85; 0.85 / 0.85 / 0.70 | 76.6 % |
| `served` (adapter default) | `Context:` / `Question:` / `Options:` headers, options as `A: name: description` | 0.80; 0.80 / 0.90 / 0.95 | 76.4 % |

The 10 suites: judge proxy, hard proxy, stated-distribution probability, realistic transfer set, MuSiQue (multiple choice), SQuAD 2.0 unanswerable questions, ContractNLI, PAWS-X (English), a held-out hard decision set and a consistency set. None of them is a JevBench tier, and no JevBench item was used to choose the wording or the temperatures.
To use `native` with its v2-fitted temperatures, pass `--prompt-wording native --native-system-prompt none --default-temperature 0.85 --temperature-by-type choice=0.85,noul=0.85,score=0.70`.

## Changes in v2.2

v2.2 continues training from v2.1 with additional decision data. Questions with more than 26 options now use the labels `A`–`Z`, then `AA`, `AB`, …; the `server/` code in [StandardOne-8B](https://huggingface.co/StandardThinking/StandardOne-8B) uses this order by default (`--label-scheme upper`). Both versions were measured the same way: merged BF16 weights through SGLang 0.5.20 and `jev-adapter`, `served` wording, one option order, accuracy of the most probable answer; measured 1–3 October 2026.

| Suite | v2.1 | **v2.2** | Change (points) |
|---|---:|---:|---:|
| many-option questions, 53–151 options (18,000) | 69.77 % | **82.67 %** | +12.90 |
| the same question set, at most 26 options (750) | 83.87 % | **88.13 %** | +4.26 |
| long-document questions (150) | 23.33 % | **40.00 %** | +16.67 |
| held-out decision set (600) | 78.33 % | **82.50 %** | +4.17 |
| hard proxy (600) | 51.83 % | **53.50 %** | +1.67 |
| realistic transfer set (600) | 90.50 % | **91.50 %** | +1.00 |
| JevBench public easy (48) | 100.00 % | **100.00 %** | 0.00 |
| JevBench public standard (72) | 98.61 % | **98.61 %** | 0.00 |
| JevBench public hard (111) | 58.56 % | **56.76 %** | −1.80 |

Decision Index 0.2.1 (balanced skill): **41.14**, measured through that `server/` adapter (`served` wording, default temperature 1.65). The tables below are the v2 measurements with `native` wording and are not directly comparable with the table above.

## Benchmarks

**Served endpoint results (the v2 release configuration).** Merged BF16 weights through SGLang 0.5.20
and `jev-adapter`, native wording, no system prompt, one option order, per-answer-type temperatures (choice 0.85, noul 0.85, score 0.70). The wording and the temperatures were chosen on non-JevBench data. Jev 1.13 was
measured on the same items through its hosted endpoint; its probabilities are raw, with no
temperature applied. These are our measurements, not official sealed-set JevBench scores.

| Suite | **Standard One 8B** | Jev 1.13 |
|---|---|---|
| JevBench public easy (48) | **100.00 %** | 100.00 % |
| JevBench public standard (72) | **93.06 %** | 98.61 % |
| JevBench public hard (111) | **54.95 %** | 72.07 % |
| judge proxy (600: routing + answer adequacy) | **89.33 %** | 90.50 % |
| realistic transfer set (600) | **90.83 %** | 86.67 % |
| stated-distribution probability (1,036) | **81.18 %** | 72.97 % |
| hard proxy (600) | **53.00 %** | 54.83 % |

**Offline comparison with the untuned base.** This separate `transformers` runner used native
wording, no system prompt, one option order, T=1. The base and tuned checkpoint were scored by the
same offline path; these numbers are indicative of the base-model change, not the served scores above.

| Suite | Untuned base | **Standard One 8B** |
|---|---|---|
| JevBench public easy (48) | 100.00 % | **97.92 %** |
| JevBench public standard (72) | 79.17 % | **97.22 %** |
| JevBench public hard (111) | 60.36 % | **55.86 %** |
| judge proxy (600: routing + answer adequacy) | 79.33 % | **88.33 %** |
| realistic transfer set (600) | 72.83 % | **90.00 %** |
| stated-distribution probability (1,036) | 34.85 % | **82.63 %** |

Against the untuned base, four suites improve, and public easy falls by 2.08 percentage points and public hard falls by 4.50 percentage points on this offline run. Served and offline probabilities differ even on identical
prompts, so use the served table for expected endpoint behavior. Hard-tier ECE at the served temperatures is
0.184 against Jev 1.13's 0.099 raw, and mean TV to the stated distributions is 0.108 at served T
against Jev's 0.192 raw; full calibration table: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#calibration).

**Speed** — raw serial latency on one H200 with SGLang 0.5.20, using a 242-decision profile averaging
about 280 input tokens per decision. The 25.8 ms figure is p50 for this profile, not a latency
guarantee for other request lengths, concurrency or hardware. Qwen checkpoints are untuned and shown
for speed only; no accuracy comparison is implied.

| Model | p50 | p95 | Input tokens/decision |
|---|---|---|---|
| Standard One 3B | 22.6 ms | 33.2 ms | ≈280 |
| **Standard One 8B** | **25.8 ms** | **41.9 ms** | ≈280 |
| Qwen3-8B (untuned) | 28.5 ms | 57.2 ms | 278 |
| Qwen3.5-4B (untuned) | 48.8 ms | 72.7 ms | 283 |

Throughput on one H200 (hard+standard mix, 1,322 tokens/request): 29.3k tok/s at concurrency 1, rising
to 39.5k tok/s at concurrency 64 (≈40 decisions/s at concurrency 8 on the 280-token profile above).

On the public classification and decision suites (400 cases/suite, seed 13, served endpoints): AG News
84.2 %, typed decisions 71.1 %, MASSIVE intent mean 86.1 %, email spam 91.8 %, phishing 85.2 %. Full
table, per-language and per-workflow breakdown: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#public-classification-and-decision-suites)
and [`docs/public-classification-suites.md`](docs/public-classification-suites.md).

A JevBench v1.4.1 run has been requested; the sealed-set result is not yet available.

Full report: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Model details

- **Base model:** `mistralai/Ministral-3-8B-Instruct-2512-BF16`, revision `f6fae9795746f63c9be8344932f01275f3c63734` (Apache-2.0).
- **Adapter:** LoRA r=16, α=32, dropout 0, on `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` of the language-model projections only (vision tower and multimodal projector excluded), **44,564,480** trainable parameters, PEFT 0.21.0. Adapter file `adapter_model.safetensors`, 214,559,872 bytes, sha256 `123ddd039f4053e82e8ca18d7c247691dc49cbcabb6e1b7d98077bb7c5c7446e`.
- **Merged BF16 checkpoint** (as published in `StandardThinking/StandardOne-8B`): merging this adapter into the base changes **238 tensors** (293 unchanged), none outside the language-model projections, maximum absolute weight change **0.00211**.
- **Serving details:** `served` wording (the adapter default), no system prompt, `--default-temperature 1.65` for every answer type (the value used for the v2.2 measurements; the adapter itself defaults to 1.0; no temperature was refit for v2.2), labels `A`–`Z`, then `AA`, `AB`, … (`--label-scheme upper`, the default; up to 255 options); served model name `standard-one-8b` behind stock SGLang 0.5.20 via `jev-adapter` (`POST /v1/systemone`); single caller-supplied option order, no rotation ensemble; 32,768-token context.

| Path | Contents |
|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | The LoRA adapter |
| `merge.py` | Loads the base model, applies this adapter, saves the merged BF16 checkpoint |
| `docs/`, `docs/public-classification-suites.md` | Full benchmark report, figures, per-language/per-workflow numbers |
| `SHA256SUMS`, `LICENSE`, `NOTICE`, `README.md` | File hashes, licence, notice, this card |

The merged BF16 checkpoint, the server code and the full quick-start guide are published in
`StandardThinking/StandardOne-8B`.

## Training data

Training data is synthetic and format-augmented decision data plus decision items converted from public
datasets (listed below); the JevBench public tiers used only for evaluation carry MIT. Full per-cohort breakdown (row counts, what each covers, licence):
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#training-data-provenance).

Public datasets used (train splits where the dataset has one; licence as stated by each dataset; labels come from the
datasets, distractor options are generated by code):

| Dataset | Licence |
|---|---|
| SQuAD 2.0 | CC BY-SA 4.0 |
| ARC | CC BY-SA 4.0 |
| BoolQ | CC BY-SA 3.0 |
| CommonsenseQA | MIT |
| HellaSwag | MIT |
| Banking77 | CC BY 4.0 |
| Bias in Bios | MIT |
| Bitext customer support | CDLA-Sharing-1.0 |
| CLINC150 | CC BY 3.0 |
| Amazon Counterfactual | CC BY 4.0 |
| DBpedia-14 | CC BY-SA 3.0 |
| Dolly 15k | CC BY-SA 3.0 |
| GoEmotions | Apache-2.0 |
| MASSIVE | CC BY 4.0 |
| Twitter Financial News Sentiment | MIT |
| HelpSteer3 | CC BY 4.0 |
| HelpSteer2 | CC BY 4.0 |
| 2WikiMultihopQA | Apache-2.0 |
| HotpotQA | CC BY-SA 4.0 |
| MuSiQue | CC BY 4.0 |
| QASC | CC BY 4.0 |
| DROP | CC BY-SA 4.0 |
| GSM8K | MIT |
| TempReason | CC BY-SA 3.0 |
| MultiNLI | OANC / CC BY-SA 3.0 / CC BY 3.0 |
| PAWS | Google terms, free for any purpose |
| PAWS-X | Google terms, free for any purpose |
| SNLI | CC BY-SA 4.0 |
| WANLI | CC BY 4.0 |
| ContractNLI | CC BY 4.0 |
| CUAD | CC BY 4.0 |
| ShARC | CC BY-SA 3.0 |
| Jailbreak classification | Apache-2.0 |
| Prompt injections | Apache-2.0 |
| Aegis AI Content Safety 2.0 | CC BY 4.0 |
| Jigsaw Toxic Comment Classification (mirror of the Kaggle data) | CC0 (data); comment text CC BY-SA 3.0 (Wikipedia) |
| Measuring Hate Speech | CC BY 4.0 |
| Image safety classes | MIT |
| WinoGrande | CC BY |
| Lichess puzzles and games | CC0 |
| ClinicalTrials.gov records | Public domain (U.S. Government work) |

Upstream ids and the cohort each one feeds: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#training-data-provenance).

An exact-text overlap audit of the v2 training mixture against the public JevBench tiers found 0 exact
scenario matches and 181
exact instruction matches — rows in two adequacy-rubric cohorts whose entire instruction field, a
generic 58-character adequacy question, is byte-identical to one public hard-tier instruction (0.03 %
of that 520,754-row mixture). These rows are kept and disclosed here rather than regenerated,
since the overlap is limited to one rubric question's wording and never touches a scenario or an
answer.

## Limitations

- **Public hard tier:** the served v2.2 8B score is **56.76 %**, versus **72.07 %** for Jev 1.13. In the
  separate offline base comparison (measured on v2), Standard One 8B scores **55.86 %**, below the untuned base's
  **60.36 %**.
- Served probabilities are temperature-scaled by a fixed default (1.65 in the commands above, not refit for
  v2.2); if you apply this model to a materially different question distribution, re-fitting that
  temperature is advisable rather than assuming this value transfers.
- Up to 255 options per question with `served` wording (labels `A`–`Z`, then `AA`, `AB`, …, each one token);
  v2.2 was evaluated with up to 151 options. `native` wording accepts at most 26.
- The sealed JevBench set has not been measured for this model.
- Served and offline probabilities can differ on identical prompts (mean total-variation ≈0.06 on the
  hard tier, measured on v2); served numbers are treated as authoritative.
- Korean is a small share of multilingual training relative to the other seven languages.
- The card reports text benchmarks; it does not establish decision accuracy on image inputs.
- Ten-way support triage (36 %) and RAG passage relevance (59 %) were weak zero-shot on an earlier version; fine-tune
  for those.

## Licence

Adapter weights, merge recipe and this card: Apache-2.0. Base model
`mistralai/Ministral-3-8B-Instruct-2512` (and `-BF16`): Apache-2.0 per its Hugging Face model card,
which adds that the model must not be used in a way that infringes, misappropriates, or otherwise
violates any third party's rights. `jev-adapter` and SGLang: Apache-2.0. The JevBench harness and
public tiers used for evaluation: MIT; other benchmark items keep their own upstream terms.

## Citation

`StandardThinking/StandardOne-8B-LoRA` (this repository, adapter + merge recipe) · `StandardThinking/StandardOne-8B`
(merged weights + server code).
