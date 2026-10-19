# Training and model details: Vela 2.0 0.3B

[Back to the model card](README.md) · [Usage](USAGE.md) · [Evaluation](EVALUATION.md)

## Backbone and runtime readouts

One token sequence contains the non-span questions and at most one span question; additional span questions use separate sequences:

```
<bos> ([Q] question ([O] name: description)* [ABS]?)* ([E] label: description)* [SEP_SCHEMA] ([SEG_role] part)* <eos>
```

Eight marker tokens were added to the Vela tokenizer (ids 256,000–256,007). Text is encoded with the unmodified tokenizer and markers are inserted by id, so a user who types "[Q]" never produces a marker. The three part roles are `user`, `context` and `answer`; tool outputs go in `context`.

- **Encoder.** The `Vela-1.0-Encoder-307M` geometry (ModernBERT, mmBERT-base lineage, 22 layers, hidden 768, YaRN-extended positions), initialised from the Choice-path trunk of `Decision-1.0-Kai-0.6B`.
- **Choice / Noul / Score / Set.** Each option's logit is a cosine term between the option and question markers and the mean of the part(s) the question is over, plus an MLP term on the option marker. Choice, Noul and Score are a softmax over the shown options, Set a sigmoid per option. Each question pools only its own part(s), so a safety question over the request and a span question over the answer share a pass.
- **Span.** Each word's first sub-word state is compared with each `[E]` label marker. A sigmoid produces word × label probabilities; decoding selects the highest-probability label per word above threshold, merges neighbouring words and trims span edges.
- **Long inputs.** If the parts do not fit in 8,192 tokens, the part no question is asked over is shortened first; a labelled part that still does not fit is scored in windows with up to 512 tokens of overlap, each carrying the full schema (logits averaged for choice, score and span, max-pooled for set).
- **Calibration.** One temperature per question type, fitted on dev NLL: Choice 1.379, Score 1.128, Set 1.015 and Span 0.293. Dev thresholds are 0.3 for Set, 0.55 for hallucination spans and 0.1 for toxic spans.
- **PII rule.** The length curve is piecewise-linear in log(token count) and log(threshold), with a sparse-document gate: a document with at most 3 spans at 0.5 is decoded at `max(rule, 0.1)`. The curve and gate were fitted on dev; the shipped minimum threshold floor was lowered after the test effect had been seen. [Evaluation](EVALUATION.md#pii-calibration-and-scoring) distinguishes the resulting metrics. A caller-supplied threshold overrides the default.

| tokens | 21 | 126 | 254 | 510 | 1,022 | 2,046 | 3,070 | 4,094 | 8,190 |
|---|---|---|---|---|---|---|---|---|---|
| PII threshold | 0.75 | 0.05 | 0.035 | 0.012 | 0.0059 | 0.0020 | 0.0014 | 0.00075 | 0.00032 |

## Repository files

| path | what |
|---|---|
| `model.safetensors` | encoder + readout heads, fp32 |
| `configuration_vela2.py`, `modeling_vela2.py` | `AutoConfig` / `AutoModel` remote code (`trust_remote_code=True`) |
| `vela2_inference.py` | package-free engine: SystemOne, `predict()` and the shortcuts, torch or ONNX backend |
| `vela2_serve.py` | optional SystemOne HTTP server (FastAPI + uvicorn) |
| `config.json`, `calibration.json`, `pii_calibration_fit.json` | geometry, markers, window settings; temperatures, thresholds, PII rule and gate, trained schemas; the full dev-only PII fit |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | the Vela base tokenizer plus the 8 marker tokens |
| `onnx/model.onnx`, `onnx/model_fp16.onnx` | the same graph in fp32, and with an fp16 encoder and fp32 heads |
| `PARITY.md` | export parity against the research scorer, and dev / test metrics re-scored through the export |
| `LICENSE`, `NOTICE`, `LICENSES/`, `DISTRIBUTION_TERMS.md`, `LICENSING_STATUS.md`, `NOTICE_SOURCES.json`, `MODIFICATIONS.md` | licence and attribution |

## Training recipe

The model trains for 101,000 steps from Kai's Choice-path trunk.

| Setting | Value |
| --- | --- |
| Batch size | 16 |
| Token budget | 16,384 |
| Gradient accumulation | 2 |
| Encoder learning rate | 3e-5 |
| Head learning rate | 1e-4 |
| Maximum input length | 8,192 |
| Choice / Score loss | Cross-entropy |
| Set loss | BCE |
| Span loss | Focal BCE, α 0.7 and γ 2 |

Labels are resampled at every draw using the GLiNER2 recipe:

- Options are anonymised with probability 0.5, and a random fraction is dropped.
- If the gold option is dropped, it is replaced by `[ABS]` or an all-negative Set.
- Question wording is paraphrased with probability 0.5.
- Score questions are never anonymised, dropped or shuffled.

## Training data

| task | sources | share of steps |
|---|---|---|
| safety / attack / hazard | Shield mix: AEGIS 2.0, PolyGuardMix, Nemotron-Safety-Guard v3, LLMail-Inject, Salad-Data | 13.2% |
| hallucination spans | LettuceDetect prose + code (rows whose context fits the 8,192 window) | 12.2% |
| PII | generated long-document PII (10,457 rows, all 17 types; short PII is in the next row) | 30.0% |
| domain / fact-check / modality / feedback, short PII | Global-MMLU, Aya, DiffusionDB, WildFeedback; the router's synthetic PII recipe and a Presidio replay pool | 4.1% |
| general decisions | the Decision-1.0-Kai source families in our own wording: SNLI, BoolQ, ARC, MASSIVE, HelpSteer2, KLUE, JGLUE, SocialIQA, CosmosQA, TyDi QA, OASST1, AfriSenti, Super-NI, SentiMix | 13.5% |
| synthetic router-style decisions | generated for this model with an open LLM (Qwen3-30B-A3B-Instruct-2507), 21 languages, kept only when a blind re-label agrees | 25.0% |
| small in-house set | | ~2% |

General-decision and synthetic rows are deduplicated against every evaluation set, and a hashed dev carve (hallucination 2%, toxic 5%, Shield 0.4%) is held out so that thresholds are picked on rows the model never saw.

The curve-fitting data were held out from training. The shipped PII minimum-floor adjustment nevertheless considered the test effect, as disclosed in [Evaluation](EVALUATION.md#pii-calibration-and-scoring).

[Back to the model card](README.md)
