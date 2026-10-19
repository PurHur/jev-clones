# Jeeves (PostHog)

A Kev-style Jev clone on Qwen3.5-9B: LoRA, a pointer head over `</opt>` tokens, and Kev's five reused Qwen marker tokens. The difference is that it **writes an autoregressive reasoning chain before deciding** whenever its no-think answer is unsure. A block-4 diffusion drafter (Orthrus-style, adapted to Gated DeltaNet) speeds up those chains. It is Jev-compatible on the wire, but it gives up Jev's "no generation" property, by design.

| ref | what | where |
|---|---|---|
| `G` = GitHub `PostHog/jeeves` @ `8c6ab532` (2026-10-16, "Merge PR #11 public-v2-p3") | shallow clone, read only, never executed | `/home/ai/projects/jev-clones/sources/jeeves` |
| `HF` = `PostHog/jeeves` @ `04b9f87a` (v2 on main, v1 at tag `v1`); `PostHog/jeeves-fp8` @ `d106b56c` | API JSON, tree, README, config.json, export.json, drafter configs (no weights) | `/home/ai/projects/jev-clones/notes/hf-raw/PostHog__jeeves/`, `.../PostHog__jeeves-fp8/` |
| HN | "Jeeves. Reasoning improves Jev-like decision models", item 49891290, 242 points, 95 comments | news.ycombinator.com/item?id=49891290 |

---

## Identity

| | |
|---|---|
| Maker | Nicholas P. Waltz, PostHog AI research (`G:LICENSE:3`, `G:README.md:302-308`). Coverage dates the release to 2026-09-29 ([eesel.ai](https://www.eesel.ai/blog/posthog-jeeves)) |
| Lineage | "Inspired by Kev" (`G:README.md:14`). Follows Hume's *Jev's Architecture Unmasked* (`G:README.md:313`) |
| Weights | `PostHog/jeeves` v2: fused bf16, 8.95B params (`HF:api.json` safetensors). `PostHog/jeeves-fp8`: F8_E4M3 linears + bf16 (`HF:jeeves-fp8/api.json`). HF has 363 + 74 downloads and 7 likes; GitHub has 425 stars (`notes/families.json`) |
| Versions | v1 = SFT + CISPO (HF tag `v1`). v2 = SFT only, step 16,867 (`HF:export.json` `stage: sft`). Model id `jeeves-latest` (`G:inference/api.py:10`) |
| Decision Index 0.3.2 | not listed (`notes/decision_index_0.3.2_entries.tsv`) |

## Architecture

### Backbone
- **Qwen3.5-9B** in the post-trained (not `-Base`) variant (`HF:README.md:3`, `HF:export.json` `base_model`). 32 layers, Gated DeltaNet:full attention at 3:1, d=4096, 16 q-heads / 4 KV, head_dim 256, partial RoPE 0.25, 262,144 positions (`HF:config.json`, `G:model/config.py:40-64`).
- Jeeves has **its own reimplementation** of Qwen3.5 (`G:model/model.py`, 1,017 lines), plus Triton kernels, Metal kernels and FP8 linears (`G:model/triton_kernels.py`, `G:model/metal.py`, `G:inference/fp8.py`). It does not use HF `transformers` modelling.
- LoRA r=16, alpha=32, on all projections. Head lr 1e-4 (`HF:export.json` training_config). The released weights are fused (`G:export.py`).

### Input layout
- Each question runs as a **separate row** inside the Qwen chat template (`enable_thinking=True`). Prompt: `<state>` state `\n<q>` instructions `\n` + `<opt>`option`</opt>\n` per option, then the template's `<think>\n` (`G:loader/dataloader.py:105-117`).
- After the chain (an empty `"\n"` when not thinking), the **remainder** is appended. It repeats the full option block and ends with `<decide>`: `</think>\n\n<opt>…</opt>…<decide>` (`G:loader/dataloader.py:116-117,225`). The README reports that dropping this repetition, or using plain-text markers, hurt accuracy (`G:README.md:220-221`).
- The markers are Kev's reused tokens: `<state>`=`<|fim_prefix|>`, `<q>`=`<|fim_middle|>`, `<opt>`=`<|box_start|>`, `</opt>`=`<|box_end|>`, `<decide>`=`<|fim_suffix|>` (`G:loader/dataloader.py:15`). User text containing `<|x|>` is rewritten to `<¦x¦>` (`:17-21`). Encoder format id: `markers-v3-plainchains` (`:80`).
- noul is encoded as the options `[no(: false-desc), yes(: true-desc)]`; score levels become options in order (`G:prep/format.py:66-71`).

### Attention & question isolation
- Isolation is **by construction**: every question is its own causal row, with no packed block mask and no cross-question tokens (`G:loader/dataloader.py:119-128`).
- In serving, `group()` computes the **common token prefix** of the rows in a group (chat header plus state) once and shares it. A request's text is shared only in a group that holds no other request (`G:inference/engine.py:551-573`). Chains for the questions then decode in parallel as a batch.
- Options are read jointly and in order, causally (as in Kev). Nothing enforces permutation invariance.

### Readout head
- `PointerHead` (`G:model/head.py:4-23`): `logits_j = (W_k h[</opt>_j]) · (W_q h[<decide>]) / sqrt(256)`, then a softmax over the K options. The `</opt>` positions are taken **after the last `</think>`**, i.e. from the repeated option block (`G:loader/dataloader.py:153-156`).
- One head serves all three types. There are two temperatures: `temperature` for direct answers and `think_temperature` for answers after a chain (`G:model/head.py:10-11,22`).

### Question types
| type | answer (`G:inference/api.py:85-93`) | confidence |
|---|---|---|
| noul | `noul = p[yes]` | none |
| choice, 1..255 options | argmax, `probabilities` | `(p_max − 1/K)/(1 − 1/K)` |
| score, 1..255 levels | `Σ i·p_i`, `legend`, `probabilities` | `1 − Σ p_i·|i − mode|/(K−1)`. This is Kev's L-era stand-in, **not** TypeSafe's MAD_unif formula (`:78-82`) |

Probabilities are rounded to 2 decimals (`:69-70`). Unknown request fields are rejected with a 422 (`:41-43`, `G:inference/serve.py:66`).

### Limits
- Serving defaults are `--max-len 8192` per row and `--max-rows 8` (`G:inference/serve.py:80-81`). Training uses `max_len` 4096 and `max_seq_len` 8192; third-party states are capped at 24k characters during prep (`G:trainer.py:38-39`, `G:prep/thirdparty.py:18`). The number of questions is unbounded; rows are packed into groups of at most `max_rows` (`G:inference/engine.py:554`).
- Extra `options` (ignored by Jev clients): `think` (default true), `max_think` (2,560 tokens), `nothink_threshold`, `return_reasoning` (`G:inference/api.py:13-32`). Usage also reports `reasoning_tokens` (`:99-101`).
- The `nothink_threshold` gate compares the **max no-think probability** against the threshold (`G:inference/engine.py:590`). The README describes it as "calibrated confidence"; in code it is `max p`, not the choice-confidence formula.

### Calibration
- `calibrate.py` fits **one** global temperature on dev no-think rows. It grid-searches 351 log-spaced T in e^-1.5..e^2 and minimises NLL (`G:calibrate.py:15`, `G:predictor.py:137-154`).
- Released v2 values: **T = 1.522 direct, T = 5.420 after reasoning** (`HF:export.json`). Chains in SFT always reach the right answer, so post-chain answers come out overconfident (`G:README.md:227`). The script that fits `think_temperature` is not in `calibrate.py`; it is only read in `G:checkpoint.py:39` (inferred: fitted by a separate, unpublished step).

## Training
- **Data (v2).** `train.jsonl` holds 413,780 records / 542,399 questions; dev 5,138; rl 12,623; test 2,698 (`G:data/manifest.json`, `G:prep/manifests/v2/manifest.json`). The README counts 453,531 questions from about 175 public sources (`G:README.md:226`).
  - Public NLP/QA sets, plus the 15 sets new in v2 (GSM8K, AQuA, RuleTaker, WinoGrande, HellaSwag, ToolACE, When2Call, RAGTruth, ESCI, SGD …).
  - Kev-style programmatic policy, compositional and contrastive generators (`G:prep/composition.py`, `G:prep/contrastive.py`).
  - **Third-party clone data pinned by revision:** ZefanCai/Open-Jev v1/v1.1, jaredpalmer/kev-suites and the kev GitHub repo, Mapika/decider's converters (downloaded and run unchanged) and teacher data, spark, LocalLLaMA/typed-decisions, and SemIf/WANLI (`G:prep/thirdparty.py:18-25`, `G:prep/decider.py:1-25`).
  - About 20 prompt-injection sources (`pi_*` in `HF:export.json` sources).
  - No PostHog or customer data, with 8-gram decontamination against eval/dev (`HF:README.md:91`).
- **SFT.** 1 epoch, 16,867 steps on 4 GPUs. lr 5e-5 (head 1e-4), batch 8, warmup 20, clip 1.0 (`HF:export.json`).
  - Loss: head NLL over the options, with soft targets when present (`G:loader/dataloader.py:203-206`), **plus 0.5 × chain LM NLL** (`chain_lm_weight` 0.5). The head loss alone never trains generation (`G:trainer.py:420-428`).
  - `sft_think_frac` 0.15: about 15% of rows carry a self-sampled chain from earlier rounds, kept only if it closes within 1,024 tokens, does not loop, and reaches the right answer (`G:README.md:226`). `sft_truncate_frac` 0.2.
- **CISPO** (v1 only; "did not improve v2", `G:README.md:229`). Settings: 8 prompts × group 8 rollouts.
  - Reward = **head probability of the correct option after the chain** × (1 − length penalty), where the penalty is a hinge cosine-annealed from 2,048 → 1,024 tokens, capped at 0.1 (`G:trainer.py:524-533,450-452`).
  - Group-mean baseline advantage. Truncated IS weight `exp(logp − old).clamp(max=1+0.2)`, detached, and multiplied into the token log-prob (the CISPO form). Plus 0.5 × rollout head CE and 0.5 × anchor CE (`G:trainer.py:534-551`, defaults `:48-56`).
- **Drafter.** A diffusion view of the frozen fused model. Block 4 or 8, distilled by KL on chains sampled from v1 (20k steps, lr 3e-4, `delta_mode: kvattn`) (`HF:drafter_k4.config.json`, `HF:README.md:85`). Mask tokens cross-attend to the DeltaNet layers' post-conv keys and values (`G:README.md:233-235`). The v2 repo still ships v1's drafters.

## Evals
Gate `nothink_threshold` 0.9, greedy, 2,560-token cap. The percentage is the share of questions that reasoned. The Kev-9B and Jev columns are Kev's published numbers on **different items from the same sources**; only JevBench uses identical items (`G:README.md:32-55,292`).

| | Kev-9B | Jev | JevK5 | Jeeves v1 | Jeeves v2 |
|---|---|---|---|---|---|
| Test overall (OOD + held-out) | 0.822 | 0.857 | 0.802 | 0.884 (41%) | **0.905** (39%) |
| MMLU-Pro + buried state | 0.579 | **0.800** | 0.561 | 0.728 | 0.780 |
| JevBench public tiers (231 items) | 0.715* | 0.866 | 0.861 | **0.913** | 0.896 |
| JevBench hard (111) | 0.451* | 0.730 | 0.748 | **0.820** | 0.793 |
| JevBench ECE | – | 0.049 | **0.014** | 0.028 | 0.059 |
| MMLU / MMLU-Pro | 0.738 / 0.515 | **0.900 / 0.840** | 0.690 / 0.504 | 0.810 / 0.726 | 0.845 / 0.770 |
| Unknowable answered at p ≥ 0.9 | 0.000 | 0.090 | 0.009 | 0.000 | 0.000 |

\* Kev-8B (Qwen3).

- v2 **without thinking**: 0.812 on its own test split vs 0.851 when reasoning only if unsure (`G:README.md:57`). On 325 dev questions: no-think 0.806 vs full thinking 0.818 (`G:README.md:186-191`). On dev the reasoning gain is about 1–2 pp; the large gains are on OOD and JevBench.
- v2 dev at export: acc 0.831, NLL 0.496. Unknowable items have mean max p 0.50, and 1.8% of them are answered at p ≥ 0.9 (`HF:export.json` metrics).
- Secondary coverage quotes older (v1-era) README numbers, e.g. JevBench 0.935 and MMLU 0.793 ([braindetox.kr](https://braindetox.kr/en/posts/posthog_jeeves_reasoning_decision_model_2026.html), [AI Weekly](https://aiweekly.co/alerts/posthog-ships-jeeves-9b-decision-model-beating-jev-on-jevbench)). Treat those as superseded.
- HN user TN1ck, on German irony tweets: Jeeves 68 correct vs Jev 79, "above other open decision models". A 394-item moderation run took about 2 h. Critics cite p90 latency ("defeats the point of a Jev-class model") (HN 49891290).

## Serving
- `python -m inference.serve` is a stdlib `ThreadingHTTPServer` on 127.0.0.1:8009 with no auth. It serves `POST /v1/systemone` and `GET /v1/models` (`G:inference/serve.py:53-92`). `sdk/jeeves_sdk` is a drop-in for `typesafe-sdk` (`G:README.md:147-171`).
- Batched speculative engine: CUDA graphs, merged QKV GEMMs, FP8 w8a16 on CC ≥ 8.9 and on Metal (`G:inference/engine.py:154-171`).
- Latency on 1×H100 fp8 (`G:README.md:186-191`):
  - no think: 22 ms median / 46 ms p90;
  - gate 0.9: 1.2 s / 8.2 s;
  - full think: 1.9 s / 8.1 s.
- Drafter speed: 109 → 176 tok/s with block 4 (1.6×), and about 960 tok/s total for 8 batched questions (`:237-242`). M4 Pro: 0.3–0.5 s no-think, about 20 tok/s thinking in bf16 (38 tok/s fp8). Weights are 21 GB bf16 / 11.5 GB fp8 (`:74-82`).

## Licence
- Code: **MIT** (`G:LICENSE`). Weights: **Apache-2.0**, inherited from Qwen3.5-9B (`HF:README.md:108-110`). Each public dataset keeps its own licence. The SFT reasoning chains are not published, so `prep.prep` rebuilds the records but not the chains (`G:README.md:295`).

## What is novel vs Jev and kev
1. **System 1.5: reason only when unsure.** A no-think pointer readout gates an autoregressive `<think>` chain, and the options are then re-read after the chain. This trades Jev's 70–500 ms bound for accuracy. It is the clearest open evidence that test-time reasoning lifts Jev-style accuracy on OOD and JevBench hard (0.73 Jev → 0.82 v1). The dev gain is only 1–2 pp, and the price is a heavy tail (p90 about 8 s).
2. **Head reward for the RL step.** In CISPO, the reward is the pointer head's calibrated probability of the gold option, so the chain is optimised for the decision head rather than for text correctness.
3. **Two temperatures** (direct vs post-chain, 1.52 vs 5.42). Rejection-sampled training chains make post-reasoning answers overconfident. A single shared T would not fix that.
4. **Diffusion drafter for hybrid DeltaNet models.** It extends Orthrus from attention-only models to Gated DeltaNet via cross-attention to post-conv K/V.
5. **Same skeleton as Kev:** markers, LoRA r16/α32, pointer head, per-question rows on a hybrid base. Kev itself is used as **training data** (kev-suites, kev repo generators) alongside Open-Jev and decider sets, so the clones are now cross-training on each other's data.
6. **Gaps.** Score confidence uses the old non-TypeSafe formula. The `nothink_threshold` is `max p`, not confidence. The v2 drafter does not match v2 weights. No published think-temperature fitting script. English only.
