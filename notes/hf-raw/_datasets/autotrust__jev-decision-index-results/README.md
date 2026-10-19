---
license: apache-2.0
pretty_name: JEV Decision Index results
tags:
  - decision-index
  - evaluation
---

# JEV models on the Decision Index

Complete Decision Index 0.2.1 runs of the JEV typed-decision models by AutoTrust AI, for the
[Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) board
([kit](https://github.com/apolinario/decision-index)). Only engine outputs are published here (compact result rows: run
ids, statuses, answers and probabilities, timings). No benchmark inputs are included; rebuild the suite with the kit.

| run | model | Decision Index 0.2.1 | raw | complete |
|---|---|---:|---:|---|
| `runs/jev-9b` | [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B) @ `4ab5dfb` | **43.14** | 56.91 | yes (150,317 of 150,317 scoreable requests, 0 errors, 0 unsupported) |
| `runs/jev-27b` | [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) @ `9b1f6fd` | **53.30** | 64.32 | yes (150,317 of 150,317 scoreable requests, 0 errors, 0 unsupported) |
| `runs/jev-gemma4-26b-a4b` | [autotrust/JEV-Gemma4-26B-A4B](https://huggingface.co/autotrust/JEV-Gemma4-26B-A4B) @ `603bfed` | **58.05** | 67.35 | yes (150,317 of 150,317 scoreable requests, 0 errors, 0 unsupported) |

## `runs/jev-9b`

| file | contents |
|---|---|
| `results.jsonl.gz` | 150,759 result rows in the kit's format (`--compact`: no payload / raw output); the 442 excluded questions were run as well |
| `scores.json`, `index.json`, `benchmark-summary.json` | output of the kit's `score --edition 0.2.1` (re-scoring `results.jsonl.gz` reproduces them exactly) |
| `environment-sequential.json`, `environment-batched.json`, `status.json` | the environment records of the two phases of the run (below) and the final status |
| `source/jev_engine.py` | the engine (`jev_engine:JevEngine`), sha256 `8ee479f6…` |
| `source/fast_run.py` | the batched driver used for the second phase, sha256 `14056c7f…` (published with relative paths; otherwise the file that ran, plus the later, unused vLLM backend) |
| `training-overlap-check.json` | verbatim-overlap counts between the training corpus and the suite |

**Suite.** Rebuilt locally with `suite rebuild` and imported with `suite import`: `selected-rows` uncompressed sha256
`b2b56d6f…` and `added-rows` `7429f3c9…`, exclusions `331df32d…`, all subset files matching. (RouterBench's normalizer
averages floats with the built-in `sum()`, whose algorithm changed in Python 3.12; rebuilding it under 3.11 gives
different state digits, so the suite was frozen under Python 3.12, which reproduces the pinned hashes.)

**Engine.** The model's own System 1 path, as on its model card: Qwen3.5-9B text backbone in bf16 (fp32 tensors kept
fp32), the `adapter/` LoRA merged in memory, the 24-slot fp32 decision head from `head.safetensors`, per-kind
temperatures from `calibration.json`, one prefill pass per question, read-out at the last token of the model's
`bare-v1` template. On the model's own held-out test rows the engine matches the project's reference inference to
max |Δp| 3.6e-07 with identical argmax (300 of 300). Mechanical translation, identical for every benchmark:

* `state`, `instructions` and option descriptions verbatim when strings, JSON otherwise; a choice option is rendered
  `"<key>: <description>"`, or `"<key>"` when the description is empty or equal to the key;
* a `noul` statement is phrased `"Is this scenario one where: <statement>?"`, the form in which all 150,515 TypeSafe Jev
  1.13-labelled `noul` rows of the training corpus present a statement (already-questions are passed verbatim). This was
  fixed before any suite result was seen; it only affects RAGTruth among scored benchmarks.

**Declared capacity.** The decision head has 16 choice slots (A–P). A question with more than 16 options is answered
with every option read by the model and none pruned, in ceil(n/16) + 1 passes: contiguous groups of near-equal size
(≤ 16) in the given order, then a final of 16 (each group's top option, free places filled by in-group probability);
finalists keep the final's distribution times the chance the answer is among them, every other option gets its group's
share of the final times its in-group probability. Context: 262,144 tokens; nothing is truncated (a longer prompt would
be refused; none was). Question types: `choice` and `noul`.

**How the run was executed (two phases, same engine, same output file).**

1. Rows 1–35,590 (BFCL through the start of POP909-CL) ran with the kit's own sequential runner (`run`, then
   `pipeline` resuming on the imported suite): `environment-sequential.json`. Every finished row's `payload_sha256`
   was checked against the imported suite before resuming (0 mismatches).
2. Rows 35,591–150,759 ran with `source/fast_run.py`, which answers 64 requests at a time: the prompts of all 64 are
   rendered by the same engine code and share length-sorted forward passes (prompts over 16,384 tokens run alone;
   right padding without an attention mask, which is exact for this causal model because pad tokens follow the last
   real token). `environment-batched.json`. The script's vLLM backend was added after this run and was not used; its
   default (`--backend hf`) path is the one that ran.

Agreement of the two execution paths, measured by re-running 3,000 randomly chosen phase-1 requests through the
batched driver: 7,044 of 7,051 answers identical (99.90 %); the 7 differences are near-ties (probability differences
of 0.004–0.013). Because of the batching, `total_wall_ms` in phase-2 rows is the batch wall time divided by 64, not a
per-request latency. Sequential single-request latency on the same GPU (compatibility pass, kit runner): median 77 ms.

**Hardware.** 1 × NVIDIA B200 (183 GB), torch 2.13.0 + CUDA 13.0, transformers 5.16.1, peft 0.21.0, kit `87d4650`.

**Training data and overlap.** The model was trained only on the public
[`SargeDev/jev-distill-corpus-v3`](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) (streams `yuri_v3`,
TypeSafe Jev 1.13 output distributions over synthetic operational scenarios; `openjev_v2`, Open-Jev synthetic
games/workflow tasks; `yuri_v1`, memory-relevance pairs with placeholder labels). Checkpoint selection and temperature
fitting used the corpus's own validation and calibration splits; no suite data was used. Check over the train,
validation and calibration splits (683,683 rows) against every suite state, instruction and option: **no whole state or
question of the suite appears in the training data**; zero sentence-level hits in `yuri_v3` and `openjev_v2`. 547
`yuri_v1` rows share Wikipedia sentences (≥ 60 characters) with suite contexts (HoVer 401, ANLI 154, BRIGHT 7,
ToolRet 1); those rows carry uniform `[0.5, 0.5]` placeholder labels, so they hold no answer information.

## `runs/jev-27b`

Same files, engine, suite, mechanical translation, capacity and training data as `runs/jev-9b` (the model is the same
recipe on `Qwen/Qwen3.8-27B` @ `1d4bf0f2`: rank-16 LoRA of 108,789,760 parameters and a 122,904-parameter head;
25,624,722,968 served parameters; temperatures noul 1.0143, choice 1.0161). On the model's own held-out test rows the
engine matches our reference inference to max |Δp| 5.1e-07 with identical argmax (300 of 300).

| area (skill) | Knowledge & Reasoning | Language | Retrieval & Classification | Tools | Arts |
|---|---:|---:|---:|---:|---:|
| JEV-27B | 0.398 | 0.574 | 0.537 | 0.735 | 0.399 |

Two phases, same output file: rows 1–2,361 (BFCL and the start of ToolRet) with the kit's sequential runner
(`environment-sequential.json`; `payload_sha256` of every finished row checked against the imported suite before
resuming, 0 mismatches), rows 2,362–150,759 with `source/fast_run.py`, 64 requests per batch
(`environment-batched.json`). Agreement check: the 1,694 phase-1 BFCL requests re-run through the batched driver give
4,768 of 4,768 identical answers (max |Δp| 0.016). Phase-2 `total_wall_ms` is batch time / 64. Hardware as above
(1 × NVIDIA B200).

## `runs/jev-gemma4-26b-a4b`

[autotrust/JEV-Gemma4-26B-A4B](https://huggingface.co/autotrust/JEV-Gemma4-26B-A4B) @ `603bfed4`: `google/gemma-4-26B-A4B-it`
@ `4d7ae498` (mixture of experts, ≈ 4 B active parameters per token) + LoRA adapter (merged in memory at load) + 24-slot fp32 head (67,608) soft-capped at 30 like Gemma's logits; read-out at the last token of
`<bos>` + the `bare-v1` template; temperatures noul 1.0027, choice 1.0172.

| area (skill) | Knowledge & Reasoning | Language | Retrieval & Classification | Tools | Arts |
|---|---:|---:|---:|---:|---:|
| JEV-Gemma4-26B-A4B | 0.430 | 0.636 | 0.679 | 0.697 | 0.415 |

**Execution.** One phase: all 150,759 rows with `source/fast_run.py` (64 requests per batch, shared length-sorted
forward passes, right padding without an attention mask, exact for this causal model), `environment-batched.json`.
`source/jev_engine.py` is the file that ran (sha256 `9f01ca19…`); the kit submission carries a later version that only
adds an option for unmerged-adapter bundles (not used by this model). Phase `total_wall_ms` is batch time / 64. Hardware
1 × NVIDIA B200, torch 2.13 + CUDA 13.0, transformers 5.16.1, peft 0.21.0, kit `87d4650`.

**Training data.** TypeSafe Jev 1.13 teacher distributions (the public `jev-distill-corpus-v3`) and ground-truth decision data from public datasets. The ground-truth data includes the public **training splits** of 12 datasets whose test splits the suite uses: BANKING77, CLINC150+OOS, WinoGrande, HellaSwag, GSM8K, ContractNLI, Amazon ESCI, HoVer, RAGTruth, When2Call, New Yorker caption matching and Habermas Machine. No test split of any suite benchmark was used; suite items were excluded before training; checkpoint selection and temperature fitting never used suite data. `training-overlap-check.json`: no whole suite state appears in the training data; the remaining instruction and sentence matches are the fixed per-benchmark templates.

Not affiliated with TypeSafe AI.
