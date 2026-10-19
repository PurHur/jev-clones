---
license: apache-2.0
base_model: google/gemma-4-31B-it
language: [en]
pipeline_tag: text-classification
tags: [decision-model, calibrated, structured-output, system-one, one-pass, nvfp4]
---


# decider-31b: typed decisions with calibrated probabilities on Gemma-4-31B, one pass, under one second

decider-31b answers typed decisions with a probability for every option. It reads each question in one forward pass and
generates no text.

- **Input:** a state and one or more typed questions (Choice, Noul yes/no, Score), each with an explicit option list, in the
  System One wire format (`POST /v1/systemone`).
- **Output:** a probability distribution over the options of every question. Nothing needs parsing.
- **Weights:** [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) with a fine-tune merged into the
  attention and MLP projections, quantized to NVFP4 (MLP weights and activations in NVFP4, attention projections in bf16,
  FP8 KV cache). About 31 GB.
- **Code:** `decider.serve_vllm` in [decider-ai](https://github.com/Mapika/decider) 1.9.0. It reads this repository's
  `decider_config.json`, which holds the configuration measured below.
- **No thinking.** This model never generates a reasoning trace. Every request is answered in one or two forward passes.

## Results

Decision Index 0.3, public suite, all 140,178 requests (kit `--edition 0.3`), run by us on 2026-10-07/08 with exactly this repository (commit cf50c0b) and decider-ai 1.9.0 from PyPI, one server per B300. The
board's full score also includes private tests that only the maintainers run; this is the public part only.

| | decider-31b |
|---|---|
| Public index (chance-corrected) | **64.27** |
| Public raw | 73.24 |
| Coverage | 1.0 |
| Calibration, ECE_bw | 0.039 |

| Area | Skill |
|---|---|
| Knowledge & Reasoning | 50.32 |
| Language Understanding | 66.69 |
| Retrieval & Classification | 65.47 |
| Tools & Automation | 84.23 |
| Arts & Human Taste | 55.19 |

Public-index scores of other entries, as shown on the board on 2026-10-06 (their own runs, not ours):

| Entry | Public index |
|---|---|
| decider-31b (this, our run) | 64.27 |
| Torchcast Decision 27B | 65.10 |
| Perplexity Decider v1.1 (27B) | 62.25 |
| Quyet-1.0-Large | 60.79 |
| Fastino GLiDE no-thinking (28B) | 59.06 |
| Jev | 57.96 |
| Decider chat · Gemma-4-31B (stock weights, our previous entry) | 57.79 |

Notes on the numbers:

- The run was not verified by the Decision Index maintainers and is not on the board.
- ECE_bw is the benchmark-weighted expected calibration error over the 214,304 scored fields whose rows are in the suite's
  `selected-rows` file. Rows outside that file are not part of the calibration figure.
- Results: [Mapika/decision-index-results, runs/decider-31b](https://huggingface.co/datasets/Mapika/decision-index-results/tree/main/runs/decider-31b)
  (gated). Our development server gave 64.23 and ECE_bw 0.035 on the same suite; the figures above are from the released code.
- The temperatures were fitted on our own rows only. No Decision Index row was used to fit or choose them.
- Training used the public train splits of several benchmarks that the index also tests (see Training data). No Decision
  Index test row was used: every training row was compared with all suite rows and dropped on a match.

## Latency

Measured one request at a time over HTTP on one B300, 400 requests drawn from the Decision Index 0.3 suite:

| | median | mean | p80 | p95 | max |
|---|---|---|---|---|---|
| decider-31b (`decider.serve_vllm`, this repository) | 39 ms | 116 ms | 203 ms | 531 ms | 1.45 s |
| stock Gemma-4-31B-it NVFP4, one reading, for reference | 31 ms | 112 ms | 207 ms | 526 ms | 1.56 s |

The Decision Index measures latency on an RTX PRO 6000. We have not measured on that GPU. Our previous entry (stock
Gemma-4-31B) measured 3.2 times slower there on the mean than on our B300; on that factor this model's mean and p80 stay
below one second. Treat that as an estimate.

## Usage

vLLM 0.29.0 pins its own torch, so install in a separate environment:

```bash
python -m venv decider-vllm && . decider-vllm/bin/activate
pip install vllm==0.29.0 fastapi "uvicorn[standard]" jinja2 huggingface_hub
pip install --no-deps "decider-ai>=1.9.0"
DECIDER_MODEL=Mapika/decider-31b uvicorn decider.serve_vllm:app --host 127.0.0.1 --port 8000
```

It needs one GPU with NVFP4 support (Blackwell: B200/B300, RTX PRO 6000, RTX 50 series with enough memory).

```python
import httpx
r = httpx.post("http://127.0.0.1:8000/v1/systemone", timeout=60, json={
    "state": {"ticket": "Order #1182 arrived broken. Customer asks for a refund; order is 40 days old; policy: 30 days."},
    "questions": {"refund": {"type": "noul", "instructions": "Is the refund allowed under the policy?",
                             "criteria": {"true": "allowed", "false": "not allowed"}}}})
print(r.json()["answers"]["refund"])
```

## How a request is answered

1. **First reading.** Every question is rendered in the model's chat template with thinking off, and the log-probabilities of
   the option letters are read at the answer slot. A softmax at T1(n) = 2.042 + 0.003 ln n (n = number of options) gives
   the probabilities.
2. **Second reading, only when unsure.** A question whose top probability is below 0.7 is read again with its options in
   reversed order. The two log-softmaxes are averaged and a softmax at T2(n) = 1.953 + 0.010 ln n gives the final
   probabilities. On the Decision Index public suite this touches 28% of the questions; it removes part of the option-order
   bias at a small latency cost.
3. Questions above 0.7 keep the first reading.

The thresholds and temperatures are in `decider_config.json` (`temperature_by_options`, `second_reading`). They belong to
these weights. `DECIDER_SECOND_READING_BELOW=0` turns the second reading off (one reading per question; slightly faster, less
accurate). The config also selects vLLM's Triton attention backend (`vllm_attention`): with this checkpoint's FP8 KV cache,
the FlashInfer backend returned NaN in batched readouts in our tests.

## Training data

The fine-tune was trained on decision items only, in the same layout as serving. The data:

- **New for this model:** rows from public train splits of tasks where our previous entry was weak, rendered in the Decision
  Index request format, and decontaminated against every Decision Index 0.3 suite row:
  - CLINC150 (intent classification, including out-of-scope), train split;
  - iSarcasmEval (task A, English), train file;
  - VAST (stance), train split;
  - ACOS (aspect-category sentiment, laptop and restaurant), train splits;
  - POP909-CL (chord labels), songs not used by the index;
  - GSM8K, train split; NLI4CT, train split;
  - generated smart-home requests (device states, household policies) from our own generator, with homes, rooms, devices
    and request texts drawn at random.
- **From our earlier decider models:** generated state-tracking decisions and decision families written by our own code,
  document questions written by Qwen3.6-27B, and labelled rows from public datasets (MMLU, ARC, CommonsenseQA, BoolQ, MNLI,
  SNLI, BANKING77, RACE, OpenBookQA, LogiQA 2.0, MedQA, WinoGrande).

Not used: Decision Index test rows, JevBench items.

## Limitations

- **One pass only.** Tasks that need several steps of reasoning (GPQA 0.35, ChessBench 0.16, HLE 0.00 skill on the index) are
  not solved by reading harder. A thinking variant is a separate model.
- **Calibration:** ECE_bw 0.039 on the index. The model is underconfident on the tasks it was trained on (language area:
  accuracy 0.85 at confidence 0.74) and closer to calibrated elsewhere. Check calibration on your own data before relying on
  the probabilities.
- **Trained skills vs. new tasks.** On a 4,500-request index sample, compared with our previous readout adapter, the 8
  benchmarks whose train splits were used gained 17 points of skill on average; the other 29 were unchanged on average
  (+0.6), with losses of 0.03 to 0.09 on SATA-Bench, Amazon ESCI, HoVer, GPQA and ChessBench (about 88 requests each, so
  partly noise).
- **Hardware:** NVFP4 needs a Blackwell GPU. We measured only on B300.
- **vLLM version:** the server patches a vLLM 0.29 internal and refuses other versions.
- **Independent questions only:** `/decide` and the schema cache of `decider.serve` are not served.

## License

The weights are released under Apache 2.0, as the base model. Training sources and their terms:

- CLINC150: CC BY 3.0 (Larson et al., 2019). Credit: "An Evaluation Dataset for Intent Classification and Out-of-Scope
  Prediction", clinc/oos-eval.
- iSarcasmEval: MIT (Abu Farha et al., 2022). The texts are tweets.
- POP909-CL: MIT; built on POP909 (Wang et al., 2020), MIT.
- VAST (Allaway and McKeown, 2020) and ACOS (Cai et al., 2021): no licence stated by the authors.
- GSM8K (Cobbe et al., 2021): MIT.
- NLI4CT (SemEval-2024 Task 2, Jullien et al.): no licence file in the task repository; released for the shared task.
- RACE: non-commercial research use. It is a small part of the replayed data (250 rows).
- ARC, SNLI, LogiQA 2.0 (copy dependent): CC BY-SA 4.0; BoolQ: CC BY-SA 3.0; BANKING77, MedQA: CC BY 4.0; MMLU,
  CommonsenseQA: MIT; MNLI: mixed per genre.

