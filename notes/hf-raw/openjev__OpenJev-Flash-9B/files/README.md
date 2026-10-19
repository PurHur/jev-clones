---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3.5-9B
language:
- en
tags:
- decision-model
- zero-shot-classification
- classification
- calibrated-probabilities
- vllm
---

# OpenJev Flash 9B

**Typed decisions about text with up to 52 options, answered in one forward pass per question. About 40 ms per decision on one H100, 1.6x faster than OpenJev 27B on the same setup.**

**On par with Cloudflare's Clef-Flash and ahead of Kev-9B and Nimble 9B on JevBench.**

No free-form text to parse. No chain of thought. No training per task.

![OpenJev Flash 9B in five numbers](assets/hero_stats.png)

OpenJev Flash 9B is the small, fast member of the [OpenJev](https://huggingface.co/openjev/openjev) family: an open-weights **decision model** built on [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). You describe the decision in plain words at request time, with your own labels. It answers with a choice, a yes / no probability or a score. It uses the same API and the same helper as OpenJev 27B, so a client written for one works with the other.

```bash
curl -s http://localhost:3000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "openjev-flash-9b",
  "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
  "questions": {
    "route":   {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": null, "shipping": null, "technical": null}},
    "angry":   {"type": "noul",   "instructions": "Is the customer angry?"},
    "urgency": {"type": "score",  "instructions": "How urgent is this?",
                "criteria": ["can wait", "this week", "today", "right now"]}
  }
}'
```

This request asks three questions and gets back three typed answers: a choice and a score with a probability for every option, and a yes / no probability. Each question takes one forward pass (up to 52 options).

## Why use it

- **Labels live in the request.** You need no labelled data, no training run and no fixed label set. For a new task, new labels or a new domain, change the JSON, not the model.
- **Not a chatbot you have to parse.** The answer is read from the scores at the first output position, one score per option. Fixed calibration settings turn those scores into probabilities.
- **Small and cheap to run.** About 19 GB of 16-bit weights, served in FP8 on one GPU. It takes 40.4 ms per decision on one H100 when requests run one at a time, about 4.6 US cents per 1,000 decisions at $4.09 per GPU-hour. An 8-bit MLX build for Apple silicon is about 9.5 GB.
- **Same API as OpenJev 27B.** Start with the 9B and move to the 27B where you need the extra accuracy, without changing client code.

## Use cases

| job | example decision |
|---|---|
| Routing and triage | intent, topic, team, priority |
| Moderation and safety | toxicity, spam, hate, policy violation |
| Judging other models | does the response satisfy the request, is it grounded, does it follow the rubric |
| Documents and ops | apply a written policy to a case, invoice approve / hold / reject, alert severity |
| Scoring | ordered levels with an expected value and a confidence |

## Results

### JevBench: on par with Clef-Flash, ahead of Kev-9B and Nimble 9B

[JevBench](https://github.com/fstandhartinger/jevbench) is the public benchmark for Jev-style decision models. On its 231 public items:

| model | correct | accuracy |
|---|---|---|
| Clef-Flash (Cloudflare) | 190 of 231 | 82.3% |
| **OpenJev Flash 9B** | **188 of 231** | **81.4%** |
| Kev-9B | 183 of 231 | 79.2% |
| Nimble 9B (Bespoke Labs) | 183 of 231 | 79.2% |

OpenJev Flash 9B is on par with Cloudflare's Clef-Flash and ahead of Kev-9B and Nimble 9B. No JevBench item was used to train or tune it.

![JevBench public items by family: OpenJev Flash 9B leads on ambiguous cases (6 of 7) and on judging responses (11 of 17)](assets/jevbench_families.png)

Where it stands out: **ambiguous cases** (6 of 7, the others 3 to 4) and **judging whether a response does what was asked** (11 of 17, the others 9 to 10). It gets every easy item, every trap and every routing item right.

### 10,000 text questions

The same 10,000 questions from 34 public sources:

| model | accuracy |
|---|---|
| Jev (hosted API) | 85.4% |
| OpenJev 27B | 84.1% |
| **OpenJev Flash 9B** | **79.4%** |

### Speed and cost

![Time and cost per decision on one H100: OpenJev Flash 9B 40.4 ms and 4.6 US cents per 1,000 decisions, OpenJev 27B 65.7 ms and 7.5 cents](assets/latency.png)

- **40 ms per decision** on one H100 (FP8), 1.6x faster than OpenJev 27B.
- **About 4.6 US cents per 1,000 decisions** at $4.09 per GPU-hour.
- **About a quarter of a second per decision on a Mac** with the MLX build.

## Quick start

```bash
pip install "vllm==0.29.0" "openai==3.26.0" "httpx==0.28.1"
hf download openjev/OpenJev-Flash-9B helper/shim.py --local-dir openjev-flash-9b
```

Terminal 1, the model (downloads the weights on first start):

```bash
vllm serve openjev/OpenJev-Flash-9B --host 127.0.0.1 --served-model-name qwen --port 8000 \
  --enable-prefix-caching --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":1}' --trust-remote-code --max-num-seqs 16 \
  --max-logprobs 64 --gdn-prefill-backend triton --quantization fp8
```

Terminal 2, the decision API in front of it:

```bash
VLLM=http://127.0.0.1:8000/v1 TOKENIZER=openjev/OpenJev-Flash-9B \
READOUT_T=1.07 READOUT_NOUL_T=1.074766 READOUT_NOUL_BIAS=0 \
READOUT_TARGETED=1 READOUT_INSTR_STYLE=pyrepr SHIM_STAGGER=1 \
python openjev-flash-9b/helper/shim.py --host 127.0.0.1 --port 3000
```

Then `POST /v1/systemone` as in the example at the top. `curl -s http://127.0.0.1:3000/v1/version` should show `T: 1.07`, `noul_t: 1.074766`, `noul_bias: 0`, targeted readout, `instr_style: "pyrepr"`, and a helper SHA256 beginning `81a22f1b`. The helper's built-in defaults belong to OpenJev 27B, so always pass the five `READOUT_*` settings above. Keep `--served-model-name qwen` and `--max-logprobs 64`, because the helper requests the option scores from that model name. For the full serving guide, the Apple silicon path and the security notes for exposing the port, see [`serve/SERVE.md`](serve/SERVE.md).

The request and response shapes follow the hosted Jev API, so you can point a client written for it at your own server.

## Formats

| format | repository | for |
|---|---|---|
| 16-bit (bfloat16), about 19 GB | `openjev/OpenJev-Flash-9B` | This repository. vLLM on one GPU, served in FP8. |
| FP8, 11.9 GB | [`openjev/OpenJev-Flash-9B-FP8`](https://huggingface.co/openjev/OpenJev-Flash-9B-FP8) | vLLM with nothing quantized at load time; matches the 16-bit model. |
| MLX 8-bit, about 9.5 GB | [`openjev/OpenJev-Flash-9B-MLX`](https://huggingface.co/openjev/OpenJev-Flash-9B-MLX) | Apple silicon. The build behind the JevBench numbers above. |
| MLX 4-bit, about 5.0 GB | [`openjev/OpenJev-Flash-9B-MLX-4bit`](https://huggingface.co/openjev/OpenJev-Flash-9B-MLX-4bit) | Apple silicon with less memory. |
| GGUF Q4_K_M to Q8_0, 5.6 to 9.5 GB | [`openjev/OpenJev-Flash-9B-GGUF`](https://huggingface.co/openjev/OpenJev-Flash-9B-GGUF) | llama.cpp on laptops, consumer GPUs and CPUs. |

## Request limits

- Up to 52 options per question in a single pass; larger option sets use several passes.
- Prompts up to 16,384 tokens in the documented vLLM setup.
- Text, JSON and DOM input.

## How it works, in one paragraph

A language model computes a score for every possible next token before it writes anything. OpenJev Flash 9B gives each of your options a letter, asks the question, and reads the scores of exactly those letters at the first output position. The server is asked for a single token, never a free-form answer. For questions with up to 52 options, one forward pass produces one number per option, and a fixed calibration step turns those numbers into probabilities you can threshold. The base model was fine-tuned for typed decisions with supervised and teacher-guided fine-tuning. The training recipe and data are not released.

## Licence

OpenJev Flash 9B weights are released under **CC BY-NC 4.0**: free for research and other non-commercial use, with attribution. For a commercial licence, email [support@loopai.com](mailto:support@loopai.com).

The files in `helper/` and `serve/` are Apache 2.0. OpenJev Flash 9B is built on [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`), which is Apache 2.0. The required attribution is in [`NOTICE`](NOTICE) and the Apache text is in `LICENSE-APACHE-2.0`.

OpenJev is an independent project, not affiliated with TypeSafe; Jev is their product.
