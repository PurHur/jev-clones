---
license: cc-by-nc-4.0
language:
- en
tags:
- decision-model
- typed-decisions
- classification
- multimodal
---

# StartLux-Decision-v1.1-27B-Instant

StartLux-Decision-v1.1-27B-Instant is a 27B decision model. You give it a state (text or JSON, with up to 8 images) and typed questions about it: pick one of several options (`choice`), yes or no (`noul`), or a level on a scale of 2 to 10 (`score`). It returns a probability for every option of every question from one forward pass, without generating text. The request and response format is compatible with the System One `/v1/systemone` API.

Formats: **BF16** · [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant-MLX-4bit) · Code: [StartLuxLabs/StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision) · Paper: [arXiv 2610.17309](https://arxiv.org/abs/2610.17309)

Family: [0.8B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant) · [2B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant) · [4B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant) · [4B AutoThink](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink) · [9B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant) · [9B AutoThink](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-AutoThink) · **27B Instant** · [27B AutoThink](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink)

<p align="center"><img src="https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant/resolve/main/figures/score_vs_size.png" width="860" alt="Decision Index 0.3 public score against model size"></p>

<p align="center"><em>Decision Index 0.3, public subset, as of 11 October 2026: the StartLux-Decision Instant models against other decision models of each size. Figure 1 of the <a href="https://arxiv.org/abs/2610.17309">technical report</a>.</em></p>

## How it answers

Each question is read as its own prompt that holds the whole state. At the last prompt position the logits of the option letters are divided by a temperature for the question type and normalised over the listed options. In the full-attention layers, one in four, prompt tokens also attend to later prompt tokens, so the state is read with the question in view, while the linear-attention layers stay causal. `config.json` declares this with `"jev_prompt_attention": "bidirectional"`, and an engine that reads the prompt causally serves a different model without an error.

Stage 1 fine-tuned all weights on about 0.8 million decision examples. Stage 2 trained an adapter with bidirectional prompt attention on about 1.7 million further examples and merged it. The released weights are the uniform average of four such runs that differ only in their random seeds.

## Results

| Model | Decision Index 0.3, public | DecisionPivot, pair accuracy |
|---|---|---|
| 27B AutoThink | 73.37 | 86.9 |
| **27B Instant (this model)** | **72.83** | **75.4** |
| 9B AutoThink | 70.45 | 59.9 |
| 9B Instant | 69.96 | 47.5 |
| 4B AutoThink | 68.21 | 59.9 |
| 4B Instant | 67.88 | 41.5 |
| 2B Instant | 59.14 | 21.1 |
| 0.8B Instant | 53.75 | 8.1 |

Decision Index 0.3 scores are our own runs of the public kit on its public subset, not leaderboard submissions. We compared training recipes and seed averages by these scores, so they reflect model selection and are not independent test results. At every size the Instant model leads the strongest evaluated decision model of comparable scale by 7.4 to 16.9 points. [DecisionPivot](https://huggingface.co/datasets/startlux-models/DecisionPivot) was not used for any data, checkpoint or configuration choice. Its core track has 7,800 program-generated examples in 13 families, mostly pairs that differ in one answer-changing detail, and a pair counts only when both examples are right. AutoThink rows use a gate of 0.9.

## Speed

Median latency of one read-out in milliseconds, one request in flight, by prompt length. vLLM 0.31 on one NVIDIA H200, MLX on an Apple M5 Pro with 64 GB of memory (a dash: does not fit in memory).

| Engine and precision | 256 tokens | 1,024 | 4,096 | 8,192 |
|---|---|---|---|---|
| vLLM BF16 | 52 | 98 | 358 | 737 |
| vLLM FP8 | 51 | 72 | 251 | 534 |
| MLX BF16 | – | – | – | – |
| MLX 8-bit | 1,384 | 4,294 | 17,008 | 32,002 |
| MLX 4-bit | 946 | 3,308 | 13,993 | 28,697 |

## Quick start

```bash
git clone https://github.com/StartLuxLabs/StartLux-Decision.git
cd StartLux-Decision
pip install -e '.[vllm]'      # vLLM 0.19.1, the version of the published scores; vLLM 0.31 is also supported
hf download startlux-models/StartLux-Decision-v1.1-27B-Instant --local-dir models/StartLux-Decision-v1.1-27B-Instant
startlux-decide serve --model models/StartLux-Decision-v1.1-27B-Instant --launch-vllm --gpus 0 --port 8090
```

Images and prompts over 65,536 tokens need vLLM 0.31. With several GPUs, `--gpus 0,1,2,3` starts one vLLM server per GPU behind the same decision server.

```bash
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
  "questions": {
    "team":     {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments and refunds", "shipping": "Delivery", "technical": "App and login"}},
    "urgent":   {"type": "noul", "instructions": "Should this ticket be answered today?"},
    "severity": {"type": "score", "instructions": "How severe is the impact?",
                 "criteria": ["cosmetic", "annoying", "blocks the customer"]}
  }
}'
```

The response has this shape:

```text
{"answers": {
   "team":     {"type": "choice", "choice": <id>, "confidence": <c>,
                "probabilities": {"billing": <p>, "shipping": <p>, "technical": <p>},
                "none_probability": <p>},
   "urgent":   {"type": "noul", "noul": <P(yes)>},
   "severity": {"type": "score", "score": <expected level>, "confidence": <c>,
                "legend": {"0": "cosmetic", "1": "annoying", "2": "blocks the customer"},
                "probabilities": {"0": <p>, "1": <p>, "2": <p>}}},
 "usage": {"input_tokens": <n>, "output_tokens": <0>},
 "model": "StartLux-Decision-v1.1-27B-Instant"}
```

This model never thinks: a request with `think_mode` `auto` or `always` gets status 422. `return_details` adds the read-out and its timing.

Every choice answer carries `none_probability`, the share of the read-out that goes to an extra "None of the above" option the server adds internally. `choice` stays one of your options, and `probabilities` covers only your options. With `"allow_none": true`, `choice` is null when "None of the above" is the most likely option. [docs/none_probability.md](https://github.com/StartLuxLabs/StartLux-Decision/blob/main/docs/none_probability.md) explains how to read it and why we suggest acting on it from 0.2. [docs/api.md](https://github.com/StartLuxLabs/StartLux-Decision/blob/main/docs/api.md) has the full format, including images (base64 strings or data URIs, at most 8 per request).

## Engines and formats

On vLLM, `serve --launch-vllm` enables the package's plugin, which reads the prompt bidirectionally, and stops unless every vLLM server confirms it. The GGUF files carry llama.cpp's non-causal flag and need the server settings of the GGUF repository, and the package's MLX backend applies the bidirectional mask itself. Plain vLLM, llama.cpp, mlx-lm or mlx-vlm read the prompt causally and give a different model.

| Build | Decision Index 0.3 | DecisionPivot, chance-corrected |
|---|---|---|
| BF16, vLLM | 72.89 | 80.5 |
| FP8 at load (`--quantization fp8`), vLLM | 72.91 | 80.5 |
| GGUF Q8_0, llama.cpp | | 80.7 |
| GGUF Q4_K_M, llama.cpp | | 80.2 |

These are runs of the served engines, so the BF16 row differs slightly from Results. The DecisionPivot column here is the mean chance-corrected accuracy over the benchmark's seven scoring layers, not pair accuracy. Q8_0 is the recommended GGUF file. MLX 8-bit builds agree with BF16 on the top option for 99.5% to 99.6% of questions, 4-bit builds for 96.3% to 97.5%.

## Limitations

One read covers at most 26 options. Longer choice lists go through group rounds (groups of at most 25, the three best of each to a final read, the rest sharing 0.001 of the probability). `none_probability` is an alarm score, not a calibrated probability, and it is weak where no single answer is right (preferences, moves in a game) and on questions the model cannot answer itself. A confidence used to select or route answers overstates their accuracy, so set thresholds on the answers your own policy returns.

## License

The weights are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): free for research and other non-commercial use, with attribution. Commercial use requires a separate license from StartLux Labs (contact@startlux.com). The inference package is Apache-2.0. See [LICENSE](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant/blob/main/LICENSE) and [NOTICE](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant/blob/main/NOTICE).

## Citation

```bibtex
@misc{startlux2026decision,
  title         = {StartLux-Decision: AutoThink Decision Models with Reinforcement Learning for Calibrated Deliberative Decisions},
  author        = {Zhuang, Xialie and Su, Zunhai and Pan, Jia and Wong, Ngai and Chen, Guanhua and Kuo, Chuan-Wei},
  year          = {2026},
  eprint        = {2610.17309},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.17309}
}
```
