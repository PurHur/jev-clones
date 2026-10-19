---
license: cc-by-nc-4.0
language:
- zh
- en
library_name: vllm
pipeline_tag: audio-text-to-text
base_model:
- Qwen/Qwen3-4B
- Qwen/Qwen3-ASR-0.6B
tags:
- duplexjev
- speech-to-decision
- full-duplex
- speech-llm
- paralinguistics
- vllm
---

# DuplexJev-4B

> **Archived.** Superseded by [DuplexJev-4B-Para](https://huggingface.co/adventists-ai/DuplexJev-4B-Para). This is the version before the turn-taking upgrade (paper model, total 78.6); 4B-Para scores 82.5. This repository stays online for existing users and paper reproducibility, but is no longer updated.

📄 Paper: [arXiv:2610.02638](https://arxiv.org/abs/2610.02638) · Code: [github.com/adventists-ai/duplexjev](https://github.com/adventists-ai/duplexjev)

**Built on [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) + the [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) audio encoder.**

DuplexJev reads typed, closed-set decisions from speech (*has the user finished? which filler fits? who is speaking,
in what mood?*) as a **single-token distribution** of a frozen LLM: zero decode steps, many questions about the same
clip in parallel. This repository is the **complete model** (encoder + trained connector + LLM, 4.2 B parameters, bf16)
and runs on [vLLM](https://github.com/vllm-project/vllm) with a small plugin.

| | |
|---|---|
| Audio encoder (frozen) | Qwen3-ASR-0.6B encoder, 12.5 frames/s |
| Connector (trained, 12.6 M) | 2 frames stacked → 6.25 audio tokens/s → SwiGLU MLP → LLM width |
| LLM (frozen) | Qwen3-4B, used without thinking |
| Training | content alignment (transcript distillation), then turn-taking + gender + emotion under the mixed objective (2026-10-01 update) |
| Try it online | [api.adventists.cn/duplexjev](https://api.adventists.cn/duplexjev/) (web demo and trial API, this model) |

## Serve with vLLM

```bash
pip install "vllm[audio]>=0.29" duplexjev-vllm
vllm serve adventists-ai/DuplexJev-4B --max-model-len 4096
```

`duplexjev-vllm` registers the model with vLLM (Qwen3-ASR encoder + connector; the LLM runs on vLLM's own Qwen3
kernels). Tested with vLLM 0.29 on one GPU; the model needs about 8 GB plus KV cache.

Each question is one chat request with `max_tokens=1`: list the options under letters and read the log-probabilities
of the letters. Requests about the same clip share the prompt prefix (chat header + audio), which vLLM's prefix cache
computes once. A ready-made client (`openai` + standard library only) is
[`examples/vllm_client.py`](https://github.com/adventists-ai/duplexjev/blob/main/examples/vllm_client.py):

```python
from vllm_client import DuplexJevClient

dj = DuplexJevClient("http://localhost:8000/v1")
dj.decide(open("call.wav", "rb").read(), {
    "turn":    ("Has the user finished speaking?", ["finished", "not finished"]),
    "gender":  ("What is the perceived gender of the speaker?", ["female", "male"]),
    "emotion": ("What is the speaker's emotional state?", ["neutral", "happy", "angry", "sad"]),
})
# {'turn': {'answer': 'finished', 'confidence': 0.88, 'probs': {...}}, 'gender': {...}, 'emotion': {...}}
```

The prompt it sends, which is the format the connector was trained on:

```
The user said: <|audio|>

Question: What is the perceived gender of the speaker?

Options:
A. female
B. male

Answer with only the letter of the correct option.
```

For Chinese speech ask in Chinese: `用户说：<|audio|>`, `问题：…`, `选项：`, `请只回答正确选项的字母。`
Send the audio as an `input_audio` part next to that text; restrict the answer with `allowed_token_ids` (the letter
tokens) and set `logprobs=True`. The chat template turns thinking off by default.

## Results

Paper protocol (single-token readout; % correct):

| version | qa100 | ZJU-ML | Easy-Turn | gender (800) | emotion (800, 4-way) | total |
|---|---:|---:|---:|---:|---:|---:|
| **current (2026-10-01)** | 74 | 50 | **78.1**\* | 88.6 | 91.2 | **78.6** |
| first release (2026-09-28, in the repo history) | 72 | 51 | 60.2 | 89.4 | 91.9 | 75.9 |

\* The current version's decision stage includes the Easy-Turn **training** split (disjoint from the 800-item test set,
different wording), so Easy-Turn is in-domain for it. Total = mean of main language (qa100, ZJU-ML, Easy-Turn) and
paralinguistics (gender, emotion).

Served by vLLM 0.29 with `duplexjev-vllm`, questions in the wording of the client above: qa100 74, ZJU-ML 52,
Easy-Turn 77.9, gender 87.8, emotion 86.0. One decision event (8 questions about one clip) takes about 80 ms on a
shared H100.

qa100: 100 bilingual spoken multiple-choice questions ([`adventists-ai/qa100`](https://huggingface.co/datasets/adventists-ai/qa100)).
ZJU-ML: main-language part of the ZJU audio benchmark v2.0.0. Easy-Turn: 800-item four-way turn-state test set
(see the [GitHub README](https://github.com/adventists-ai/duplexjev#6-benchmarks) for the reference).
Gender: 800 real utterances (AISHELL-1, LibriSpeech). Emotion: 800 utterances (ESD, CREMA-D).

## Limitations

- Evaluated on short read or acted speech; not on streaming input.
- Spoken factual QA is much weaker than in [DuplexJev-32B](https://huggingface.co/adventists-ai/DuplexJev-32B).
- Emotion labels come from acted corpora; do not use the outputs to make decisions about individuals.

## License

**CC BY-NC 4.0, research use only.** The connector was trained on the ESD emotional speech corpus, which is licensed
for research purposes only, and on CREMA-D (ODbL). The base models keep their own licences: Qwen3-4B and
Qwen3-ASR-0.6B are Apache-2.0 (Copyright Alibaba Cloud); this repository redistributes their weights unchanged.

## Citation

```bibtex
@misc{jin2026duplexjev,
  title         = {Batched Speech Decisions Without Decoding: Single-Token Supervision Lets a Frozen {LLM} Hear Beyond the Transcript},
  author        = {Jin, Jie and Ma, Ziyin and Yin, Min and Chen, Jinyu and Song, Haigang and Pang, Zhikun and Zhang, Xiaowen},
  year          = {2026},
  eprint        = {2610.02638},
  archivePrefix = {arXiv},
  note          = {Submitted to IEEE ICASSP 2027}
}
```

Built by [Adventists.ai](https://github.com/adventists-ai/duplexjev). Claude (Anthropic) assisted with code.
