---
license: cc-by-nc-4.0
language:
- zh
- en
library_name: vllm
pipeline_tag: audio-text-to-text
base_model:
- Qwen/Qwen3-VL-32B-Instruct
- Qwen/Qwen3-ASR-0.6B
tags:
- duplexjev
- speech-to-decision
- full-duplex
- speech-llm
- paralinguistics
- vllm
---

# DuplexJev-32B

> **Archived.** Superseded by [DuplexJev-32B-Para](https://huggingface.co/adventists-ai/DuplexJev-32B-Para). This is the version before the turn-taking upgrade (paper model, total 87.7); 32B-Para scores 92.3. This repository stays online for existing users and paper reproducibility, but is no longer updated.

📄 Paper: [arXiv:2610.02638](https://arxiv.org/abs/2610.02638) · Code: [github.com/adventists-ai/duplexjev](https://github.com/adventists-ai/duplexjev)

**Built on the language model of [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) + the
[Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) audio encoder.** Our recommended DuplexJev model.

DuplexJev reads typed, closed-set decisions from speech (*has the user finished? which filler fits? who is speaking,
in what mood?*) as a **single-token distribution** of a frozen LLM: zero decode steps, many questions about the same
clip in parallel. This repository is the **complete model** (encoder + trained connector + LLM, 33.0 B parameters,
bf16) and runs on [vLLM](https://github.com/vllm-project/vllm) with a small plugin.

| | |
|---|---|
| Audio encoder (frozen) | Qwen3-ASR-0.6B encoder, 12.5 frames/s |
| Connector (trained) | 2 frames stacked → 6.25 audio tokens/s → SwiGLU MLP → LLM width |
| LLM (frozen) | the language model of Qwen3-VL-32B-Instruct, weights unchanged, run as a text Qwen3 model (for text tokens the interleaved M-RoPE equals 1-D RoPE) |
| Training | content alignment (transcript distillation), then turn-taking + gender + emotion under the mixed objective |
| Inputs | audio + text. The vision encoder of Qwen3-VL is not included; image input is planned. |

## Serve with vLLM

```bash
pip install "vllm[audio]>=0.29" duplexjev-vllm
vllm serve adventists-ai/DuplexJev-32B --max-model-len 4096
```

The weights take about 62 GB, so one 80 GB GPU (H100, A100 80 GB) is enough; or split over two GPUs with
`--tensor-parallel-size 2`. `duplexjev-vllm` registers the model with vLLM (Qwen3-ASR encoder + connector; the LLM
runs on vLLM's own Qwen3 kernels).

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
# {'turn': {'answer': 'finished', 'confidence': 0.9, 'probs': {...}}, 'gender': {...}, 'emotion': {...}}
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
tokens) and set `logprobs=True`.

## Results

Paper protocol (single-token readout; % correct):

| model | qa100 | ZJU-ML | Easy-Turn | gender | emotion | total |
|---|---:|---:|---:|---:|---:|---:|
| **DuplexJev-32B** (this repo) | 90 | **87** | **79.1**\* | 89.4 | 90.6 | **87.7** |
| same LLM, content training only | 91 | 88 | 71.4 | 54.3 | 25.5 | 61.7 |
| [DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B) | 74 | 50 | 78.1\* | 88.6 | 91.2 | 78.6 |

Total = mean of main language (qa100, ZJU-ML, Easy-Turn) and paralinguistics (gender, emotion). Decision training
lifts gender and emotion from chance to about 90% while spoken QA stays where content-only training left it.
\* The decision stage includes the Easy-Turn **training** split (disjoint from the 800-item test set, different
wording), so Easy-Turn is in-domain for this model; without it the same LLM scores 71.3.

Spoken [VoiceBench](https://github.com/matthewcym/voicebench) (question spoken, options also given as text; same
single-token readout): OpenBookQA **83.7**, MMSU **70.8**. Reading the transcript as text, the frozen LLM scores
95.4 and 79.3.

The same weights served by vLLM 0.29 with `duplexjev-vllm`, questions in the wording of the client above (one H200,
bf16): qa100 94, ZJU-ML 88, Easy-Turn 79.4, gender 89.8, emotion 89.9. One decision event (10 questions about one
4.5 s clip, sent concurrently) through the OpenAI-compatible server takes about 240 ms end to end.

qa100: 100 bilingual spoken multiple-choice questions ([`adventists-ai/qa100`](https://huggingface.co/datasets/adventists-ai/qa100)).
ZJU-ML: main-language part of the ZJU audio benchmark v2.0.0. Easy-Turn: 800-item four-way turn-state test set
(see the [GitHub README](https://github.com/adventists-ai/duplexjev#6-benchmarks) for the reference).
Gender: 800 real utterances (AISHELL-1, LibriSpeech). Emotion: 800 utterances (ESD, CREMA-D).

## Limitations

- Evaluated on short read or acted speech; not on streaming input.
- Spoken QA is still below reading the transcript as text (about 9–12 points on VoiceBench).
- Emotion labels come from acted corpora; do not use the outputs to make decisions about individuals.

## License

**CC BY-NC 4.0, research use only.** The connector was trained on the ESD emotional speech corpus, which is licensed
for research purposes only, and on CREMA-D (ODbL). The base models keep their own licences: Qwen3-VL-32B-Instruct and
Qwen3-ASR-0.6B are Apache-2.0 (Copyright Alibaba Cloud); this repository redistributes their weights unchanged
(the language-model weights of Qwen3-VL-32B-Instruct, renamed; the vision encoder is not included).

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
