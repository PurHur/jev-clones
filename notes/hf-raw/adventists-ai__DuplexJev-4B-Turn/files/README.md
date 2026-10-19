---
license: cc-by-nc-4.0
language:
- zh
- en
library_name: vllm
pipeline_tag: audio-text-to-text
base_model:
- adventists-ai/DuplexJev-4B
- Qwen/Qwen3-4B
- Qwen/Qwen3-ASR-0.6B
tags:
- duplexjev
- speech-to-decision
- full-duplex
- turn-taking
- speech-llm
- vllm
---

# DuplexJev-4B-Turn

> **Archived.** Superseded by [DuplexJev-4B-Para](https://huggingface.co/adventists-ai/DuplexJev-4B-Para). 4B-Para keeps the same prompts and plugin and adds non-verbal sounds, speaking style, speaker, stress and normal conversation (total 82.5 vs 80.0). For live end-of-turn on a small GPU, use [DuplexJev-4B-Para-LiveTurn](https://huggingface.co/adventists-ai/DuplexJev-4B-Para-LiveTurn). This repository stays online for existing users and paper reproducibility, but is no longer updated.

📄 Paper: [arXiv:2610.02638](https://arxiv.org/abs/2610.02638) · Code: [github.com/adventists-ai/duplexjev](https://github.com/adventists-ai/duplexjev)

**[DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B) with a turn-taking upgrade.** Same size, same
serving stack, same prompt format; it decides much better *when to speak, when to keep talking and when to stop*,
while spoken QA, gender and emotion stay where they were.

DuplexJev reads typed, closed-set decisions from speech as a **single-token distribution** of the LLM: zero decode
steps, many questions about the same clip in parallel. This repository is the complete model (Qwen3-ASR-0.6B encoder
+ connector + Qwen3-4B, 4.2 B parameters, bf16).

## What changed

Starting from DuplexJev-4B we trained the connector together with a small **LoRA on the LLM** (rank 16 on q/k/v/o,
0.6 % of the parameters; merged into the weights here) on 110 k turn-taking questions rewritten from public
corpora, mixed with the original content and paralinguistic data:

| source | clips | what it teaches |
|---|---:|---|
| Easy-Turn training set (synthetic + real) | 44 k | Chinese: finished / unfinished / backchannel / "wait" |
| SmoothConv | 30 k | Chinese multi-party conversation with turn labels, with text history |
| otoSpeech + DualTurn frame labels | 12.5 k | English two-channel conversation: end of turn, pause, backchannel, interruption |
| DuplexGen Spoken | 14 k | English human–AI dialogues with text history |
| MUSAN noise, distant background voices | 9 k | sounds the assistant should ignore |

Most questions give the **system state** (the assistant is speaking / idle) and ask for an action
(keep talking / stop and listen; reply / keep waiting); the rest ask for the user's turn state.

## Results

| | qa100 | ZJU-ML | gender | emotion | VoiceBench OBQA / MMSU | Easy-Turn | CoDeTT zh / en | TurnBench-dev clips† |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DuplexJev-4B | 74 | 50 | 88.6 | 91.2 | 47.3 / 41.3 | 78.1 | 62.0 / 53.5 | 49.4 |
| **DuplexJev-4B-Turn** | 74 | 52 | 89.4 | 92.0 | 46.4 / 41.7 | **92.5** | **67.0 / 68.6** | **87.3** |

Paper protocol (single-token readout, PyTorch), % correct. Through vLLM with `duplexjev-vllm` in the client wording:
qa100 77, ZJU-ML 49, gender 87.0, emotion 88.4, Easy-Turn 92.8.

- **Easy-Turn**: 800-item four-way turn-state test set; the training mix contains the Easy-Turn training split
  (disjoint from the test set), so this score is in-domain.
- **CoDeTT** (arXiv 2603.25434): 18 k items, system state given, four actions averaged; zero-shot (no CoDeTT data
  in training). Our protocol: current utterance as audio, no history. For reference, published results: GPT-4o-audio
  66.6 / 71.9, Qwen3-Omni 70.4 / 70.9.
- † **TurnBench-dev clips**: our own clip-level protocol on the TurnBench development conversations (the official test
  labels are withheld): end of turn, pause inside a turn, backchannel, floor-taking interruption, noise / channel
  bleed; mean of four action accuracies. Not comparable with the official TurnBench leaderboard.
- Weakest remaining case: a pause in the middle of a long turn (the model takes the turn too early about half the time).

## Serve with vLLM

```bash
pip install "vllm[audio]>=0.29" duplexjev-vllm
vllm serve adventists-ai/DuplexJev-4B-Turn --max-model-len 4096
```

Prompt format, client and decision tables are the same as for
[DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B); see
[`examples/vllm_client.py`](https://github.com/adventists-ai/duplexjev/blob/main/examples/vllm_client.py).
A state-aware question, for example:

```
The user said: <|audio|>

Question: The user said this while the assistant was speaking. What should the assistant do?

Options:
A. keep talking (backchannel, noise, or not addressed to the assistant)
B. stop and listen (interruption or a new request)

Answer with only the letter of the correct option.
```

## Limitations

- Decides on one clip at a time; it is not a streaming end-of-turn detector by itself (call it at each pause).
- Pauses inside long turns are still often taken as the end of the turn.
- Emotion labels come from acted corpora; do not use the outputs to make decisions about individuals.

## License

**CC BY-NC 4.0, research use only** (emotion data ESD is research-only; SmoothConv is CC BY-NC). Base models keep their
licences: Qwen3-4B and Qwen3-ASR-0.6B are Apache-2.0 (Alibaba Cloud). The LLM weights here are Qwen3-4B with a LoRA
merged in.

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
