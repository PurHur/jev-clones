---
license: cc-by-nc-4.0
language:
- zh
- en
library_name: vllm
pipeline_tag: audio-text-to-text
base_model:
- adventists-ai/DuplexJev-4B-Turn
- Qwen/Qwen3-4B
- Qwen/Qwen3-ASR-0.6B
tags:
- duplexjev
- speech-to-decision
- spoken-dialogue
- full-duplex
- paralinguistics
- nonverbal
- speech-llm
- vllm
---

# DuplexJev-4B-Para (v1.1)

📄 Paper: [arXiv:2610.02638](https://arxiv.org/abs/2610.02638) · Code: [github.com/adventists-ai/duplexjev](https://github.com/adventists-ai/duplexjev) · Research note: [keeping a speech-decision LLM able to talk](https://github.com/adventists-ai/duplexjev/blob/main/research/2026-10-chat-replay.md)

**The small DuplexJev that hears the most beyond the words, and can still hold a normal conversation.**

> **v1.1 (2026-10-10)** replaces v1.0 (2026-10-09). v1.0 stays available as revision `v1.0`
> (`--revision v1.0` in vLLM, `revision="v1.0"` in `from_pretrained`).

Single-token decision training (one letter per answer) makes a speech LLM drift towards terse, one-line replies, and the
drift grows with every LoRA stacked on top. v1.1 mixes **conversation replay** into the same LoRA stage so the
model keeps the decision skills of v1.0 (spoken QA, gender, emotion, turn-taking, stress, non-verbal sounds, speaking
style, same-speaker) while answering free-form turns, multi-turn conversations included, like the original LLM.

Architecture: Qwen3-ASR-0.6B encoder + connector + Qwen3-4B, 4.2 B parameters, bf16,
~10 GB GPU memory.

**Which 4B to pick.** This one by default (it is the `duplexjev` package default). v1.0 if you only read single-token
decisions and need the last 1.5–2.6 points of emotion / VoiceBench or the higher sigh score; 4B-Turn if ZJU-ML and
turn-taking matter most; 4B-Stress if stress is the main thing you need.

## What changed

Starting from DuplexJev-4B-Turn: connector + a new rank-16 LoRA on the LLM (merged here), 4,000 steps. The mix is the v1.0 mix (4B-Turn mix, stress / pauses, speaker pairs, non-verbal sounds, speaking style, spoken multiple-choice replay; see revision `v1.0` for details) plus one more dataset (17.9 k items):

| added data | what it teaches |
|---|---|
| conversation replay: 13.4 k user turns (paralinguistic_dialogues with up to 6 earlier turns as text history, SmoothConv, AISHELL-1), target = the original Qwen3-4B's own reply to the transcript | keep normal, full-length replies and multi-turn behaviour |
| in-house chat items: 4.5 k, spoken (CosyVoice2) and typed | short everyday turns, repeat back what was said |

Two training details matter (both are in the open-source patches):

- replay rows get plain cross-entropy on the full reply. With the default mixed loss they would have gone to KL against
  a teacher that saw an empty message, and the model learned "you didn't send anything";
- **per-row** loss averaging. With token averaging, 256-token replies swamp the 1-token decision rows; stress scores fell
  12–20 points before this fix.

## Results

| | v1.0 | **v1.1** |
|---|---:|---:|
| qa100 / ZJU-ML | 79 / 50 | 77 / **58** |
| gender / emotion | 89.5 / 91.5 | 89.2 / 90.0 |
| VoiceBench OBQA / MMSU | 57.8 / 41.8 | 55.2 / 42.1 |
| Easy-Turn* | 91.5 | 91.0 |
| stress: DEBATE meaning / StressTest / StressPresso | 89.0 / 81.7 / 84.7 | 89.9 / 81.2 / 84.2 |
| laughter / breathing / cough / sigh (AUC, NonverbalTTS test) | 0.80 / 0.76 / 0.75 / 0.81 | 0.81 / 0.77 / 0.77 / **0.71** |
| laughter / breathing in natural Mandarin conversation (AUC) | 0.84 / 0.84 | 0.85 / **0.74** |
| speaking style, 7 classes (EARS) | 67% | 67% |
| multi-turn spoken conversation (mean reply tokens, cap 300) | 287 | 300 |

Paper protocol (single-token readout, PyTorch) for the decision rows. Through vLLM with `duplexjev-vllm` in the client
wording: qa100 76, ZJU-ML 51, gender 87.4, emotion 87.9, Easy-Turn 91.4 (v1.0: 76, 52, 87.2, 89.5, 91.2).

- \* Easy-Turn is in-domain.
- Single-turn spoken questions are still answered fairly briefly (about 40–50 tokens; the base LLM uses ~86). Multi-turn
  and typed chat are back to normal length.

## Serve with vLLM

```bash
pip install "vllm[audio]>=0.29" duplexjev-vllm
vllm serve adventists-ai/DuplexJev-4B-Para --max-model-len 4096
# previous version: add --revision v1.0
```

Decision prompts are the same as for [DuplexJev-4B-Turn](https://huggingface.co/adventists-ai/DuplexJev-4B-Turn); see [`examples/vllm_client.py`](https://github.com/adventists-ai/duplexjev/blob/main/examples/vllm_client.py). Always offer a "none" option for stress and non-verbal questions. For
conversation, send the user audio as the user turn (`<|audio|>`) with ordinary chat history and sample as usual.

## Limitations

- Emotion is 1.5, VoiceBench OBQA 2.6 and spoken qa100 2 points below v1.0; sigh (NonverbalTTS) and
  breathing in natural conversation lose about 0.1 AUC.
- Asked about itself, it may describe abilities (seeing images, speaking) that this checkpoint does not have on its own:
  it takes speech and text and replies in text.
- Asked what it can do, it sometimes lists generic assistant skills (writing code, writing emails) beyond the training
  answers.
- Speaker judgments are coarse and must not be used to identify or verify people; emotion outputs must not be used to
  make decisions about individuals.

## License

**CC BY-NC 4.0, non-commercial use only.**

- Non-commercial data:
  - Stress: DEBATE (github.com/SmileHnu/DEBATE), Stress-17K and TinyStress-15K (slprl);
  - Expresso (Meta), EARS (Meta) and NonVerbalSpeech-38K, all CC BY-NC 4.0;
  - ESD (research only); SmoothConv (CC BY-NC).
  - Conversation replay: user turns from wilzzzz/paralinguistic_dialogues (no licence stated on the dataset card; used for non-commercial research), SmoothConv (CC BY-NC), AISHELL-1 (Apache-2.0), LibriSpeech (CC BY 4.0); target replies generated by Qwen3-4B (Apache-2.0); spoken chat questions synthesised with CosyVoice2 (Apache-2.0).
- Other data:
  - VocalSound (CC BY-SA 4.0), NonverbalTTS (audio from VoxCeleb and Expresso);
  - VoxCeleb1 (CC BY 4.0), AISHELL-1 (Apache-2.0), LibriSpeech (CC BY 4.0);
  - ARC (CC BY-SA 4.0), OpenBookQA, CommonsenseQA (MIT).
- Base models keep their licences: Qwen3-4B and Qwen3-ASR-0.6B are Apache-2.0 (Alibaba Cloud); the Qwen3-4B licence
  is included. The LLM weights are Qwen3-4B with LoRAs merged in.

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

Built by AI降临派 / [Adventists.ai](https://github.com/adventists-ai/duplexjev). Claude (Anthropic) assisted with code.
