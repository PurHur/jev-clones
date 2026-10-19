---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
library_name: transformers
language:
- en
- ja
tags:
- decision-engine
- decision-model
- system-one
- classification
- text-classification
- zero-shot-classification
- gemma
---

# Shisa DE-2

Shisa DE-2 (Decision Engine 2) is a general-purpose decision model. It takes one state and a
list of named questions and returns one typed answer per question: yes/no, or one of up to
256 options, each with a probability. No answer text is generated. Each answer is read from
the next-token distribution at the answer boundary, restricted to the option codes.

DE-2 is based on [Gemma 4 26B A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) with a 
rank-16 LoRA on the language model's attention projections merged into the weights. It was 
trained with softmax cross-entropy over the restricted option distribution, which is the same 
readout used at inference. The vision encoder is present in the weights and was not 
specifically trained, but which has been evaluated to function extremely well for decision tasks.

## What changed from DE-1

DE-2 supercedes our prior [Shisa DE-1](https://huggingface.co/shisa-ai/shisa-de-1) model with some important upgrades:

| | DE-1 | DE-2 |
| --- | --- | --- |
| Options per question | 26 (letters `A`–`Z`) | 256 (letters, then two-letter single-token codes `AA`, `AB`, …) |
| Option order in training | As authored | Shuffled |
| Read-Out| One pass, compound | Adaptive Inference Policy |
| Training data | 3,632 EN/JA samples | 14,637 samples per epoch from 28 families, EN/JA |
| Decision Index 0.3, public index | 50.19 | **58.96** |

The Shisa DE models are designed to run on existing inference engines (vLLM, SGLang, llama.cpp, etc)
without requiring any modifications. You can manually interact with the models using the readout contract
and inference policy described below, however we also have a [Shisa DE SDK](https://github.com/shisa-ai/shisa-de) 
that makes interaction easier.

## Model details

| Field | Value |
| --- | --- |
| Model id | `shisa-ai/shisa-de-2` |
| Developed by | Shisa AI |
| Model type | Causal language model; no task head is added to the graph |
| Base model | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| Architecture | Gemma 4 mixture of experts: 30 layers, 128 experts with 8 active plus 1 shared, 2,816 hidden size, 262,144-token vocabulary |
| Parameters | 25.2B total and 3.8B active per the base model card |
| Trained parameters | 11.5M (LoRA rank 16, alpha 32, on `q_proj`, `k_proj`, `v_proj`, `o_proj` of the language model), merged |
| Precision | bfloat16 |
| Size on disk | 51,644,315,363 bytes (48.1 GiB) in 3 safetensors shards |
| Max context used in evaluation | 262,144 tokens |
| License | Apache-2.0 |

## Serving

### Readout contract

Send one request per question. Render the state, the question and the option list with the
chat template, then read `logprobs` at the answer position with `max_tokens=1`. Restrict the
logits to the valid option codes and normalize. Take the argmax for a choice answer, or the
probability of the Yes row for a `noul` answer. The model's unrestricted next-token
distribution is not part of the contract.

The scaffold is a system instruction and a JSON payload in the user message:

```text
<bos><|turn>system
Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.<turn|>
<|turn>user
{"evidence": <state>, "criterion": <question>, "options": [{"letter": "A", "description": ...}, {"letter": "B", "description": ...}]}<turn|>
<|turn>model
```

Options are labelled `A` to `Z`, then `AA`, `AB` and so on up to 256. Every code is a single
token. A `noul` question uses the option pair Yes/No.

### Launch

DE-2 serves on mainline vLLM with no special options:

```
vllm serve shisa-ai/shisa-de-2
```

Then use the [Shisa DE SDK](https://github.com/shisa-ai/shisa-de), which applies the readout
contract, the adaptive policy and the calibration below:

```python
from shisa_de import DecisionModel

de = DecisionModel.from_endpoint("http://127.0.0.1:8000/v1", model="shisa-ai/shisa-de-2", api_key="")
de.classify({"ticket": "I was charged twice."}, {"intent": ["refund", "cancel"]})
```

### Requirements

- **vLLM attention backend.** Pass `attention_config={"backend": "TRITON_ATTN"}` on Hopper
  GPUs (H100, H200, H20) vLLM 0.30.0 otherwise runs this base model's 512-wide attention
  heads [through a FlashAttention 4 path that lowers accuracy](https://github.com/vllm-project/flash-attention/pull/203): 48.59 against 52.58 on a Decision Index run for the same weights. Triton and FlashInfer tested on SM90, SM120 did not have any accuracy issues.
- **String content for text questions sent through the chat endpoint.** The SDK reads text
  questions through `/v1/completions` and needs nothing here. If you send text questions
  through `/v1/chat/completions` yourself, serve with `--chat-template-content-format string`:
  the unmodified Gemma 4 chat template writes a space after the system text when the system
  message arrives as content parts and none when it arrives as a string, and DE-2 was trained
  and evaluated on the string rendering. Leave the option unset if you send images; with it
  set, vLLM 0.30.0 rejects image requests.
- **Sampling defaults.** Read probabilities with `temperature` 0 or 1 and without `top_k` or
  `top_p` truncation.

### Adaptive inference policy

Based on empirical testing, we recommend the following adaptive inference policity (applied by the SDK automatically):
1. Write the user turn twice, separated by `\n\nRead the same input again before answering:\n`,
   and read the option codes.
2. If the top probability is below 0.7 and the question has 26 options or fewer, ask again
   with thinking on, let the model reason for up to 1,024 tokens, close the thought and read
   the codes after it. Otherwise keep the answer from step 1.

The SDK's two steps are worth 4.3 points on the Decision Index. The same weights score:

| Decision Index 0.3, skill by area | Single read | Read twice | Full policy |
| --- | ---: | ---: | ---: |
| Knowledge & Reasoning | 41.4 | 45.4 | **49.2** |
| Language Understanding | 59.8 | 61.8 | **62.6** |
| Retrieval & Classification | 60.7 | 63.8 | **63.9** |
| Tools & Automation | 70.9 | **72.8** | **72.8** |
| Arts & Human Taste | 34.2 | 39.4 | **39.7** |
| Public index | 54.69 | 57.71 | **58.96** |

Reading twice helps every area. Thinking adds most where the task needs working out:
Knowledge & Reasoning gains 3.8 points from it, and skill on GSM8K goes from 70.2 to 81.4.
The policy thinks on *4% of questions*.

There is of course slight latency penalty so if fast or predictable response is required, one or both of these policies can be disabled:

| Reading | p50 | Avg |
| --- | ---: | ---: |
| Single read | 41 ms | 86 ms |
| User turn read twice | 74 ms | 160 ms |
| Read twice, think when unsure | 74 ms | 610 ms |

Response times are measured one request at a time on one H20-3e with a cold prefix cache.

A chat template with an opt-in `repeat_input=true` argument is provided for callers who
serve through a plain chat endpoint. It covers step 1 only, is off by default, repeats the
last user turn only, and must not be combined with a client that already repeats the input.

### Calibration

Raw DE-2 probabilities are overconfident, and by a different amount on each read. A post-hoc
temperature is fitted per read and per answer kind by NLL grid search, and the SDK applies the
one for the read each answer came from:

| Read | T, `noul` | T, choice | ECE, `noul` | ECE, choice |
| --- | ---: | ---: | --- | --- |
| Single read | 2.69 | 1.81 | 0.059 → 0.032 | 0.121 → 0.070 |
| Read twice | 2.76 | 1.89 | 0.059 → 0.023 | 0.110 → 0.048 |
| After thinking | 6.8 | 6.8 | 0.411 → 0.184 (both kinds) | |

Fitted on 9,931 questions from 53 of our evaluation suites (84% English, 16% Japanese), each
suite weighted equally, served by vLLM 0.30.0 on one RTX PRO 6000. The single-read and
read-twice temperatures were fitted on half of the question groups and measured on the other
half (1,895 `noul` and 3,140 choice questions); the thinking temperature is one value fitted on
475 thoughts and measured by cross-validation. Over the adaptive policy's whole output, ECE
moves from 0.091 to 0.021. Accuracy is unchanged, because tempering does not move the argmax.

Probabilities read after thinking remain the least reliable: those answers are right 52% of
the time at 92% raw confidence. For questions scored on the probability itself, such as
forecasts, use the single read: reading twice and thinking both shift forecasting answers
toward "yes", which a temperature does not correct. Refit on your own rows before using these
probabilities to drive a threshold.

## Training

| Setting | Value |
| --- | --- |
| Method | LoRA rank 16, alpha 32, dropout 0.05, language-model attention projections |
| Objective | Cross-entropy over the listed option codes at the answer boundary |
| Schedule | 3 epochs, peak learning rate 1.5e-4, 10% warmup then cosine decay |
| Batch | Up to 16 examples and 4,096 padded tokens, gradient accumulation 2 |
| Data per epoch | 14,637 questions: 8,745 English, 5,892 Japanese |

The 28 training families, by group:

- **Distilled and procedural decision data** in English and Japanese.
- **Japanese judgement families:** leakage of internal information, nuance, policy, register,
  and indirect refusals in context.
- **Fraud and safety:** antiscam messages with link and sender cues, prompt injection and
  hidden text.
- **Agent workflow:** compaction, guardrails, review triage and progress checks.
- **Wide-option and ordinal tasks:** large-option financial intents, relation and sentiment,
  product, emotion and sentiment ratings.
- **Exactness and logic:** sibling criteria where one detail decides the answer, compound
  conditions, counting and arithmetic probes.
- **A second classifier-benchmark suite** of 49 short tasks.

## Evaluation

Unless a table says otherwise, each system is shown at its best measured reading: DE-2 through
the SDK's reading policy, DE-1 with the SDK's scaffold, Jev as the hosted `jev-1.13.0`
API, and [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions) (`gpt-6-luna`) via the latest OpenAI SDK. All DE-2 and DE-1 figures are our own local runs. A dash means the system was not run.

### Summary

| Benchmark | Measure | DE-2 | DE-1 | Jev 1.13.0 | OpenAI Decisions |
| --- | --- | ---: | ---: | ---: | ---: |
| Decision Index 0.3 | Public index | **58.96** | 50.19 | 57.96 | 47.32 |
| Plan-RewardBench, text only | Macro accuracy | 72.8% | **75.6%** | 75.1% | 69.4% |

- **Decision Index 0.3.** Full public suite, 140,178 scored requests, official scorer; every
  request is answered. These are not board entries. The board ranks by a Full score that adds
  private tests only its maintainers run; Jev's public-index figure is from the board on
  2026-10-06. Other public-index figures there: Clef 61.71, Rune 26B-A4B v3 58.28, Decider
  chat on Gemma-4-31B 57.79.
- **Plan-RewardBench.** 1,171 pairs of tool-agent trajectories, both presentation orders
  averaged, following the public JEV-27B-DEMO script. The stock base scores 76.1%, and the
  [JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) card reports 73.2%.
- **Image benchmarks** are in "Vision" below. Jev 1.13.0 has no vision encoder, so the
  comparisons there are with vision-language decision models.

### JevBench

The 231 public JevBench items, by tier:

| System | Easy (48) | Medium (72) | Hard (111) |
| --- |  ---: | ---: | ---: |
| DE-2, adaptive | **48** | **72** | 86 |
| DE-1, read twice | **48** | **72** | 93 |
| Jev 1.13.0 | **48** | 71 | 81 |
| OpenAI Decisions | **48** | 70 | **94** |

"Medium" is JevBench's standard tier. DE-2 and DE-1 use the Shisa DE SDK, with the best policy results chosen. 
Jev and OpenAI Decisions (`gpt-6-luna`) are their hosted APIs on the same items, one request per item. The easy and medium tiers are
saturated for all four models.

### Harness suites

67 internal suites were run against DE-2, DE-1 and hosted Jev and OpenAI Decisions on the same questions, read
once. Primary metric per suite; a selection:

| Suite | Metric | DE-2 | DE-1 | Jev 1.13.0 | OpenAI Decisions |
| --- | --- | ---: | ---: | ---: | ---: |
| Hidden text v1 | accuracy | **0.930** | 0.578 | 0.556 | 0.556 |
| Antiscam v5 | accuracy | **0.844** | 0.745 | 0.734 | 0.740 |
| Counting probe | accuracy | **0.967** | 0.627 | 0.867 | 0.793 |
| Fresh-math probe | accuracy | **0.950** | 0.583 | 0.717 | 0.375 |
| Emotion, 6 classes | accuracy | **0.670** | 0.440 | 0.465 | 0.440 |
| Japanese indirect refusals | AUROC | **0.919** | 0.852 | 0.844 | 0.806 |
| Japanese leakage v3 | AUROC | **0.983** | 0.936 | 0.951 | 0.955 |
| Japanese nuance | AUROC | **0.965** | 0.947 | 0.921 | 0.898 |
| Kev transfer v9 | accuracy | 0.766 | 0.737 | **0.824** | 0.734 |
| Antiscam, original | accuracy | 0.949 | **0.992** | 0.924 | 0.848 |
| Slop | AUROC | 0.966 | **0.997** | 0.970 | 0.888 |
| Guardrails | AUROC | 0.979 | **0.996** | 0.975 | 0.956 |
| Compaction | AUROC | 0.850 | 0.869 | **0.923** | 0.886 |
| Agent progress v2 | AUROC | 0.924 | 0.983 | **0.992** | 0.971 |
| Agent routing, 16 cases | accuracy | 0.812 | **0.938** | **0.938** | 0.812 |

On the 64 suites all four systems answered, DE-2 has the top result on 38, Jev on 29, DE-1 on
18 and OpenAI Decisions on 8, counting a tie for each system that shares it. Several of the
suites where DE-2 leads share a task family with its training data. DE-1 keeps an edge on the
original four use cases it was built for.

Response time per case on those 64 suites (1.4 questions per case on average):

| | DE-2 | DE-1 | Jev 1.13.0 | OpenAI Decisions |
| --- | ---: | ---: | ---: | ---: |
| Median | 27 ms | 26 ms | 195 ms | 201 ms |
| Mean | 40 ms | 33 ms | 204 ms | 229 ms |
| 95th percentile | 114 ms | 74 ms | 280 ms | 312 ms |

DE-2 and DE-1 were served locally by vLLM on one H20-3e, read once, with one client sending
requests in sequence. Jev and OpenAI Decisions are hosted APIs measured from our client,
network round trip included; the OpenAI runs had up to 16 requests in flight.

### Vision

Like DE-1, DE-2's vision encoder is the base model's and was not trained, but the decision
readout carries over to images. Jev 1.13.0 takes no image input. OpenAI Decisions
(`gpt-6-luna`) was run by us on the same items; the Qwen3-VL-4B and JEV-27B-VL figures are their
authors' published numbers.

Image classification, [glance](https://github.com/yoheinakajima/glance) suites (one forward pass per question):

| Suite | Measure | DE-2 | DE-1 | Qwen3-VL-4B |
| --- | --- | ---: | ---: | ---: |
| POPE, object presence | Accuracy | 0.865 | 0.860 | **0.880** |
| POPE, object presence | AUROC | 0.938 | **0.960** | 0.955 |
| GQA, compositional yes/no | Accuracy | 0.675 | 0.690 | **0.732** |
| Oxford-IIIT Pet, 37 breeds | Accuracy | **0.92** | **0.92** | 0.892 |
| Caltech-101, 26-option form | Accuracy | 0.96 | 0.96 | **0.977** |

Multimodal judging, following the public JEV-27B-DEMO protocols with both presentation orders
averaged:

| Benchmark | Measure | DE-2 | DE-1 | OpenAI Decisions | JEV-27B-VL |
| --- | --- | ---: | ---: | ---: | ---: |
| VL-RewardBench, 1,247 pairs | Overall accuracy | **80.0%** | 77.9% | **80.0%** | 78.3% |
| Multimodal RewardBench 2, 4 × 1,000 pairs | Average accuracy | 61.8% | 58.9% | **64.3%** | 63.8% |
| AgentRewardBench, 1,302 trajectories with screenshots | AUROC | 0.910 | **0.922** | 0.859 | 0.91 |
| AgentRewardBench | Precision / recall at 0.5 | 75.1 / 66.9% | 72.1 / **77.8%** | **82.0** / 58.7% | 78.4 / 70.2% |

The stock base model scores 80.0%, 60.0% and 0.907 on the three judging benchmarks, so on
images DE-2 is the base model's vision with DE-2's readout; training neither helped nor hurt.
On Multimodal RewardBench 2, JEV-27B-VL is ahead on text-to-image (69.2% against 64.1%) and
interleaved judging (67.8% against 57.8%), and DE-2 on image editing (65.0% against 57.8%).

## Limitations

- **Overconfident raw probabilities.** The SDK's calibration corrects most of this for the
  single and read-twice answers. Probabilities read after thinking stay poorly calibrated
  (ECE 0.18 after tempering), and calibration was not fitted for image questions.
- **Forecasting questions.** On ForecastBench (retired from the index in edition 0.3) skill
  was 5.5 read once and 0 under the full policy, because the model says "yes" too often. Use
  the single read with its calibration when the probability is what you need.
- **Rare-positive detection.** When very few candidates are true, as in aspect extraction
  with dozens of pairs per input, the model raises many false alarms: precision is about 10%
  on ACOS.
- **Order sensitivity on multi-image comparisons.** Swapping two candidate responses changes
  the verdict on 45% of Multimodal RewardBench 2 pairs. Ask both orders and average.
- **Agent workflow and routing.** DE-2 is 3 points below the stock base and DE-1 on
  Plan-RewardBench, behind Jev on the compaction, guardrails and progress suites, and tends
  to route simple tasks to a larger tier than needed.
- **Knowledge limits.** Scores on expert-knowledge and game benchmarks (HLE, GPQA Diamond,
  ChessBench) do not improve with either reading.
- **Thinking is limited to 26 options.** Wider questions use the read-twice answer.
- **Prompts longer than 65,536 tokens** were not evaluated, however runs with the max-context 
  length (256K) vs restricted (32K) showed no regressions.
