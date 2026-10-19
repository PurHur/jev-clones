---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
library_name:
- peft
- transformers
base_model:
- ibm-granite/granite-4.0-micro
- ibm-granite/granite-4.1-3b
- ibm-granite/granite-4.1-8b
- ibm-granite/granite-4.1-30b
tags:
- granite
- guardian
- uncertainty-quantification
- lora
- peft
---

# Uncertainty Quantification

## Model Summary

**Uncertainty** is a family of LoRA/aLoRA adapters that provide calibrated certainty scores for Granite 4.1 models. The adapter responds with a certainty score from 0 to 9, which maps to a calibrated likelihood via `confidence = 0.1 * score + 0.05`, yielding 10 possible values (5%, 15%, 25%, ..., 95%). This percentage is *calibrated* in the following sense: given a set of answers assigned a certainty score of X%, approximately X% of these answers should be correct. We provide uncertainty adapters for Granite-4.0-micro, Granite-4.1-3B, Granite-4.1-8B, and Granite-4.1-30B.

- **Developer:** IBM Research
- **HF Collection:** [Granite Libraries](https://huggingface.co/collections/ibm-granite/granite-libraries)
- **GitHub Repository:** https://github.com/ibm-granite
- **Last Updated:** May 5th, 2026
- **Model type:** LoRA/aLoRA adapters for:
  - [ibm-granite/granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro)
  - [ibm-granite/granite-4.1-3b](https://huggingface.co/ibm-granite/granite-4.1-3b)
  - [ibm-granite/granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b)
  - [ibm-granite/granite-4.1-30b](https://huggingface.co/ibm-granite/granite-4.1-30b)
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Paper:** [[Shen et al. ICML 2024] Thermometer: Towards Universal Calibration for Large Language Models](https://arxiv.org/abs/2403.08819)

## Intended Use

Uncertainty is a LoRA adapter that enables Granite 4.1 base models to express calibrated self-assessments of their own answer correctness. The adapter is designed to be used as part of the Granite inference pipeline, activated via the `<certainty>` invocation token after the model generates a response.

**Adapter input:** An OpenAI-compatible chat completion request containing the user query and the model's response, followed by the `<certainty>` invocation token.

**Adapter output:** A JSON object `{"score": "X"}` where X is 0-9, mapping to calibrated confidence via `0.1 * score + 0.05`.

**Scoring:** When called via Mellea, the discrete 0–9 output is converted to a calibrated certainty float using the token-level probabilities over all 10 score categories, weighted by their mapped values (5%, 15%, …, 95%). This yields a continuous confidence estimate rather than a coarse 10-bin discretization. The threshold for "confident enough" is left to the user and may need to be tuned for your use-case.

### Use Cases
* **Human usage:** Certainty scores give human users an indication of when to trust answers from the model.
* **Model routing/guards:** If the model has low certainty (below a chosen threshold), route the request to a larger model or decline to show the response.
* **RAG:** Certainty scores can be applied to answers created using RAG, predicting overall correctness based on both the documents and the model's own knowledge.

## Usage Examples

### via Mellea

The recommended way to call this adapter is through the [Mellea](https://mellea.ai) framework:

```python
from mellea import model_ids, start_backend
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components.intrinsic import core

ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_1_3B, context_type="chat"
)

response, ctx = mfuncs.chat("What is IBM Research?", ctx, backend)
print(f"Response: {response.content}")

result = core.check_certainty(ctx, backend)
print(f"Certainty score: {result}")
```

### via HF+PEFT

```python
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_NAME = "ibm-granite/granite-4.1-3b"
LORA_REPO = "ibm-granite/granitelib-core-r1.0"
LORA_SUBFOLDER = "uncertainty/granite-4.1-3b/lora"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(BASE_NAME, padding_side="left")
model_base = AutoModelForCausalLM.from_pretrained(BASE_NAME, device_map="auto", torch_dtype=torch.bfloat16)
model_uq = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(BASE_NAME, device_map="auto", torch_dtype=torch.bfloat16),
    LORA_REPO,
    subfolder=LORA_SUBFOLDER,
)

question = "What is IBM Research?"

# Step 1: Generate answer with base model
messages = [{"role": "user", "content": question}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(device)
output = model_base.generate(**inputs, max_new_tokens=600, do_sample=False)
answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# Step 2: Generate certainty score with LoRA adapter
uq_messages = [
    {"role": "user", "content": question},
    {"role": "assistant", "content": answer},
    {"role": "user", "content": "<certainty>"},
]
uq_text = tokenizer.apply_chat_template(uq_messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(uq_text, return_tensors="pt").to(device)
output = model_uq.generate(**inputs, max_new_tokens=15, do_sample=False)
uq_response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# Parse score and map to confidence
match = re.search(r'\{[^}]*"score"\s*:\s*"?(\d)"?[^}]*\}', uq_response)
if match:
    score = int(match.group(1))
    confidence = 0.1 * score + 0.05
    print(f"Score: {score}, Certainty: {confidence*100:.0f}%")
```

For more comprehensive examples, refer to the [Mellea intrinsics examples](https://github.com/generative-computing/mellea/tree/main/docs/examples/intrinsics).

## Evaluation

The adapters were evaluated on the [MMLU](https://huggingface.co/datasets/cais/mmlu) dataset (57 subsets, 14,042 total samples, not used in training). We report [Expected Calibration Error (ECE)](https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d), Brier Score, AUROC, and Sharpness. Lower ECE and Brier scores indicate better calibration; higher AUROC indicates better discrimination between correct and incorrect answers.

### Results

| Model | Method | ECE | Brier Score | AUROC | Sharpness |
|---|---|---|---|---|---|
| Granite-4.0-micro | Base (seq prob) | 0.1606 | 0.2535 | 0.6748 | 0.2888 |
| | LoRA | 0.0565 | 0.2131 | 0.6903 | 0.1607 |
| Granite-4.1-3B | Base (seq prob) | 0.2879 | 0.3011 | 0.6291 | 0.0648 |
| | LoRA | 0.0998 | 0.1845 | 0.7870 | 0.1971 |
| | aLoRA | 0.0986 | 0.1877 | 0.7713 | 0.2027 |
| Granite-4.1-8B | Base (seq prob) | 0.1123 | 0.1932 | 0.6439 | 0.0438 |
| | LoRA | 0.0455 | 0.1487 | 0.7912 | 0.1898 |
| | aLoRA | 0.0381 | 0.1515 | 0.7778 | 0.1924 |
| Granite-4.1-30B | Base (seq prob) | 0.0921 | 0.1559 | 0.6507 | 0.0358 |
| | LoRA | 0.1006 | 0.1287 | 0.8116 | 0.1900 |
| | aLoRA | 0.1002 | 0.1317 | 0.8017 | 0.1927 |

The LoRA/aLoRA adapters consistently improve calibration (lower Brier scores) and discrimination (higher AUROC) across all model sizes, with zero-shot MMLU performance preserved.

## Training Details

The uncertainty adapters are finetuned to provide certainty scores mimicking the output of a calibrator trained via the Thermometer method ([Shen et al. ICML 2024](https://arxiv.org/abs/2403.08819)).

**Training Data:** Each adapter was trained on ~240K question-answer pairs generated by its respective base model, where each pair is annotated with a certainty score (0-9) derived from a calibrated thermometer model. The following datasets were used for calibration and/or fine-tuning:

* [BigBench](https://huggingface.co/datasets/tasksource/bigbench), [MRQA](https://huggingface.co/datasets/mrqa-workshop/mrqa), [newsqa](https://huggingface.co/datasets/lucadiliello/newsqa), [trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa), [search_qa](https://huggingface.co/datasets/lucadiliello/searchqa), [openbookqa](https://huggingface.co/datasets/allenai/openbookqa), [web_questions](https://huggingface.co/datasets/Stanford/web_questions), [cosmos_qa](https://huggingface.co/datasets/allenai/cosmos_qa), [kilt_tasks](https://huggingface.co/datasets/facebook/kilt_tasks), [smiles-qa](https://huggingface.co/datasets/alxfgh/ChEMBL_Drug_Instruction_Tuning), [orca-math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k), [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc), [commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa), [social_i_qa](https://huggingface.co/datasets/allenai/social_i_qa), [super_glue](https://huggingface.co/datasets/aps/super_glue), [figqa](https://huggingface.co/datasets/nightingal3/fig-qa), [riddle_sense](https://huggingface.co/datasets/INK-USC/riddle_sense), [ag_news](https://huggingface.co/datasets/fancyzhx/ag_news), [medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa), [dream](https://huggingface.co/datasets/dataset-org/dream), [codah](https://huggingface.co/datasets/jaredfern/codah), [piqa](https://huggingface.co/datasets/ybisk/piqa)

### Adapter Details

| Property | Granite-4.0-micro | Granite-4.1-3B | Granite-4.1-8B | Granite-4.1-30B |
|---|---|---|---|---|
| **PEFT Type** | LoRA | LoRA/aLoRA | LoRA/aLoRA | LoRA/aLoRA |
| **Rank (r)** | 32 | 32 | 32 | 32 |
| **Alpha** | 64 | 64 | 64 | 64 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, input_linear, output_linear | q_proj, k_proj, v_proj, o_proj | q_proj, k_proj, v_proj, o_proj | q_proj, k_proj, v_proj, o_proj |
| **Invocation token (aLoRA)** | — | `<certainty>` | `<certainty>` | `<certainty>` |
| **Output format** | `{"score": "X"}` (X: 0-9) | `{"score": "X"}` (X: 0-9) | `{"score": "X"}` (X: 0-9) | `{"score": "X"}` (X: 0-9) |
| **Confidence mapping** | `0.1 * score + 0.05` | `0.1 * score + 0.05` | `0.1 * score + 0.05` | `0.1 * score + 0.05` |
| **Max completion tokens** | 15 | 15 | 15 | 15 |

**Infrastructure:** Training was completed using H100 GPUs.

**Ethical Considerations:** Certainty is inherently an intrinsic property of a model and its abilities. The Uncertainty adapter is not intended to predict the certainty of responses generated by any other models besides its corresponding base model. Certainty scores are *distributional* quantities and will do well on realistic questions in aggregate, but may have surprising scores on individual red-teamed examples. Certainty scores may be biased towards moderate values because: (1) calibrated models are less likely to output very high or very low confidence, (2) smaller models cannot perfectly self-assess, and (3) generalization to novel question difficulties requires hedging.

**Resources:**
- Learn about the latest updates with Granite: https://www.ibm.com/granite
- Get started with tutorials: https://www.ibm.com/granite/docs/
- Learn about the latest Granite learning resources: https://research.ibm.com/blog/granite-4-1-ai-foundation-models
