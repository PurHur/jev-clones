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
---

# Requirement Checking

## Model Summary

**Requirement Checking** is a family of LoRA adapters fine-tuned for binary
requirement satisfaction classification. The model takes as input a user prompt,
an assistant response, and a set of user-specified requirements, and classifies
whether the assistant's response satisfies the given constraints. The model
outputs a JSON object `{"score": "yes"}` or `{"score": "no"}` indicating whether
the constraints are satisfied. We provide requirement checking capabilities
implemented as LoRA adapters trained over Granite-4.0-micro, Granite-4.1-3B, Granite-4.1-8B, and
Granite-4.1-30B.

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

## Intended Use

The Requirement Checking adapter is designed for evaluating whether LLM-generated
responses satisfy user-specified constraints. Key use cases include:

- **Instruction-following evaluation**: Assessing whether a model's output
  adheres to specific formatting, content, or structural requirements provided
  in the prompt.
- **Constraint satisfaction checking**: Verifying that generated text meets
  multiple simultaneous constraints (e.g., length limits, style requirements,
  content inclusion/exclusion).
- **Quality assurance pipelines**: Automated checking of LLM outputs against
  predefined acceptance criteria.

**Adapter input**: The input to the requirement checking adapter is an
OpenAI-compatible chat completion request containing a user message, an assistant
response, and a follow-up user turn with the requirements to evaluate wrapped in
a `<requirements>` tag.

**Adapter output**: `{"score": "yes"}` if the constraints are satisfied, `{"score": "no"}` otherwise.

**Scoring:** When called via Mellea, the binary output is converted to a float between 0.0 and 1.0 using the model's token-level probability of "yes". This provides a continuous confidence score rather than a hard binary decision. The threshold for converting this score to a binary outcome is left to the user and may need to be tuned for your use-case.

## Usage Examples

### via Mellea

The recommended way to call this adapter is through the [Mellea](https://mellea.ai) framework:

```python
from mellea import model_ids, start_backend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import core

ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_1_3B, context_type="chat"
)

ctx = (
    ctx.add(Message("user", "Invite for an IBM office party."))
       .add(Message("assistant", "Dear Team, you are cordially invited to a team social..."))
)

result = core.requirement_check(ctx, backend, "Use a professional tone.")
print(f"Requirements satisfied: {result}")  # float between 0.0 and 1.0
```

For more comprehensive examples and detailed documentation, refer to the [Mellea intrinsics examples](https://github.com/generative-computing/mellea/tree/main/docs/examples/intrinsics).

### via HF+PEFT

```python
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "ibm-granite/granite-4.1-3b"
adapter_repo = "ibm-granite/granitelib-core-r1.0"
adapter_subfolder = "requirement-check/granite-4.1-3b/lora"

# Load base model and LoRA adapter
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_repo, subfolder=adapter_subfolder)
model.eval()

# Define the user prompt, assistant response, and constraints to check
user_text = "Invite for an IBM office party."
response_text = """
Dear Team,

To celebrate our recent successes and take a well-deserved moment to recharge,
you are cordially invited to a team social. Please join us for an evening of
live music, appetizers, and drinks as we recognize our collective wins.

Event Details
* **Date:** Saturday, April 25, 2026
* **Time:** 6:00 PM
* **Location:** Ryan's Bar, Chelsea, NY
* **Highlights:** Live entertainment and refreshments

RSVP
To ensure we have an accurate headcount for catering, please confirm your
attendance by **Friday, April 10, 2026**.

We look forward to seeing everyone there and celebrating our hard work together.

**Best regards,**
[Your Name/Management Team]
"""

constraints = "Use a professional tone."

# Build the evaluation prompt
evaluation_prompt = (
    "Please verify if the assistant's generation satisfies the user's "
    "requirements or not and reply with a binary label accordingly. "
    'Respond with a json {"score": "yes"} if the constraints are '
    'satisfied or respond with {"score": "no"} if the constraints are not '
    "satisfied."
)

messages = [
    {"role": "user", "content": user_text},
    {"role": "assistant", "content": response_text},
    {"role": "user", "content": f"<requirements> {constraints}\n{evaluation_prompt}"},
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[1]

stop_token_id = tokenizer.encode("}", add_special_tokens=False)[0]

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        eos_token_id=[tokenizer.eos_token_id, stop_token_id],
    )

response = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()

brace_idx = response.find("}")
if brace_idx != -1:
    response = response[:brace_idx + 1]

print(f"Response: {response}")  # {"score": "yes"}

result = json.loads(response)
print(f"Constraints satisfied: {result['score']}")  # yes
```

## Evaluation

We evaluated the adapters on binary requirement satisfaction classification
using four benchmarks: [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3),
[IFEval Multi-Constraint](https://arxiv.org/abs/2311.07911), and
[InfoBench](https://arxiv.org/abs/2401.03601) (GPT-4-annotated and
human-annotated splits).

### Results

The table below reports Balanced Accuracy across all benchmarks for prompting
baselines and trained adapters (LoRA and aLoRA) at each model size.

| Model | HelpSteer3 | IFEval | InfoBench-GPT4 | InfoBench-Human |
|---|---|---|---|---|
| Granite-4.1-3B (prompting) | 0.5090 | 0.5118 | 0.6237 | 0.5866 |
| Granite-4.1-8B (prompting) | 0.5162 | 0.5689 | 0.6564 | 0.6018 |
| Granite-4.1-30B (prompting) | 0.5234 | 0.6237 | 0.6863 | 0.6236 |
| Granite-4.0-micro LoRA | 0.7434 | 0.8403 | 0.6950 | 0.6799 |
| Granite-4.1-3B LoRA | 0.6726 | 0.8153 | 0.6960 | 0.6728 |
| Granite-4.1-3B aLoRA | 0.6957 | 0.8425 | 0.7150 | 0.6475 |
| Granite-4.1-8B LoRA | 0.7051 | 0.8360 | 0.7373 | 0.6546 |
| Granite-4.1-8B aLoRA | 0.6941 | 0.8554 | 0.7119 | 0.6761 |
| Granite-4.1-30B LoRA | 0.7119 | 0.8406 | 0.7512 | 0.7038 |
| Granite-4.1-30B aLoRA | 0.7178 | 0.8492 | 0.7432 | 0.7052 |

The evaluations on HelpSteer3 and IFEval above are on the validation set and a heldout set, respectively.

### Performance Comparison to Decision Models

| Benchmark | Metric | Best prompted Granite-4.1 | Best trained adapter | [jev-1.13](https://openrouter.ai/typesafe/jev-1.13) | [microsoft-decision-1](https://openrouter.ai/microsoft/microsoft-decision-1) | [gpt-6-luna-decisions](https://openrouter.ai/openai/gpt-6-luna-decisions) | [clef-flash](https://openrouter.ai/cloudflare/clef-flash) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| IFEval | BAcc | 0.624 | 0.855 | 0.685 | 0.678 | 0.633 | 0.663 |
| InfoBench-GPT4 | BAcc | 0.686 | 0.751 | 0.768 | 0.769 | 0.783 | 0.730 | 
| InfoBench-Human | BAcc | 0.624 | 0.705 | 0.699 | 0.698 | 0.713 | 0.693 | 

## Training Details

### Training Data

The Requirement Checking adapter is fine-tuned using a combination of
instruction-following evaluation data. Training data includes samples with
explicit user requirements and constraints paired with assistant responses,
annotated for whether the constraints are satisfied.

The two sources of training data are:

1. [IF-RLVR training data](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5),
   with up to 5 constraints per instruction. Each prompt (with constraints) was
   passed through [Mixtral 8x22b Instruct](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1).
   Then, the [prompt, constraints, response] are all passed through a
   [programmatic evaluation pipeline](https://github.com/allenai/IFBench?tab=readme-ov-file)
   which provides yes/no labels corresponding to whether the constraints were
   followed or not.
2. [Helpsteer3-Preference](https://huggingface.co/datasets/nvidia/HelpSteer3).
   The original dataset consists of [prompt, response1, response2, feedback].
   The `feedback` column contains individual rankings (a score from 1-5) of each
   response from multiple annotators. We map scores 4-5 to a label of yes and
   scores 1-3 to a label of no, take the average of the annotators' scores and
   map each of the two responses to a binary yes/no label accordingly.

### Training Hyperparameters

The adapters were fine-tuned using PEFT under the following regime: rank = 64
(LoRA) / 16 (aLoRA), alpha = 64, learning rates between 5e-5 and 5e-4 (tuned
per base model).

### Adapter Details

| Property | Granite-4.0-micro | Granite-4.1-3B | Granite-4.1-8B | Granite-4.1-30B |
|---|---|---|---|---|
| **PEFT Type** | LoRA | LoRA/aLoRA | LoRA/aLoRA | LoRA/aLoRA |
| **Rank (r) - LoRA** | 64 | 64 | 64 | 64 |
| **Rank (r) - aLoRA** | — | 16 | 16 | 16 |
| **Alpha** | 64 | 64 | 64 | 64 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, input_linear, output_linear | q_proj, k_proj, v_proj, o_proj | q_proj, k_proj, v_proj, o_proj | q_proj, k_proj, v_proj, o_proj |
| **Invocation Token (aLoRA)** | — | `<requirements>` | `<requirements>` | `<requirements>` |

**Infrastructure:** Training was completed using H100 GPUs.

**Ethical Considerations:** The requirement checking adapter outputs are not guaranteed to be factually accurate or complete. All outputs should be independently validated before use in decision-making or downstream applications. The model has been trained and evaluated on English data only. The model must only be used in the prescribed evaluation mode, outputting JSON responses (`{"score": "yes"/"no"}`) based on the specified prompt template.

**Resources**
- Learn about the latest updates with Granite: https://www.ibm.com/granite
- Get started with tutorials and best practices: https://www.ibm.com/granite/docs/
- Learn about the latest Granite learning resources: https://research.ibm.com/blog/granite-4-1-ai-foundation-models
