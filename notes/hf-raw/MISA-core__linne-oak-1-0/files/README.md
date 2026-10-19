---
license: apache-2.0
base_model: google/gemma-4-31B-it
base_model_relation: finetune
library_name: transformers
language:
- en
- vi
- ar
tags:
- decision-model
- unsloth 
- decision-index
- jev
- jev-decision
- classification
- calibrated
- noncausal
- nli
- fact-verification
- tool-use
- routing
- guard-model
- agent-safety
- gemma4
---



# linne-oak-1.0

**linne-oak-1.0** is a 31B decision model. Give it a state (text, a document, JSON, a conversation or an agent trajectory)
and a question with a set of options, and it returns one option together with a probability for every option, in a single
forward pass.

It is the larger sibling of [linne-fern-1.0](https://huggingface.co/MISA-core/linne-fern-1-0): linne-fern is a 27B causal model
that also chats, linne-oak is a dedicated decision model.

<p align="center">
  <img src="linne-oak.jpg" width="200" alt="linne-oak">
</p>

### Highlights

- **Tied #1 on the Decision Index 0.3.2** (as of 2026-10-14, 166 models): Full score **65.61**,
  and 2nd-best on the private tasks from new domains (60.12).
- **Reads the whole prompt at once**: the full-attention layers attend in both directions over the
  prompt (sliding-window layers keep their native behavior).
- **Strong across the board**: knowledge, language understanding, retrieval, tools and automation.
- **Agent-safe**: a strong step-level guard for agents with a low false-block rate.
- **Fast**: one forward pass per decision; median 105 ms per request on one RTX PRO 6000
  (leaderboard measurement), 63 ms on one H200.

## Decision Index 0.3.2

Full score = 20% public benchmarks + 50% private tests of the same skills + 30% private tasks from
new domains, all run by the leaderboard maintainers. Snapshot of 2026-10-14.

| Model | Full | Public | Private, same skills | Private, new domains |
|---|---:|---:|---:|---:|
| vLLM-SR d3 (27B) | **66.26** | **65.45** | **62.83** | **60.51** |
| **linne-oak-1.0 (31B)** | 65.61 | 63.94 | 62.44 | 60.12 |
| Decider 31B (NVFP4) | 64.62 | 64.27 | 61.79 | 58.32 |
| Sieve 31B | 63.81 | 62.52 | 60.89 | 58.31 |
| Perplexity Decider v1.1 (27B) | 62.75 | 62.25 | 61.10 | 55.57 |
| linne-fern-1.0 (27B) | 57.81 | 58.92 | 56.52 | 50.71 |

The private parts were run by the maintainers with our serving code; their answers agreed with our
submitted run on 99.51% of the validation rows.

## Training data

Fine-tuned from `google/gemma-4-31B-it` on decision-format data (state, question, options) covering
classification, NLI, retrieval, tool selection and agent safety, in English and Vietnamese.
The training data is not released.

## Independence and Data Provenance

*This model is an independently developed personal project. It is not commissioned, controlled, 
endorsed, or officially maintained by any company or organization. The training datasets were 
independently collected, curated, and constructed by the individual developer. 
The project does not represent the official work or position of any affiliated organization.*

## Usage

The prompt is one user message (chat template, thinking off): the state (compact JSON when not a string), a blank line,
`Question: <instructions>`, one line `<label>: <name> - <description>` per option, and
`Answer with the letter of one option only.` The answer is the softmax over the next-token logits of the option letters
(with the model's final logit soft-capping), read at the last prompt position, **with the full-attention layers bidirectional**.

`linne_engine.py` (with `nc_hook.py`) in this repository implements exactly this, as a Decision Index engine and as a library:

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("MISA-core/linne-oak-1-0")
sys.path.insert(0, path)
from linne_engine import LinneEngine

model = LinneEngine(model=path, device="cuda:0", noncausal=True)
result, _ = model(
    "My Stripe integration keeps failing. Please help ASAP.",
    {"team": {"type": "choice", "instructions": "Which team should handle this request?",
              "criteria": {"billing": "Charges and refunds", "technical_support": "Integration errors",
                           "sales": "Questions about buying a product"}}},
)
print(result["answers"]["team"])  # chosen option and a probability for every option
```

Run the Decision Index public suite:

```
PYTHONPATH=<path> python -m decision_index pipeline --engine linne_engine:LinneEngine \
  --option model=<path> --option device=cuda:0 --option noncausal=1 --out runs/linne-oak-1-0
```

BF16 weights are about 62 GB and fit one 80 GB GPU.

## License

Apache 2.0, following the base model's license.

## Citation

If you use `linne-oak-1.0` in your research or applications, please cite this model:

```bibtex
@misc{linne_oak_1_0,
  title        = {linne-oak-1.0: A 31B Non-Causal Decision Model},
  author       = {Tran, Minh Duc},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/MISA-core/linne-oak-1-0}},
  note         = {Version 1.0}
}
```
