---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
- lora
- transformers
- structured-prediction
- classification
- probabilities
- base_model:adapter:Qwen/Qwen3.5-9B
---

# Bespoke-Nimble-9B-v2

> **Probability temperature:** load with the included `ParallelScorer(model_dir)`
> to automatically use **T=2.179078721266035**. For a custom runtime, compute
> `softmax(candidate_logits / 2.179078721266035)` exactly once. Loading only the
> LoRA weights does not apply this step; a plain softmax silently gives T=1.0.
> The helper rejects T=1.0 unless `allow_uncalibrated=True` is explicitly supplied.

A **PEFT LoRA adapter** for text-based, schema-conditioned decisions. 
It scores allowed answer-token logits to return
boolean judgments, categorical choices, and ordinal scores with probability
distributions. It does not generate reasoning or free-form answers.


## Files and base model

The approximately 173 MB adapter requires the separately downloaded base model:

- Base: `Qwen/Qwen3.5-9B`
- Base revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Adapter SHA-256: `1bd126be997be6d9a0c25ce483ccf858c31b3422d480c33f02ba47b614be68ae`
- Context contract: 8,192 tokens; oversized inputs are rejected, not truncated.
- Serving support: up to **255 choices per field**, subject to the 8,192-token context limit.
- Training provenance: the original A–Z training contract remains unchanged; largest training choice set was 24.

This is an unmerged adapter, not standalone base-model weights. The package
includes the tokenizer, pinned prompt contract, reference scoring code, training
and evaluation summaries, provenance, license, and file checksums.

## Expanded choice support

The included `inference.py` now supports 1–255 enum choices per field. It retains
A–Z codes and the exact original prompts for schemas with at most 26 choices.
Larger schemas use additional existing uppercase vocabulary tokens, each checked
to be exactly one token at the assistant answer boundary. No vocabulary expansion,
new embeddings, retraining, or weight changes are required.

Download the complete latest package, including `serving_schema.py`,
`extended_schema.py`, and `serving_config.json`; replacing only `inference.py`
is insufficient. `serving_config.json` records the verified 255-code mapping and
source hashes separately from the unchanged `schema_config.json` training record.
Custom serving code must use the expanded prompt builder and request validator;
loading new weights alone cannot remove a server's old 26-choice validation rule.

The 255-choice limit describes runtime support, not measured accuracy or
calibration at every candidate count. The existing transferred temperature is
unchanged and has not been fitted specifically for wide-choice inputs. Long
schemas can reach the context limit before reaching 255 options.

## Quickstart: CUDA reference scorer

Use Python 3.11 with a CUDA GPU supporting BF16 and enough memory for the 9B base
plus adapter and activations. The measured run used an H100 80GB. Authenticate
with `hf auth login` if the repository is private.

```sh
pip install huggingface_hub
hf download bespokelabs/Bespoke-Nimble-9B-v2 --local-dir ./Bespoke-Nimble-9B-v2
pip install -r ./Bespoke-Nimble-9B-v2/requirements.txt
```

```python
import sys
from pathlib import Path

model_dir = Path("Bespoke-Nimble-9B-v2").resolve()
sys.path.insert(0, str(model_dir))
from inference import ParallelScorer

model = ParallelScorer(model_dir)  # Automatically applies T=2.179078721266035
assert model.temperature == 2.179078721266035
result = model.score(
    context="The tracking record says delivered. The customer requests help finding the package.",
    schema={
        "delivered": {
            "type": "boolean",
            "description": "Does the tracking record say the package was delivered?",
        },
        "route": {
            "type": "enum",
            "choices": ["support", "sales"],
            "description": "Which team should handle this request?",
        },
    },
)
print(result["output"])
print(result["fields"]["delivered"]["probabilities"])
```

`score()` returns allowed-choice probabilities and native logits. For an ordinal
field, use integer-valued strings as enum choices and pass its name through
`score_fields=("field_name",)`; the result includes an expected score as well as
an argmax level. Describe every choice/level precisely in the schema.

This portable reference wrapper evaluates fields sequentially with independent
forwards. It does not use an optimized shared-prefix serving backend. No
`trust_remote_code` or generative decoding is required. Its default temperature
is **2.179078721266035**, applied once as `softmax(candidate_logits / temperature)`.
Raw logits and the top-choice answer remain unchanged; probabilities, Noul
values, and expected ordinal scores use the scaled distribution. Override with
`ParallelScorer(model_dir, temperature=1.0, allow_uncalibrated=True)` only for
an intentional comparison with the original raw probabilities. Without that
explicit opt-in, requesting T=1.0 raises an error.

This is a **transferred inference default, not a temperature fitted for v2**.
The [GitHub calibration table](https://github.com/bespokelabsai/nimble/blob/54c2e82e8f613a7ab3a4d4072c50dfb9376db4e3/nimble/scoring/calibration.py)
fits 2.179078721266035 only to the original 9B checkpoint. The updated
[GitHub Mac/CUDA loaders](https://github.com/bespokelabsai/nimble/pull/9)
automatically use the same transferred default when `model_id` is
`bespokelabs/Bespoke-Nimble-9B-v2` or a local merge's `READY.json` identifies the
v2 adapter. **Update your GitHub checkout before loading v2.** Older checkouts
require an explicit `temperature=2.179078721266035`; current loaders reject
`temperature=1.0` unless `allow_uncalibrated=True` is also supplied.

Temperature is postprocessing in the scorer, not a setting stored in the LoRA
weights or a generative sampling parameter. Loading the adapter through a
different runtime does not automatically apply it. The saved
`evaluation_summary.json` still describes the original **T=1** measurements;
it is not a report for this new default. `temperature_config.json` records the
chosen value and checkpoint provenance. No new fit or GPU inference was run.

See https://huggingface.co/bespokelabs/Bespoke-Nimble-9B for more information.

## Limits and verification

This text-only decision adapter has not been evaluated for image/audio inputs,
free-form generation, autonomous actions, or legal compliance. Synthetic
references and template coverage limit generalization. The reported probabilities
are conditional on the supplied choices and are not guaranteed calibrated.
These held-out sets have been repeatedly inspected across development runs, which
limits independence of subsequent comparisons. No claim of universal improvement
over the earlier checkpoint is made.

The original training process reloaded this adapter and exactly reproduced logits
on two training probes. Packaging verified all seven original adapter files,
retained weight bytes unchanged, and pinned the previously-null PEFT base revision.
Offline checks replay saved evaluation logits and verify the prompt code and
original scoring functions against the training snapshot. The temperature update
adds explicit scaling and separate offline regression tests. A new GPU inference run was not
performed for publication. `SHA256SUMS` verifies package files; `provenance.json`
records the source checkpoint and packaging changes.
