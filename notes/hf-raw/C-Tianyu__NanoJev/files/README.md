---
language:
- en
base_model: Qwen/Qwen3-0.6B
library_name: pytorch
tags:
- nanojev
- decision-model
- reinforcement-learning-environment
- supervised-fine-tuning
- maze
- snake
- vizdoom
---
# NanoJev — A nano replica of Jev

One 0.6B model returns action distributions for Maze, Snake, ViZDoom Basic and
Predict Position. This public release contains the current unified
step-400 model, its exact training initialization and the full inference source.
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) inspired the
state, question and candidate-set interface.

Each candidate is encoded with its state and question. A shared Qwen3-0.6B
backbone, end-of-sequence representations and an attention-based Choice head
score the offered set. Softmax returns all candidate probabilities together,
without output-token decoding. Boolean and ordered-score request types remain
available through the same structured interface.

| Matched test task | NanoJev | Jev | Untuned Qwen |
|---|---:|---:|---:|
| Maze | 4/10 | 7/10 | 2/10 |
| Snake | 8/8 | 8/8 | 0/8 |
| Basic | 128/128 | 56/128 | 56/128 |
| Predict Position | 27/128 | 11/128 | 11/128 |

All systems share the recorded structured observation interface, frozen cases,
controller epsilon 0.1 and sampling seed 17. Full test and OOD reports, source
trajectories and independent replay checks are in the linked dataset. The hard
50x50 Maze and 12x12 Snake demonstrations are a separately recorded showcase
using common code planners: current NanoJev reaches the maze goal in 225 attempts
and collects 30 food items while surviving all 256 Snake steps.

## Load the model

[Complete mixed dataset and recordings](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data)
· [Training recipe](TRAINING_RECIPE.md)
· [File hashes](SHA256_MANIFEST.json)

The release is available without authentication.
The root directory is the current complete checkpoint bundle; it is loaded by
NanoJev's `DecisionPredictor`.

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("C-Tianyu/NanoJev", revision="unified-games-v1",
                         token=False, allow_patterns=["best.safetensors", "config.json",
                         "backbone_config/*", "tokenizer/*", "source/*"])
sys.path.insert(0, path + "/source/scripts")
from predict_toy_decisions import DecisionPredictor

model = DecisionPredictor(path, device_name="cuda:0", precision="bf16",
                          disable_native_triton=True)
request = {"states": [{"id": "example", "state": "The target is left of the aim.",
    "questions": {"action": {"type": "choice", "instructions": "Choose the next action.",
        "criteria": {"left": "Move the aim left.", "right": "Move the aim right.",
                     "shoot": "Fire.", "noop": "Wait."}}}}]}
answer = model.predict(request)
print(answer)
```

The saved format contains `best.safetensors`, `config.json`, `backbone_config/`
and `tokenizer/`, plus the original training audits. It is the project's decision
model format. The base revision is
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.

Training uses mixed-task supervised cross entropy. Predict Position action
supervision comes from the released Sonic Doom visual expert. The existing
Maze, Snake and Basic data remain in their original splits. Four matched runs
compare hard/soft PP targets and two learning rates; development performance
selects `hard_lr1e5` at step 400. The selected weight SHA256 is
`f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b`. The full original 15-file bundle is preserved.

Source code carries the included MIT license; the Qwen base model retains its
upstream license. Model and data snapshots are published as `unified-games-v1`.

## Release history

The inference download above includes the current model and source. The full training
package also includes `training_initialization/`; use the manifest allowlist in
[the recipe](TRAINING_RECIPE.md) to download all current files.

The current release is verified by `SHA256_MANIFEST.json`. Retained
`MODEL_MANIFEST.json`, `GAMES_MODEL_MANIFEST.json`, `stage1/` and `variants/`
belong to earlier releases. The prior root snapshot remains available at
[`legacy-before-unified-games-v1`](https://huggingface.co/C-Tianyu/NanoJev/tree/legacy-before-unified-games-v1).
