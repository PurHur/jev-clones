# Reproduce the unified supervised training run

The selected artifact is `hard_lr1e5`, seed 17, checkpoint step 400 from a
600-update run. Selection uses development data: each arm's minimum weighted
dev CE, then weighted game success across the four arms and initialization.
The test and OOD cohorts do not select a checkpoint. No RL or TD update is active
in this release. The loss is complete-question categorical cross entropy.

## Exact inputs

- Model root: selected step-400 bundle, SHA256 `f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b`.
- `training_initialization/`: the exact prior unified Basic-supervised bundle,
  SHA256 `38116340795de1c82369b7fe15819d92d79600a7b4dc7a3cd0d4390cb6782639`; a fresh optimizer is created for this round.
- Dataset `unified/hard/`: all five original mixed-task split files, unchanged.
  `unified/soft/` is the matched comparison arm's input.
- Raw expert data: `expert/predict_position/episodes.jsonl` (896 episodes).
- Preparation input: `preparation/original_unified_hard/`, containing the exact
  previous mixed dataset. Maze, Snake and Basic rows are retained byte-for-byte.
- `configs/sonic_unified_sft_v1.json`: all four arms and the frozen selection rule.
- `evaluation/experiment/`: initial and four-arm dev trajectories, selection,
  selected/initial full test trajectories, metrics and replay checks.
- `evaluation/{jev,native}/`: the other full 548-case test/OOD recordings.
- `demonstrations/`: fixed 50x50 Maze / 12x12 Snake case registry, six source
  recordings and validated export. These use their recorded code planners and
  remain separate from the 548-case benchmark.

Stored questions across all five splits total 18,760 per target variant. The trainer quarantines
11 invalid-target questions across the splits, leaving 10,893 eligible training
questions out of 10,898 stored training rows. The eligible training pools are
Maze 651, Snake 400, Basic 3,054 and Predict Position 6,788.

## Download the complete current package

Download the exact manifest-listed files so retained historical checkpoints are
not included in the training package download. No authentication is required.

```python
import json
from huggingface_hub import hf_hub_download, snapshot_download

for repo, kind, name in [("C-Tianyu/NanoJev", "model", "MODEL_DIR"),
                         ("C-Tianyu/NanoJev-Data", "dataset", "DATA_DIR")]:
    manifest = hf_hub_download(repo, "SHA256_MANIFEST.json", repo_type=kind,
                               revision="unified-games-v1", token=False)
    with open(manifest) as handle:
        names = [row["path"] for row in json.load(handle)["files"]]
    path = snapshot_download(repo, repo_type=kind, revision="unified-games-v1",
                             token=False, allow_patterns=names + ["SHA256_MANIFEST.json"])
    print(name + "=" + path)
```

## Environment and selected arm

The recorded environment is Python 3.14.4, PyTorch 2.14.0, Transformers 5.17.0,
Safetensors 0.8.0, NumPy 2.5.3, ViZDoom 1.3.0 and A100 80GB.
FP32 weights are trained with BF16 forward computation and gradient checkpointing.
Install the matching packages from `source/requirements-toy.txt` and
`source/requirements-vizdoom.txt` in a GPU environment.

After downloading the public release snapshots with the allowlist below,
set `MODEL_DIR` and `DATA_DIR` to those absolute local paths and run from the model folder:

```bash
python source/scripts/train_unified_games.py \
  --input "$DATA_DIR/unified/hard" \
  --init-checkpoint "$MODEL_DIR/training_initialization" \
  --output-dir ./reproduced_hard_lr1e5 --stage sft --loss ce --balance task \
  --policy-pool-weights "$DATA_DIR/configs/sonic_policy_pool_weights.json" \
  --steps 600 --head-steps 0 --seed 17 \
  --batch-questions 24 --microbatch-questions 8 --max-microbatch-tokens 32768 \
  --max-length 8192 --eval-every 100 \
  --backbone-lr 1e-5 --head-lr 1e-4 --weight-decay 0.01 \
  --precision bf16 --gradient-checkpointing --disable-native-triton
```

Every batch contains 8 Maze, 8 Snake, 4 Basic and 4 Predict Position questions.
Their loss weights are respectively 1/3, 1/3, 1/6 and 1/6. Rows are sampled with
replacement. Across 600 updates the PP sampler draws 2,400 questions covering
2,021 distinct questions; the selected step-400 checkpoint sees the matching
shorter prefix. The backbone/head learning-rate comparison is 1e-5/1e-4 versus
2e-5/2e-4, for both hard and soft PP targets. All arms keep the same sample order.

The complete original configs, hashes, logs, target audit, predictions and
selection receipts are included. Run the data-only check before training:

```bash
python source/scripts/train_unified_games.py \
  --input "$DATA_DIR/unified/hard" --stage sft --loss ce \
  --policy-pool-weights "$DATA_DIR/configs/sonic_policy_pool_weights.json" \
  --validate-only
```

To replay the six hard navigation examples without model or API calls, change
into `$DATA_DIR/demonstrations` and run:

```bash
ln -s "$MODEL_DIR/source/scripts" scripts
python "$MODEL_DIR/source/scripts/build_hard_navigation_demo.py" --output ./rebuilt_hard_navigation.json --receipt ./rebuilt_hard_navigation_receipt.json
```

The `scripts` link lets the replay validator resolve the original relative runner
paths while checking their exact source hashes. The frozen source paths are
preserved under that folder. Source paths, recorded protocols
and all referenced source hashes appear in its `results/hard_navigation_demo_v1`
receipt. Inference uses `source/scripts/predict_toy_decisions.py`, whose
`DecisionPredictor` loads the custom dynamic candidate-scoring head and backbone.
