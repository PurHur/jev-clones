---
license: mit
library_name: contrastive-lm
tags:
- contrastive-learning
- verifier
- best-of-n
- swe-agent
- deepswe
- clm
---

# DeepSWE CLM fixed-split head (Qwen3-8B, 8K)

This repository contains the single CLM projection-head checkpoint used for the
DeepSWE panel in the release chart.

- Split: seed 42, stratified by candidate count and pass count
- Training/evaluation tasks: 75/38, task-disjoint
- Training tasks with successful trajectories: 66 (59 train, 7 validation)
- Candidate budget: Bo4
- Trajectory score: mean of the final 12 available step scores
- Heldout result: 31/38 = 81.579%
- Heldout pass@1: 28/38 = 73.684%
- Heldout oracle: 34/38 = 89.474%
- Checkpoint SHA-256: `554989fe88635606cb978dc45a1ce083be1990c4a51e551ea3b6055ead1a029a`
- Heldout-list SHA-256: `d4e2e7639f9eace09fba50318a266111c080d591bfbe39d213c3ea611e7b65c1`

## Reproduce

From a clone of [`Contrastive-LM/CLM`](https://github.com/Contrastive-LM/CLM)
([`evaluation/bon_eval.py`](https://github.com/Contrastive-LM/CLM/blob/main/evaluation/bon_eval.py)):

```bash
hf download Contrastive-LM/deepswe-clm-heads-8k --local-dir heads/deepswe
python evaluation/bon_eval.py \
  --hf-dataset Contrastive-LM/deepswe-clm-embeddings-8k \
  --checkpoint heads/deepswe/best_head.pt \
  --tasks-file heads/deepswe/heldout_tasks.json \
  --n 4 --window 12
```

The checkpoint was initialized from
`Contrastive-LM/CLM-v0.1-8B/CLM_v0.1-8B.pt` and trained with the unified
`train/finetune.py` CLM path using batch size 512, seed 1234, up to 20 epochs,
and patience 5.
