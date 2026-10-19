---
license: apache-2.0
base_model: google/gemma-4-12B-it
language: [en]
pipeline_tag: text-classification
tags: [decision-model, calibrated, structured-output, system-one, one-pass]
---

# decider-12b: typed decisions with calibrated probabilities in one forward pass, on Gemma-4-12B-it

decider-12b is [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) (revision `707f0a3b`) read through
the decider readout.

- **Input:** a state and one or more typed questions (Choice, Noul yes/no, Score), each with an explicit option list.
- **Output:** a probability distribution over the options for every question, from one forward pass. There is no decoding and
  no parsing.
- **Prompt:** the model's chat template, with thinking off. The option letter is read at the answer slot, with the model's
  final-logit softcapping applied.
- **Temperatures:** one per answer type, in `decider_config.json`.

**Version 2 (2026-09-29, this revision):** a merged LoRA fine-tune (state tracking) on top of the stock weights.
- Only the 328 attention and MLP projection matrices differ from Google's weights. The config, tokenizer and chat template are
  Google's files unchanged.
- **Version 1** is stock Gemma-4-12B-it with no training. It is kept under the tag `v1`:
  `Decider("Mapika/decider-12b", revision="v1")`.

## Usage

Needs `decider-ai>=1.7.0` (1.7.0 adds Gemma's final-logit softcapping; older versions read this model too sharply).

```python
from decider.infer import Decider
d = Decider("Mapika/decider-12b")
d.system_one(state, {"refund": {"type": "noul", "instructions": "Is the refund allowed under the policy?",
                                "criteria": {"true": "allowed", "false": "not allowed"}}})
```

Serving: `DECIDER_MODEL=Mapika/decider-12b uvicorn decider.serve:app` (POST `/v1/systemone`, the System One wire format).

## Training (version 2)

- **Method:** LoRA rank 32, alpha 64, on q/k/v/o/gate/up/down. One epoch of 10,000 items (10.6M tokens), LR 5e-5 cosine,
  one B300, 45 minutes. The loss is cross-entropy over the option letters at the answer slot, in the same chat layout as serving.
- **Data:**
  - **6,000 generated state-tracking decisions:** event logs over seven invented domains (library loans, parking permits, hotel
    rooms, sprint boards, vehicle fleets, licence seats, lab samples). They include rejected entries, VOID and CORRECTION
    entries, aliases and out-of-order exports, rendered as logs, tables, prose and JSON.
    - The gold answers are computed by replaying the log.
    - Of 104 yes/no rows answered blind by GLM-5.3-Flash, 101 agree with the computed gold.
  - **4,000 rows from our earlier training data, replayed:**
    - 836 human-labelled public datasets;
    - 2,047 from our own generated decision families, written without reading JevBench items;
    - 1,117 teacher-written document questions.
- **Not used for training or selection:** JevBench items, public or sealed; Decision Index rows. The checkpoint and the
  temperatures were chosen on our own held-out sets only.

## Temperatures

| type | v2 T | v1 T | fitted on (our own rows) |
|---|---|---|---|
| Choice | 1.5 | 4.0 | NLL and top-label ECE over 2,113 held-out Choice rows |
| Noul | 0.05 | 1.0 | chance-corrected v1.5 Noul score on 685 rows plus the 204-item fit half of a fresh yes/no set |
| Score | 1.0 | 3.5 | chance-corrected Score competence on 246 rows |

- The fine-tune leaves the model less confident, so every temperature is lower than in v1.
- Noul is read sharp. JevBench v1.5 counts a yes/no answer with P(yes) between 0.2 and 0.8 as wrong, and at T 0.05 about 1% of
  answers fall in that band.
- **For calibrated yes/no probabilities, use Noul T 1.0.** At T 0.05 most answers round to exactly 0 or 1 at 4 decimals, so the
  probabilities do not rank or calibrate. On the fresh 399-item yes/no set (bf16) the accuracy is 0.847 at every temperature:

  | Noul T | AUROC | ECE | log loss | exactly 0 or 1 | JevBench v1.5-style Noul score |
  |---|---|---|---|---|---|
  | 0.05 (config) | 0.878 | 0.151 | 1.277 | 92% | 68.1 |
  | 1.0 | 0.934 | 0.038 | 0.326 | 5% | 32.2 |
  | 1.5 | 0.934 | 0.049 | 0.333 | 5% | 3.6 |

  In Python: `Decider("Mapika/decider-12b", temperature_by_type={"choice": 1.5, "noul": 1.0, "score": 1.0})`. Reported in
  [decider issue #18](https://github.com/Mapika/decider/issues/18), where the same effect was measured on 2,975 items.

## Measurements (decider-ai 1.8.0, one B300, each version at its own temperatures)

| set | v2 | v1 |
|---|---|---|
| fresh 399-item yes/no set, test half (195), v1.5 chance-corrected Noul score | 71.6 | 58.8 |
| same, argmax accuracy / share of answers between 0.2 and 0.8 / ECE | 0.856 / 1 % / 0.144 | 0.805 / 6 % / 0.181 |
| same set, 80 generated state-tracking logs (another generator than training), argmax | 0.750 | 0.675 |
| same set, 319 written items, argmax | 0.871 | 0.859 |
| held-out generated decisions (1,500) and teacher questions (944): Choice accuracy | 0.615 | 0.588 |
| same rows: Noul accuracy / chance-corrected Noul score | 0.790 / 57.1 | 0.740 / 38.7 |
| same rows: chance-corrected Score competence | 61.2 | 60.1 |
| human-labelled Choice rows (600): accuracy / ECE | 0.847 / 0.015 | 0.845 / 0.024 |
| JevBench public items, accuracy easy / standard / hard | 1.000 / 0.986 / 0.712 | 1.000 / 0.986 / 0.730 |
| JevBench public hard tier, top-label ECE | 0.147 | 0.098 |

- The public JevBench items were read for measurement only.
- On the hard tier (111 items), v2 answers 79 correctly and v1 answers 81.
- Hard-tier calibration is worse in v2.

## Limitations

- The base model's strengths and failure modes carry over.
- The fine-tune targets state tracking over event logs. Gains elsewhere come from the replayed rows and are 1-5 points.
- On the JevBench public hard tier, v2 is not better than v1: 79 vs 81 correct of 111, and ECE 0.147 vs 0.098.
- Per-dataset calibration still varies: knowledge multiple choice is underconfident, and hard reasoning is overconfident.

## Licence

Weights: derived from google/gemma-4-12B-it, Apache-2.0 as published by Google.
Readout code: Apache-2.0 (github.com/Mapika/decider).
