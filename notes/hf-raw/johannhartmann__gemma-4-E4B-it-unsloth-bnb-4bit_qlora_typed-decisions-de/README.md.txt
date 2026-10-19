---
license: apache-2.0
base_model: google/gemma-4-E4B-it
datasets:
- johannhartmann/typed-decisions-de
- LocalLLaMA/typed-decisions
language:
- de
pipeline_tag: text-classification
tags:
- decision-model
- structured-decisions
- system-one
- calibration
- clef
- unsloth
- german
---

# gemma-4-E4B-it-unsloth-bnb-4bit_qlora_typed-decisions-de

A German decision model: [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)
fine-tuned with QLoRA (via [`unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`](https://huggingface.co/unsloth/gemma-4-E4B-it-unsloth-bnb-4bit))
and a fresh Clef-style joint-schema decision head on
[`johannhartmann/typed-decisions-de`](https://huggingface.co/datasets/johannhartmann/typed-decisions-de),
the German translation of [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
in Unsloth Studio. Given a state and a set of typed questions (`choice`, `noul`, `score`), it
returns a calibrated probability distribution over the options of every question in one
forward pass, in the System One / Decision API answer format.

This is an independent fine-tune by johannhartmann. It is not affiliated with or endorsed by
Google or Unsloth; the name only refers to the base model and how it was trained. The weights
in this repository are **not** 4-bit: the LoRA was merged into a dequantized bf16 copy of the
base, so they are a regular bf16 checkpoint (about 16 GB).

## Results

**German `test` split** (400 cases, 2,000 decisions), measured with a small script on the final
model:

| Metric | Fine-tuned |
|---|---|
| Accuracy (per decision, argmax vs. gold label) | **0.776** |
| Record accuracy (all decisions of a case right) | 0.333 |
| Mean KL(gold ‖ model) per decision | 0.496 |

**Studio held-out set** (80 cases, 400 decisions held out of the `train` split), as reported by
Unsloth Studio before and after fine-tuning:

| Metric | Base (fresh head) | Fine-tuned |
|---|---|---|
| Accuracy | 0.393 | 0.800 |
| Expected calibration error | 0.070 | 0.037 |

The `test` split was not used during training, for evaluation or for calibration. The
Studio numbers come from 80 cases Studio held out of the `train` split; the calibration
temperatures were fitted on those same cases, so that ECE is optimistic.

Please read these numbers with the following caveats:

- **This is a specialist.** It was trained on the benchmark's `train` split for these four
  workflows. The `typed-decisions` card asks for specialist and zero-shot general models to be
  reported separately; numbers are not comparable across the two.
- **Gold is a teacher's distribution, not ground truth.** Gold is the mean of three samples
  from a ~4B-class teacher, generated on the **English** text and carried over unchanged to the
  German translation. Teacher self-agreement is 0.735; accuracy above that level suggests the
  model is also fitting the teacher's tendencies.
- These are not the benchmark's official scorer. The original leaderboard is for the English
  version and does not transfer to the German data.

## Training

| Setting | Value |
|---|---|
| Base | `google/gemma-4-E4B-it` (text backbone hidden size 2560, 42 layers), loaded as `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` |
| Head | new Clef joint-schema head (width 512, 4 layers, 2 routing layers, 8 heads) |
| Method | QLoRA r=64, alpha=64, dropout 0, on q/k/v/o/gate/up/down projections; merged into bf16 weights |
| Objective | soft cross-entropy against the gold distributions |
| Data | `johannhartmann/typed-decisions-de`, config `all`, split `train` (1,200 cases, 6,000 decisions), of which 80 cases (400 decisions) were held out by Studio for evaluation and calibration |
| Schedule | 3 epochs, 210 steps, batch 8 x grad. accumulation 2, lr 1e-4 linear, 5 warmup steps, weight decay 0.01, AdamW |
| Max length | 4,096 tokens |
| Seed | 3407 |
| Hardware | 1x RTX A6000, about 27 minutes, peak memory 19.1 GB |
| Calibration | per question type temperatures [1.073, 0.874, 1.211] (`unsloth_decision_config.json`), fitted on the 400 held-out train decisions |

Only text is used. The decision head reads text tokens; the vision and audio towers of Gemma 4
are included in the merged weights but not used.

## Usage

Requires [Unsloth](https://github.com/unslothai/unsloth) with decision-model support and a
`transformers` version with the `gemma4` architecture. Inference in bf16 needs about 16 GB of
GPU memory.

```python
from unsloth import FastDecisionModel

model, tokenizer = FastDecisionModel.from_pretrained(
    "johannhartmann/gemma-4-E4B-it-unsloth-bnb-4bit_qlora_typed-decisions-de")
FastDecisionModel.for_inference(model)

state = {"account": {"tier": "Business", "seats": 25, "tenure_months": 30},
         "thread": [{"role": "Kunde", "text": "Seit gestern Abend kann sich unser ganzes Team nicht mehr "
                                              "anmelden. Morgen früh haben wir eine Präsentation beim Kunden."}]}
questions = {
    "category": {"type": "choice", "instructions": "Worum geht es in diesem Kundengespräch hauptsächlich?",
                 "criteria": {"account": "Anmeldung, Tarifwechsel, Profil- oder Zugriffsverwaltung.",
                              "billing": "Ein Problem mit einer Abbuchung, Rechnung, einem Abonnement oder einer Zahlung.",
                              "technical": "Das Produkt oder der Dienst funktioniert nicht wie erwartet."}},
    "urgency": {"type": "score", "instructions": "Wie zeitkritisch ist dieses Gespräch?",
                "criteria": ["Kein Zeitdruck.", "Routine; normale Warteschlange.",
                             "Erhöht; heute bearbeiten.", "Kritisch; sofort handeln."]},
    "needs_human": {"type": "noul", "instructions": "Dieses Gespräch sollte an einen Menschen übergeben werden."},
}
print(FastDecisionModel.predict(model, tokenizer, state, questions))
```

In Unsloth Studio the model can also be served through the Decision API
(`POST /v1/systemone`).

**No GGUF.** llama.cpp's decision graph (`/v1/systemone`) currently supports Qwen3.5 backbones
only, so this Gemma 4 model cannot be exported to a GGUF that keeps its decision head. It serves
through PyTorch.

The model was trained on four workflows (agent trace observability, customer service, invoice
processing, security incidents). Expect weaker results on other domains. As in the dataset,
question and option keys stay English; instructions and option descriptions are German.

## License

Apache-2.0, matching the base model ([`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it), Apache-2.0)
and the training data ([`johannhartmann/typed-decisions-de`](https://huggingface.co/datasets/johannhartmann/typed-decisions-de)
and [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), both Apache-2.0).
