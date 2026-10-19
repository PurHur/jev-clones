---
license: apache-2.0
base_model: IFM/K2-Horizon-3.7B
library_name: transformers
language:
- en
tags:
- decision-model
- classification
- lora
- k2-horizon
- decision-index
datasets:
- theaviv/lev1-decisions
- openai/gsm8k
---

# Lev1 (K2-Horizon-3.7B)

Lev1 is a System One decision model, in the line of Typesafe AI's Jev: fast, intuitive calls on structured input. You send a state and typed questions, and Lev1 returns a probability for every option in one forward pass. It never generates text.

- **Fast.** 28.8 ms median, 169.9 ms p95 per request on one RTX PRO 6000 Blackwell Server Edition.
- **Scored.** 42.28 on Decision Index 0.2.1, from a complete run with the official harness.
- **Best in class on tools.** 68.8 on Tools & Automation, above every model under 10B parameters on the board (as of 2026-09-28).
- **Small and open.** A 3.7B model: a LoRA on [IFM/K2-Horizon-3.7B](https://huggingface.co/IFM/K2-Horizon-3.7B), shipped as merged weights and as the adapter, under Apache-2.0.
- **Clean data.** Training sources carry commercial-use licences and passed decontamination against the benchmark suite. The synthetic part is public as [theaviv/lev1-decisions](https://huggingface.co/datasets/theaviv/lev1-decisions).

## Training

I trained a LoRA (rank 32, alpha 64) on questions sampled from a 163,611-question mixture. Each short question also appears in a second option order, and a consistency loss pulls the two answers together. For 15,591 questions the target blends the gold label with Gemma-4-31B-it's probabilities, on sources where Gemma matched gold at least 60% of the time. MiMo-V2.6-Flash wrote the 4 LLM-written sets; I kept a text only when Qwen3-235B-A22B agreed with its label, and Gemma-4-31B-it too for all but the phishing emails.

| Data | Licence | Role | Train questions |
|---|---|---|---:|
| [theaviv/lev1-decisions](https://huggingface.co/datasets/theaviv/lev1-decisions) (17 generators, 4 LLM-written sets) | CC-BY-4.0; CC-BY-SA-3.0 (hallucination set) | synthetic | 31,837 |
| [Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) | CC0-1.0 | general | 43,311 |
| [multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli), [snli](https://huggingface.co/datasets/stanfordnlp/snli), [boolq](https://huggingface.co/datasets/google/boolq), [contract-nli](https://huggingface.co/datasets/presencesw/contract-nli), [SemEval_NLI4CT](https://huggingface.co/datasets/araag2/SemEval_NLI4CT), [RAGTruth-processed](https://huggingface.co/datasets/wandb/RAGTruth-processed) | OANC/CC-BY/CC-BY-SA/MIT; CC-BY-SA-4.0; CC-BY-SA-3.0; CC-BY-4.0; MIT (annotations); Summary task only (news contexts) | inference | 19,686 |
| [winogrande](https://huggingface.co/datasets/allenai/winogrande), [social_i_qa](https://huggingface.co/datasets/allenai/social_i_qa), [piqa](https://huggingface.co/datasets/ybisk/piqa), [openbookqa](https://huggingface.co/datasets/allenai/openbookqa), [ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | Apache-2.0; CC-BY-4.0; AFL-3.0; CC-BY-SA-4.0 | commonsense | 18,315 |
| [esci](https://huggingface.co/datasets/tasksource/esci), [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2), [newyorker_caption_contest](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest), [chess-puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles) | Apache-2.0; CC-BY-4.0; CC0-1.0 | mixed | 16,950 |
| [clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos), [banking77](https://huggingface.co/datasets/legacy-datasets/banking77), [hwu_64](https://huggingface.co/datasets/FastFit/hwu_64), [amazon_massive_intent](https://huggingface.co/datasets/mteb/amazon_massive_intent) | CC-BY-3.0; CC-BY-4.0 | intent detection | 13,061 |
| [esci](https://huggingface.co/datasets/tasksource/esci) | Apache-2.0 | retrieval | 12,280 |
| [gsm8k](https://huggingface.co/datasets/openai/gsm8k) | MIT | reasoning | 6,026 |
| [When2Call](https://huggingface.co/datasets/nvidia/When2Call) | CC-BY-4.0 | tool use | 2,145 |

## Results

| | **Lev1** | Decision 1.0 Lux | JPT-4B | Jet v6.2 | Decider 4B | Kev 4B |
|---|---:|---:|---:|---:|---:|---:|
| **Decision Index** | **42.28** | **43.49** | **43.04** | **42.60** | **40.70** | **34.64** |
| Knowledge & Reasoning | 22.7 | 30.9 | 28.7 | 28.7 | 25.7 | 22.9 |
| Language Understanding | 44.0 | 48.0 | 52.5 | 43.9 | 46.0 | 35.3 |
| Retrieval & Classification | 47.1 | 50.0 | 45.0 | 48.2 | 44.7 | 41.0 |
| Tools & Automation | 68.8 | 57.2 | 57.2 | 62.9 | 58.6 | 52.6 |
| Arts & Human Taste | 30.5 | 26.4 | 25.8 | 27.0 | 25.0 | 17.9 |

The other columns are entrants near Lev1 on the board. Scores are chance-corrected skill: 0 is guessing, 100 is perfect. The run answered all 150,759 requests; the harness scores 150,317 of them and excludes 442.

## Try it

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("theaviv/Lev1-3.7B")
sys.path.insert(0, path)  # the repo ships its scorer as the `lev1` package
from lev1.engine import Lev1Scorer

scorer = Lev1Scorer.from_pretrained(path)  # adapter=True loads the base plus the LoRA instead
probs = scorer.score(
    {"channel": "email", "message": "Checkout has returned a 500 error for an hour and orders are failing."},
    {
        "urgent": {"type": "noul", "instructions": "This message needs a reply within the hour."},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle it?",
            "criteria": {"billing": "Payments, refunds and invoices.", "engineering": "Outages and bugs.", "sales": "Pricing and new accounts."},
        },
    },
)
print(probs["urgent"]["yes"], probs["team"])
```

Decision Index harness (bring your own copy of the suite):

```bash
PYTHONPATH="$(python -c 'from huggingface_hub import snapshot_download as d; print(d("theaviv/Lev1-3.7B"))')" python -m decision_index pipeline --engine lev1.engine:Lev1Engine --edition 0.2.1 --out runs/lev1
```

## Limits

- Two question types, `choice` and `noul`, up to 255 options. A request that does not fit raises `Unsupported`; Lev1 never truncates or drops an option.
- States above 16,384 tokens take a slower chunked path.
- English only. Weakest on Knowledge & Reasoning (22.7) and Arts & Human Taste (30.5).
- The probabilities come straight from the softmax, without a fitted temperature. Check calibration on your data before you set thresholds.

---

Apache-2.0. Built on [IFM/K2-Horizon-3.7B](https://huggingface.co/IFM/K2-Horizon-3.7B) (Apache-2.0, revision `fe504ef19c7e`). By Aviv Dozorets.

Lev1 is not `lev`, the Interfaze entrant on the Decision Index board.
