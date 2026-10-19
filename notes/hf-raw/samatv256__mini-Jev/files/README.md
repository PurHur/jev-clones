---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
library_name: custom
datasets:
  - samatv256/jev-decisions-v1
tags:
  - mini-jev
  - decision-model
  - tool-selection
  - tool-use
  - agents
  - action-selection
  - decision-making
  - finite-choice
  - qwen3
---

# mini-Jev Qwen3-0.6B - step 10,626

mini-Jev is a **0.6B agentic decision model specialized for finite-choice tool and action selection**. Give it a state, a question, and candidate actions; it returns a probability for each candidate. It uses a Qwen3-0.6B backbone with a PEFT/LoRA adapter and a compact decision head. Candidate scoring is permutation equivariant, and the model returns finite-choice probabilities rather than generated text or tool arguments.

This is an **early 10,626-update checkpoint**. Training is planned through 50,000 updates; more training and broader decision supervision are coming. This interim release is not the final checkpoint.

## Release files and revisions

The current [`step-010626`](https://huggingface.co/samatv256/mini-Jev/tree/step-010626) checkpoint loads its adapter from [`adapter/`](adapter/), its decision-head weights from [`decision_head.safetensors`](decision_head.safetensors), and its inference loader from [`inference.py`](inference.py). Use these with `config.json` and the separately downloaded Qwen3-0.6B base model.

The retained root `model.safetensors` and `odm_mini.py` belong to the previous v1 baseline. They are preserved only for history and backward reference; they are not the step 10,626 weights or loader. To use that baseline, pin the [previous revision](https://huggingface.co/samatv256/mini-Jev/tree/02acc03cc28ec0b99705a8459ea987274fa7987f).

## Measured results

| Evaluation | Result | Scope |
|---|---:|---|
| BFCL V1 Multiple Function - Jev function-selection adaptation | **96.5% (193/200)** | Selects the supplied function identity from 2-4 definitions. **Not an official BFCL leaderboard score**; does not generate arguments or execute calls. |
| Strict held-out Choice | **65.6% (63/96)** | A selected, bucket-balanced validation slice, not the full validation split. |
| Bespoke OOD typed decisions | **72.0% (36/50)** | Small synthetic sanity check with straightforward distractors; not a public benchmark. |
| LocalLLaMA/typed-decisions, full test set | **34.0% (680/2,000)** | Choice 27.0%, Noul 52.2%, Score 25.6%. General typed decisions remain a weakness. |

![BFCL selection accuracy at step 0 and step 10,626](assets/bfcl-step-comparison.svg)

![BFCL selection accuracy by candidate count](assets/bfcl-by-candidates.svg)

![Accuracy snapshot across separate evaluations](assets/evaluation-summary.svg)

These evaluations have different datasets and denominators; their percentages are not directly comparable. The BFCL adaptation uses the [BFCL V1 Multiple Function AST examples](https://github.com/EnlightenedAI/BFCL/blob/main/berkeley-function-call-leaderboard/bfcl_eval/data/README.md). The typed-decision result uses the [LocalLLaMA/typed-decisions test split](https://huggingface.co/datasets/LocalLLaMA/typed-decisions).

## Intended use and limits

The current strength is **finite-choice tool and action selection** from state and supplied candidates. The model is less reliable for general typed decisions and for large candidate sets. It is not a general probability or classification model, and its probabilities have not been calibrated for arbitrary domains. A selected 17+ candidate held-out slice scored 4/17, so large sets need particular care.

## Use

A CUDA GPU with BF16 support is required for this 4-bit package. Install the tested dependencies from requirements.txt. The [Qwen3-0.6B base model](https://huggingface.co/Qwen/Qwen3-0.6B) is downloaded separately on first load; the current checkpoint consists of the adapter and decision-head files described above.

Fetch the inference module and dependency list into a working directory first:

```bash
python -m pip install "huggingface_hub==1.33.0"
hf download samatv256/mini-Jev inference.py requirements.txt --local-dir .
python -m pip install -r requirements.txt
```

    from inference import MiniJev

    model = MiniJev.load()  # Defaults to samatv256/mini-Jev using the HF cache.
    result = model.predict(
        state={"user_goal": "Find tomorrow's weather forecast for Paris."},
        question="Given the current state and available options,\nwhich option should be selected?",
        question_type="choice",
        answer_options=[
            {"id": "weather", "type": "tool", "label": "get_weather_forecast",
             "description": "Look up the weather forecast for a place and date."},
            {"id": "invoice", "type": "tool", "label": "calculate_invoice_total",
             "description": "Add amounts on an invoice."},
            {"id": "email", "type": "tool", "label": "send_email",
             "description": "Send an email message."},
        ],
    )
    print(result["selected_id"])
    print(result["options"])

To pin a release, use `MiniJev.load("samatv256/mini-Jev", revision="step-010626")`.
Existing local directories and `Path` arguments remain supported, for example
`MiniJev.load(".")` after downloading the release files. Existing local paths take
precedence over repo IDs. `revision` applies only to Hub loading. Set
`HF_HUB_OFFLINE=1` to use cached files offline; both the release and its pinned Qwen
base/tokenizer must already be cached. Hub loading fetches only `config.json`,
`decision_head.safetensors`, `adapter/adapter_config.json`, and
`adapter/adapter_model.safetensors` from one revision, excluding legacy root weights.

Option IDs are bookkeeping and are excluded from model text. The model scores each supplied option and normalizes probabilities over the finite set. Input branches are limited to 8,192 tokens.

## Data attribution

Decision supervision for this checkpoint draws in part on [Jev Decisions v1](https://huggingface.co/datasets/samatv256/jev-decisions-v1), a public dataset released under CC BY 4.0. Its public upstream datasets carry their own attribution and licensing terms; see the [dataset card](https://huggingface.co/datasets/samatv256/jev-decisions-v1) and [upstream license notes](https://huggingface.co/datasets/samatv256/jev-decisions-v1/blob/main/SOURCE_LICENSES.md).

## License

Apache-2.0. The Qwen3-0.6B base model is separately licensed by Qwen under Apache-2.0. See LICENSE and the [base model card](https://huggingface.co/Qwen/Qwen3-0.6B).
