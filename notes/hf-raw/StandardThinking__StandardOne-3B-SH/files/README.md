---
base_model: StandardThinking/StandardOne-3B
library_name: transformers
license: apache-2.0
pipeline_tag: text-classification
language:
  - en
tags:
  - mistral3
  - decision-model
  - typed-decisions
  - schema-head
  - jev
  - calibration
  - decode-free
---




# Standard One 3B SH




> **Updated weights (v3, 2026-10-13).** v3 is trained further on decision data, mostly our own synthetic items. Decision Index 0.3 public index 51.53 (self-scored with the official kit; v2 42.90).
> Earlier releases stay available under the tags `v2` and `v1`.




**Version:** v3




**Standard One 3B SH** is Standard One 3B with a **joint schema head** (SH). It reads the scenario once and
scores every option of every question in a single forward pass, then returns probabilities through the same
`POST /v1/systemone` contract as Standard One. It does not generate text.




| | |
|---|---|
| Backbone | Standard One 3B v2.2 (`mistral3`, Ministral 3 3B text + Pixtral vision tower), fine-tuned for the head and merged; weights in this repository |
| Head | joint schema head, float32 (`head/`), reading the final layer and layer 19 |
| Question types | `choice` (2–255 options), `noul` (yes/no), `score` (ordinal scale) |
| Context | up to 32,768 tokens per request |
| Prompt format | the chat template with **no default system message** (see "Prompt format") |
| Other formats | [StandardOne-3B-SH-FP8](https://huggingface.co/StandardThinking/StandardOne-3B-SH-FP8) · [StandardOne-3B-SH-GGUF](https://huggingface.co/StandardThinking/StandardOne-3B-SH-GGUF) (both v3) |
| Larger model | [StandardOne-8B-SH](https://huggingface.co/StandardThinking/StandardOne-8B-SH) |
| License | Apache-2.0 |




## Why a schema head




Standard One reads one answer letter per option, so a question is limited by the label alphabet and every
question in a request needs its own pass over the scenario. The schema head instead attends over the hidden states
of the whole request (the final layer and one intermediate layer), places every question and option in one joint
graph, and returns a calibrated distribution for each question at once. In practice:




- **Many options.** Intent and routing questions with 100–255 options keep nearly the accuracy of small option sets.
- **Many questions per scenario.** All questions of a request share one pass over the scenario.
- **Long scenarios.** Decisions buried in long documents are found far more reliably than with Standard One v2.2.




## Results




| Benchmark | SH v2 | **SH v3 (this release)** |
|---|---:|---:|
| Decision Index 0.3, public index | 42.90 | **51.53** |
| Areas: Knowledge / Language / Retrieval / Tools / Arts | 25.2 / 44.2 / 59.7 / 60.3 / 19.8 | 30.3 / 49.5 / 63.3 / 79.8 / 36.5 |
| HoVer, multi-hop claim verification (DI) | 62.7 | 63.9 |
| BRIGHT, reasoning-heavy retrieval over long documents, ~12k tokens per request (DI) | 25.7 | 23.9 |
| Decision hidden in a long document (our set, 150 items) | 82.00 | 85.33 |
| JevBench public: easy / original / hard (48 / 72 / 111 items) | 100.00 / 98.61 / 63.96 | 100.00 / 98.61 / 63.96 |




Decision Index rows (DI) are chance-corrected skill scores (0 = random guessing, 100 = perfect) from our own runs of the public
0.3 suite with the official kit, all 140,178 scoreable requests; every benchmark of both runs is in [standardone-3b-sh-v2](https://huggingface.co/datasets/StandardThinking/StandardOne-SH-decision-index/tree/main/runs/standardone-3b-sh-v2)
and [standardone-3b-sh-v3](https://huggingface.co/datasets/StandardThinking/StandardOne-SH-decision-index/tree/main/runs/standardone-3b-sh-v3). For some DI benchmarks the public train split or the benchmark's public
generator was part of training (HoVer among them); evaluated items never were. The other rows are accuracy in percent
on identical items: the v2 column with the reference server in `code/`, the v3 column with the Standard One engine.




## Benchmark charts




Decision Index 0.3 public index, board data of 2026-10-12; Standard One self-scored with the official kit (see the footnote of each chart).
The same charts with notes: [Standard One SH on the Decision Index](https://standardthinking.ai/resources/guides/standard-one-sh-decision-index/).




![Public index against model size](https://huggingface.co/StandardThinking/StandardOne-3B-SH/resolve/main/assets/charts/01-index-vs-size.png)




![Retrieval & Classification](https://huggingface.co/StandardThinking/StandardOne-3B-SH/resolve/main/assets/charts/02-retrieval-classification.png)




![Tools & Automation](https://huggingface.co/StandardThinking/StandardOne-3B-SH/resolve/main/assets/charts/03-tools-automation.png)




![Product search relevance (Amazon ESCI)](https://huggingface.co/StandardThinking/StandardOne-3B-SH/resolve/main/assets/charts/04-product-search.png)




![Multi-hop claim verification (HoVer)](https://huggingface.co/StandardThinking/StandardOne-3B-SH/resolve/main/assets/charts/05-claim-verification.png)




![Smart-home request handling (Home appliance simulator)](https://huggingface.co/StandardThinking/StandardOne-3B-SH/resolve/main/assets/charts/06-home-appliance.png)




![Contract understanding (ContractNLI)](https://huggingface.co/StandardThinking/StandardOne-3B-SH/resolve/main/assets/charts/07-contract-nli.png)




![Reading chords from a score (POP909-CL)](https://huggingface.co/StandardThinking/StandardOne-3B-SH/resolve/main/assets/charts/08-chords.png)




## Changes in v3




- **Further training on synthetic decision data.** v3 continues from v2 on decision items built for this round, mostly our own
  synthetic items (code-checked, LLM-written in varied styles), with replay of earlier data. See "Results" for the
  Decision Index 0.3 and the long-context checks.




## Changes in v2




- **Hard decision tasks.** v2 is fine-tuned on about 2,200 new decision items written in the families where v1 was
  weakest: long policies with exceptions and overrides, multi-criteria trade-offs, judging a response against a
  rubric, and date and number reasoning. The items were authored by large language models, every gold answer was
  checked by an independent model that answered without seeing it (and recomputed where they disagreed), and items
  sharing text with the JevBench public set or our evaluation sets were removed. No JevBench item was used. These
  items are trained with softened targets (0.8 on the gold option), which keeps the head from becoming over-confident
  on tasks it cannot always solve.
- **Effect.** The JevBench hard tier rises from 54.05 to 63.96 (long policies 7 → 13 of 19 items). Confidence on that
  tier tracks accuracy closely (mean confidence 72.6 for 64.0 correct).
- **Trade-offs.** Policy-judgement repair is 3.5 points below v1 (64.58 vs 68.08); WinoGrande, games and the intent
  label sets are within one point of v1. Keep `v1` if policy-judgement repair matters most to you.




## Prompt format




The included `chat_template.jinja` has an empty default system message, and the reference server renders requests
without one (`SH_SYSTEM_PROMPT=none`, the default in `code/render.py`). Do not add the Ministral default system
prompt: the head was trained without it, and adding it changes the hidden states it reads.




## Quick start




```bash
python -m pip install --upgrade huggingface_hub
hf download StandardThinking/StandardOne-3B-SH --local-dir StandardOne-3B-SH
python -m pip install -r StandardOne-3B-SH/code/requirements.txt




python StandardOne-3B-SH/code/serve_head.py \
  --snapshot StandardOne-3B-SH --head StandardOne-3B-SH/head --port 30171 \
  --served-model-name standardthinking/standard-one-3b-sh
```




```bash
curl -s localhost:30171/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"message": "Please close my card, I lost it yesterday."},
  "questions": {
    "intent": {"type": "choice", "instructions": "What does the customer want?",
               "criteria": {"cancel": "close the card", "limit": "change the limit", "other": null}},
    "urgent": {"type": "noul", "instructions": "The request is urgent."},
    "mood":   {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]}
  }
}'
```




The server batches concurrent requests (`--max-batch-requests`, `--max-batch-tokens`) and binds `127.0.0.1` by
default. It runs on one GPU with the backbone in bfloat16 and the head in float32. On Blackwell GPUs (B200/B300)
install a CUDA 13 build of PyTorch first (`pip install torch --index-url https://download.pytorch.org/whl/cu130`).
For CPU or Apple-silicon use, see [StandardOne-3B-SH-GGUF](https://huggingface.co/StandardThinking/StandardOne-3B-SH-GGUF).






## Run locally with llama.cpp


For CPU or Apple-silicon use, follow the download, build and server instructions in [StandardOne-3B-SH-GGUF](https://huggingface.co/StandardThinking/StandardOne-3B-SH-GGUF#quick-start-with-llamacpp). The SH head reads hidden states from the backbone, so it requires the supplied state extractor and Python runner; stock `llama-server -hf` alone does not execute the schema head. Once the runner is listening on port 30171, use the requests below unchanged.


## Practical examples

The following invented records illustrate four integrations: receiving inventory, diagnosing an import failure, evaluating an inspection note, and checking backup evidence. They are example inputs, not benchmark results. Start either the reference server or the SH GGUF runner on port 30171 before sending them.

### Warehouse receiving: reconcile a delivery record

An inventory application can ask several questions about one receiving note. Here, the model compares the recorded quantities and checks whether the paperwork supports closing the receipt.

```bash
curl --fail-with-body http://127.0.0.1:30171/v1/systemone \
  -H "Content-Type: application/json" --data-binary @- <<'JSON'
{
  "state": "Purchase order R-417 lists 48 cable reels. The receiving scan recorded 46 reels. Two pallet tags cannot be read, and the receiver has not signed the count sheet.",
  "questions": {
    "receipt_action": {
      "type": "choice",
      "instructions": "Select the next inventory step supported by the receiving note.",
      "criteria": {
        "close_receipt": "The delivered quantity is reconciled and the count sheet is signed.",
        "reconcile_count": "The recorded quantity differs from the order or pallet identification is unresolved.",
        "await_delivery": "The shipment has not reached the receiving area."
      }
    },
    "quantities_match": {
      "type": "noul",
      "instructions": "Do the receiving scan and purchase order report the same number of reels?"
    },
    "record_completeness": {
      "type": "score",
      "instructions": "Evaluate how complete the receiving evidence is.",
      "criteria": [
        "No quantity or identification evidence is available.",
        "Some evidence is recorded, but discrepancies or missing sign-off remain.",
        "Quantities and identifiers are reconciled and sign-off is recorded."
      ]
    }
  }
}
JSON
```

This record describes an unresolved count, unequal quantities and partial paperwork. Use those facts to inspect the returned answers. A `choice` question compares the descriptions you provide, a `noul` question returns a yes/no probability, and a `score` question returns an expected position on its ordered, zero-based scale. The score can lie between two levels.

### Python client

Use this small standard-library client for the remaining examples. It displays the response as received so that you can examine the selected values and probability estimates.

```python
import json
from urllib.request import Request, urlopen

def evaluate_record(record, questions):
    payload = json.dumps({"state": record, "questions": questions}).encode("utf-8")
    request = Request(
        "http://127.0.0.1:30171/v1/systemone",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        print(json.dumps(json.load(response), indent=2))
```

### Data import: identify a column mismatch

An import monitor can classify a diagnostic record before selecting a repair playbook. This input contains rows, but the expected timestamp column is absent.

```python
evaluate_record(
    "Import job J-82 read 1,200 CSV rows. Required column event_time was not found. The file contains a column named timestamp_ms. UTF-8 decoding succeeded.",
    {"failure_kind": {
        "type": "choice",
        "instructions": "Identify the import failure described in this diagnostic record.",
        "criteria": {
            "column_mismatch": "The available column names differ from the required schema.",
            "empty_input": "The file contains no data rows.",
            "text_encoding": "The importer cannot decode the file contents.",
            "unresolved": "The record does not establish a failure category."
        }
    }}
)
```

### Equipment inspection: evaluate a filter reading

A maintenance log can map a written observation onto a defined condition scale. The criterion here is the gauge marking recorded by the inspector.

```python
evaluate_record(
    "Inspection of ventilation unit V-12: the filter pressure gauge is above the marked replacement line. The fan guard is secured and the housing has no visible cracks.",
    {"filter_condition": {
        "type": "score",
        "instructions": "Assess the filter condition using the reported gauge position.",
        "criteria": [
            "The gauge is inside the normal operating band.",
            "The gauge is above normal but below the replacement line.",
            "The gauge has reached or exceeded the replacement line."
        ]
    }}
)
```

### Backup verification: compare evidence with an acceptance rule

A backup catalog can check whether a verification record satisfies a supplied rule. This example requires both a matching checksum and a successful restore check.

```python
evaluate_record(
    "Archive B-206 was written to storage. Verification reported a checksum mismatch. A restore check has not been attempted.",
    {"backup_verified": {
        "type": "noul",
        "instructions": "Does this record satisfy the rule that a verified backup must have a matching checksum AND a successful restore check?"
    }}
)
```

These calls only request decisions; the surrounding application decides what to do with them. Measure accuracy and choose probability thresholds using records from your own workflow. All examples send text, which is the validated input path for these releases.

## Files




| Path | Contents |
|---|---|
| `model-0000{1,2}-of-00002.safetensors`, `config.json`, tokenizer files | merged backbone, bfloat16 (`Mistral3ForConditionalGeneration`) |
| `chat_template.jinja` | chat template with an empty default system message |
| `head/` | `schema_head.safetensors` (float32) and `schema_head_config.json` (taps: final layer and layer 19) |
| `code/` | reference `/v1/systemone` server: request rendering, head, scoring |
| `SHA256SUMS`, `release-manifest.json` | checksums; source and merge details |




## Limitations




- Requests longer than 32,768 tokens are refused, not truncated.
- On our hard held-out decision suite (600 items; long policies, multi-step lookups, date and number reasoning)
  it scores 41.67 against 44.67 for Standard One 3B v2.2. Prefer Standard One 3B for that kind of workload.
- Text input only has been validated for this release.
- The reference server is a PyTorch server; throughput under heavy load is lower than an inference engine.
- Probabilities are calibrated on our held-out data; refit thresholds on your own distribution before relying on them.




## License




Apache-2.0. Built on [Standard One 3B](https://huggingface.co/StandardThinking/StandardOne-3B) (Apache-2.0), which is
built on [Ministral 3 3B](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) (Apache-2.0).



