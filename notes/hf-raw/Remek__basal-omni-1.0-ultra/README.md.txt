---
license: apache-2.0
language:
- pl
- en
base_model: google/gemma-4-31B-it
library_name: transformers
pipeline_tag: text-generation
tags:
- typed-decisions
- decision-making
- classification
- calibration
- polish
- gemma4
- experimental
---

# basal-omni-1.0-ultra (experimental)

[![GitHub](https://img.shields.io/badge/GitHub-rkinas%2Fbasal%20%E2%80%94%20inference%20engine-black?logo=github)](https://github.com/rkinas/basal)
[![Website](https://img.shields.io/badge/web-basal.si5.pl-0a7d5a)](https://basal.si5.pl/)

**basal-omni-1.0-ultra (31B)** is the largest model of **basal-omni**, basal's line on Google Gemma 4. It is an
**experimental** release: it is trained on text decisions only, and it also answers the same questions about
**images**, read by Gemma 4's own image encoder without media training (zero-shot; media training comes in the next
releases). The 31B base has no audio encoder, so this tier takes no audio. It answers the same typed questions as
basal-1.5, over the same API.

**What it is.** Inspired by *System 1* (fast, intuitive) decision models such as Jev. Instead of writing an answer,
the model reads a *state* (a message, a document, retrieved passages, an agent trace, a web page as JSON). It then
answers a typed question about it (`choice`, yes/no `noul`, `score`, and in the basal engine `multi`, `act` and
`tool`; a `tool` question over OpenAI or MCP tool definitions picks the tool, not its arguments). The answer is a
**calibrated probability for each allowed answer**, in one forward pass, without generating text. The answer never
falls outside the options you give, and the probability says how sure the model is.

**What is new against basal-1.5-max.**
- A Gemma 4 base (31B) instead of Bielik (11B).
- Up to **255 options** per question (basal-1.5: 10).
- One option order by default (basal-1.5 averaged two).
- **Images** in the same request (`"media"`, served with `--media on`), zero-shot; no audio.
- Trained on about 640,000 Polish and English typed decisions, with soft targets from basal-1.5.

## Results

Served with the basal engine: basal-omni-1.0-ultra on one H200 and basal-omni-1.0-max on one H100, both with `--mode eager`, bf16, one option order; basal-1.5-max as released (`--mode fast`, both option orders). Same items and the same clients for all three. Measured with basal 2.0; basal 2.1 gives the same top answer on 23,278 of 23,278 served-check rows.

| benchmark | basal-omni-1.0-ultra (31B) | basal-omni-1.0-max (12B) | basal-1.5-max (11B) |
|---|---|---|---|
| Werdykt v1, macro (hidden PL + EN, 10 categories × 2 languages) | **0.879** | 0.823 | 0.773 |
| PL decisions (held-out test of basal-1.0, 7,081 items) | **0.979** | 0.954 | 0.940 |
| PL general (knowledge, exams, reading; 3,079 items) | **0.823** | 0.810 | 0.797 |
| PL-Real (300 real Polish documents) | 0.973 | 0.980 | 0.970 |
| EN external panel (10 public English decision tasks, 3,842 items) | **0.855** | 0.789 | 0.737 |
| Coverage at 1% error (share decided automatically with the shipped threshold) | 56.8%¹ | 47.4% | 62.2% |

- **Against basal-omni-1.0-max:** ahead on PL decisions (+2.5 points, 95% interval +2.0 to +3.1), the English corpus test of 1,479 items (0.751 vs 0.724, +0.6 to +4.7) and the English external panel (+6.9 points over its items; ahead on 9 of 10 tasks: WinoGrande 0.863 vs 0.710, BIG-Bench Hard 0.883 vs 0.809, JudgeBench 0.700 vs 0.600; behind on FinancialPhraseBank, 0.947 vs 0.967). Level on Polish general knowledge (+1.3, 0.0 to +2.7) and PL-Real (−0.7), whose intervals include zero.
- **Against basal-1.5-max:** PL decisions +3.9 points (+3.3 to +4.5), Polish general knowledge +2.6 (+1.1 to +4.1), English corpus test 0.751 vs 0.659, English panel +11.8; level on PL-Real (+0.3).
- ¹ **Coverage:** per question type, at 0.47% observed error (max: 47.4% at 0.47%). A type without a certified threshold is deferred: at the 1% target, score questions go to a person. basal-1.5-max's 62.2% used one threshold pooled over all types; under that rule this model reaches 77.5% at 0.75% error.
- **Werdykt v1:** 0.879 (95% interval 0.871–0.887; Polish 0.881, English 0.877), 5.6 points above basal-omni-1.0-max (0.823) and 10.6 above basal-1.5-max, served with `--deterministic` on one H200 (one request per forward). Ahead of max in all 10 categories; weakest categories: rules (0.550; compute exact thresholds and dates in code), contracts (0.822) and long documents (0.854).

## Quick start

**1. Install** into a fresh [uv](https://docs.astral.sh/uv/) environment:

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[media] @ https://github.com/rkinas/basal/archive/refs/tags/v2.2.0.tar.gz"
```

**2. Start the server.** It prints `basal: model ... ready`:

```bash
basal-serve --model Remek/basal-omni-1.0-ultra --mode eager --media on --port 8000
```

The `eager` mode runs the model in PyTorch, bf16 (about 62 GB of GPU memory for the weights; on one 80 GB GPU, states up to about 10,000 tokens, see Limitations).
For text only, the FP8 checkpoint [basal-omni-1.0-ultra-FP8](https://huggingface.co/Remek/basal-omni-1.0-ultra-FP8)
runs on `--mode vllm`. The `fast` mode (shared prefix, CUDA graphs) comes with a later release of this line.

**3. Ask questions**, exactly as with basal-1.5:

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej, system pokazuje błąd hasła.",
  "questions": {
    "dept":   {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
               "criteria": {"cards": "Reklamacje kart", "online": "Wsparcie bankowości elektronicznej", "loans": "Kredyty"}},
    "urgent": {"type": "noul", "instructions": "Czy klient nie może korzystać z usługi?"}}}'
```

Full API (`multi`, `act`, `tool`, option keys, `max_error`): [API](https://github.com/rkinas/basal#api).

## Images

With `--media on`, a request may carry images with its state. They are placed where the state text goes,
before it, in the order given; the state text may be empty.

```bash
IMG=$(base64 < paragon.jpg | tr -d '\n')
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "", "lang": "pl",
  "media": [{"type": "image", "data": "'"$IMG"'"}],
  "questions": {
    "card": {"type": "noul", "instructions": "Czy zapłacono kartą?"},
    "vat":  {"type": "choice", "instructions": "Jaka jest najwyższa stawka VAT na paragonie?",
             "criteria": {"23": "23%", "8": "8%", "5": "5%", "0": "0% lub zwolniony"}}}}'
```

- `"media"`: a list of `{"type": "image", "data": "<base64>"}` (a `data:` URL works too), any format Pillow reads.
  Audio is refused with HTTP 422 (`this model does not take audio (it takes: image)`): the 31B base has no audio encoder.
- `"detail"`: `"default"` (280 image tokens) or `"high"` (1,120; small print, slower).
- `"lang"`: `"pl"` or `"en"`, the prompt language when the state text does not show it (a state that is only media).
- Limits: 8 items and 20 MB each per request, 50 MP per image.
- Every question of a request sees all of its images. For the steadiest answers, send each question with only the
  images it is about.
- Answers carry `"calibration": "text"`: the temperatures and the certified thresholds were fitted on text; validate
  the thresholds on your own media before automating with them.

On omni-probe-pl, our small Polish probe (each item asked from the image and from its text):

| probe | items | from the image | from the text | untrained Gemma 4 31B, from the image |
|---|---|---|---|---|
| decision from an image of a rendered document | 160 | 0.894 | 0.931 (the document's text) | 0.769 |
| decision about a real photo | 150 | 0.800 | 0.613 (the question alone) | 0.833 |

- Rendered documents are read almost as well as their text (4 points behind) and 12.5 points better than by the untrained base (95% interval +5.6 to +19.4).
- Photos add 19 points over the question alone. The text training did not improve photos: 3.3 points below the untrained base, within the probe's noise (95% interval −9.3 to +2.7).

## Speed

One question per request, one option order, on one H200 (`--mode eager`, bf16, `--media on`):

| measurement | result |
|---|---|
| sequential, 200 requests | 80 ms p50 / 188 ms p95 per question |
| 32 concurrent clients, 1,000 requests | 8.6 decisions/s |

- basal-omni-1.0-max on one H100, same test: 66 ms p50 and 19.0 decisions/s; basal-1.5-max in its `fast` engine: 27.3 ms p50 and 33.4 decisions/s.
- `--orders 2` costs about twice as much per question.
- Long states are prefilled in 2,048-token chunks; the longest state a GPU serves is under Limitations.

## Files

- `model-*.safetensors`, tokenizer, `chat_template.jinja`, `processor_config.json`: the model.
  - Gemma 4 architecture, 31B, 262k vocabulary.
  - Gemma's image encoder is included and unchanged (this release is trained on text); the engine reads images
    through it with `--media on` (`processor_config.json`). The 31B base has no audio encoder.
- `CALIBRATION.json`: per-type temperatures and certified confidence thresholds for 1% / 5% error, fitted on the
  served engine (`--mode eager`, bf16, one option order).
- `CALIBRATION.o2.json`: the same for `--orders 2` (both option orders averaged).
- `basal.json`: prompt format 2 and the readout protocol.

## How it works

The model receives a fixed chat prompt with the state, the question and coded options (A–Z, then two-character codes,
up to 255). The assistant turn is prefilled with `{"answer": "`, and the decision is the softmax over the next-token
logits of the option codes only: one forward pass, no text generation. The server applies the calibrated temperature of
the question type from `CALIBRATION.json`.

**Use the confidence.** `CALIBRATION.json` stores thresholds for a target error of 1% or 5% among accepted decisions.
They were chosen on calibration data before testing; on the test data they accept 56.8% / 94.2% of decisions at 0.47% / 3.8% observed error (at the 1% target score questions have no certified threshold and are deferred). Accept decisions above the threshold
automatically and route the rest to a person.

## Training

- **Base and method:** fine-tuned from [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) with LoRA
  (rank 32) on the language-model layers, merged into the weights. The image encoder was not trained.
- **Data:** about 640,000 typed decisions in Polish and English: rules and deadlines, real documents, long
  documents, retrieval decisions, contracts, routing, ordinal scores, judging answers, agent and tool decisions, and
  "cannot be determined" answers.
- **Distillation:** basal-1.5 provides soft targets on the decisions it answers correctly.
- **Schedule:** 10,000 of the 12,741 planned steps of 64 decisions (640,000 of the basal-1.7 mix's 815,386, without its 1,931 items longer than 6,144 tokens) at peak learning rate 3e-5 with cosine decay; stopped at step 10,000 because the dev panel had stopped improving (0.908, 0.909 and 0.909 at steps 8,000, 9,000 and 10,000), then the LoRA was merged into the weights in float64 and rounded once to bf16.

## Limitations

- **Experimental.** It is the first model of a new base; validate it on your own documents before relying on it.
- **Images are zero-shot.** Gemma 4's encoder reads them, but this model was trained on text decisions only; image
  answers are weaker than text where the image itself is hard (phone photos), see "Images". Their calibration is the
  text calibration. **No audio:** the 31B base has no audio encoder.
- **Size and speed:** 31B parameters, about 62 GB in bf16, about 2× slower than basal-omni-1.0-max (see Speed). The longest state it answered with one option order: about 10,000 tokens with the memory capped at an 80 GB H100's (8,000 with `--orders 2`), and about 32,000 tokens on one 141 GB H200. A longer state fails with an out-of-memory error (HTTP 422), and the server keeps serving.
- **Exact arithmetic and thresholds** are a weak spot of one-pass decisions: compute them in code.
- **Option order:** one order is asked by default; `--orders 2` reduces sensitivity to option order at twice the cost.
- **`tool` with no fitting tool:** ultra usually answers "no suitable tool" (4 of 6 out-of-scope requests), but is sometimes unsure between that and "ask the user" (1 of 6 near 0.5).
- **Review:** decisions with serious consequences for people should be reviewed by a person.

## Citation

```bibtex
@software{kinas2026basal20,
  title   = {basal-2.0: typed-decision models and inference engine for Polish and English},
  author  = {Kinas, Remigiusz},
  year    = {2026},
  version = {2.0.0},
  url     = {https://github.com/rkinas/basal}
}

@techreport{kinas2026basal,
  title       = {basal-1.0: Reliable, Highly Optimized Typed Decisions for Polish},
  author      = {Kinas, Remigiusz},
  institution = {ai5},
  year        = {2026},
  type        = {Technical report},
  doi         = {10.5281/zenodo.23022986},
  url         = {https://doi.org/10.5281/zenodo.23022986}
}
```

## License and attribution

Apache-2.0. Fine-tuned from [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) (Apache-2.0).
