---
license: apache-2.0
language:
- pl
- en
base_model: google/gemma-4-12B-it
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

# basal-omni-1.0-max (experimental)

[![GitHub](https://img.shields.io/badge/GitHub-rkinas%2Fbasal%20%E2%80%94%20inference%20engine-black?logo=github)](https://github.com/rkinas/basal)
[![Website](https://img.shields.io/badge/web-basal.si5.pl-0a7d5a)](https://basal.si5.pl/)

**basal-omni-1.0-max (12B)** is the first model of **basal-omni**, basal's line on Google Gemma 4. It is an
**experimental** release: it is trained on text decisions only, and it also answers the same questions about
**images and audio**, read by Gemma 4's own encoders without media training (zero-shot; media training comes in the
next releases). It answers the same typed questions as basal-1.5, over the same API.

**What it is.** Inspired by *System 1* (fast, intuitive) decision models such as Jev. Instead of writing an answer,
the model reads a *state* (a message, a document, retrieved passages, an agent trace, a web page as JSON). It then
answers a typed question about it (`choice`, yes/no `noul`, `score`, and in the basal engine `multi`, `act` and
`tool`; a `tool` question over OpenAI or MCP tool definitions picks the tool, not its arguments). The answer is a
**calibrated probability for each allowed answer**, in one forward pass, without generating text. The answer never
falls outside the options you give, and the probability says how sure the model is.

**What is new against basal-1.5-max.**
- A Gemma 4 base (12B) instead of Bielik (11B).
- Up to **255 options** per question (basal-1.5: 10).
- One option order by default (basal-1.5 averaged two).
- **Images and audio** in the same request (`"media"`, served with `--media on`), zero-shot.
- Trained on about 815,000 Polish and English typed decisions, with soft targets from basal-1.5.

## Results

Served with the basal engine on one H100: basal-omni-1.0-max with `--mode eager`, bf16, one option order; basal-1.5-max as released (`--mode fast`, both option orders). Same items and the same clients for both. Measured with basal 2.0; basal 2.1 gives the same top answer on 23,276 of 23,278 served-check rows.

| benchmark | basal-omni-1.0-max (12B) | basal-1.5-max (11B) |
|---|---|---|
| Werdykt v1, macro (hidden PL + EN, 10 categories × 2 languages) | **0.823** | 0.773 |
| PL decisions (held-out test of basal-1.0, 7,081 items) | **0.954** | 0.940 |
| PL general (knowledge, exams, reading; 3,079 items) | 0.810 | 0.797 |
| PL-Real (300 real Polish documents) | 0.980 | 0.970 |
| EN external panel (10 public English decision tasks, 3,842 items) | **0.789** | 0.737 |
| Coverage at 1% error (share decided automatically with the shipped threshold) | 47.4%¹ | 62.2% |

- **Werdykt v1:** 0.823 (95% interval 0.813–0.832; Polish 0.824, English 0.821), 5.0 points above basal-1.5-max, served with `--deterministic` (one request per forward; at most 0.1 point from the default mode on our tests). Weakest categories: rules (0.418; compute exact thresholds and dates in code), long documents (0.742) and contracts (0.762).
- **English:** it is ahead on the English external panel (+5.2 points; BIG-Bench Hard 0.809 vs 0.644, TabFact 0.873 vs 0.750; behind on ContractNLI, 0.720 vs 0.740), and on the English corpus test of 1,479 items (0.724 vs 0.659).
- **Polish:** it is ahead on the PL decisions test (+1.4 points, 95% interval +0.8 to +2.0), and level on Polish general knowledge (+1.3) and PL-Real (+1.0), whose intervals include zero.
- ¹ **Coverage:** per question type, at 0.47% observed error. A type without a certified threshold is deferred: at the 1% target, score questions go to a person. basal-1.5-max's 62.2% used one threshold pooled over all types; under that rule this model reaches 65.1% at 1.02% error.

## Quick start

**1. Install** into a fresh [uv](https://docs.astral.sh/uv/) environment:

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[media] @ https://github.com/rkinas/basal/archive/refs/tags/v2.2.0.tar.gz"
```

**2. Start the server.** It prints `basal: model ... ready`:

```bash
basal-serve --model Remek/basal-omni-1.0-max --mode eager --media on --port 8000
```

The `eager` mode runs the model in PyTorch, bf16 (about 24 GB of GPU memory), with text, images and audio. For text
only, basal 2.2 also serves it through vLLM (`--mode vllm`; FP8 checkpoint:
[basal-omni-1.0-max-FP8](https://huggingface.co/Remek/basal-omni-1.0-max-FP8)), llama.cpp and Ollama
([basal-omni-1.0-max-GGUF](https://huggingface.co/Remek/basal-omni-1.0-max-GGUF)) and MLX on Apple Silicon
(`--mode mlx-q8`, or the 8-bit port [basal-omni-1.0-max-MLX](https://huggingface.co/Remek/basal-omni-1.0-max-MLX));
details: [OMNI.md](https://github.com/rkinas/basal/blob/v2.2.0/docs/OMNI.md). The `fast` mode (shared prefix, CUDA
graphs) comes with a later release of this line.

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

## Tasks (experimental, basal 2.1)

basal 2.1 adds four task types on the same `POST /v1/basal` API:
- **`classify`:** labels with a rubric each.
- **`verify`:** statements about the text; each one is supported, contradicted or not stated.
- **`compress`:** keeps the sentences, lines or paragraphs that a question needs.
- **`extract`:** fills the fields of a JSON schema, copied exactly from the text with character offsets.

The model does not write text. Each task is expanded into the model's own decisions over one shared input, so every
answer comes with calibrated probabilities. *Experimental:* the API may change in a minor release. Details:
[TASKS.md](https://github.com/rkinas/basal/blob/v2.1.0/docs/TASKS.md).

With the server from the quick start:

```bash
curl -s localhost:8000/v1/basal -H 'content-type: application/json' -d '{
  "state": "I was charged twice for my card payment yesterday.",
  "questions": {"intent": {"type": "classify", "none": true, "labels": [
    {"name": "card_payment_duplicate", "rubric": "a card payment was taken more than once"},
    {"name": "card_lost", "rubric": "the card is lost or stolen"},
    {"name": "refund_status", "rubric": "asks about a refund in progress"}]}}}'
```

The answer (`answers.intent`, rounded to 4 places):

```json
{"type": "classify", "label": "card_payment_duplicate", "top_label": "card_payment_duplicate",
 "scores": {"card_payment_duplicate": 0.9996, "card_lost": 0.0002, "refund_status": 0.0000, "none": 0.0002},
 "confidence": 0.9996, "automated": {"0.01": null, "0.05": null}, "experimental": true}
```

basal-omni-1.0-max was not trained on these task formats. Measured with basal-serve 2.1 (eager, bf16, one H100):

| task | set | n | basal-omni-1.0-max |
|---|---|---|---|
| classify | CLINC150 "plus" test: accuracy (151 labels, including out-of-scope) | 1,000 | 85.4% |
| classify | BANKING77 test: accuracy (77 labels) | 1,000 | 74.6% |
| verify | VitaminC (3-way): accuracy | 1,000 | 80.3% |
| verify | RAGTruth, response vs source (binary): accuracy | 1,000 | 80.3% |
| compress | QASPER paragraphs, keep 30% of the characters: needed paragraphs kept | 248 | 90.1% |
| extract | synthetic PL / EN invoices, 12 fields: fields correct | 400 | 90.1% |
| extract | documents in layouts never used in training: fields correct | 400 | 84.8% |

- **`automated`:** this model's calibration file has no task thresholds, so `automated` is always `null` for task
  answers. Fit `task_thresholds` on your own labelled data (TASKS.md).
- **Limits:** currency is the weakest invoice field (62%). `extract` does not yet return arrays of strings or
  objects (line items), or text fields read from images.

## Images and audio

With `--media on`, a request may carry images and audio with its state. They are placed where the state text goes,
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

- `"media"`: a list of `{"type": "image" | "audio", "data": "<base64>"}` (a `data:` URL works too). Images: any format
  Pillow reads. Audio: WAV, FLAC, OGG or MP3, any sample rate (mixed to mono, resampled to 16 kHz), at most 30 s a clip.
- `"detail"`: `"default"` (280 image tokens) or `"high"` (1,120; small print, slower).
- `"lang"`: `"pl"` or `"en"`, the prompt language when the state text does not show it (a state that is only media).
- Limits: 8 items and 20 MB each per request, 50 MP per image.
- **Mixed media:** one request may mix images and audio, and every question sees all of them. In our test (20 pairs of a
  document image and a spoken request), adding the other medium changed 10–25% of
  the answers. For the steadiest answers, send each question with only the media it is about.
- Answers carry `"calibration": "text"`: the temperatures and the certified thresholds were fitted on text; validate
  the thresholds on your own media before automating with them.

On omni-probe-pl, our small Polish probe (each item asked from the media and from its text):

| probe | items | from the media | from the text | untrained Gemma 4 12B, from the media |
|---|---|---|---|---|
| decision from an image of a rendered document | 160 | 0.838 | 0.900 (the document's text) | 0.675 |
| decision about a real photo | 150 | 0.660 | 0.527 (the question alone) | 0.680 |
| intent of a spoken request (MINDS-14) | 140 | 0.714 | 0.929 (transcript) | 0.700 |
| topic of a spoken passage | 100 | 0.510 | 0.930 (transcript) | 0.550 |
| spoken decision | 45 | 0.556 | 0.956 (transcript) | 0.578 |

- Rendered documents are read almost as well as their text (6 points behind), and photos add 13 points over the question alone.
- Speech scores 22–42 points below its transcript and stays at the untrained base's level: send a transcript when you have one.

## Speed

One question per request, one option order, on one H100 (`--mode eager`, bf16, `--media on`):

| measurement | result |
|---|---|
| sequential, 200 requests | 66 ms p50 / 95 ms p95 per question |
| 32 concurrent clients, 1,000 requests | 19.0 decisions/s |

- basal-1.5-max in its `fast` engine: 27.3 ms p50 and 33.4 decisions/s on the same test.
- On Werdykt's 5,000 items (longer states, 16 concurrent clients): 4.2 decisions/s, 76 ms p50 / 1.36 s p95 (basal-1.5-max in its `fast` engine: 4.6 decisions/s).
- `--orders 2` costs about twice as much per question.
- Long states (up to ~30k tokens measured) are prefilled in 2,048-token chunks.

## Files

- `model-*.safetensors`, tokenizer, `chat_template.jinja`, `processor_config.json`: the model.
  - Gemma 4 architecture, 12B, 262k vocabulary.
  - Gemma's image and audio encoders are included and unchanged (this release is trained on text); the engine reads
    media through them with `--media on` (`processor_config.json`).
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
They were chosen on calibration data before testing; on the test data they accept 47.4% / 82.9% of decisions at 0.47% / 3.9% observed error (at the 1% target score questions have no certified threshold and are deferred). Accept decisions above the threshold
automatically and route the rest to a person.

## Training

- **Base and method:** fine-tuned from [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) with LoRA
  (rank 32) on the language-model layers. The vision and audio parts were not trained.
- **Data:** about 815,000 typed decisions in Polish and English: rules and deadlines, real documents, long
  documents, retrieval decisions, contracts, routing, ordinal scores, judging answers, agent and tool decisions, and
  "cannot be determined" answers.
- **Distillation:** basal-1.5 provides soft targets on the decisions it answers correctly.
- **Schedule:** one epoch of 815,386 decisions (12,741 steps of 64; the basal-1.7 mix without its 1,931 items longer than 6,144 tokens) at learning rate 3e-5.

## Limitations

- **Experimental.** It is the first model of a new base; validate it on your own documents before relying on it.
- **Images and audio are zero-shot.** Gemma 4's encoders read them, but this model was trained on text decisions only;
  media answers are weaker than text where the media itself is hard (phone photos, speech), see "Images and audio".
  Their calibration is the text calibration.
- **Speed:** in the `eager` engine (needed for images and audio) it is slower than basal-1.5-max in the `fast` mode;
  text also runs on vLLM, llama.cpp and MLX (see Quick start).
- **Exact arithmetic and thresholds** are a weak spot of one-pass decisions: compute them in code.
- **Option order:** one order is asked by default; `--orders 2` reduces sensitivity to option order at twice the cost.
- **`tool` with no fitting tool:** max usually answers "no suitable tool", but is often unsure between that and
  "ask the user" (2 of 6 out-of-scope requests near 0.5).
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

Apache-2.0. Fine-tuned from [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) (Apache-2.0).
