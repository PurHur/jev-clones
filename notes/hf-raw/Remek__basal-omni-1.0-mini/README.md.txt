---
license: apache-2.0
language:
- pl
- en
base_model: google/gemma-4-E4B-it
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

# basal-omni-1.0-mini (experimental)

[![GitHub](https://img.shields.io/badge/GitHub-rkinas%2Fbasal%20%E2%80%94%20inference%20engine-black?logo=github)](https://github.com/rkinas/basal)
[![Website](https://img.shields.io/badge/web-basal.si5.pl-0a7d5a)](https://basal.si5.pl/)

**basal-omni-1.0-mini (Gemma 4 E4B)** is the small model of **basal-omni**, basal's line on Google Gemma 4 (with
[basal-omni-1.0-max](https://huggingface.co/Remek/basal-omni-1.0-max), 12B). It is an **experimental** release: it is
trained on text decisions only, and it also answers the same questions about
**images and audio**, read by Gemma 4's own encoders without media training (zero-shot; media training comes in the
next releases). It answers the same typed questions as basal-1.5, over the same API.

**What it is.** Inspired by *System 1* (fast, intuitive) decision models such as Jev. Instead of writing an answer,
the model reads a *state* (a message, a document, retrieved passages, an agent trace, a web page as JSON). It then
answers a typed question about it (`choice`, yes/no `noul`, `score`, and in the basal engine `multi`, `act` and
`tool`; a `tool` question over OpenAI or MCP tool definitions picks the tool, not its arguments). The answer is a
**calibrated probability for each allowed answer**, in one forward pass, without generating text. The answer never
falls outside the options you give, and the probability says how sure the model is.

**What is new against basal-1.5 (4.5B).**
- A Gemma 4 base, 4B (Gemma 4 E4B), instead of Bielik (4.5B).
- Up to **255 options** per question (basal-1.5: 10).
- One option order by default (basal-1.5 averaged two).
- **Images and audio** in the same request (`"media"`, served with `--media on`), zero-shot.
- Trained on two stages of about 815,000 and 518,000 Polish and English typed decisions, with soft targets from basal-1.5.

## Results

Served with the basal engine on one H100: basal-omni-1.0-mini with `--mode eager`, bf16, one option order; basal-1.5 (4.5B) as released (`--mode fast`, both option orders). Same items and the same clients for both. Measured with basal 2.0; basal 2.1 gives the same top answer on 23,278 of 23,278 served-check rows.

| benchmark | basal-omni-1.0-mini (E4B) | basal-1.5 (4.5B) |
|---|---|---|
| Werdykt v1, macro (hidden PL + EN, 10 categories × 2 languages) | **0.769** | 0.721 |
| PL decisions (held-out test of basal-1.0, 7,081 items) | **0.932** | 0.922 |
| PL general (knowledge, exams, reading; 3,079 items) | 0.751 | 0.746 |
| PL-Real (300 real Polish documents) | 0.950 | 0.963 |
| EN external panel (10 public English decision tasks, 3,842 items) | **0.740** | 0.688 |
| Coverage at 1% error (share decided automatically with the shipped threshold) | 42.0%¹ | 51.0% |

- **Werdykt v1:** 0.769 (95% interval 0.758–0.778; Polish 0.775, English 0.763), 4.8 points above basal-1.5 (4.5B) and level with basal-1.5-max (11B, 0.772), served with `--deterministic` (one request per forward; no difference from the default mode on our tests). Weakest categories: rules (0.352; compute exact thresholds and dates in code), contracts (0.664) and long documents (0.686).
- **English:** it is ahead on the English external panel (+5.2 points; BIG-Bench Hard 0.726 vs 0.556, TabFact 0.780 vs 0.647, Circa 0.750 vs 0.633, RAGTruth 0.760 vs 0.653; behind on JudgeBench, 0.537 vs 0.567), and on the English corpus test of 1,479 items (0.675 vs 0.643).
- **Polish:** it is ahead on the PL decisions test (+1.0 points, 95% interval +0.3 to +1.6), and level on Polish general knowledge (+0.4) and PL-Real (−1.3), whose intervals include zero.
- **Against basal-1.5-max (11B):** level on the English panel (+0.3 points); behind on PL decisions (−0.8) and Polish general knowledge (−4.6).
- ¹ **Coverage:** per question type, at 0.70% observed error. A type without a certified threshold is deferred: at the 1% target, score questions go to a person. basal-1.5's 51.0% used one threshold pooled over all types; under that rule this model reaches 54.5% at 0.82% error.

## Quick start

**1. Install** into a fresh [uv](https://docs.astral.sh/uv/) environment:

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[media] @ https://github.com/rkinas/basal/archive/refs/tags/v2.2.0.tar.gz"
```

**2. Start the server.** It prints `basal: model ... ready`:

```bash
basal-serve --model Remek/basal-omni-1.0-mini --mode eager --media on --port 8000
```

The `eager` mode runs the model in PyTorch, bf16 (about 16 GB of GPU memory), with text, images and audio. For text
only, basal 2.2 also serves it through vLLM (`--mode vllm`; FP8 checkpoint:
[basal-omni-1.0-mini-FP8](https://huggingface.co/Remek/basal-omni-1.0-mini-FP8)), llama.cpp and Ollama
([basal-omni-1.0-mini-GGUF](https://huggingface.co/Remek/basal-omni-1.0-mini-GGUF)) and MLX on Apple Silicon
(`--mode mlx` or `mlx-q8`, or the 8-bit port
[basal-omni-1.0-mini-MLX](https://huggingface.co/Remek/basal-omni-1.0-mini-MLX)); details:
[OMNI.md](https://github.com/rkinas/basal/blob/v2.2.0/docs/OMNI.md). The `fast` mode (shared prefix, CUDA graphs)
comes with a later release of this line.

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
  document image and a spoken request), adding the other medium changed 5–15% of
  the answers. For the steadiest answers, send each question with only the media it is about.
- Answers carry `"calibration": "text"`: the temperatures and the certified thresholds were fitted on text; validate
  the thresholds on your own media before automating with them.

On omni-probe-pl, our small Polish probe (each item asked from the media and from its text):

| probe | items | from the media | from the text | untrained Gemma 4 E4B, from the media |
|---|---|---|---|---|
| decision from an image of a rendered document | 160 | 0.725 | 0.831 (the document's text) | 0.500 |
| decision about a real photo | 150 | 0.667 | 0.507 (the question alone) | 0.640 |
| intent of a spoken request (MINDS-14) | 140 | 0.921 | 0.943 (transcript) | 0.900 |
| topic of a spoken passage | 100 | 0.870 | 0.900 (transcript) | 0.860 |
| spoken decision | 45 | 0.844 | 0.956 (transcript) | 0.600 |

- Rendered documents are read 11 points behind their text (the untrained model: 0.500), and photos add 16 points over the question alone.
- Speech is read well: intent and topic 2–3 points behind the transcript, spoken decisions 11 points behind (basal-omni-1.0-max, 12B: 22–42 points behind).

## Speed

One question per request, one option order, on one H100 (`--mode eager`, bf16, `--media on`):

| measurement | result |
|---|---|
| sequential, 200 requests | 61 ms p50 / 80 ms p95 per question |
| 32 concurrent clients, 1,000 requests | 35.2 decisions/s |

- basal-omni-1.0-max (12B) on the same test: 66 ms p50 and 19.0 decisions/s.
- On Werdykt's 5,000 items (longer states, 16 concurrent clients): 8.7 decisions/s, 67 ms p50 / 0.68 s p95 one request at a time (basal-1.5 in the basal engine's `fast` mode: 8.0 decisions/s).
- `--orders 2` costs about twice as much per question.
- Long states are prefilled in 2,048-token chunks.

## Files

- `model-*.safetensors`, tokenizer, `chat_template.jinja`, `processor_config.json`: the model.
  - Gemma 4 architecture (E4B: KV-shared layers, per-layer embeddings), 262k vocabulary.
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
They were chosen on calibration data before testing; on the test data they accept 42.0% / 77.4% of decisions at 0.70% / 3.7% observed error (at the 1% target score questions have no certified threshold and are deferred). Accept decisions above the threshold
automatically and route the rest to a person.

## Training

- **Base and method:** fine-tuned from [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) with LoRA
  (rank 32) on the language-model layers. The vision and audio parts were not trained.
- **Data:** two stages of about 815,000 and 518,000 typed decisions in Polish and English: rules and deadlines, real documents, long
  documents, retrieval decisions, contracts, routing, ordinal scores, judging answers, agent and tool decisions, and
  "cannot be determined" answers.
- **Distillation:** basal-1.5 provides soft targets on the decisions it answers correctly.
- **Schedule:** two stages at learning rate 3e-5: one epoch of 815,390 decisions of the basal-1.7 mix (12,741 steps of 64; without its 1,927 items longer than 6,144 tokens), then one epoch of 517,732 decisions of the basal-1.8 mix (8,090 steps of 64; without its 5,376 longer items).

## Limitations

- **Experimental.** It is the first model of a new base; validate it on your own documents before relying on it.
- **Images and audio are zero-shot.** Gemma 4's encoders read them, but this model was trained on text decisions only;
  media answers are weaker than text where the media itself is hard (phone photos, speech), see "Images and audio".
  Their calibration is the text calibration.
- **Speed:** images and audio need the `eager` engine; text also runs on vLLM, llama.cpp and MLX (see Quick start).
  The `fast` mode comes later.
- **Exact arithmetic and thresholds** are a weak spot of one-pass decisions: compute them in code.
- **Option order:** one order is asked by default; `--orders 2` reduces sensitivity to option order at twice the cost.
- **`tool` with no fitting tool:** mini often answers "ask the user" instead of "no suitable tool" (4 of 6
  out-of-scope requests in our test); do not rely on it to refuse requests no tool covers.
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

Apache-2.0. Fine-tuned from [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) (Apache-2.0).
