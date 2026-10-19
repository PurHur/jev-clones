---
license: apache-2.0
base_model: h2oai/h2o-lightning-4b
library_name: llama-cpp
tags:
  - jev
  - typed-decisions
  - decision-model
  - gguf
  - llama-cpp
---

![H2O Lightning: Decision Models](assets/banner.png)

# H2O Lightning 4B GGUF

GGUF files of [H2O Lightning 4B](https://huggingface.co/h2oai/h2o-lightning-4b) for llama.cpp. For the model description, training and general benchmarks, see the model page.

The model answers typed decisions with probabilities: named `choice` options, true/false `noul`, and ordered `score` levels. There are two ways to run it, each with its own files:

- **llama.cpp with [PR #30361](https://github.com/ggml-org/llama.cpp/pull/30361)** (the default files). `llama-server` answers decisions itself on `POST /v1/systemone`. Until the pull request is merged, build `llama-server` from its branch ([below](#llamacpp-with-pr-30361)).
- **Standard llama.cpp** (the `stock-` files). The included Python shim serves the decision API on top of an unmodified `llama-server`. The [one-command launcher](#run-with-one-command) builds pinned stock llama.cpp and starts both.

## Which file

| Precision | Size (GB) | Public questions correct (of 231) | Accuracy change vs BF16 (95% CI) | Same top answer as BF16 | Use |
|---|---:|---:|---:|---:|---|
| BF16 | 8.42 | 205 | 0.0 pts (−0.2 to +0.2) | 99.5% | Reference |
| **Q8_0** | **4.48** | **204** | **0.0 pts (−0.3 to +0.3)** | **98.6%** | **Recommended** |
| Q5_K_M | 3.07 | 204 | −0.2 pts (−0.7 to +0.3) | 96.7% | Same accuracy, smaller |
| Q4_K_M | 2.71 | 204 | −0.2 pts (−0.9 to +0.5) | 94.6% | Same accuracy, smallest such file |
| IQ3_XXS, mixed | 2.16 | 206 | −2.5 pts (−3.5 to −1.5) | 88.6% | Experimental; loses some accuracy |
| IQ3_XXS | 1.90 | 203 | −3.5 pts (−4.6 to −2.6) | 87.3% | Experimental; loses some accuracy |
| IQ2_XXS | 1.54 | 189 | −10.5 pts (−11.7 to −9.1) | 76.1% | Experimental; loses accuracy |
| IQ1_M | 1.43 | 158 | −21.1 pts (−22.8 to −19.4) | 61.0% | Experimental; loses accuracy |

Accuracy and answer agreement are measured on 4,074 held-out decision questions, each GGUF run with standard llama.cpp and the shim, against the BF16 model served with vLLM. The BF16 row is the BF16 GGUF run in llama.cpp; its 99.5% agreement shows how closely llama.cpp itself matches vLLM. Q5_K_M and Q4_K_M match BF16 accuracy within the measurement error: they change about 3% and 5% of answers (roughly 1 in 30 and 1 in 18), about as often from wrong to right as from right to wrong. Just over half of those changes (60% and 53%) are on questions where BF16's top answer led by less than 0.2, and about a third are on questions BF16 answered confidently (a lead of 0.6 or more). IQ2_XXS and IQ1_M lose accuracy, and most of their changes are on questions BF16 answers confidently; for the IQ3_XXS files about 40% are. The Q5_K_M and Q4_K_M files with BF16 embeddings (`-BF16embed`) measured no better than the plain files. The mixed IQ3_XXS file keeps a few small projections at 6 to 8 bits: in the recurrent layers, the decay and update-strength projections (8-bit) and the output gate and output projections (6-bit); in the attention layers, the output projection (6-bit). Everything else stays at IQ3_XXS's default types; it is 1.1 points more accurate than plain IQ3_XXS (95% CI +0.4 to +1.7). The two IQ3_XXS files were measured later than the other rows, on the same questions against the same reference. On the 231 public questions, differences of a few answers are within noise; the mixed IQ3_XXS file's 206 does not mean it is more accurate than BF16.

## Files

| Precision | llama.cpp with PR #30361 | Standard llama.cpp + shim |
|---|---|---|
| BF16 | `h2o-lightning-4b-BF16.gguf` (`-hf` tag `:BF16`) | `h2o-lightning-4b-stock-BF16.gguf` (launcher default) |
| Q8_0 | `h2o-lightning-4b-Q8_0.gguf` (`:Q8_0`, also the default with no tag) | `h2o-lightning-4b-stock-Q8_0.gguf` (launcher `--variant Q8_0`) |
| Q5_K_M | — | `h2o-lightning-4b-stock-experimental-Q5KM.gguf` (3.07 GB), `…-Q5KM-BF16embed.gguf` (3.82 GB) |
| Q4_K_M | — | `h2o-lightning-4b-stock-experimental-Q4KM.gguf` (2.71 GB), `…-Q4KM-BF16embed.gguf` (3.46 GB) |
| IQ3_XXS, mixed | — | `h2o-lightning-4b-stock-experimental-mixed-IQ3XXS.gguf` (2.16 GB) |
| IQ3_XXS | — | `h2o-lightning-4b-stock-experimental-IQ3XXS.gguf` (1.90 GB) |
| IQ2_XXS | — | `h2o-lightning-4b-stock-experimental-IQ2XXS.gguf` |
| IQ1_M | — | `h2o-lightning-4b-stock-experimental-IQ1M.gguf` |
| Vision projector (0.67 GB) | `mmproj-lightning-f16.gguf` (loaded by `-hf`) | same file (launcher `--experimental-images`) |

The files below Q8_0 are for standard llama.cpp with the shim; download them by name. Their names keep them from being picked by a plain `-hf` tag. The Q5_K_M and Q4_K_M names still contain `experimental` from before they were measured; only the IQ3_XXS, IQ2_XXS and IQ1_M files lose accuracy. The one-command launcher offers BF16 and Q8_0 only.

The PR #30361 files have the same weights as the `stock-` files of the same precision, tensor for tensor, plus decision metadata (the decision type `h2o-lightning`, the option labels, the per-type temperatures, the temperature for requests with images, the prompt template and the Qwen2 pre-tokenizer setting). Standard llama.cpp refuses to load them, because it does not know the decision type; with standard llama.cpp, use a `stock-` file. Ordinary chat with a `stock-` file does not implement the decision API; the shim does. Both use decision-merged weights; they cannot disable the decision adapter to recover the original base model.

Storage sizes exclude runtime memory. Sizes and SHA-256 hashes of all files are in [artifact_manifest.json](artifact_manifest.json). Until October 12, 2026, `h2o-lightning-4b-BF16.gguf` and `h2o-lightning-4b-Q8_0.gguf` in this repository were the files now named `stock-`; if your copy's SHA-256 does not match artifact_manifest.json, download it again.

## llama.cpp with PR #30361

### Build llama-server (until the pull request is merged)

Git, CMake and a C++ compiler; CUDA also needs a CUDA toolkit; `-hf` downloads need OpenSSL headers (for example `libssl-dev`).

```bash
git clone --branch h2o-lightning-decision --depth 1 https://github.com/pseudotensor/llama.cpp llama.cpp-h2o-lightning
cd llama.cpp-h2o-lightning

# CPU
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j --target llama-server

# or CUDA
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j --target llama-server
```

This is the pull request's branch. Once the pull request is merged as it stands, the first llama.cpp release that includes it, and any later one, works without building.

### Run

```bash
./build/bin/llama-server -hf h2oai/h2o-lightning-4b-gguf:Q8_0 --alias h2oai/h2o-lightning-4b \
    -c 40960 -np 1 --no-kv-unified --image-max-tokens 1024 --port 8080

# or with downloaded files
hf download h2oai/h2o-lightning-4b-gguf h2o-lightning-4b-Q8_0.gguf mmproj-lightning-f16.gguf --local-dir .
./build/bin/llama-server -m h2o-lightning-4b-Q8_0.gguf --mmproj mmproj-lightning-f16.gguf --alias h2oai/h2o-lightning-4b \
    -c 40960 -np 1 --no-kv-unified --image-max-tokens 1024 --port 8080
```

These are the settings the response and parity numbers in this section were measured with. `-c 40960` leaves room for a 32,000-token state plus the question; `-np 1 --no-kv-unified` gives one slot the whole context (with more slots, the questions of one request share the state's prompt and their probabilities can differ slightly); `--image-max-tokens 1024` is the image token budget the model was validated with; `--alias` sets the `model` field of the response. Use `:BF16` for the reference precision. llama.cpp offloads to a GPU when it finds one; `-ngl` controls it. `-hf` also loads the vision projector; pass `--no-mmproj` for text only.

### Request

```bash
curl http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"customer": "I was charged twice for order 1182 and nobody has replied.", "plan": "pro"},
  "questions": {
    "route":   {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "payments and refunds", "shipping": "delivery problems", "technical": "app or website errors"}},
    "angry":   {"type": "noul",   "instructions": "Is the customer upset?"},
    "urgency": {"type": "score",  "instructions": "How urgent is this?", "criteria": ["can wait", "this week", "today", "right now"]}}}'
```

Response, measured with the Q8_0 file on an RTX A6000 (numbers rounded to 4 digits here; a CPU build gives slightly different last digits, for example a `score` of 1.8339):

```json
{
  "model": "h2oai/h2o-lightning-4b",
  "answers": {
    "route":   {"type": "choice", "choice": "billing",
                "probabilities": {"billing": 0.9657, "shipping": 0.0170, "technical": 0.0173}, "confidence": 0.9486},
    "angry":   {"type": "noul", "noul": 0.7888},
    "urgency": {"type": "score", "score": 1.8261,
                "legend": {"0": "can wait", "1": "this week", "2": "today", "3": "right now"},
                "probabilities": {"0": 0.0194, "1": 0.3811, "2": 0.3535, "3": 0.2460}, "confidence": 0.1351}
  },
  "usage": {"input_tokens": 321, "output_tokens": 0}
}
```

Images go in a top-level `images` array of data URLs (`"images": ["data:image/png;base64,..."]`).

### Yes/no floor

`noul` probabilities are the raw calibrated values. The shim's default applies a floor of 0.801: a `noul` probability between 0.199 and 0.801 is moved to the nearer of the two, so the answer does not change; the leaderboard results of this model were produced with it. To get the shim's numbers, apply it to the returned value: `if max(p, 1 - p) < 0.801: p = 0.801 if p >= 0.5 else 0.199` (in the example above, `angry` becomes `0.801`). A request option for it, `"noul_floor": 0.801`, is proposed separately on the [`decision-noul-floor`](https://github.com/pseudotensor/llama.cpp/tree/decision-noul-floor) branch; the PR #30361 build does not have it and ignores the field.

### Parity with the shim

On the 231 public JevBench text questions, with the same prompts read on the same server, llama-server's endpoint and the shim agree on 231/231 answers for BF16 and Q8_0, with probabilities within 2.4e-6, without the floor and with it (the same arithmetic as the formula above); on 30 synthetic image questions (receipts, dashboards, charts), 30/30 within 1.7e-7. Run as each is deployed with the settings above, both score 205/231 (BF16) and 204/231 (Q8_0), answers agree on 231/231 and probabilities stay within 0.037 (Q8_0) and 0.005 (BF16): differences from how llama-server evaluates a prompt on the shim's completion path. Median / p95 latency on an RTX A6000 (a different GPU from the shim measurements further down): 43 / 419 ms (Q8_0) natively, 184 / 699 ms through the shim. The end-to-end scores and the latency were measured on the same weights with an earlier build of the branch.

### Differences from the shim

- Text: the shim NFC-normalizes text; llama.cpp tokenizes the text as sent. Send NFC-normalized text (most text already is). Integers beyond 64 bits in a JSON state are written as floats.
- Response: llama.cpp's shape, `model`, `answers` and `usage` (`input_tokens`, `output_tokens`). The shim's extra fields (`probability_source`, `effort_used`, `usage.latency_ms`, `threshold`/`decision` for a `noul` threshold) are not returned.
- Floor: not applied by llama-server; apply it to the returned value (see above).
- Images: send them as data URLs in the top-level `images` array (or as `image_url` parts of chat messages in `state`). Images elsewhere in `state`, http(s) URLs, multipart uploads and the shim's other image fields (`image`, `image_url`, `image_urls`, `image_data`, `image_base64`) are not read as images; there is no 4-image cap (llama.cpp takes up to 8) and no `[image N]` placeholder in the state. Image decisions match the shim on the 30-question development fixture only; broader image parity is not established, as for the shim.

## Standard llama.cpp with the shim

### Run with one command

Requires Python 3.12, Git, CMake and a C++ compiler. Install `huggingface_hub` with `python -m pip install huggingface_hub`. CUDA additionally requires a compatible CUDA toolkit. No Hugging Face login is required.

```bash
python -c 'import hashlib,runpy; from huggingface_hub import hf_hub_download; p=hf_hub_download("h2oai/h2o-lightning-4b-gguf","bootstrap.py",revision="6867bcf29370f6590d25d407d8e6c1e74adb88ae",token=False); assert hashlib.sha256(open(p,"rb").read()).hexdigest()=="94c659982add42534409a79541ceeb4b6a0255e09670e5373afb814b7bdca738"; runpy.run_path(p,run_name="__main__")' --backend cpu
```

BF16 is the default. Add `--variant Q8_0` for the smaller model. Use `--backend cuda --gpu 0` for CUDA. Add `--experimental-images` to download the projector for image decisions. `--prepare-only` downloads and verifies without building or starting a server.

The launcher downloads pinned weights and code (the same bytes as `h2o-lightning-4b-stock-BF16.gguf` / `h2o-lightning-4b-stock-Q8_0.gguf`, checked by SHA-256), builds stock llama.cpp at `10a60cf303566e10d6a7a2774c17d2085503d87b`, checks provenance and 24 tokenizer probes, then starts the stock backend and the shim. Loading a `stock-` file directly into `llama-server` (for example with `-hf h2oai/h2o-lightning-4b-gguf:stock-Q8_0`) gives ordinary chat, not the decision API; use the launcher for decisions. The first CUDA build took 35.79 minutes on the tested machine with cached weights; subsequent runs reuse it. Ctrl-C stops its owned children. Logs remain in `lightning-llamacpp-logs/`.

In another terminal:

```bash
curl http://127.0.0.1:8761/v1/systemone -H 'Content-Type: application/json' -d '{"state":{"status":"paid"},"questions":{"status":{"type":"choice","instructions":"What status is recorded?","criteria":{"a":"paid","b":"unpaid"}}}}'
```

The backend has one 40,960-token slot; concurrent requests queue. State text is capped at 32,000 tokens with head/tail truncation. Full question/instruction overhead still needs to fit. Reserved control strings are escaped and the required Qwen2/NFC tokenizer normalization is applied by the shim; direct completion calls do not apply that contract.

### Image references and the Box question

Images can be sent by reference: put each image once under `"assets"`
(`{"<ref>": {"type": "image", "format": "png", "data_base64": "..."}}`, `"jpg"` for JPEG) and write `<ref>` wherever
the image belongs in `state`; the shim replaces the reference with the image. References match as plain text, so
every occurrence counts, even inside a longer word (`photo` also matches "a photo of"); each distinct reference is
sent to the model once, and every occurrence points at that image. Pick references that cannot occur in your text,
such as `IMG_1`. PNG and JPEG only, 16 MiB per file,
every asset referenced at least once, no URL fetching; a malformed asset map is a 400.

A `box` question, ``{"type": "box", "instructions": "the coffee mug in `photo`"}``, answers with a bounding box
`{"left", "top", "right", "bottom"}` in fractions of the image, `confidence` (the probability that the box tightly
contains the target), `present` (the probability that the target is in the image) and `found` (whether a box was
located; if not, the box is the whole image with confidence 0). Box is off by default: a box question gets 400 until
the shim runs with `LIGHTNING_BOX_ENABLED=1` (the launcher verifies `serve_config.json` against its pinned hash, so use the environment variable). `box.mode` chooses how the box is found: `"generate"` (the default: one constrained generation of the box),
`"edges"` (each edge read as a distribution over positions, the coordinate its expected value) or `"hybrid"` (both,
the tighter box answered). `edges` and `hybrid` need a model trained on edge readouts; keep `generate` with this
release.

Both go through the shim. On llama.cpp they need the image projector (`--experimental-images`), and the Box path is not yet validated there (it was checked with the shim on vLLM). The `/v1/systemone` endpoint of the PR #30361 llama-server build reads neither.

### Text and image accuracy / latency

Measurements below use an **RTX PRO 4500 Blackwell, 32 GB**, with serial client requests through the shim's decision endpoint. llama.cpp uses the corrected Qwen2/NFC tokenizer contract and one 40,960-token slot.

Text, 231 public JevBench questions:

| Runtime | Correct answers | Median latency | p95 latency |
|---|---:|---:|---:|
| vLLM BF16 reference | 204/231 (88.31%) | 29.6 ms | 219.3 ms |
| llama.cpp BF16 | 205/231 (88.74%) | 166.2 ms | 545.4 ms |
| llama.cpp Q8_0 | 204/231 (88.31%) | 161.5 ms | 543.7 ms |

Accuracy includes all 231 questions. Latency is a reanalysis of the completed run: the first **10 requests per runtime are excluded**, leaving 221 measured requests; model loading, compilation and startup are excluded. These are archived development measurements, not a new benchmark run. The same 231 distinct task IDs were used for all three runtimes. Cache behavior follows the original runtime settings; these numbers do not establish concurrent throughput.

Images, 30 synthetic development questions:

| Runtime | Correct answers | Median latency | p95 latency |
|---|---:|---:|---:|
| llama.cpp BF16 + vision projector | 30/30 (100%) | 193.3 ms | 204.3 ms |
| llama.cpp Q8_0 + vision projector | 30/30 (100%) | 189.8 ms | 200.9 ms |

Image accuracy includes all 30 questions. Timings exclude the first request per variant, leaving 29 measured requests; startup is excluded. This small fixture reuses images across questions and is **not the eight-question public ImageJevBench evaluation**. Images require `--experimental-images`.

Decision Index 0.3, public suite, text: **51.38** (the kit's 0-100 index, not percent correct) with the one-command launcher above (Q8_0, CUDA, as shipped), all 140,620 requests answered, scored with the Decision Index's own scorer on NVIDIA L40S GPUs. The same weights on vLLM (the [H2O Lightning 4B](https://huggingface.co/h2oai/h2o-lightning-4b) v1.2.3 package) score 51.41. Images were not part of this run.

### Smaller files

Q8_0 is the recommended smaller file. Q5_K_M and Q4_K_M match BF16 accuracy (see [Which file](#which-file)) but change about 3% and 5% of answers, some of them confident ones, so check them against your own routing thresholds before relying on their answers or probabilities. Every file smaller than Q4_K_M loses accuracy. If you need a smaller file, use the mixed IQ3_XXS (2.16 GB, 2.5 points lower) or, if that does not fit, plain IQ3_XXS (1.90 GB, 3.5 points lower); IQ2_XXS and IQ1_M lose much more. All of them remain experimental. The one-slot llama.cpp deployment trades speed for local runtime portability and smaller storage.

## License

Apache-2.0. Built on [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0, Qwen team, Alibaba Cloud); runs on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
