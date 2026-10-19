# Laya

Profile checked against upstream `NandhaKishorM/laya` at **ffc1420** (tag v0.4.12, 2026-10-19). Our
pin was **ea9f899** (2026-09-18, package version 0.1.6). Code citations are `file:line` at ffc1420
inside `sources/laya/`. Hub metadata, configs and the model card were fetched through the HF API on
2026-10-19 and saved as text in `notes/laya-hf/`. No weights were downloaded and no upstream code was run.

## Identity

| | |
|---|---|
| Repo | https://github.com/NandhaKishorM/laya (Python package `laya` on PyPI, 0.4.12; about 33.5k stars) |
| Author | Nandha Kishor M / "Convai Innovations" (`pyproject.toml` authors) |
| Weights | HF org **`convaiinnovations`** (the author's own HF account is `nandakishorm`, which holds only the vega line, see below) |
| Checkpoints | `convaiinnovations/laya` (English, at the repo root, which also bundles the `multilingual/` and `typed-decisions/` subfolders), `convaiinnovations/laya-multilingual`, `convaiinnovations/laya-typed-decisions` (`laya/router.py:6-8,51,60-62`) |
| Weight files | root `model.safetensors` 842,609,210 B (fp16, about 421M params) sha256 `891102d3…`; `typed-decisions/` 842,609,220 B `4fa56de7…`; `multilingual/` 643,835,514 B `9d628fd9…` |
| Weights changed since our pin? | **No.** Every Hub commit after the 2026-09-18/19 initial uploads touches only README, assets or `config.json`. The LFS sha256 values above match the blobs already in our `~/.cache/huggingface/hub/models--convaiinnovations--laya` |
| Pitch | "Non-autoregressive System 1 decision engine": typed `choice`/`score`/`noul` questions over a state, one batched forward pass, no generation |

**The author's siblings and newer lines.** These come from `gh repo list NandhaKishorM` and the HF API with `?author=`.
- **vegaml** (github.com/NandhaKishorM/vegaml, created 2026-10-09, Apache-2.0, about 270 stars). This is a *different architecture*: a frozen causal LM "reader" (Qwen3.5-0.8B/4B, gemma-3-1b, and Qwen3.8-27B attached cold) feeds a 14.3M-parameter trained "engine". It reports conformal sets, an abstain flag and a claimed 73,728-token context. HF: `nandakishorm/vega-U` (2026-10-17) and `nandakishorm/vega-08b-public-intents` (2026-10-09). It is the author's successor direction and deserves its own profile (`clones/vegaml.md`, not done yet).
- The other `convaiinnovations` models (medgemma-ECG, hindi-*, gemma3 sales coach, …) have nothing to do with decision models. **No newer Laya checkpoint exists.** All three are still the September uploads.
- Third-party derivatives on HF (`?search=laya`) include `ggml-org/Laya-GGUF`, `rupeshs/laya-typed-decisions-ov-int8`, `cklxx/laya-browser` (a browser-agent fine-tune that upstream links from `docs/finetune_browser_agent.md`; it is not ours), plus several language fine-tunes (tr, fa, he, ar, indic).

## Architecture

### Backbone
- The model is `DecisionModel(encoder, head_layers=2, n_act=2)` (`laya/common.py:575-608`). The encoder is `AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")` (`common.py:694-717`), fully fine-tuned.
- English and typed-decisions use **`answerdotai/ModernBERT-large`**: 28 layers, d=1024, 16 heads, FFN 2624, vocab 50,368, max positions 8192, a global-attention layer every 3rd layer, the other layers on a 128-token local window, RoPE θ 160k global / 10k local. Sources are the Hub `encoder/config.json` and `rl_agent_config.json`.
- Multilingual uses **`jhu-clsp/mmBERT-base`**: 22 layers, d=768, vocab 256k, both RoPE θ = 160k. `common.py:662-682` exists to carry those thetas onto transformers 4.x.
- Decision head, trained from scratch, about 26M params:
  - 2 × `nn.TransformerEncoderLayer(d, d/64 heads, 4d, dropout 0.1, norm_first)` (`common.py:591-596`)
  - `type_emb = Embedding(3, d)` (`:597`)
  - `scorer = LN → Linear(d,d) → GELU → Linear(d,1)` (`:598`)
  - `act_head = Linear(d+4,256) → GELU → Linear(256,n_act)` (`:599`)
  - a `temperature` buffer of length 3 (`:600`)
- Total is about 421M (English) / 322M (multilingual), per the model card.

### Input encoding
This is a **listwise cross-encoder, one sequence per (state, question)**. All options of a question sit in the same sequence as the state. Questions do not share a sequence: Q questions become Q rows of one padded batch, and the state is re-encoded per question (`agent.py:1133-1184`; the state is tokenized once and sliced per question, `:1150-1154`).

`build_sequence` / `build_head` (`common.py:157-277`), docstring `:170`:

```
[CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 ... [MASK] optK-1 [SEP] <state> [SEP]
  |      '---- head_ids, cut to max(8, opt_budget) ----'  '-- each [MASK]+≤48 option tokens --'     '- room = max_len-len(head)-1 -'
  |                                                        ^ marker positions -> scorer logits
  `-- type_emb(qtype) is added to every hidden state before the head layers
```

- Option text (`render_options`, `common.py:128-154`):
  - `choice`: `"label: description"`, or the bare label when there is no description, in the caller's dict order. There is no sorting: commit 3ebc23db "sort choice criteria keys" only added tests.
  - `score`: `"level i: text"`.
  - `noul`: exactly two slots in fixed order [false, true]. They read `"false: <crit.false | 'no, the statement does not hold'>"` and `"true: <crit.true | 'yes, the statement holds'>"`, and an optional `labels` map can replace the visible words "false"/"true" (`:114-125, 146-154`).
- Special tokens: ModernBERT's `[CLS] [SEP] [MASK] [PAD]` (Hub `tokenizer/tokenizer_config.json`, `model_max_length` 8192). `[MASK]` strings inside user text are replaced with spaces (`common.py:202, 238, 247`).
- The state is a string, or `json.dumps(dict|list)` (`common.py:95-98`). It is **right-truncated** silently. A conversation list is **left-truncated** to keep the newest turns (`agent.py:1142-1145`, `common.py:205`). Since 0.3.x, `usage` reports the truncation (`common.py:211-218`).
- Instructions are cut **after** the options are budgeted: `head_ids[:max(8, head_max_len - Σ option tokens)]` (`common.py:253-260`). Long instructions therefore get truncated, down to 8 tokens when there are many options.
- `option_order` (since 0.3.x) lets the caller permute slots, and probabilities are un-permuted afterwards (`common.py:558-572`).

### Attention between options
- **Default "sequential" layout** (every published checkpoint): full bidirectional attention. Options attend to each other, to the instruction and to the state, inside ModernBERT's alternating local(128)/global layers. An option's absolute position therefore changes its embedding. Upstream measures slot logits of `[+1.68, +1.47, +0.11, -1.35, -1.91]` for five *identical* options (README "Option order", `research/eval/presentation_checks.py`).
- The 2 head layers attend over **the whole sequence** (padding-masked only), not just over the markers (`common.py:623-631`), and they carry no positional encoding.
- **"parallel" layout**, opt-in and new since our pin (`common.py:280-348`, commit 9aaf4af9, 2026-10-05):
  - every option restarts at the same position id, and the state continues after the longest option;
  - the encoder masks option↔other-option attention, and the local window is measured in position ids;
  - the head layers still see all tokens, but they are permutation-equivariant.
  - It needs transformers>=5, works on the eager backend only, and is set by `"option_layout": "parallel"` in the checkpoint config (`common.py:312-327`).
  - **No published checkpoint uses it.** A README fine-tune gives 0.764 accuracy with 100% order stability, against 0.779/91.7% for sequential.

### Readout head
- `h = encoder(...) + type_emb(qtype)` goes through 2 transformer layers. The hidden states at the `[MASK]` markers are gathered, each becomes one scalar logit through `scorer`, and padding slots are filled with -1e4 (`common.py:620-635`).
- At decode time the logits are divided by the temperature, softmaxed over that question's markers and un-permuted (`agent.py:1342-1351`). The head is the **same for every type**; only the `type_emb` and the decode differ.
- **Act head** (`common.py:637-658`): input is pooled `[CLS]` plus 4 detached features (top-1 prob, top1−top2 margin, normalised entropy, `k/255`). There are `n_act = 1 + len(act_costs)` outputs, i.e. {answer, escalate} with config `act_costs={"escalate":0.5}, cost_wrong_act=3.0`. Exposed as `action.act_probability = softmax(act)[0]` (`agent.py:1330,1366`).
  - The public trainer discards it (`train.py:635-647` returns logits only), so it is never trained there.
  - The README says it carries no usable signal (#185: reads ≈1.0, AUROC 0.30).

### Question types
| type | slots | answer | `confidence` | `answer_confidence` |
|---|---|---|---|---|
| `choice` | one per criterion | `keys[argmax p]` + `probabilities` (`agent.py:1369-1377`) | 1 − H(p)/log k (`common.py:814-835`) | max p (`common.py:791-811`), optional histogram binning |
| `score` | one per level | **expected level index** Σ i·pᵢ, a float in [0, k−1], not normalised to 0..1 (`agent.py:1378-1395`) | normalised entropy | max p |
| `noul` | 2, [false, true] | P(true) = p[1] (`agent.py:1396-1404`) | max(p, 1−p) | max p |

`QTYPES = {"choice":0, "score":1, "noul":2}` (`common.py:19`). The core accepts only these three type names; an unknown type such as `boolean` is refused (`agent.py:985-987`). The server has no `boolean` alias either. The served API adds `x_jev_confidence`, computed with TypeSafe's formulas: choice `(max p − 1/n)/(1 − 1/n)`, score from a distance-to-mode measure (`laya/confidence.py:38-57`, `serve.py:256`).

### Context & limits
- Shipped configs (Hub `rl_agent_config.json`): English **max_len 512 / head_max_len 192**; typed-decisions and multilingual **1024 / 256**. The runtime defaults to 512/192 (`agent.py:1140-1141`).
- The encoder supports 8192 positions. Since 0.3.x, `max_len` and `head_max_len` can be overridden per call (`agent.py:1416-1417`), and the server caps them at `LAYA_MAX_TOKEN_BUDGET=8192` (`serve.py:151`). The model card recommends `max_len=8192` only for *multilingual*: 16-18 of 20 correct up to about 4k tokens, then 8-17 of 20.
  - The English checkpoint was trained at 512. Longer inputs are out of distribution: our laya-browser untuned 4096/3584 gives top-1 0.369, against 0.263 at 512.
- Option budget per option (`common.py:242-259`): at most 48 tokens. If the options together leave less than 16 tokens of the head budget, each is cut to `max(4, (head_max_len−16)//k)`. Beyond about `head_max_len/4` options the head therefore *exceeds* the cap and eats into the state room. If any marker falls past `max_len`, the request is refused (`agent.py:1163-1178`).
  - Banking77 (77 labels) scores 0.425 on both checkpoints, a budget ceiling (BENCHMARKS.md "On the public datasets").
- Workarounds added since our pin:
  - `predict_long`: overlapping windows, answers aggregated per question (`common.py:373-461`, `agent.py:1601`);
  - `predict_shortlist`: an embedding shortlist (`shortlist.py:68`);
  - `predict_tournament`: groups of at most 16 and then a final round, up to 256 labels in two calls (`shortlist.py:139`; served behind `LAYA_TOURNAMENT_GROUP`, which raises the server cap from 100 to 255 options, `serve.py:126,130`).
- Server limits: 64 questions, 50k state chars, 32 score levels, 512 total options (`serve.py:90-132`).

### Calibration
- Temperature scaling. There is one temperature per type, `temperature[qtype]`, plus per-bucket overrides `temperature_by_options["<type>:<2|3-5|6-10|11+>"]`, and the bucket wins (`common.py:838-840`, `agent.py:1342`).
- Optional extras: `lang_temperatures` per language (`common.py:889-955`, `agent.py:1343-1345`) and a histogram `binning_map` on `answer_confidence` (`calibrate.py:519-567`).
- The fitter is LBFGS on log T minimising NLL (`calibrate.py:166-182, 331`).
- Shipped English values: `temperature=[1.637, 1.251, 1.983]` (choice, score, noul) with buckets choice:2 1.906, choice:3-5 1.760, choice:6-10 1.000, **choice:11+ 0.1006**, score:3-5 1.251, noul:2 1.983.
  - typed-decisions ships **the same bucket table copied from English**, which masks its own fitted per-type temperatures of about 1.01-1.06. The README fine-tuning section admits this inheritance bug.
  - Multilingual ships all 1.0 and no buckets.
- **Clamp to [0.5, 5.0]**, added 2026-09-21 in commit 8e58a6cd (`common.py:843-866`). It stops the 0.1006 temperature from sharpening the logits 10×.
  - **At our pin ea9f899 the raw values were applied unclamped** (`git show ea9f899:laya/agent.py` lines 158-159, 254). Every choice question with 11 or more options was published about 10× over-sharpened. This is very likely part of why our jevpilot Laya variant saw over-confident wrong answers on large option sets.
- Upstream's own numbers: shipped ECE 0.466 (English) and 0.314 (multilingual), against 0.081 and 0.106 after refitting per (type, k) on held-out data. "Ships over-confident" (README "Calibration").

## Training

- **Base checkpoints.** The pre-training corpus and script for the base RL run are not published.
  - What the configs record: English `training: {updates 7313, epochs 1, 1.96 h, world_size 1, fine_tuned_from_checkpoint: true}`. The English root is itself a fine-tune of an earlier, unpublished checkpoint. Multilingual: `{updates 15987, epochs 4, 4.97 h, from scratch}`.
  - Task families listed in the Hub `eval/results.md` "in-task" set: conversation outcomes, email triage, emotion, NLI and fact checking, instruction-following, intent/routing, moderation, reading comprehension, response-quality scoring, robustness, search relevance, sentiment, topic.
  - AG News, BoolQ, spam, phishing, RAG relevance and support triage are "in training mix" (BENCHMARKS.md "Themes").
  - The notebook claims the data is "100% permissively licensed public datasets".
- **The original encoder code** is on the Hub as `rl_common.py` (saved as text in `notes/laya-hf/laya/rl_common.py.txt`; it was not run). It shows that training **shuffled option order for choice and noul** but not for score (`encode_record` lines 249-256). Conversations become noul items over ≤`max_prefixes=6` prefix slices with TD(λ=1) soft targets (lines 228-247, 319; `common.py:749-764`).
- **Objective "RLCD"** (`train.py:499-518`): zero-mean Gaussian noise (σ annealed 0.4→0.1) is added to detached logits, G=4 samples are drawn, and each sample is rewarded by a strictly proper score: log score + w_sph · spherical, minus w_rps · RPS for `score` (`common.py:720-746`). The update is REINFORCE with a group-mean baseline (GRPO-style).
  - **plus** soft cross-entropy to the target distribution (`:518`). The old Colab notebook claimed "no supervised CE".
  - Upstream's own #741 measured **no gain from the RL term** over soft-CE alone on typed-decisions (`train.py:7-9`).
- **Fine-tune hyperparameters** (`TrainConfig`, `train.py:78-124`, matching the Kaggle 2×T4 notebook):
  - 4 epochs, effective batch 64 (8 × 8 accumulation);
  - AdamW, encoder lr 2.5e-5 / head lr 1e-4, cosine schedule to 1e-6, weight decay 0.01, gradient clip 1.0;
  - w_sph 0.75, w_rps 1.0;
  - a calibration hold-out (≤400 items or 10%), and fp16 AMP plus gradient checkpointing on CUDA.
  - `shuffle_options` (opt-in) and `option_layout` are configurable since 0.3.28/0.4.x.
- **laya-typed-decisions** is fine-tuned from the English root on HF dataset `LocalLLaMA/typed-decisions` (Apache-2.0) at 1024/256, with targets taken from a teacher's probabilities ("gold"). Runtime is about 4-5 h on 2×T4 for about 30k questions (README "Fine-Tuning").
- What we learned in our own fork (`/home/ai/projects/laya-browser/README.md`):
  - fine-tuning the same `DecisionModel` at 4096/3584 with up to 150 options plus "none" took top-1 from 0.263 (untuned 512) to 0.862 and ECE from 0.632 to 0.033;
  - the head and readout work far beyond the shipped budgets; the 512/192 limit is a fine-tuning choice, not an architectural one;
  - in upstream's browser write-up, moving candidates from the state into the options was the change that mattered most (`docs/finetune_browser_agent.md:52-57`).

## Evals

All Jev figures in this section are **third-party published, never measured by upstream** (README "Laya (with routing) vs Jev").

| benchmark | Laya | Jev 1.13.0 | note |
|---|---|---|---|
| typed-decisions, 2,000 decisions | **0.766** (laya-typed-decisions, fine-tuned on its own train split) | 0.727 | base `laya` 0.362, multilingual 0.352, majority class 0.461, random 0.318: **the base checkpoints sit below the majority-class baseline** |
| typed-decisions soft accuracy | 0.471 | **0.580** | |
| typed-decisions ECE / Brier | 0.213 / **0.062** | **0.144** / 0.148 | |
| AG News 4-way | **0.950** | 0.910 | AG News was in training |
| DAIR Emotion 6-way | **0.595** | 0.480 | held out |
| Banking77 | 0.425 | **0.870** | option-budget ceiling |
| p50 latency, 1 question | 32.8-39.5 ms (T4) | 236-276 ms (API, third-party) | |

- Further results:
  - MASSIVE intent across 51 languages: English checkpoint macro 0.227, multilingual 0.401; the English checkpoint gives 0.000 on Khmer at 0.95 confidence;
  - XNLI English 0.860;
  - held-out toxic-chat moderation 0.530 (around chance), prompt-injections 0.698, SST-5 0.372 (`score` is the weakest primitive).
- The Hub `eval/results.md` gives overall in-task accuracy 0.753 / ECE 0.030, and zero-shot families 0.651 / ECE 0.204.
- Known failure modes stated upstream that match what we saw:
  - **a `noul` without criteria answers "no" whatever the state on the English checkpoint**: the default option text carries the decision (#156);
  - `choice` keys like true/false or yes/no override the descriptions;
  - negation fails (#377);
  - multilingual has a position bias on `score` (#131).

## Serving

- `laya-serve` (FastAPI, `pip install laya[serve]`) exposes **`POST /v1/systemone`** and **`POST /v1/systemone/batch`**, plus `GET /health` (`serve.py:1265,1331,1428`). It is presented as wire-compatible with TypeSafe Jev's request and response (`serve.py:1-15`).
  - Extra fields: `routing`, `action`, `answer_confidence`, `x_jev_confidence`. `LAYA_JEV_STRICT=1` strips them (`serve.py:52-56`).
  - `usage` = `{input_tokens, output_tokens: 0}` and truncation stats.
  - Optional `Authorization: Bearer` via `LAYA_API_KEY`.
- The `Router` picks a checkpoint per request by script and language detection: English goes to `laya`; non-English and, since 0.4.0, undecided text go to multilingual.
- Python API: `laya.load(...)`, `Agent.predict / predict_batch / predict_long / decide` (pydantic schema).
- Backends: eager, `torch.compile`, a TileLang CUDA fast path (`fast.py:28`), and ONNX with optional INT8. SDKs exist in TS, Go, Java, Rust and .NET, plus an MCP server and LangChain, LlamaIndex, CrewAI and Agno integrations.
- Our jevpilot `variants/laya/server.py` is a 64-line stdlib shim over `laya.load()` at 0.1.6 (the pinned ea9f899). Upstream now ships the equivalent server itself.

## Licence

Apache-2.0 for code (`LICENSE`, `pyproject.toml`) and weights (HF `cardData.license: apache-2.0` on all three repos). The backbones are Apache-2.0 (`answerdotai/ModernBERT-large`) and MIT (`jhu-clsp/mmBERT-base`), both checked via the HF API. The fine-tune dataset `LocalLLaMA/typed-decisions` is Apache-2.0. The base-training data mix is claimed permissive but not itemised.

## Changelog (ea9f899 → ffc1420)

The pin is 2026-09-18 (0.1.6); HEAD is 2026-10-19 (0.4.12): **2,395 commits** and 44 tags (v0.2.0 … v0.4.12); `laya/` grew from 5 to 47 files (+24.9k lines).
- **Model and weights: unchanged.** The architecture in `common.py` (`DecisionModel`, sequence format, losses) is identical to 0.1.6 apart from the opt-in parallel layout.
- Relevant behaviour changes:
  - **temperature clamp [0.5, 5]** (0.3.x, 2026-09-21), which neutralises the 0.1006 choice:11+ temperature;
  - `answer_confidence` = max p on every type (0.3.12);
  - per-call `max_len` / `head_max_len`;
  - `option_order` (2026-09-24);
  - `predict_long` windows (2026-09-24);
  - truncation stats in `usage`;
  - `noul` `labels`, with refusal of non-true/false criteria keys (which used to be dropped silently);
  - `lang_temperatures`; binning maps; fit, save and load calibration;
  - abstention `min_confidence`;
  - parallel option layout (2026-10-05);
  - `laya.train` / `laya-train` CLI, one shared fine-tuning loop with soft-ce or rlcd, option shuffling and calibration fitting (2026-10-03..07);
  - shortlist and tournament for large label sets;
  - Router default switched to multilingual for undecided text (0.4.0);
  - `OptionStabilityHook`, a reorder/relabel re-ask stability signal (0.4.12).
- Serving and tooling: Router and 3-checkpoint bundle (0.2-0.3.0), `laya-serve` /v1/systemone and its batch endpoint, Jev-strict mode, `x_jev_confidence`, ONNX, TileLang, compile, MCP, the framework integrations, five language SDKs, an eval harness (`laya-evals`), and SHA-256 and revision pinning.

## Notable design choices vs Jev

- **One listwise cross-encoder row per question.** Every option is a `[MASK]` slot in the same sequence as the state and gets one scalar logit, then a softmax over the question's slots. `noul` is a 2-slot choice and `score` is a k-slot choice read out as an expectation. There is no per-type head, only a type embedding. Cost scales as Q × (state + options).
- **Options compete for a small shared head budget** (192/256 tokens, at most 48 per option, a 4-token floor), and instructions are truncated *after* the options. That gives the documented ceiling past about 20 options, where Jev reportedly handles 255. `k/255` in the act features echoes that same 255 limit.
- **Options are order-dependent by default**: sequential positions and cross-option attention cause a measurable slot prior. Training-time shuffling did not remove it. The parallel layout fixes it but has never shipped in a checkpoint.
- **Calibration is post-hoc temperature scaling**, per (type, option-count bucket). The "RLCD" proper-scoring objective does not make the shipped models calibrated: shipped ECE is 0.47 and upstream measured no gain over soft-CE.
- **Fine-tuning is required.** The base checkpoints are a fast backbone, not a zero-shot decider: they score below majority class on typed-decisions. The headline wins over Jev come from a fine-tune on the benchmark's own train split.
- **Short trained context** (512 English) on a backbone that supports 8192. Long states are cut silently at the right, which `predict_long` works around.
- **The served API mirrors Jev's `/v1/systemone`**, but semantics differ in two places:
  - `score` is an expected *index*, not a 0..1 value;
  - `confidence` is normalised entropy, with the Jev formula offered as `x_jev_confidence`.
- **Open weights (Apache-2.0)**, about 840 MB fp16, CPU-runnable. Jev is a closed API.
