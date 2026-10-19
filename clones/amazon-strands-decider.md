# Amazon / Strands Agents: strands-decider

Open Jev replication from the Strands Agents team (Amazon's open-source agent SDK). A frozen decoder LLM loses its LM head and gets a rank-16 LoRA plus a ~1M-parameter **pointer head**. The head scores each option by comparing the hidden state at `<answer>` with the hidden state at the last token of that option's line. The state is encoded once, and every question forks the state's KV/recurrent cache as its own causal row. It serves TypeSafe's `POST /v1/systemone` shape. Each release is a **3-seed soup** with **one temperature per question type**.

Source pins used for this profile:

| ref | what | where |
|---|---|---|
| `R` = github `strands-labs/strands-decider` @ `930e6d5d` (2026-10-16, "feat(evaluate): prepare the model as serving does…", #79) | shallow clone; read only, never executed | `/home/ai/projects/jev-clones/sources/strands-decider` |
| `HF:<size>` = `amazon/strands-decider-<size>-v1-2610`, all five repos | API JSON, `tree.json`, README, `strands_decider_config.json`, `train_config.json`, `provenance.json`, `lora/adapter_config.json`, `eval/**` | `/home/ai/projects/jev-clones/notes/hf-raw/amazon__strands-decider-*-v1-2610/` |
| `ST:<size>` | safetensors **headers only**: an HTTP range request for the first 8+N bytes of `lora/adapter_model.safetensors` and `head.safetensors`. These give tensor names, shapes and dtypes; no weights were downloaded | `*.hdr` in the same folders (the 2B ones in `…2B-qwen3.5…/st-headers/`) |
| `G:<base>` | Gemma 4 base `config.json` | `notes/hf-raw/google__gemma-4-*/files/config.json` |

Citation form: `R:src/strands_decider/infer.py:315` means file:line at `930e6d5d`. `HF:2B/train_config.json` means a fetched file. The task brief names the GitHub org as `strands-agents`. The code actually lives at **`strands-labs/strands-decider`**, as linked from every card (`HF:*/README.md`). `strands-agents/sdk-python` is only the agent SDK that the README points to for usage (`R:README.md`).

> **Main finding.** All five `amazon/*-v1-2610` repos are plain PEFT LoRA adapters plus a pointer `head.safetensors`. Each config shows rank 48 and `pointer_dim` 768, but **every seed was trained at rank 16 with `pointer_dim` 256**. The soup concatenates the three seeds' A/B matrices and heads, so that the soup's ΔW and its logit are each the exact mean of the seeds' (`R:training/soup.py:1-20,92-105`, `HF:2B/train_config.json` vs `HF:2B/strands_decider_config.json`). The Qwen3.5-2B "v1" and the four Gemma 4 "v1"s use **different recipes**:
> - 2B: 246,678 rows, teacher gemma-4-31B-it.
> - Gemma: hobson-v21's 135,602 rows, teacher Qwen3.5-4B.
>
> Both use a frozen-KL anchor to the released hobson-v19 checkpoint. The readout is trained on ≤ 24 options (`num_slots`), but the pointer itself has no option cap, and the schema allows 255.

---

## Identity

| | |
|---|---|
| Org / repos | Code: `github.com/strands-labs/strands-decider` (Apache-2.0). Main HF home: `StrandsAgents/*`. The five v1 models are mirrored byte-identical under `amazon/*` (`HF:*/README.md` line 1, `R:README.md` "Model names") |
| HF repos | `amazon/strands-decider-{E2B,E4B,12B,26B-A4B}-gemma4-v1-2610` and `amazon/strands-decider-2B-qwen3.5-v1-2610`. All were created 2026-10-09 22:53–22:55Z with downloads 0. The HF API returned `sha: null` for every mirror (`HF:*/api.json`). Earlier releases: `StrandsAgents/strands-decider-2B-hobson-v19` (09-30) and `-hobson-v21` (10-05) |
| Name scheme | `strands-decider-<size>-<base>-v<N>-<YYMM>`. `v<N>` counts releases **per base family**, and "releases with the same N on different bases can use different recipes" (`R:README.md`, `R:docs/naming.md`). The internal project name is **"hobson"**: `hobson-hf-export/1` and `hobson_config.json` (`HF:*/provenance.json`, `R:src/strands_decider/modeling.py:681`) |
| Run ids | 2B: `v1-qwen3.5-2b-soup` (data generation `g28`). Gemma: `g26-{e2b,e4b,12b,26b-a4b}-code-soup3`, configs written 2026-10-06 (`HF:*/provenance.json`) |
| History | Research generations v1–v21 ran 2026-09-20 → 10-05 on **one RTX 3090** (`R:research/generations.md`). Release training ran on AWS: p4de (8×A100) for the 2B and 8×H100 for Gemma (`HF:*/README.md` Training) |
| Lineage | The cache-fork code says it follows "decider-2b's (`decider/shared_prefix.py`, Apache-2.0)" (`R:src/strands_decider/infer.py:100`). On the board, decider-2b is `Mapika/decider-2b`. The pointer head carries a LayerNorm that "is not in the reference implementation" (`R:modeling.py:245-247`). That reference is plausibly kev's `PointerHead`, which has no LN (inferred). The docs adopt the confidence bands of the Jev docs (`R:docs/architecture.md` "The routing convention") |
| Package | `pip install strands-decider`. The PyPI build lacks Gemma 4 support, so the Gemma cards install from git `main` (`HF:26B-A4B/README.md` Use) |

---

## Architecture

### Backbone ("torso")

| Checkpoint | Base (revision, per `provenance.json`; "inferred: hosts did not pin it") | Layers | d | LoRA modules actually present (`ST:`) | Trained params in the release (fp32) | Head params |
|---|---|---|---|---|---|---|
| 2B-qwen3.5 | `Qwen/Qwen3.5-2B-Base` @`b1485b2f` | 24 = 18 Gated DeltaNet + 6 full attn | 2048 | DeltaNet `in_proj_{qkv,z,a,b}`, `out_proj` ×18; `q,k,v,o` ×6; MLP `gate,up,down` ×24 | 50.46M (r48 soup = 3 × 16.8M) | 3.15M |
| E2B-gemma4 | `google/gemma-4-E2B-it` @`3e22461f` | 35 (28 sliding-512 / 7 full), 1 KV head, 20 KV-shared layers | 1536 | `q,o` + MLP ×35; `k,v` only ×15, i.e. not on the KV-shared layers (inferred from `G:` `num_kv_shared_layers 20`) | 72.47M | 2.36M |
| E4B-gemma4 | `google/gemma-4-E4B-it` @`ee0ef602` | 42 (35 sliding / 7 full), 18 KV-shared | 2560 | `q,o` + MLP ×42; `k,v` ×24 | 104.6M | 3.94M |
| 12B-gemma4 | `google/gemma-4-12B-it` @`707f0a3b` (`gemma4_unified`) | 48 (40 sliding-1024 / 8 full), `attention_k_eq_v` | 3840 | all ×48 except `v_proj` ×40, i.e. absent on the 8 global layers, consistent with k = v there (inferred) | 196.7M | 5.91M |
| 26B-A4B-gemma4 | `google/gemma-4-26B-A4B-it` @`4d7ae498` (MoE, 128 experts, top-8) | 30 (25 sliding / 5 full) | 2816 | attention + the dense `mlp.{gate,up,down}_proj` (down_proj in-dim 2112) ×30; `v_proj` ×25. **No LoRA on the experts** (inferred: expert weights are not named `*_proj` Linear modules) | 55.76M | 4.33M |

- The **LM head is never loaded.**
  - Qwen3.5 loads through `Qwen3_5ForCausalLM(...).model` and drops the vision tower (`R:modeling.py:399-409`).
  - Gemma 4 loads the full `Gemma4(Unified)ForConditionalGeneration`, keeps `.model.language_model`, and drops the 0.47B of vision and audio encoders. `Gemma4ForCausalLM` "leaves every text weight randomly initialised" (`R:modeling.py:410-424`).
- **LoRA** (`R:modeling.py:444-456`): `task_type=FEATURE_EXTRACTION`, dropout 0.05, no DoRA or rsLoRA. Each seed trained at **r 16, α 32**. The release adapter is r 48 with α 96, which keeps s = α/r = 2 (`HF:*/lora/adapter_config.json`, `R:training/soup.py:156-164`). Targets are q/k/v/o/gate/up/down, plus the five DeltaNet projections on Qwen3.5, because otherwise "LoRA reaches only its few full-attention layers" (`R:training/train.py:58-62`).
- **Gemma quirks.**
  - `force_bos: true`: gemma-4-E2B-it's tokenizer leaves BOS to the chat template. "Without it the frozen -it torso reads JevBench at chance (73/231, against 135 with it)" (`R:modeling.py:110-114`).
  - `host_embeddings` (the per-layer embedding table, 4.4 GiB in bf16, kept in host RAM) is off in the releases (`HF:E2B/strands_decider_config.json`).
  - The Gemma bases are **instruction-tuned** (`-it`), whereas the Qwen torso is `-Base`.
- Dtypes: the torso runs in bf16 (`torch_dtype`), and the **head always runs in fp32** (`R:modeling.py:265-284`). On CPU the torso is upcast to fp32, because "bf16 is slower than fp32" there (`R:infer.py:171-184`).

### Input layout

`render_state` + `render_question` (`R:src/strands_decider/prompting.py:90-150`) produce plain text. **No reserved tokens** are used: delimiters are ordinary-text pseudo-XML, and options are numbered lines.

```
<state>
{state: string as-is, or JSON.dumps(indent=2, ensure_ascii=False)}
</state>
<question type="choice">                       <- type is noul | choice | score
Select exactly one option.                     <- per-type header (see table)
{instructions, rendered like state}
<options>
1. billing — payments, invoices, payouts       <- one option = one line; desc collapsed to one line
2. technical — bugs, outages, API errors
3. sales                                        <- null description: bare name
</options>
</question>
<answer>                                        <- last token = query position
```

| type | rendered options | header |
|---|---|---|
| noul | `1. false — {criteria.false or "the statement does not hold for this state"}` and `2. true — …` (`R:prompting.py:27-31,102-106`) | "Decide whether the statement is true of the state." |
| choice | criteria keys in request order, `name — desc` | "Select exactly one option." |
| score | `1. 0 — level0 text`, `2. 1 — level1 text`, … (the level index is the label) | "Rate the state against the ordered levels below (lowest first)." |

Readout positions within one question row:

```
tokens:  [BOS?] <state> … </state>\n | <question …> header instr <options>\n 1. A …a  \n 2. B …b  \n 3. C …c  \n </options></question>\n<answer>
                └──── shared prefix ──┘ └────────────────────── per-question suffix ────────────────────────────────┘
                (encoded once, cached)                                ^k1          ^k2          ^k3                     ^q
                                                         key = last token whose char offsets lie inside option j's line;
                                                         query = last unmasked token (the end of "<answer>")
```

- Option key positions are computed from the tokenizer's **character offset mapping** against each line's `(start, end)` span (`R:infer.py:467-494`). If truncation removed every token of an option, the engine raises instead of scoring it from a neighbour.
- **Window fitting** (`_fit`, `R:infer.py:197-248`): the **question gets first claim**, at most 75% of `max_length` (4096 in the releases). It is truncated from the *front* if needed, so the options and `<answer>` survive. The **state is truncated** to the remaining budget with `truncation=True`. The docs say the state "is cut from the front" (`R:docs/architecture.md` "Drop, never truncate"), but neither the code nor the shipped `tokenizer_config.json` sets `truncation_side`, so the HF default (right, i.e. the *end* of the state is cut) would apply. This is inferred, and the doc and the code disagree. The motivation: before this, "all 19 `long_policy` states hit the cap exactly" and accuracy sat at chance. `strict_window` refuses instead of truncating, for benchmarks. Training and eval drop over-length rows ("drop, never truncate", `R:docs/architecture.md`).
- No chat template is used, even on the `-it` Gemma bases (inferred from `prompting.py`, which never calls `apply_chat_template`). `chat_template.jinja` ships only because it is part of the saved tokenizer.

### Attention, mask and isolation between questions

- Attention is the base model's own, **causal**, with no custom mask. "Converting a causal LLM to bidirectional attention requires substantial retraining" (`R:docs/architecture.md` "Design decisions").
- **Exact isolation by construction** (`_slot_probs_shared_prefix`, `R:infer.py:314-370`):
  1. The state prefix runs once with `use_cache=True`.
  2. `_expand_cache` forks it into *m* rows. It copies attention K/V and also the Gated DeltaNet `conv_states` and `recurrent_states`, as new tensors, because "a DeltaNet layer updates its states … in place". The Gemma sliding-window size tensor is shared (`R:infer.py:80-139`).
  3. All question suffixes forward in one right-padded batch over the forked cache.

  Each row is therefore exactly `state ‖ one question`: no question sees another, and the result does not depend on order. Unknown cache layouts raise `UnforkableCache` and fall back to the batched path.
- **Single-question requests skip the prefix cache.** The full prompt is encoded in one pass, because "p50 0.111 s batched, 0.204 s through the prefix path" (`R:infer.py:408-412`). Both paths agree within float tolerance (`tests/test_prefix_cache.py`, not run here).
- Within a question, options attend causally to earlier options, so the readout is **order sensitive**: "reversing the option list moves the distribution by ~0.016… per-option attention isolation… is not implemented" (`R:docs/architecture.md`).
- Chunking: at most `max_batch` = 32 questions per forward, with an optional `max_batch_tokens` cap. "Long-context Gemma torsos need it at a 32,768 window" (`R:infer.py:56-73,443-461`).
- **Gemma E2B attention bug.** Torch 2.7's CUDA SDPA returns wrong attention for a forward over a KV cache with 64k+1 query tokens, with one KV head. Served E2B forces `eager` (`R:modeling.py:46-60`). The E2B card warns that its Decision Index was measured before this fix, so multi-question noul answers there "may be off" (`HF:E2B/README.md`).

### Readout head

`PointerHead` (`R:src/strands_decider/modeling.py:228-262`):

```
x̂     = LayerNorm(x)                               (one shared LN, d → d, fp32)
q     = W_q · LN(h[<answer>]) + b_q                W_q: d → dp
k_j   = W_k · LN(h[last token of option j]) + b_k  W_k: d → dp
z_j   = (k_j · q) / sqrt(dp)                       dp = 256 per seed (768 in the soup, see below)
z_j  /= T_type                                     T from temperature_by_kind[noul|choice|score]
p     = softmax over the first N slots             (masked_log_softmax, mask value −1e4; R:modeling.py:314-325)
```

- There is **no per-option parameter**, so "nothing can learn that the first option is usually right". The design claims "nothing caps how many options a question may carry", but see the limits below. Measured on frozen v6 features, the pointer head beat a refitted slot head by +0.063 at dp = 256 and +0.058 at dp = 16, while a 2.1M-parameter MLP slot head gained only +0.032: "the gain is the addressing rather than capacity" (`R:modeling.py:240-243`).
- **Soup head**: each seed's LN is folded into its q/k projections, the heads are concatenated, and `k` is scaled by 1/√n, so the soup's logit is exactly the mean of the 3 seeds' logits. The soup's LN is identity (`R:training/soup.py:52-89`). This is why the released config says `pointer_dim 768` (`ST:2B/head`: `q.weight [768, 2048]`, `k.weight [768, 2048]`, `norm.{weight,bias} [2048]`).
- The legacy **slot head** (`Linear(LN(h_last)) → 24 slots`) was the v1–v6 readout and stays loadable (`R:modeling.py:203-225`).

### Question types and per-type handling

| type | slots | answer (`_to_answer`, `R:infer.py:497-525`) | confidence | T (E2B / E4B / 12B / 26B-A4B / 2B) |
|---|---|---|---|---|
| noul | 2 (`false`, `true`) | `noul = p[true]`, wherever "true" landed | none ("with two outcomes the probability *is* the uncertainty", `R:schema.py:38-39`) | 1.070 / 0.592 / 0.775 / 0.531 / 1.070 |
| choice | 2..255 | argmax, `probabilities` by key, 4 decimals | `(N·p_max − 1)/(N − 1)`, the Jev formula (`R:schema.py:185-198`) | 0.561 / 0.775 / 0.625 / 0.592 / 0.863 |
| score | 2..10 levels | `score = Σ i·p_i` after re-keying by true level index; `legend`; `probabilities` | **its own formula, not Jev's**: `1 − (σ − σ_floor)/(σ_max − σ_floor)` with σ the std of the level distribution, σ_max = (L−1)/2, and σ_floor = √ordinal_smoothing = √0.1 (`R:schema.py:141-182`) | 1.479 / 0.863 / 0.911 / 0.911 / 1.834 |

(T values from `HF:*/strands_decider_config.json` `temperature_by_kind`.) The score-confidence choice is argued at length:
- Max-probability "reads spread as doubt". It also ranks bimodal distributions above uniform ones.
- Without the smoothing floor, "a 3-level score capped at 0.68 confidence" (`R:docs/architecture.md`).

Jev uses the mean absolute deviation from the mode instead (`jev-reference.md` §2).

### Calibration

- Calibration is post-hoc temperature only, fitted **per primitive** by `fit_temperature_by_kind`:
  - **objective ECE**, not NLL. "Fitting noul to NLL… chose T = 2.63, which left it under-confident by 0.29";
  - a **60-point log grid on [0.25, 6]**, which is why the same values recur across models (e.g. 0.8630, 0.5919, 0.9107): ECE is piecewise constant, so golden-section search fails on it;
  - at least 200 rows per kind, falling back to the global `temperature` (`R:training/evaluate.py:17-21,223-255,348-380`).
- Data: the calibration half of `data/holdout_v5_norule.jsonl`, a deterministic calib/test split of four held-out short tasks (emotion, MASSIVE intent, sarcasm, hate-speech severity) (`R:training/recipe.sh:194`, `HF:*/eval/internal/recipe_heldout.json` `by_task`).
- The soup resets the temperatures to 1 and is then calibrated as a fresh checkpoint (`R:training/soup.py:18-19,172`).
- Per-row temperatures let one batch mix types (`R:modeling.py:328-342`, `R:infer.py:186-194`).
- The card's own caveat: "Calibration is one temperature per primitive, fitted on held-out short classification… measure on your own traffic before you trust a threshold." Noul ECE on the held-out test half is 0.115–0.190 across sizes, against 0.030–0.043 for choice (`HF:*/eval/internal/recipe_heldout.json`).

### Limits

| | value | source |
|---|---|---|
| Context (`max_length`) | 4,096 tokens per row (state + one question). The Decision Index runs used a 32,768 window | `HF:*/strands_decider_config.json`; DI row in each card |
| Options trained | ≤ 24 (`num_slots`). The corpus is built with `--max-options 24`, and the KL tables are 24 wide | `R:modeling.py:32-36`, `R:training/train.py:281-343` |
| Options accepted | choice 2..255, score 2..10. The pointer path skips the slot check, so above 24 options it extrapolates (inferred: untrained) | `R:schema.py:30-32,66-90`, `R:infer.py:383-393` |
| Questions per request | ≥ 1, unbounded; chunked by 32 | `R:schema.py:99`, `R:infer.py:60` |
| Images | Qwen3.5-2B only (`serve --vision`): the base's vision tower with the adapter and head unchanged, never trained on images. Gemma releases are text only | `HF:2B/README.md` Images, `R:README.md` |

---

## Training

**Objective** (`R:training/train.py:1-8,281-372,600-655`, `R:docs/architecture.md` Training):

```
L = CE(gold over masked slots; score targets with 10% mass on adjacent levels = ordinal_smoothing 0.1)
  + 0.3 · KL_fwd(ref || student)                 ref = hobson-v19 checkpoint at T = 1 (kl_frozen_reference)
  + 1.0 · KL_fwd(teacher || student)             only rows that have a teacher distribution
```

- **Frozen-KL anchor.**
  - By default the reference is the adapter-disabled torso's own readout of the option-number tokens "1".."9", read through the tied input embedding (with Gemma's logit soft-cap of 30 applied).
  - In v1 it is replaced by the released **hobson-v19** checkpoint (`kl_frozen_reference: checkpoints/v19-ref`). This is a Qwen3.5-2B model, even for the Gemma students. The KL is "over option slots", so another tokenizer works.
  - It is precomputed before step 1, and only rows with ≤ 9 options are eligible (`R:modeling.py:484-569`, `R:training/train.py:76-92,281-343,566-646`).
- **Teacher.**
  - **2B v1**: `google/gemma-4-31B-it` distributions on noul/choice rows where its argmax equals gold. Multi-step rows keep "an earlier model's own distributions" (replay, `data/replay.py`). Generated-document and adequacy rows use gold only (`HF:2B/README.md` Training data).
  - **Gemma v1s**: a frozen `Qwen/Qwen3.5-4B` read without training, where it agrees with gold (`teacher_v20.jsonl`, `HF:26B-A4B/README.md`).
  - History: distilling the 4B on short classification "moved nothing" (v12); on multi-step documents it helped (v14) (`R:docs/architecture.md`).
- **Augmentation**:
  - option order re-shuffled every example and every epoch, with the label remapped. This is "the one detail [that] matters more than any of the loss terms";
  - score rubrics only **reversed**, with p = 0.5 (`reverse_score_prob`), never permuted;
  - length-grouped batches (`R:docs/architecture.md`, `HF:2B/train_config.json`).
- **Optimiser** (all v1 seeds): AdamW, lr 1e-4 (LoRA), **head_lr 1e-3**, wd 0.01, warm-up 3%, clip 1.0, 1 epoch, micro 8 × accum 4. Gemma uses gradient checkpointing. A GaLore option exists but is unused (`HF:*/train_config.json`).

**Data**

| | 2B-qwen3.5 v1 (`g28`) | Gemma 4 v1s (`g26`) |
|---|---|---|
| Rows | **246,678** in 9 files: short tasks 80,449 (v21's minus AG News, Yahoo, Yelp and language-ID, dropped for non-commercial terms); multi-step docs 12,909 (ContractNLI, MuSiQue, BoardgameQA); generated docs 3,815; adequacy 6,166 (HelpSteer2 + generated); code 24,533; gap fill 96,638 (tool choice, agent steps, preference, legal, routing, NLI, MC, 119 sources of `tasksource/tasksource-jev-typed-decisions`); rule application 22,168 | **135,602** = hobson-v21's 123,339 (`train_v5`, which still includes the NC sets per the file name, inferred) + 12,263 code rows (`code_v27`) |
| Teacher | gemma-4-31B-it (agree-with-gold) | Qwen3.5-4B (agree-with-gold) |
| Hardware | 8× A100 80 GB (p4de.24xlarge), ~79 min per seed | 8× H100 80 GB, 4,109 steps per seed |
| Source | `HF:2B/README.md`, `HF:2B/training/sources.md`, `HF:2B/train_config.json` | `HF:*-gemma4*/README.md`, `HF:*/train_config.json` |

- **Rule-application rows** (2B only): the rules are templates; gemma-4-31B-it wrote the policy text and the cases; the gold is computed by code. **Claude was used only as a filter**: a row was kept where both Gemma's and Claude's answers matched the code, and "Claude's text is not in the data" (`HF:2B/README.md`).
- **Decontamination**: a 13-gram overlap check against JevBench public, the Decision Index suite, HotpotQA dev and the internal held-out sets. Rows over 4,096 tokens are dropped (`HF:2B/README.md`). Calibration and eval only: emotion, MASSIVE, sarcasm, measuring-hate-speech, HotpotQA (`HF:26B-A4B/README.md`).
- Synthetic generators use OpenRouter or Amazon Bedrock clients (`R:data/generators/llm_client.py`). Committed synthetic files and model labels carry SHA256 sums (`R:data/SHA256SUMS`, `HF:2B/training/data_sha256.txt`).
- **Reproducibility caveat**: "The builders of this data mix are not yet [on main], so the training data cannot be rebuilt from `main`" (2B card). The research recipe (v19) is fully scripted (`R:training/recipe.sh all`): about 11 h on one RTX 3090, or 1 h 10 min on 8×H100 (`R:README.md` Training).
- **Process**: preregistrations from v9 to v20 record predictions and failure conditions before each run, with outcomes appended afterwards. "Five runs were promoted by the maintainers' decision anyway" (`R:README.md` The research, `R:research/preregistrations/`).

---

## Evals

**Card numbers** (served through `main` on the files; JevBench = third-party 231 public tasks, window 4096, tiers easy 48 / standard 72 / hard 111; `HF:*/eval/summary.json`, `HF:*/eval/jevbench-w4096/run_meta.json`):

| | E2B | E4B | 12B | 26B-A4B | 2B-qwen3.5 |
|---|---|---|---|---|---|
| JevBench acc (n/231) | 0.779 (180) | 0.810 (187) | 0.866 (200) | **0.896 (207)** | 0.779 (180) |
| JevBench Brier / ECE | 0.313 / 0.053 | 0.243 / 0.037 | 0.173 / 0.061 | 0.162 / 0.051 | 0.280 / 0.072 |
| tiers easy / std / hard | 48 / 68 / 64 | 48 / 69 / 70 | 48 / 71 / 81 | 48 / 72 / 87 | 48 / 67 / 65 |
| held-out short tasks acc / ECE (6,000) | 0.652 / 0.068 | 0.684 / 0.054 | 0.708 / 0.047 | 0.708 / 0.050 | 0.653 / 0.054 |
| … by kind: noul / choice / score acc | 0.611 / 0.738 / 0.477 | 0.686 / 0.754 / 0.514 | 0.727 / 0.752 / 0.586 | 0.766 / 0.755 / 0.547 | 0.644 / 0.728 / 0.477 |
| held-out code tasks | 0.773 | 0.815 | 0.870 | 0.859 | 0.836 (1,996) |
| **Decision Index 0.2.1** (card, window 32,768, balanced skill) | 29.78 (pre-fix, "may be off") | 39.42 | 47.14 | **48.69** | 30.82 (training harness) |
| median / p95 latency per JevBench question | 102 / 318 ms (H100) | 139 / 531 ms | 136 / 340 ms | 245 / 440 ms | 70 / 467 ms (L4) |
| weights GPU memory (bf16) | ~9 GB | ~15 GB | ~23 GB | ~51 GB | – |

- **Decision Index board (0.3.2, `notes/hf-raw/_spaces/dix_leaderboard.md`).** No `amazon/*-v1-2610` model is on the board. The only Strands entry is **rank 135, 21.34, ECE 0.044, "Strands Decider 2B" = `StrandsAgents/strands-decider-2B-hobson-v21`, kind "LoRA + head"**. Related rows: `Mapika/decider-2b` 25.80, Kev 4B 39.5, Jev 1.14.0 61.53, Jev 1.13 60.11. The cards' DI 0.2.1 figures are not comparable to the 0.3.2 board.
- **Seed soup vs seeds (2B)**: the soup's 180 sits within the seed spread (178/177/184, mean 179.7 ± 3.8), while its Brier of 0.280 is better than every seed (0.305/0.288/0.289). The soup is "at or above its seeds' mean on most measures". The run-to-run SD of about 3–4 tasks means "treat a difference under about 10 tasks between two single runs as unresolved" (`HF:2B/README.md`, `R:README.md` Performance).
- **2B v1 vs v19**: 180 vs 167, McNemar p = 0.024; the gain is all in the standard (+4) and hard (+9) tiers (`HF:2B/eval/jevbench-w4096/paired.txt`).
- **Confidence bands** (2B held-out short tasks): p ≥ 0.9 → acc 0.972 (n 1,429); 0.5–0.9 → 0.686; < 0.5 → 0.462 (`HF:2B/eval/internal/recipe_heldout.json`).
- **Multi-step and adequacy, 2B / 26B-A4B**: MuSiQue 0.882 / 0.922, ContractNLI 0.878 / 0.870, BoardgameQA 0.810 / 0.869, HotpotQA (never trained) 0.832 / 0.897, HelpSteer2 adequacy 0.709 / 0.808 (`HF:*/README.md`). Note that the train splits of MuSiQue, ContractNLI and BoardgameQA are in the training data.
- **Vision (2B, untrained on images)**: NaturalBench 0.776, ECE 0.039; POPE-adv 0.872, ECE 0.023. With the image removed it is still confident (POPE ECE 0.334): "Do not use its confidence to detect a missing… image" (`HF:2B/README.md` Images).
- **Documented weaknesses** (all cards):
  - "Questions are read less than documents": a changed question often gives the same answer;
  - the long multi-step hard tier;
  - poor transfer of `score` and `noul` to unfamiliar rubrics;
  - calibration only established on short classification.
- **Against neighbours on JevBench** (`R:evaluation/jevbench.md`):
  - v19 0.723 vs decider-2b 0.710;
  - Open-Jev 2B 0.645, jeff 400M 0.628, Laya 421M 0.584, Open-Jev 9B 0.775;
  - frozen bases read through the chat template: Qwen3.5-2B 0.658, Qwen3.5-4B 0.805 (`R:research/generations.md`).

---

## Serving

- **API**: FastAPI `POST /v1/systemone` with the request `{state, questions{id: {type, instructions, criteria}}, model, images[]}` (`R:src/strands_decider/server.py:98-110`, `R:schema.py:96-138`).
  - The response is `{model, answers, usage{input_tokens, output_tokens = number of questions}, latency_ms}`. `model` is the checkpoint basename, and the request's `model` field is ignored (inferred from `create_app`).
  - `GET /health` reports base, slots, max_length, temperature, device and `prefix_cache`. There is **no `/v1/models`**.
  - It binds `127.0.0.1`, with **no auth** and a single uvicorn worker. "JevBench's typesafe adapter runs against it unchanged. Compatibility with the Jev API itself is not verified" (`R:schema.py:3-5`).
  - `instructions` and `state` accept string, object or array; `criteria` values may be structured or null, as in Jev.
- **Engines**:
  - torch on CUDA, MPS or CPU. There is a custom Gated DeltaNet chunk kernel for MPS, because fla does not run on macOS, and a depthwise-conv kernel for CPU;
  - **MLX** (`--device mlx`, 1.4–1.6× faster than MPS) (`R:src/strands_decider/{mps_kernels,cpu_kernels,mlx_engine}.py`, `R:README.md`);
  - a CLI: `strands-decider ask|serve|info`.
- **Many questions are nearly free** (measured on the Qwen3 v7 torso): 1 question takes 219 ms and 6 questions 217 ms over the same state (666 vs 966 input tokens). Over a ~2,000-token state, 16 questions take 445 ms through the shared prefix against 4,086 ms batched (`R:docs/inference.md` "Asking many questions is nearly free").
- **Latency**: v19 2B on an RTX 3090 has a median of 115 ms and p95 of 299 ms per JevBench question; on an M3 Pro, 153 ms warm under 300 tokens (`R:README.md` Performance). The v1 release numbers are in the eval table above.
- **Integration**: `examples/strands/tool_call_intervention.py` is a Strands Agents `before_tool_call` hook that gates a weather-tool call on two noul questions (`R:README.md`).
- **Checkpoint format**: `strands_decider_config.json`, `lora/` (PEFT), `head.safetensors`, the tokenizer, `provenance.json` and a `MANIFEST.sha256` (verified with `python -m strands_decider.hf_export verify`). The base weights download from the Hub at load, pinned to the provenance revision (`R:modeling.py:696-704`). The adapters are **not merged**, so no vLLM or TGI path exists (inferred: there is no merged export or modelling file).

---

## Licence

- Code and all five adapters and heads: **Apache-2.0** (`R:LICENSE`, `HF:*/api.json cardData.license`, `HF:*/LICENSE.md`).
- Bases: Gemma 4 `-it` and Qwen3.5-2B-Base, both Apache-2.0 per the README (`R:README.md` Gemma 4 models).
- Data: the 2B v1 mix **dropped the four sources whose terms restrict commercial use**. The Gemma releases train on v21's data, which still contained them (inferred from the 2B card's wording and the `train_v5` vs `train_v5_nc` file names). Every source's revision and licence is listed in `HF:2B/training/sources.md` and `R:data/sources.md`.

---

## Notable design choices vs Jev

1. **Hume's pointer hypothesis, implemented directly.** The options keep their causal reading order and are read jointly, as Jev appears to read them. The head has no per-slot parameters. Jev has some first-option bias (jaggedness #8); Strands' residual order sensitivity is about 0.016.
2. **Question isolation is exact.** Each question is a separate causal row over a forked state cache, matching Jev's "add or remove questions without changing the others". KV-plus-recurrent cache forking makes this work on a hybrid DeltaNet torso.
3. **A much smaller window**: 4,096 tokens per row, against Jev's 32k per question and 64k per request. The engine truncates the *state* to make room for the question. Jev returns `max_tokens_exceeded` instead.
4. **Confidence**: choice uses Jev's formula, but score uses a std-based formula with a smoothing-floor correction, not Jev's MAD-from-mode. Score confidences are therefore not numerically comparable with Jev's.
5. **Per-type temperature, fitted to ECE** rather than NLL. This is an explicit stance: probabilities should be good for threshold routing. Jev calls its method RLCD and publishes no recipe.
6. **No box primitive and no `/v1/models`.** Images come only through the untrained Qwen3.5 vision tower, and the field is `images: [base64]`, not Jev's `assets` map.

## Notable design choices vs kev

1. **Delimiters**: kev reuses five reserved Qwen special tokens and sanitises user text. Strands uses plain-text pseudo-XML and numbered lines. A user can forge `<options>` text, which kev prevents (inferred: no sanitising code in `prompting.py`).
2. **Head**: the same q·k/√d pointer, but Strands adds a LayerNorm and keys on the **last token of each option line**, not a `</opt>` token. It trains the head at a 10× lr.
3. **Isolation**: on hybrid bases both now use "one causal row per question continuing a cached prefix". Strands' fork credits decider-2b. Strands never used kev's packed block-causal mask.
4. **Regularisation**: kev's `--anchor_w` (KL to the frozen base) was never released. Strands ships a KL anchor at 0.3, first to the frozen torso's own number-token readout and in v1 to its own earlier release (v19). It adds agree-with-gold distillation from a larger teacher.
5. **Seed soups with exact merging**: LoRA ΔW are averaged by rank concatenation, and heads by stacking in q/k space. kev's 27B used WiSE-FT between SFT and v1 instead.
6. **Per-type temperatures** (kev ships one T in `head.pt`), and score confidence by std rather than kev's MAD formula.
7. **Multiple base families from one codebase.** Gemma 4 `-it` bases are supported, with BOS forcing, per-layer embedding offload, KV-shared layers and soft-cap handling. Kev stayed on Qwen.
