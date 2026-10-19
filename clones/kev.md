# kev (Jared Palmer)

Open replication of TypeSafe's Jev. It is a frozen Qwen causal-LM backbone run prefill-only, with a shared state prefix and isolated question branches, read out by a pointer head over option-boundary tokens. It serves TypeSafe's `POST /v1/systemone` contract.

Source pins used for this profile:

| ref | what | where |
|---|---|---|
| `L` = local checkout `00aa32d1` (2026-09-19, "v7 final plan: 8B x2, 4B seed 2") | read-only, the commit this profile was asked about | `/home/ai/projects/kev` |
| `U` = upstream HEAD `1f04cfa2` (2026-10-15) | shallow, blob-filtered clone, never executed; 246 commits after `L` | `/home/ai/projects/jev-clones/sources/kev` |
| HF | API JSON, READMEs, `adapter_config.json`, `training_config.json`, `training_metrics.json`, `provenance.json`, 27B `config.json`/`interpolation.json` (no weights) | `/home/ai/projects/jev-clones/notes/kev-hf/` |

Citation form: `L:kev/model.py:30` means file:line in the pinned checkout, and `U:` means upstream HEAD. `HF:kev-8b@main/training_config.json` is a fetched file.

> **Main finding.** At `00aa32d1` kev is the **Qwen2.5/Qwen3 attention-only family** (0.5B, 0.6B, 4B, 8B) with a single packed sequence and a block-causal mask. Upstream has since moved to **Kev 1.0** (released 2026-10-01): Kev-0.8B, 4B and 9B on **Qwen3.5 hybrid Gated-DeltaNet** bases, plus a **full-weight Kev-27B on Qwen3.8-27B**. Recurrent layers cannot honour a block-causal mask, so these models drop the packed mask and run **one causal row per question that continues a cached state prefix**. `jaredpalmer/kev-4b` main is now Qwen3.5-4B; the Qwen3-4B weights are at tag `qwen3`. kev-0.6b and kev-8b are kept as the "previous generation".

---

## Identity

| | |
|---|---|
| Repo | github.com/jaredpalmer/kev (Apache-2.0). Author Jared Palmer, "built with Devin (Cognition)" (`L:MODEL_CARD.md:50`, `L:README.md:369`) |
| Weights (HF) | `jaredpalmer/kev-0.5b`, `kev-0.6b`, `kev-4b`, `kev-8b`, and upstream also `kev-0.8b`, `kev-9b`, `kev-27b`. Collection `jaredpalmer/kev-6aad9d0ea49f2589665e07cd`. Eval suites: dataset `jaredpalmer/kev-suites` |
| Lineage | Architecture follows Archer Hume, *Jev's Architecture Unmasked* (archerhume.com/posts/jevs-architecture-unmasked) (`L:MODEL_CARD.md:39`, `L:README.md:17`); API follows docs.typesafe.ai/api |
| Status at `L` | kev-0.5b = v0.1 release; 0.6b/4b/8b = "research preview", no version tag, because each fails the predeclared 70% held-out-policy-pair screen (`L:README.md:49-57`, `L:PLAN.md:145-149`) |
| Status at `U` | Kev 1.0: 0.8B / 4B / 9B / 27B tagged `v1.0` on each Hub repo, GitHub release `kev-1.0` (`U:README.md:54-65`, commit `61b041b`) |
| HF repo heads (2026-10-19) | kev-0.5b `9ce2fd39` (tag v0.1); kev-0.6b `dece6dba` (tag qwen3); kev-4b `6cfce5c2` = Qwen3.5-4B (tags qwen3, v7-base, night2-du-release, r8-documents-release, v1.0); kev-8b `c80773da` (tag qwen3); kev-0.8b `bf75a6a8`; kev-9b `db029f08` (tags v1, v1.0, branch night2-du); kev-27b `af0e6d55` (tags v1-lora, v1.0) |

---

## Architecture

### Backbone

| Checkpoint | Backbone (revision) | Layers / type | Trained params | Dtype | Source |
|---|---|---|---|---|---|
| kev-0.5b v0.1 | `Qwen/Qwen2.5-0.5B`, frozen | 24 attn, d=896 | LoRA 8.8M + head 0.46M = 9.3M (1.9%) | fp32 train + serve (MPS) | `L:MODEL_CARD.md:52-56` |
| kev-0.6b | `Qwen/Qwen3-0.6B-Base` @`da87bfb6` | attn-only | LoRA r16 + head | bf16 autocast, fp32 masters | `HF:kev-0.6b@main/training_config.json` |
| kev-4b (Qwen3, tag `qwen3`) | `Qwen/Qwen3-4B-Base` @`906bfd4b` | attn-only, d=2560 | LoRA r16 + head | same | `HF:kev-4b@qwen3/training_config.json` |
| kev-8b | `Qwen/Qwen3-8B-Base` @`49e3418f` | attn-only | LoRA r16 + head | same; serve bf16 on a 32 GB Mac (~17 GB), fp32 needs ~33 GB | `L:docs/model-cards/kev-8b.md:50,87`, `HF:kev-8b@main/training_config.json` |
| Kev-0.8B (U) | `Qwen/Qwen3.5-0.8B-Base` @`dc7cdfe2` | 24 = 18 Gated DeltaNet + 6 full attn | LoRA 11.3M + head | bf16 autocast over fp32 weights; serve bf16 (merged); eval fp32 | `U:docs/model-cards/kev-0.8b.md` |
| Kev-4B (U) | `Qwen/Qwen3.5-4B-Base` @`1001bb4d` | 32 = 24 DeltaNet + 8 attn, d=2560 | LoRA 33.8M + head | same | `HF:kev-4b@main/README.md` |
| Kev-9B (U) | `Qwen/Qwen3.5-9B-Base` @`68c46c4b` | 32 = 24 DeltaNet + 8 attn, d=4096 | LoRA 45.4M + head | same | `U:docs/model-cards/kev-9b.md` |
| Kev-27B v2 (U) | `Qwen/Qwen3.8-27B` @`1d4bf0f2` (**post-trained**, not -Base) | 64 = 48 DeltaNet + 16 attn, d=5120, 24 q-heads / 4 KV, head_dim 256, partial RoPE 0.25, window 262,144 | **all 25.6B backbone weights** + 2.6M head (vision tower, LM head, MTP not loaded) | bf16 weights (51.3 GB) | `HF:kev-27b@main/config.json`, `U:docs/model-cards/kev-27b.md` |

- The backbone is loaded **without its LM head** (`AutoModelForCausalLM(...).model`), because kev never generates (`L:kev/model.py:118-122`). Attention is SDPA on CUDA and eager on MPS/CPU (`L:kev/model.py:120`).
- **LoRA**: `r = --lora` (default 16), **`lora_alpha = 2r`** (=32), dropout 0.05, `task_type=FEATURE_EXTRACTION` (`L:kev/model.py:130`). Targets for `all`: `q,k,v,o,gate,up,down_proj` (`L:kev/model.py:128-129`). On hybrid bases upstream adds the DeltaNet projections `in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj` (`U:kev/model.py:268-274`). This matches the HF `adapter_config.json` of every LoRA checkpoint (`HF:*/adapter_config.json`: r 16, alpha 32, no DoRA/rsLoRA). Other options: `attn`, `qv`, and on U `dense` (freezes the DeltaNet projections).
- Delimiter embeddings are **not** trained by default. `--special_embeddings 1` would use peft `trainable_token_indices` on the 5 tokens (`L:kev/model.py:127`), but it leaked memory on MPS and gave no gain (`L:AGENTS.md` Notes, `L:PLAN.md:168`).
- Training dtype: fp32 by default, or `--dtype bf16` autocast with fp32 master weights (CUDA only) (`L:kev/train.py:65`). The head always runs in fp32 (`L:kev/model.py:156`). U adds `--weights_dtype bf16` for a frozen bf16 backbone and `--full_ft 1` (MasterAdamW, fp32 masters + moments, FSDP2) (`U:kev/train.py:392,418`).

### Input encoding (sequence packing)

The API request is first flattened to text by `api.to_record`. JSON objects and arrays become labelled text, and field names are kept (`L:kev/api.py:46-52,59-75`). By question type:
- **noul** becomes 2 options `["no[: desc_false]", "yes[: desc_true]"]` (`L:kev/api.py:64-66`).
- **choice** becomes `"name"` or `"name: desc"` per criteria key, in request order (`L:kev/api.py:68-69`).
- **score** becomes the rendered level texts, in order (`L:kev/api.py:71-72`).

The instruction text is `render(instructions)`. Training data goes through the same `to_record` path (`L:kev/data.py:350-359`), so train and serve text are byte-identical.

Five **reused Qwen special tokens** act as delimiters. No embedding rows are added (`L:kev/model.py:8-10`):

| role | token |
|---|---|
| `<state>` | `<\|fim_prefix\|>` |
| `<q>` | `<\|fim_middle\|>` |
| `<opt>` | `<\|box_start\|>` |
| `</opt>` | `<\|box_end\|>` |
| `<decide>` | `<\|fim_suffix\|>` |

User text cannot forge them: `user_tokens` rewrites any `<|name|>` to `<¦name¦>` before tokenizing (`L:kev/model.py:18-24`). No chat template and no BOS are used (`add_special_tokens=False`).

`encode()` (`L:kev/model.py:30-67`) produces:
- `ids = [<state>] + state[:max_state-1]`, then per question k: `[<q>] + instr + Σ_j ([<opt>] + option_j + [</opt>]) + [<decide>]`.
- `seg`: 0 for state tokens, k for every token of question k.
- `pos`: the state uses 0..Ls-1. **Every branch restarts at Ls** (`p0 = len(S)`, `br_pos = range(p0, p0+len(br))`) (`L:kev/model.py:54,60`). Every question therefore sees "state, then exactly one question", and question order is irrelevant.
- `opt_idx[k][j]` = index of option j's `</opt>`, and `decide_idx[k]` = index of `<decide>` (the last token of the branch) (`L:kev/model.py:61-65`).
- Optional `option_isolation=True`: every option span gets the **same** position ids (starting after the instruction), and `<decide>` sits at `p0+len(instr)+longest` (`L:kev/model.py:56-58`). This makes the option representations permutation-invariant by construction (measured flip rate 0.0 / 1.2e-7) (`L:PLAN.md:93-95,115`). Default is off, because it cost 5.8 pp transfer at 4B (`L:PLAN.md:176`). It is not available on hybrid bases (`U:kev/model.py:263`).

### Attention mask

`branch_mask_batch` (`L:kev/model.py:75-101`):
- `attend(i,j) = j ≤ i  ∧  (seg[j] == 0  ∨  seg[j] == seg[i])  ∧  j is not padding`. The diagonal is always allowed, so padded rows are never empty.
- It is additive, using `finfo(dtype).min` (not −inf), with shape `[B,1,L,L]`, right-padded. The mask goes straight into HF as `attention_mask` together with explicit `position_ids` (`L:kev/model.py:155-156`).
- With option isolation, an option-span token sees state + instruction + **its own span** only, and `<decide>` sees everything in its question (`L:kev/model.py:92-99`).

```
Packed sequence (attention-only bases: kev-0.5b/0.6b/4b@qwen3/8b)

idx  0       1  2  | 3   4  5     6  7      8     9  10     11      | 12  13 14    15 16     17    18 19     20
tok  <state> s  s  | <q> I  <opt> A  </opt> <opt> B  </opt> <decide>| <q> J  <opt> C  </opt> <opt> D  </opt> <decide>
seg  0       0  0  | 1   1  1     1  1      1     1  1      1       | 2   2  2     2  2      2     2  2      2
pos  0       1  2  | 3   4  5     6  7      8     9  10     11      | 3   4  5     6  7      8     9  10     11   <- restarts
                                   ^key          ^key       ^query                  ^key         ^key  ^query
                                   (opt 0)       (opt 1)    (Q1)                    (opt 0)      (opt 1)(Q2)

Block-causal mask (rows = queries, cols = keys; '\' = causal lower triangle, '#' = all visible, '.' = blocked)

             state   Q1 branch   Q2 branch
   state   [  \        .            .     ]
   Q1      [  #        \            .     ]
   Q2      [  #        .            \     ]

Row form (hybrid Qwen3.5/3.8 bases, every Kev 1.0 model): the mask is gone, each question is its own causal row
   row 1:  <state> s s  <q> I <opt> A </opt> <opt> B </opt> <decide>     (pos 0..11)
   row 2:  <state> s s  <q> J <opt> C </opt> <opt> D </opt> <decide>     (pos 0..11)
   serving / probs(): state run once -> KV + DeltaNet conv/recurrent state cached -> rows continue from a replica
```

**Upstream change (hybrid bases).** `is_hybrid()` looks for `linear_attention` in `config.layer_types`. If found, the model "cannot honour the block-causal mask" and runs the **row form** (`U:kev/model.py:69-72,259-264`). `rows_of(enc)` splits a packed encoding into the state plus per-question branch rows, with positions already continuing the state (`U:kev/model.py:188-202`). Three variants exist:
- `forward_rows_batch` recomputes the state once per row. This is what `kev.benchmark` scores, and every published number comes from it (`U:kev/model.py:364-383`, `U:README.md:316`).
- In serving, `probs()` runs the state once (`prefix()` with a `DynamicCache` holding KV plus DeltaNet conv and recurrent states). The branch rows then continue from a copied replica of the cache (`U:kev/model.py:335-362,408-426`).
- In training, `--shared_prefix` (`kev/shared_prefix.py`) runs each record's state once per decoder layer and lets branches read a functional `Prefix` (keys/values, conv window, recurrent `initial_state`). Gradients flow back into the state. It matches the row form to ~1e-5 in fp32 (`U:kev/shared_prefix.py:1-20,99-132`).

Attention-only bases also switch to rows once a packed sequence exceeds `ROW_PASS_TOKENS = 16384` (`U:kev/model.py:23,329-333`). The packed and row forms agree to fp32 noise (`U:README.md:316`).

### Readout head

`PointerHead` (`L:kev/model.py:104-112`):

```
q = W_q · h[<decide>_k] + b_q        W_q: d → dp (dp = --head_dim, default 256)
k_j = W_k · h[</opt>_kj] + b_k        W_k: d → dp
z_kj = (k_j · q) / sqrt(dp)           logits over the K options of question k
p_k = softmax_j(z_kj)                 (U: z / T at eval time, see Calibration)
```

- Option representation = the last-layer hidden state (after the final norm) of the option's **closing `</opt>` token**. The question representation = the hidden state of `<decide>`. `<decide>` comes after all options, so it has read the whole option list. The README credits this with making "none of the above" work (`L:README.md:208`).
- The head is trained from scratch. Head-dim 1024, a separate `head_lr` and weight decay made no difference (`L:PLAN.md:175`). U adds `PointerHead.many()` for batched serving and a `temperature` attribute (`U:kev/model.py:205-223`).
- K is whatever the request sends, so one head serves all three question types.

### Question types

| type | options fed | answer (`to_answers`, `L:kev/api.py:94-105`) | confidence |
|---|---|---|---|
| `noul` | `[no, yes]` (+ optional descriptions from `criteria.{false,true}`) | `noul = p[1]` | none |
| `choice` | criteria keys (`name` or `name: desc`), 1..255 | `choice = argmax`, `probabilities` by key | `(p_max − 1/K)/(1 − 1/K)` (`L:kev/api.py:78-80`) |
| `score` | ordered level texts, 2..255 at L (U: 1..255) | `score = Σ i·p_i` (expected level), `legend`, `probabilities` by index | L: stand-in `1 − E|i − mode|/(L−1)` (`L:kev/api.py:83-87`). U: `max(0, 1 − E|i−mode|/D)` with D = uniform MAD, "mirror[ing] TypeSafe's reference adapter system-one-adapter 0.2.1" (`U:kev/api.py:120-140`) |

Losses by type (`L:kev/train.py:23-31`): CE for all three types. Score optionally adds a **ranked probability score** term (`--ord_w`, CDF squared error, a proper scoring rule). It is off in every release. U adds **soft targets** (`−Σ t·log p` when the record carries `target`) and screened label smoothing, Brier and focal losses, none of which shipped (`U:kev/train.py:41-61`). Probabilities are rounded to 2 decimals at L and to 4 at U, so a 255-option distribution sums within TypeSafe's ±0.02 (`U:kev/api.py:143-146`).

### Context & limits

| | L (`00aa32d1`) | U (Kev 1.0) |
|---|---|---|
| Training state / branch / packed | 384 / 1,024 / 2,048 (`L:kev/model.py:11`, `L:kev/train.py:175-176`) | same defaults (`U:kev/model.py:14`); `--max_state` up to 65,536, branch and packed grow with it (`U:kev/model.py:31-38`). Small family's delta stages trained at ≤ 7,552; 27B at ≤ 32,768 |
| Serving | 8,192 state / 8,192 per branch row (`L:kev/serve.py:15-16`; comment: "mirrors Jev's ~32k, bounded by the base model window") | state ≤ 65,536, state+branch ≤ 73,728 (`U:kev/model.py:18-20`); over-long state gets a **422** naming its length (`KEV_TRUNCATE_STATES=1` to cut) (`U:kev/model.py:130-142`, `U:kev/serve.py:136-142`) |
| Validated context (CUAD within 3 pp of 8k at the 95% lower bound) | n/a | 8,192 for 0.8B / 4B / 9B, 65,536 for 27B (`U:README.md:30-36`, commit `3fe301d`) |
| Options per question | 1..255 choice, 2..255 score (`L:kev/api.py:11,27,34`) | 1..255 for both (`U:kev/api.py:14,30,37`) |
| Questions per request | ≥ 1, no upper bound in code (`L:kev/api.py:43`) | no limit; rows are batched `rows_per_pass` at a time under a 16,384-token budget (`U:kev/model.py:41-47`) |
| Batching | serving: one request at a time behind a global `threading.Lock` (`L:kev/serve.py:20,48`); training: padded `--batch` records + grad accumulation (`L:kev/train.py:64`) | `Server` model thread batches queued requests, LRU state-prefix cache (4 states / 65,536 tokens), CUDA graphs and fused fla DeltaNet kernels on CUDA, MLX on Apple Silicon (`U:kev/serve.py:39,94-113`, `U:AGENTS.md` Serve) |

### Calibration

- **L.** No temperature is shipped or applied in serving: `serve.py` returns the raw softmax (`L:kev/serve.py:44-53`). Temperature is only an evaluation statistic:
  - `kev.evaluate.test_temperature` fits a global T on even records and tests on odd ones, using LBFGS on log T (`L:kev/evaluate.py:202-211`).
  - The research harness fits T on the suite's **calibration partition** over an 81-point log grid 0.25..4, minimising macro per-task NLL, then applies it unchanged to dev/transfer (`L:kev/benchmark.py:114-118`, `L:kev/experiment.py:205-216`, `L:PLAN.md:73`).
  - kev-0.5b: T = 1.47, ECE 0.057 → 0.031 in-distribution (`L:MODEL_CARD.md:148-150`).
  - The PLAN states outright that "temperature fitted in-domain does not transfer" (`L:PLAN.md:15`, `L:README.md:340`).
- **U.** The **temperature is built into the checkpoint**: `head.pt["temperature"]`, applied by `PointerHead` in eval mode only, so the argmax is unchanged (`U:kev/model.py:211-218`, commit `3aa96e5`). `KEV_TEMPERATURE=1.0` returns raw probabilities.
  - Shipped values: 0.8B T = 2.35 and 4B T = 2.41, both fitted on in-distribution dev rows. 9B v2 T = 2.19 and 27B v2 T = 1.32, both fitted on a **held-out-datasets pool** (648 questions: transfer-r3 calibration + 200 MMLU-Pro) (`U:README.md:59-63,387`).
  - `scripts/calibrate_checkpoint.py` now refuses rows that share data with training unless `--allow-in-distribution`, and reports an out-of-fold grouped-CV ECE (`U:AGENTS.md` Current family).
  - Effect at 9B on new sources: ECE 0.103 → 0.041, confident errors 8.2% → 2.4% (Jev 3.7%) (`U:README.md:387`).

---

## Training

**Data and labels** (all hard labels from public datasets or code; **no Jev outputs**, `L:docs/model-cards/kev-8b.md` Training, `U:README.md:322`):
- Public sources are converted into TypeSafe-shaped requests (`L:kev/data.py:90-147,160-262`).
  - Trainable: banking77, boolq, agnews, mnli, sst5, yelp, trec, dbpedia14, amazon, imdb, arc, openbookqa, csqa.
  - **Eval-only** (enforced): mmlu, emotion, tweet_offensive, qnli, paws, sciq (`L:kev/data.py:279-281`, `L:kev/train.py` forbidden check).
  - Rendering variation: states wrapped as `{"document"}`, `{"ticket":{channel,body}}` or `[{"role","content"}]` (32%), plus null or structured option descriptions and structured instructions (`L:kev/data.py:56-73`, `L:MODEL_CARD.md:100`).
- **Programmatic policy data** with labels computed by code:
  - legacy policy minimal pairs (`kev/contrastive.py`);
  - compositional rule trees with numeric/date/AND/OR/NOT/exceptions and relevant vs irrelevant interventions (`kev/composition.py`, `L:PLAN.md:42-53`);
  - `decision-v7`: 10k public + 896 policy records (incl. 4 ordinal Score threshold families) + 1,680 records from 60 random rule trees with negation anywhere, held-out structures excluded by a canonical key (`L:PLAN.md:212-217`).
- **Augmentation**, re-drawn each epoch (`L:kev/data.py:311-333`, `L:kev/train.py:72-75`):
  - option shuffle, always;
  - p = 0.10: replace the true option with a "none of the above" option, which becomes the label;
  - p = 0.12: add a none option as a wrong alternative;
  - p = 0.15: add an irrelevant distractor;
  - `--p_none_pair 0.25`: also emit a present/absent none minimal pair (`L:kev/data.py:335-347`).
- **U delta stages** for 0.8B/4B/9B, via `--init_from` at lr 2e-5 with `--replay` of decision-v7:
  - dates + missing evidence (1,425 records);
  - `documents-v1`: CFPB complaints, labels kept where two open-weight teachers agreed with the consumer's own filing;
  - `hard-v1`: 7 programmatic skill families;
  - `devtools-v1`: CodeReviewer, CommitPackFT, FlakeFlagger, Aegis (`HF:kev-4b@main/README.md` Training data).
- **U 27B**: one epoch of full-weight SFT on a 145,840-record corpus (Kev data + document, skill and devtools suites + public sets + licensed task families + generated long-doc/tool/agent/guardrail records, states ≤ 32k). The result is then WiSE-FT-averaged, 0.85 × SFT + 0.15 × v1 (LoRA merged in fp32) (`U:README.md:330`, `HF:kev-27b@main/interpolation.json`).

**Loss and optimiser** (`L:kev/train.py`):
- Loss: CE per question, averaged over the questions of a record (`L:kev/train.py:186-187`). Optional extras:
  - `--perm_kl`: symmetric KL between two option orders (`:192-198`);
  - `--anchor_w`: KL toward the frozen base's zero-shot answers (`:34-39,188-191`);
  - `--ord_w`: RPS (`:23-31`).
  - All three default to 0 and none is in any release.
- Optimiser: AdamW, weight decay 0.01, OneCycle with 10% warm-up (`:156-159`); clip 1.0 (`:207`). The loss is weighted per source record in the accumulation group (`:202`).
- Saved artefacts: adapter via `save_pretrained`; `head.pt` = head state + base/revision/lora/head_dim/flags (`:214-218`).

**Per-checkpoint recipes:**

| ckpt | data | epochs | lr | batch × accum | steps | wall / hardware | src |
|---|---|---|---|---|---|---|---|
| kev-0.5b v0.1 | 6 sources × 1,500 = 9,000 rec / 13,500 q | 2 | 2e-4 | 1 × 8 | 2,250 | ~1h45m, Apple M5 32 GB MPS fp32 | `L:MODEL_CARD.md:89,106-117` |
| kev-0.6b (HF main) | decision-v7 (25,152 req) | 2 | 1e-4 | 8 × 1 | 3,144 | 638 s, H100 bf16, seed 2 | `HF:kev-0.6b@main/training_*.json` |
| kev-4b@qwen3 | decision-v7 | 2 | 5e-5 | 4 × 2 | 3,144 | 2,379 s, H100, seed 1 | `HF:kev-4b@qwen3/training_*.json` |
| kev-8b (HF main) | decision-v7 | 2 | 5e-5 | 2 × 4 | 3,144 | 4,972 s, H100, ckpt, seed 0 | `HF:kev-8b@main/training_*.json` |
| kev-8b per local card | decision-v6 (13k public + 2×448) | 2 | 5e-5 | eff. 8 | – | ~70 min H100 | `L:docs/model-cards/kev-8b.md:93` |
| Kev-4B 1.0 | v7 base → dates → documents → skills+devtools | 2 + 1 + 1 + 1 | 5e-5 then 2e-5 | 4×2 then 2×4 | last stage 1,915 | 56 min H100 + 9 min + 43 min H200 + 1.4 h H200 | `HF:kev-4b@main/README.md` |
| Kev-9B v2 | night2 ckpt + round-15 joint data, replay 10k | 1 | 2e-5 | 2 × 4 | 3,318 | 9,238 s | `HF:kev-9b@main/training_*.json` |
| Kev-27B v2 | 145,840 rec, full FT | 1 | 2e-6 backbone / 1e-4 head | 128 rec/step | 1,140 | 8 × H200, FSDP2 | `U:docs/model-cards/kev-27b.md` (HF README L180) |

**Research findings at L** (`L:PLAN.md:154-194`):
1. Backbone capacity dominates transfer: 0.6B → 4B is +14–19 pp, 4B → 8B is +1.5–2 pp.
2. **lr 2e-4 → 5e-5** gives +4.7 pp [+0.4, +9.6] at 4B, because the high lr erodes base knowledge (MMLU −11 pp at 8B vs the zero-shot base).
3. None-of-the-above pairs fix the "none" shortcut.
4. More public data raises dev accuracy and flattens transfer.

Option isolation, special embeddings, head_dim, perm_kl, ord_w and WiSE-FT toward the base all showed no gain. Spend for the overnight study was ~$180 of Modal H100 time.

---

## Evals

Suites are frozen, checksummed, with a locked test read once per candidate. Jev is scored on the same items through Vercel AI Gateway (`L:kev/jev.py`, listed $0.042/M input tokens, `L:kev/jev.py:13`).

**At L: transfer-v4 dev (out of domain, 764 records) / decision-v4 dev (in-distribution, 1,200 q)** (`L:README.md:49-55,308-314`, `L:docs/model-cards/*.md`):

| | kev-0.5b | kev-0.6b | kev-4b | kev-8b | Jev |
|---|---|---|---|---|---|
| in-dist acc (dev / locked test) | 0.712 / – | 0.805 / 0.819 | 0.843 / 0.852 | 0.869 / 0.869 | 0.845 / – |
| OOD acc (dev / locked test) | 0.575 / – | 0.598 / 0.631 | 0.759 / 0.794 | 0.774 / 0.799 | **0.857** / – |
| OOD Brier | 0.50 | 0.521 | 0.346 | 0.339 | **0.211** |
| confident errors (p≥0.9 & wrong) | – | 5.2% | 5.5% | 8.2% | 3.7% |
| held-out policy pairs both-correct | – | 0.11 | 0.62 | 0.61 | **0.86** |
| option-order argmax flips | 0.21 | 0.02–0.08 | 0.00–0.08 | 0.02–0.03 | 0.00 |

- kev-8b per source vs Jev: QNLI 0.93/0.93, SciQ 1.00/0.99, TweetEval 0.71/0.81, PAWS 0.76/0.79, **MMLU 0.69/0.90**, Emotion 0.57/0.59, **deadline 0.70/0.93** (`L:docs/model-cards/kev-8b.md:68`).
- kev-0.5b on its own 6-source held-out split: 0.799 acc / 0.065 ECE (`L:MODEL_CARD.md:133-146`).
- Mechanism tests for 0.5b (`L:MODEL_CARD.md:152-160`):
  - isolation: secret in sibling 0.03 / absent 0.03 / in state 0.99;
  - packed vs separate: max |Δp| 3.7e-6, 2× faster packed;
  - 7.4% argmax flips under 4 orders;
  - IIA: mean |Δ log-odds| 0.13;
  - boundary forgery: forged option p ≤ 0.09.
- After L, the **HF** kev-8b/0.6b/4b@qwen3 were re-published from **decision-v7** trials:
  - kev-8b `v7-final/00-trial-0`: OOD dev **0.796**, locked 0.780;
  - kev-0.6b: OOD dev 0.620, locked 0.642;
  - kev-4b@qwen3 (`v7-rc3`): 0.790.
  - The cards in the pinned checkout still describe the older v4/v6 trials (`HF:kev-8b@main/README.md:54-74`, `HF:kev-0.6b@main/README.md:51-76`, `L:PLAN.md:240-244`).

**At U: Kev 1.0** (dev / test; "new sources" = transfer-v4, "trained" = decision-v7; `U:README.md:28-46`):

| | Kev-0.8B | Kev-4B | Kev-9B v2 | Kev-27B v2 | Jev |
|---|---|---|---|---|---|
| new sources acc | 0.648 / 0.697 | 0.817 / 0.838 | 0.820 / 0.852 | **0.851 / 0.889** | 0.857 / – |
| trained sources acc | 0.827 / 0.838 | 0.873 / 0.865 | 0.874 / 0.873 | 0.865 / 0.866 | 0.845 / – |
| new sources Brier | 0.481 / 0.416 | 0.269 / 0.242 | 0.289 / 0.217 | 0.225 / 0.156 | 0.211 / – |
| breadth-v1 test index (Decision Index 0.2, 14 held-out datasets) | 23.3 | 38.0 | 41.0 | **52.3** | 54.0 |

Other U numbers:
- Kev-4B (`HF:kev-4b@main/README.md`):
  - MMLU-Pro 0.565 vs Jev 0.840;
  - deadline 0.65 vs 0.95 (`KEV_DATE_FACTS=1` lifts it to 0.85 on the parent);
  - unanswerable items answered with p ≥ 0.9: 0.00 vs Jev 0.09;
  - coverage at ≤ 5% error OOD: 0.62 dev vs Jev 0.70;
  - hard-v1 test 0.803; devtools-v1 test 0.756; documents-v1 test 0.903;
  - CUAD accuracy by state length 4k/8k/16k/32k/64k: 0.847/0.837/0.823/0.788/0.781.
- Kev-27B v2 is on par with AutoJev-27B (`denis-pplx/autojev-27b`, index 50.0), and its MMLU matches Jev at 0.90 (`U:docs/model-cards/kev-27b.md`, `U:README.md:232`).

---

## Serving

- **API compatibility.**
  - `POST /v1/systemone` takes `{state, model, questions{id: {type, instructions, criteria}}}` and returns `{model, answers, usage{input_tokens, output_tokens}, latency_ms}` (`L:kev/serve.py:56-62`). `output_tokens` is a billing-style count of the serialised answers; nothing is generated (`L:kev/api.py:108-111`).
  - `GET /v1/models` reports `kev-latest` with alias `jev-latest` (`L:kev/serve.py:99-101`).
  - Demo extras: `/v1/systemone/permute` and `/v1/systemone/separate`.
  - The official `typesafe-sdk` works with `base_url` changed (`L:README.md:102-117`, `L:tests/test_api.py`).
  - At L there is no auth and the server binds 127.0.0.1 (`L:kev/serve.py:170`).
  - U adds: bearer auth when `KEV_API_KEY` is set; an `x-typesafe-request-id` header and a `server-timing` header; `jev-latest` served as a model name because it is the SDK default; 422 on over-long states; `KEV_DATE_FACTS=1` (appends pairwise day counts to the state) (`U:kev/serve.py:5-9,31-35,229-241`, `U:kev/api.py:62-91`).
- **Latency at L.** One request at a time, fp32 or bf16, no cross-request KV cache, dense per-sample mask (`L:README.md:343`). Mac figures: 0.5b ~160 ms fp32, 4b ~1 s bf16, 8b ~2 s bf16. 4b on an H100 ~45 ms per packed request (`L:README.md:51-54`, `L:docs/model-cards/kev-4b.md`). The README sample takes 277 ms for 3 questions with kev-4b bf16 on an M5 (`L:README.md:96-100`).
- **Latency at U.** Model time, median of 20, new / cached state (`U:README.md:399-414`):

| model / GPU | 6 q short | 5 q, 2,200-token state | req/s @64 clients |
|---|---|---|---|
| Kev-0.8B / L4 | 22.7 / 16.1 ms | 108.6 / 32.3 ms | 62.8 |
| Kev-4B / H100 | 18.1 / 12.9 ms | 89.4 / 22.5 ms | 100.8 |
| Kev-9B / H100 | 24.0 / 16.6 ms | 88.5 / 26.4 ms | 79.5 |
| Kev-27B / B200 | 46.5 / 32.2 ms | 178.0 / 52.1 ms | 44.2 |

  - On Apple M5 through MLX, 5 questions: Kev-4B 721 ms new / 136 ms cached.
  - Over Modal HTTP, Kev-4B at 1 client: p50 36.9 ms on the direct server vs 102 ms on the regular web endpoint.
  - bf16 serving differs from the fp32 evaluation path by ≤ ~0.03 in p on GPU.
  - Deploy: one-command Modal deploy skill `skills/kev-deploy`; HF Space `jaredpalmer/kev` (ZeroGPU).

---

## Licence

- **Code**: Apache-2.0 (`L:LICENSE`, `L:pyproject.toml:6-7`). The LICENSE file is unchanged at U.
- **Weights**: the adapters, heads and 27B full weights are all tagged Apache-2.0 on the Hub (`HF:api-*.json cardData.license`). Every card says "Apache-2.0 for the adapter and head" (`L:MODEL_CARD.md:61`).
- **Bases**: Qwen2.5-0.5B, Qwen3-*-Base, Qwen3.5-*-Base and Qwen3.8-27B are Apache-2.0 per the cards. Datasets carry their own licences. Some U eval suites (breadth-v1, documents-v2, tasksource-heldout) are private mirrors because of NC or share-alike text (`U:AGENTS.md` Frozen suites).
- **PyPI**: `pip install kev` installs an unrelated 2016 ORM, and the project is not on PyPI (`U:README.md:75-76`).

---

## Changelog

**Size progression (all at or before L):**
- **kev-0.5b v0.1** (2026-09-17): Qwen2.5-0.5B, MacBook-trained, 6 sources, lr 2e-4, no extra losses (`L:MODEL_CARD.md:62,119-123`).
- **kev-0.6b** (Qwen3-0.6B-Base): the backbone switch. Qwen3 beat Qwen2.5 at the same recipe (dev 81.6/79.3 vs 74.2/65.8) (`L:PLAN.md:5-9`). The training move to Modal H100 came with frozen suites (decision-v3 → v4/v5/v6/v7, transfer-v4), compositional policy data and none pairs.
- **kev-4b / kev-8b** (Qwen3-4B/8B-Base): the same architecture. The recipe change was **lr 5e-5**. 8B got knowledge MCQ sources (v6, ARC/OBQA/CSQA). The architecture did not change between 0.5b and 8b; only the backbone, data, lr and batching did.

**Since 00aa32d1** (246 commits, 2026-09-20 → 2026-10-15; `git log 00aa32d1..1f04cfa`):
- 09-20 `dc27166`, `9e2fa34` and the HF republish: 0.6b/4b/8b retrained on decision-v7.
- 09-20 `87108c6`, `5f78968`, `38087aa`: **Qwen3.5 port**.
  - transformers 5;
  - LoRA on the DeltaNet projections;
  - **row-batched forward for hybrid backbones** and a hybrid prefix cache;
  - Kev-0.8B/4B/9B released; Qwen3 models marked previous generation.
- 09-20/21 `406f464`, `b1480e3`: soft targets, `--replay`, the `date_facts` preprocessor, and the "dates + unknowable" delta fine-tunes promoted.
- 09-21 `3aa96e5`: temperature stored in `head.pt` and applied at load.
- 09-22 `8fc092f`: MLX backend for Apple Silicon (5–7× faster than torch-MPS).
- 09-23/24 `e9cfef0`, `60c8d39`, `1b62aa2`: CUDA graphs, fused fla Triton DeltaNet kernels, batched serving (18 ms for 6 questions, ~100–121 req/s per H100).
- 09-23/24 `b8aa777`: Kev-4B real-document delta (documents-v1 test 0.804 → 0.904). Round 10 added the skills (hard-v1) and devtools deltas.
- 09-24 `09ff745`, `eb45fd2`: **full-weight SFT** (`--full_ft`, MasterAdamW, FSDP2, bit-exact resume) and `--shared_prefix` training (state once, branches from functional per-layer state).
- 09-24 `f8c914a`: breadth-v1 held-out panel, with Jev and AutoJev baselines.
- 09-25 `9689666`: **64k-token states** for train and serve. 09-26 `a0255ce`: longdoc-v1 probe (CUAD 4k–64k).
- 09-26 → 09-29: rounds 19–26, registered experiments for 27B full-weight SFT. Round 23's blend `27b-k-w85` passed.
- 09-30 `0a58b69`: **Kev-27B v2** (full weights, T 1.32); v1 LoRA at tag `v1-lora`. `29507aa`: Kev-9B v2 (T 2.19). `2ea5660`: 422 instead of silent truncation.
- 10-01 `61b041b`: **Kev 1.0** family release. `3fe301d`: validated context lengths fixed at 8,192 / 8,192 / 8,192 / 65,536.
- 10-03 → 10-15: maintenance only: device probe, compare diagnostics, Modal direct-HTTP serving (`85ccf56`), score presentation checks (`dd375f2`), fine-tune input validation (`1f04cfa`). No new checkpoints.

---

## Notable design choices vs Jev

1. **Prefill-only with a pointer readout, no decoding.** This is the same thesis as Hume's Jev reconstruction. kev scores the `</opt>` token of each option against a `<decide>` token placed after all options, so the readout sees the full option list and can handle "none of the above". It is one head for noul, choice and score.
2. **Zero new vocabulary.** It reuses five Qwen FIM/box special tokens as delimiters and sanitises user text so they cannot be forged. Their embeddings stay frozen and LoRA adapts their meaning.
3. **Exact question isolation, two ways.** On attention-only bases it uses a block-causal mask with branch positions restarting after the state. On hybrid DeltaNet bases it uses independent causal rows over a cached state. The rows are exact by construction, whereas a packed mask cannot isolate recurrent state. Jev's reported ~32k per branch is matched or exceeded: 65,536 state + 8,192 per question at U.
4. **Order sensitivity is accepted.** Within-question options are not permutation-invariant by default. The option-isolation variant (shared option positions) is exact but cost 5.8 pp at 4B. Jev shows 0 argmax flips, but kev's PLAN notes Jev's probabilities still move under permutation (`L:PLAN.md:16`).
5. **Small adapters on frozen base models, low lr.** The key recipe lesson is that fine-tuning erodes base knowledge. lr 5e-5 (2e-5 for deltas) counters it. The 27B departs from this with full-weight SFT at 2e-6 plus WiSE-FT averaging.
6. **Hard labels only, no Jev distillation.** Data is public datasets, code-labelled policy/rule/skill generators, and teacher-agreement labels only for CFPB documents. Calibration is a single post-hoc temperature in `head.pt`. Since U it must be fitted on held-out datasets, not on the training distribution.
7. **Evaluation discipline.** Frozen hashed suites, a locked test read once, registered rounds with decision rules written before the result, paired record-clustered bootstraps, and Jev scored on the same items. The remaining gaps to Jev: knowledge (MMLU-Pro), date arithmetic (`KEV_DATE_FACTS` crutch), coverage at a 5% error budget, and held-out rule composition at small sizes.
