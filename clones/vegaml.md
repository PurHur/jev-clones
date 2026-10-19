# vegaml / vega-U (Nandha Kishor M)

Laya's author's second decision model, and a **different architecture** from Laya. A frozen causal LM ("reader") is run
prefill-only up to an intermediate layer. Mean-pooled span features (state, question, final token, one per option) feed a
trained 14.3M-parameter **"physics engine"**: a particle in a 64-d latent rolls for 12 damped-Hamiltonian steps through a
potential with one Gaussian well per option, and the answer distribution is a Boltzmann softmax over well energies. The
temperature is predicted from the particle's final physical state. Each answer also carries a split-conformal set, an
abstain flag and an "unbound" OOD flag. **vega-U** (2026-10-17) moves the engine into a shared 1024-d space so that any
frozen reader plugs in through a per-reader linear "port".

Source pins used for this profile:

| ref | what | where |
|---|---|---|
| `G` = GitHub HEAD `207c04c8` (2026-10-19 14:50 +0530, "vegaml 0.12.3: auto compares the head against the engine…") | shallow clone (`--depth 1 --filter=blob:limit=5m`), read only, never executed; upstream has 103 commits and tags v0.1.0 … v0.12.3 | `/home/ai/projects/jev-clones/sources/vegaml` |
| `HF-U` = `nandakishorm/vega-U` @ `4d205028` (created 2026-10-17, modified 10-18) | API JSON, tree, README, all `vega_config.json`, `eval_report.json`, `universal/universal_config.json`, `ports.json` (no weights) | `notes/hf-raw/nandakishorm__vega-U/` |
| `HF-C` = `nandakishorm/vega-08b-public-intents` @ `42ac8d4f` (created 2026-10-09, modified 10-11) | API JSON, tree, README, configs, adapter JSON, eval JSON/MD/JSONL, the bundled `vega_api.py`/`vega_common.py`/`vega_vision.py` saved as `.txt` | `notes/hf-raw/nandakishorm__vega-08b-public-intents/` |

Citation form: `G:vegaml/_common.py:41` = file:line at `207c04c8`. `HF-U:vega_config.json` / `HF-C:run_summary.json` =
fetched Hub files (paths relative to `files/`). Card quotes are `HF-U:README.md:L` / `HF-C:README.md:L`.

> **Main findings.**
> 1. Nothing in the backbone is trained, not even LoRA. The engine has **no attention**: MLPs, a 4-head
>    memory-reader attention over ≤16 state chunks, and an explicit integrator (`G:vegaml/_common.py:608-856`).
> 2. **One prompt and one forward pass per question.** Questions are isolated by being separate rows. A
>    state is only shared ("read once") when it is ≥ 4,096 tokens. Below that, the question is also prepended
>    *before* the state (`query_first_max_state 4096`). The README's "Both questions are answered from one read
>    of the state" holds only for long states.
> 3. The dynamics add little. Readout at step 0 against step 12: 0.751 vs 0.765 held-out on the 0.8B reader,
>    0.805 vs 0.811 on the 27B. Particles never settle early (`mean_steps_halt 12.0`). The "settle" loss has
>    weight 0 (`HF-U:eval_report.json`, `HF-U:universal/universal_config.json`).
> 4. The headline "held-out acc 0.811 / ECE 0.021" for the 27B is the **in-distribution test split of the
>    training sources**. The 4-source "zero-shot" split gives the 27B 0.629 at **ECE 0.134**.
> 5. "No silent truncation" is claimed (README:356) but not implemented in the package. `build_prompt`
>    right-truncates the state and shrinks options and instructions silently (`G:vegaml/_common.py:156-167`).
> 6. Training code for the base engine is **not published**. Only the loss names and weights survive, in
>    configs.

---

## Identity

| | |
|---|---|
| Repo | github.com/NandhaKishorM/vegaml, created 2026-10-09, Apache-2.0, 274 stars / 34 forks (gh API 2026-10-19). PyPI `vegaml` 0.12.3 (`G:pyproject.toml:6-7`). Author "Nandakishor M" (`G:pyproject.toml:13`), same person as Laya (`convaiinnovations`); HF account `nandakishorm` |
| Weights | `nandakishorm/vega-U` (default since 0.11; `G:vegaml/hub.py:16`) and `nandakishorm/vega-08b-public-intents` ("classic" per-size checkpoints, 0.8B root + `4b/`) |
| Hub stats | vega-U 76 dl / 5 likes; vega-08b 812 dl / 37 likes (`HF-*/api.json`). `safetensors` param metadata is empty on both (the engine is a plain state dict, not an HF model); `config.json` is a 68-byte stub `{"model_type":"vega-universal","engine_sha16":"78f96be7c6a06abb"}` that "nothing reads" (`G:README.md:675-677`) |
| Lineage claimed | "WALT world-model recipe (frozen language model = perception, small trainable physics = dynamics)" (`G:vegaml/_common.py:3`); related work CWMI (arXiv 2507.19855) stated openly (`G:README.md:650-670`). No Jev/Hume lineage claimed, but the type set, `criteria` shape and benchmark framing are Jev's |
| Name collision | not related to `vllm-sr/Decision-2.0-Vega-27B` (hf-sweep §3.2) |
| Decision Index board | not listed (hf-sweep `_spaces/dix_leaderboard.md`). `bench/decision_index_engine.py` exists to run it (`G:bench/README.md:1-30`), but no published score was found |

---

## Architecture

```
            state (str | JSON)          question {type, instructions, criteria}
                  |                                  |
                  v                                  v
   prompt per question:  [bos] [Question to keep in mind… (if state < 4096 tok)]
                         "Observation:\n" STATE "\n\nMeasurement (<type>): " INS "\nPossible outcomes:"
                         "\n* " OPT_0 … "\n* " OPT_K-1 "\nOutcome:"
                  |
                  v
   FROZEN causal LM reader (Qwen3.5-0.8B / 4B, Qwen3.8-27B, gemma-3-1b-it), SDPA causal, forward hooks,
   StopForward raised after the deepest tapped layer (LM head and upper layers never run)
                  |  taps at ~0.6 and ~0.85 depth (0.8B: layers 13,19 of 24; 27B: 37,53 of 64)
                  v
   mean-pool spans:  state | question | last token   (both feature layers)   -> feats [Lf=2, 3, d]
                     each option span                (deepest layer only)    -> opts  [K, 1, d]
                     state cut into ≤16 chunk means  (deepest layer)         -> chunks [16, d]
                  |
     vega-U only: standardize with stats.npz, Port = one Linear(d -> 1024) per tapped layer
                  v
   ENGINE (14.34M, d_shared 1024, latent D=64, world S=256, hidden H=2048)
     ELMo scalar mix over layers -> slots s_state, s_q, s_last ; options o_k
     s  = world(s_state) (+ MemoryReader(q=[s_q,s_last], kv=chunks))      s' = foresight(s)
     probe = probe(s_q) + type_emb ; imp = impulse(s_last)
     z0 = to_z0([s,s',probe,imp]) ; p0 = to_p0([...])                       particle (z, p) in R^64
     wells: (c_k, log a_k, sigma_k) = well([o_k, probe]) (+type_emb) ; score -> centres on a 1-d rail
     12 steps symplectic Euler, damped, with a selective-SSM "thermostat" (dt, friction, memory force)
     E_k = |z_T - c_k|^2 / 2 sigma_k^2 - log a_k ;  P = softmax(-E / tau(phys))
                  v
   probs, confidence = max p, LAC conformal set, abstain, unbound, settle_steps, energy
```

### Backbone (reader)

| Checkpoint | Reader (frozen) | L / d | taps (feature / option) | trained | source |
|---|---|---|---|---|---|
| vega-U root | `Qwen/Qwen3.5-0.8B` (jointly trained; port is the identity space, `ports.json: identity true`) | 24 / 1024 | 13, 19 / 19 | engine 14,342,512 + port 2×(1024²+1024) | `HF-U:vega_config.json:20-33`, `HF-U:universal/ports/ports.json` |
| vega-U `gemma-3-1b-it/` | `google/gemma-3-1b-it` (jointly trained, gated) | – / 1152 | fractional 0.6, 0.85 | port | `ports.json` |
| vega-U `Qwen3.5-4B/` | `Qwen/Qwen3.5-4B` **attached cold** (ridge cos 0.887/0.897, 300 refine steps, 27,433 items) | 32 / 2560 | 0.6, 0.85 | port 20.98 MB | `ports.json` |
| vega-U `Qwen3.8-27B/` | `Qwen/Qwen3.8-27B` **attached cold** (ridge cos 0.929/0.923, 38,934 items) | 64 / 5120 | 37, 53 / 53 | port 41.95 MB; engine+port = 24,830,320 (`eval_report.engine.parameters`) | `HF-U:Qwen3.8-27B/vega_config.json` |
| classic 0.8B (vega-08b root) | `Qwen/Qwen3.5-0.8B`, "SHA-256 pinned" | 24 / 1024 | 13, 19 / 19 | engine 14,342,512 (57 MB) + adapters 835,937 | `HF-C:README.md:64-68`, `HF-C:vega_config.json` |
| classic 4B (`4b/`) | `Qwen/Qwen3.5-4B` | 32 / 2560 | | engine 30,879,088 (124 MB) + adapters 1,431,905 | `HF-C:README.md:66-68` |

- The reader is loaded as `AutoModelForCausalLM(..., attn_implementation="sdpa")` and every parameter is set to
  `requires_grad=False` (`G:vegaml/_common.py:341-344`). Hooks on the tapped layers capture hidden states, and the
  deepest one raises `StopForward`, so the layers above it and the LM head never run (`:354-360, 369-377`).
- Dtype: bf16 on CUDA with capability ≥ 8, fp16 below that (T4), fp32 on CPU and MPS (`:322-326`).
- Layer finder: picks the longest `*.layers` ModuleList, so multimodal wrappers work (`:286-295`). Layer specs are
  fractions of depth or indices (`:298-304`). The universal config stores `[0.6, 0.85]` / `[0.85]`
  (`HF-U:universal/universal_config.json` `feature_layers`).
- On T4-class GPUs (no fused kernel for Qwen3.5's 256-wide heads), causal SDPA is chunked into 512-query blocks for
  sequences of 2,048 tokens or more (`G:vegaml/_common.py:243-283, 362-367`).
- Images: `decide_image` runs the reader's own frozen vision tower, and the picture replaces the state text in the
  same prompt (`G:vegaml/_vision.py:24-120`, `G:README.md:325-345`).

### Input encoding

`build_prompt` (`G:vegaml/_common.py:131-182`), template 0 at inference (template 1 exists only as a training
"invariance view", `:121-125`):

```
[bos] (QUERY_FIRST: "Question to keep in mind while reading (<t>): <ins>\n\n"  if state < 4096 tokens, ≤128 tok)
"Observation:\n" <state> "\n\nMeasurement (<t>): <ins ≤256 tok>\nPossible outcomes:"
"\n* <opt_0 ≤32 tok>" … "\n* <opt_K-1>" "\nOutcome:"
 spans recorded: state (a,b) | question (a,b) | options [(a,b)…] | last token = "Outcome:" end
```

- Option text (`option_texts`, `:101-112`):
  - `choice`: `"key - description"`, or the bare key when the description is null, in caller order;
  - `score`: `"level i - text"`;
  - `boolean` (renamed from `noul` in 0.5.0, `G:README.md:368-371`): always `["no - <false|'the statement does not
    hold'>", "yes - <true|'the statement holds'>"]`, so `p[1]` = P(yes).
- The type name goes into the prompt literally (`Measurement (choice):` …). Unknown types raise an error
  (`G:vegaml/_api.py:178-182`). The shipped calibration tables are still keyed `noul`, and `normalise_conformal`
  remaps them (`G:vegaml/_common.py:82-98`; `HF-U:vega_config.json:108`).
- The state is a string, or `json.dumps` of a dict or list (`:60-62`). Field names stay as JSON.
- **Budgeting and truncation.** If the fixed part exceeds `max_len − 16`, every option is cut evenly (floor 3
  tokens) and then the instructions (floor 8) (`:156-160`). The state then gets `room = max_len − fixed` and is
  **right-truncated silently** (`st[:room]`, `:166-167`). No refusal path exists anywhere in `vegaml/` (searched
  for `refus|truncat|too long`). This contradicts the README's "No silent truncation. An input that will not fit
  is refused" (`G:README.md:356`, `HF-C:README.md:103-104`). Inferred: the refusal may exist only in the
  unpublished training/eval tree.
- Defaults: `max_len 73,728`, `opt_max_tokens 32` (`HF-U:vega_config.json:12-13`). Training used `max_len 1024`, with
  `longctx` sources at 73,728 and `hardctx` at 16,384 (`HF-U:universal/universal_config.json` `max_len_by_source`).

### Attention / isolation

- Inside the reader: plain causal attention, no custom mask. Options are read sequentially, so option i sees state,
  question and options < i but not the options after it. The last token sees everything.
- **Between questions: separate sequences.** `Vega._view` builds one prompt per question (`G:vegaml/_api.py:189-197`),
  and `Perception.pooled` runs them as one padded batch (`G:vegaml/_common.py:439-479`). Isolation is exact by
  construction, and there is no packing.
- **Prefix sharing only for long states.** Items whose state span ends at P ≥ `prefix_min_tokens` (4,096) are grouped
  by identical first-P token ids. The prefix runs once into a `DynamicCache`, and each question continues from a
  `copy.deepcopy` of it (`:394-410, 481-506`). That covers KV plus Qwen3.5 DeltaNet recurrent state, because it is
  the HF cache object. For short states the prompt begins with the query-first header, so no prefix is shared.
  Short multi-question requests cost Q full passes (`G:README.md:594`: "Three questions about one state mean three
  prompts and three passes, except for the long-input case").
- The engine sees each question independently (batch dimension B = questions). No cross-question interaction is
  possible.
- With query-first, the *state-span mean* has read the question. For long states it is question-independent. The
  docstring calls the state span "a question-independent world observation" (`:136-144`), which is true only above
  4,096 tokens.

### Readout: the physics engine

`VegaEngine` (`G:vegaml/_common.py:667-856`), config `latent_dim 64, world_dim 256, proj_hidden 2048, n_steps 12,
dt_max 0.5, settle_eps 0.01, force_max 20, ssm_state 32, memory_chunks 16` (`HF-U:vega_config.json:2-16`):

1. **Feature mix.** Per-slot softmax scalar mixes over the tapped layers (`mix [3, Lf]`, `opt_mix [Lo]`),
   standardisation buffers, and feature dropout 0.1 (`:686-693, 727-735`).
2. **World latent.** `s = MLP(s_state)` → 256-d. The optional `MemoryReader` adds a 4-head attention of
   query `[s_q, s_last]` over ≤16 chunk means with learned relative-position embeddings. Its output layer is
   zero-initialised (`:619-648, 798-803`). Then `s' = WorldDynamics(s)`, a gated residual "one-step foresight"
   (`:651-664, 804`).
3. **Particle init.** `probe = MLP(s_q) + type_emb[qtype]`, `imp = MLP(s_last)`, and
   `z0, p0 = MLP([s, s', probe, imp])` each in R^64 (`:805-809`).
4. **Landscape.** For each option, `well(MLP([o_k, probe])) → (c_k ∈ R^64, log a_k ∈ [−4,4], σ_k = 0.3 + softplus,
   gap_k)`, and `c_k += type_emb`. For `score`, the centres are replaced by a **1-d rail**:
   `c_k = mean(c) + (cumsum(softplus gap) − mean) · u`, with `u = normalize(MLP(probe))`. Ordinal neighbours are
   therefore spatial neighbours (`:737-754`).
5. **Dynamics.** Potential `U(z) = ½κ|z|² − Σ_k a_k exp(−|z−c_k|²/2σ_k²)`, with κ = exp(log 0.05) learned
   (`:706, 763-787`). Each step (`:822-850`):
   ```
   u = Linear([z, p, ∇U, |∇U|])           -> (b_u, dt_raw, fr_raw)
   dt = 0.5·sigmoid(dt_raw) ; γ = softplus(fr_raw)
   h  = exp(−A·dt)·h + dt·b_u              (diagonal SSM memory, 32 states)
   F  = 0.1·tanh(W h)                      (bounded learned force)
   g  = ∇U / (1 + |∇U|/20)                 (soft cap)
   p ← p·exp(−γ dt) − dt·g + dt·F ;  z ← z + dt·p         (damped symplectic Euler)
   ```
   Early-exit freezing applies when kinetic energy < 0.01 and |∇U| < 0.1. The loop always runs all 12 steps, to
   avoid a host sync (`:844-850`).
6. **Readout.** `E_k = |z_T − c_k|²/2σ_k² − log a_k`, padded options get 1e4, and logits = −E (`:756-761, 841`).
   Thermal features for calibration (`:991-996`): `[log1p(kinetic), log1p(nearest well exponent), steps_used/12,
   log k]`.

**Measured contribution of the dynamics** (`HF-U:*/eval_report.json`, `accuracy_step0` = readout at z0):

| reader | held-out acc step 0 → 12 | zero-shot step 0 → 12 | mean steps used |
|---|---|---|---|
| 0.8B | 0.751 → 0.765 | 0.511 → 0.534 | 12.0 of 12 |
| 27B | 0.805 → 0.811 | 0.612 → 0.629 | 12.0 of 12 |

The integration contributes 0.6-2.3 pp. Halting never triggers, and `accuracy_halt == accuracy_full` on both
readers. Engine time is ~0.006-0.014 ms per item (`engine_ms_per_item`), so the reader is the whole latency bill.
Interpretation (inferred): functionally this is a well-structured scorer, `E_k` = a Mahalanobis-like distance from a
question-conditioned point to option-conditioned centres, refined by a 12-step recurrent update. The physics framing
gives the score type a useful ordinal inductive bias (the rail) and a set of calibration features.

### Question types

| type | options fed | answer (`G:vegaml/_api.py:257-284`) | extras |
|---|---|---|---|
| `choice` | criteria keys, `key - desc` (a list is accepted as keys with null desc, `:184-185`) | `choice = keys[argmax]`, `probs` by key | – |
| `score` | `level i - text`, ordered | **`score = argmax` level (int)** plus `expected_score = Σ i·p_i` | centres on a 1-d rail |
| `boolean` | `[no, yes]` (+ descriptions) | `boolean = p[1]` | – |

Every type gets: `confidence = max p` (not Jev's normalised formula), `prediction_set` (LAC: `{k : p_k ≥ 1 − q̂}`, top-1
always kept), `abstain = conf < threshold[risk]`, `unbound = nearest-well exponent > threshold`, `settle_steps`, `energy`,
and, with adapters, `adapter`/`gate`. `views=2` re-reads with reversed options for choice/boolean and `views=3` adds a
prose rendering of the state with template 1. Probabilities are averaged (`:237-252`).

One head, the engine, serves all three types. The type enters as a type embedding (probe and well centres), as the
literal word in the prompt, as the score rail, and as a per-type calibration row.

### Calibration, conformal, abstain

- **Thermal temperature** (`G:vegaml/_common.py:1003-1053`). Per type, `log τ = b + w·phys`, clamped to [0.2, 20], and
  fitted by LBFGS on held-out NLL with a small L2 on w. `fit_thermal(use_phys=False)` is the plain-temperature baseline.
  vega-U root ships `mode "thermal"` with held-out NLL 0.4848 (thermal) against 0.4872 (scalar): a **0.5 % gain**
  (`HF-U:vega_config.json:35-91`). For the 27B, 0.4552 against 0.4878.
  - The shipped adapter (`typed_decisions`) chose **scalar** because thermal lost (`HF-C:adapters/typed_decisions.json`
    `heldout_calib_nll scalar 0.555 / thermal 0.563`).
  - The calibration pool mixes half of the zero-shot pool (800 items from 6 tasks, weight 12.7 per item;
    `zero_shot_fit`, `zs_calib_frac 0.5`, `zs_calib_weight 0.2`). The "zero-shot" eval numbers are therefore on
    tasks whose other half shaped the temperature.
- **Split-conformal LAC** per type at α ∈ {0.1, 0.05}, with q̂ = the ⌈(n+1)(1−α)⌉/n quantile of 1 − p_true
  (`:1056-1069`). Shipped q̂ for the 0.8B: choice 0.744/0.803, score 0.837/0.924, noul 0.571/0.686.
- **Abstain**: the lowest confidence at which the selective error stays ≤ risk (0.05/0.1/0.2), default risk 0.1
  (`G:vegaml/_attach.py:585-594`). For the 0.8B: 0.729/0.597/0.273.
- **Unbound**: the 0.995 quantile of the nearest-well exponent (`G:vegaml/_attach.py:557`). For the 0.8B it is 1.377.
  Unbound items are 2 % of held-out and are measurably worse: acc 0.651 vs 0.768, ECE 0.070 vs 0.010
  (`HF-U:eval_report.json` `eval_in.bound/unbound`). On zero-shot it does **not** separate: acc 0.544 unbound vs 0.533
  bound. That makes it weak evidence as an OOD detector.
- **User refit.** `vegaml.calibrate(model, examples, questions)` fits on 80 %, keeps the thermal form per type only if
  it beats the plain one on the 20 % holdout, then computes the conformal, abstain and unbound thresholds on **all** n
  (`G:vegaml/_attach.py:457-582`). Note: the conformal thresholds are fitted in-sample on rows that also fitted the
  temperature, and `ece_after` is in-sample, so the reported improvement is optimistic.
  - A second, per-question calibrator, `vegaml.calibrate.Calibration` (temperature + conformal per question id), is
    described in `G:docs/calibration.md:1-30`.
- Calibration "never transfers between readers": every reader dir ships its own (`HF-U:README.md:82-83`).

### Adapters, TTT head, cascade

- **Gated LoRA on the engine** (not the reader). `GatedLoRALinear` covers 7 engine projections (`world, probe,
  impulse, to_z0, to_p0, well.net.1, well.net.4`), with a per-question sigmoid gate MLP over `[s_q, s_last]`
  (`G:vegaml/_common.py:873-927`). The card says the gate reads "the world latent" (`HF-C:README.md:399`); the code
  reads the question and last-token slots.
  - Routing picks the most open adapter if gate ≥ its own threshold, else the base engine. Each route has its own
    calibration (`:963-988`, `G:vegaml/_api.py:165-175, 225-227`).
  - An adapter records the base engine's SHA-256 and is refused against another engine (`G:vegaml/_api.py:121-123`).
  - Shipped classic adapters: `typed_decisions` (r16, 551k params, trained on the `LocalLLaMA/typed-decisions` train
    split, gate negatives = 10k replay records) and `public_intents`. The adapters are load-bearing: on
    typed-decisions test the gate is open on 100 % of items and the base engine alone scores 0.389
    (`HF-C:run_summary.json` `gate_stats`, `generalist_before`).
- **TTT head** (`G:vegaml/ttt.py`): PCA(128) + multinomial logistic regression on the same frozen features, fitted per
  question from ≥ 6 labelled examples. It is used by `mode="auto"` only when its stratified 5-fold CV accuracy beats
  both chance and the engine's accuracy on the same examples (`G:vegaml/__init__.py:266-293`, 0.12.3). It carries no
  calibration and no abstain flag (`G:README.md:279-305`).
- **Cascade** (`G:vegaml/router.py:47-135`): run 0.8B, then escalate to 4B when the answer is "unsettled"
  (confidence < 0.75 or set size > 1).

### Context & limits

| | value | source |
|---|---|---|
| Serving `max_len` | 73,728 tokens (overridable per load) | `HF-U:vega_config.json:12`, `G:vegaml/_api.py:56-58` |
| Trained lengths | 1,024 default; `longctx` 73,728, `hardctx` 16,384; per-reader extraction | `HF-U:universal/universal_config.json` |
| Prefix sharing | state ≥ 4,096 tokens | `prefix_min_tokens` |
| Memory reader | state ≥ 128 tokens (cfg `memory_min_tokens 128`; code default 2,048) → k = clamp(n/128, 2, 16) chunk means | `G:vegaml/_common.py:423-437` |
| Option length | 32 tokens each (silent cut) | `opt_max_tokens` |
| Instruction length | 256 tokens (silent cut) | `build_prompt` |
| Options per question | no hard cap; training subsampled choice to ≤ 20 options with gold kept | `G:vegaml/_common.py:216-220`, `max_options_train 20` |
| Questions per request | no cap in `decide`; HTTP batch ≤ 256 items | `G:vegaml/serve.py:38-39` |
| Long-context accuracy | `longctx` source (to ~70k): 0.831 (0.8B) / 0.941 (27B) | `HF-U:*/eval_report.json` |

---

## Training

The **base-engine training code is not in the GitHub repo or on the Hub.** The README refers to a "training tree"
(`G:README.md:316`), `universal_config` names `attach_backbone.py`, and Kaggle paths appear in the configs. What is
recoverable:

- **vega-U run** (`HF-U:universal/universal_config.json`):
  - joint on Qwen3.5-0.8B (reference, `d_shared 1024`) and gemma-3-1b-it, with cached frozen features;
  - **710,138 / 708,657 items**, 14 epochs (best 8), **2,422 steps in 10.3 minutes**;
  - batch 4,096, lr 1e-3, wd 0.02, warm-up 300, clip 1.0, patience 5, seed 13;
  - `inv_weight 1.0` cross-reader invariance on 60k aligned records ("pulling the same item's ported features
    together across tokenizers", `G:docs/vega-u.md:101-103`).
- **Loss terms and weights**: `capture 1.0, bind 0.05, lyapunov 0.1, settle 0.0, variance 0.2, counterfactual 0.2,
  dynamics 0.3, invariance 0.5`, with `bind_radius 1.0, score_sigma 0.5, cov_weight 0.04` and `balance_noul` (≤ 4×
  weight). The definitions are unpublished. Inferred from names, docstrings and the logged values:
  - `capture` = CE / soft CE of the Boltzmann readout at z_T (it drops 0.85 → 0.47);
  - `bind` = pull the final z inside its gold well;
  - `lyapunov` = energy non-increase along the trajectory (`H_traj` is returned, `G:vegaml/_common.py:820, 842`);
  - `variance` / `cov_weight` = an anti-collapse term on latents (`eval_report.engine.collapse_ratio 10.67`);
  - `counterfactual` = items sharing an answer space (`answer_space_hash`, `G:vegaml/_common.py:115-118`) are
    contrasted;
  - `dynamics` = foresight `s'` predicts the next conversation prefix's `s` (`G:vegaml/_common.py:651-664`; episodes
    with ≤ 6 prefixes, `:185-208`);
  - `invariance` = agreement between the template-0 and template-1/prose/permuted views (`view_frac_train 0.25`,
    `:121-125, 226-234`);
  - `score_sigma` = soft ordinal targets for `score` (inferred).
- **Augmentation** (`record_items`, `G:vegaml/_common.py:191-235`): option shuffle for choice/boolean (never score);
  choice with > 20 options subsampled with gold kept; 25 % second views with a different template, prose state and a
  different permutation.
- **Data** (sources and caps, `universal_config.caps`), all public datasets:
  - NLI and verification: boolq, mnli, snli, anli, vitaminc;
  - intents: clinc150, bitext_intents;
  - quality: helpsteer2;
  - safety and spam: civil_comments, jailbreak, enron_spam, phishing_emails, phishing_urls;
  - classification: ag_news, amazon_reviews, go_emotions, support_tickets;
  - retrieval: ms_marco;
  - natural_instructions (227 tasks, 15 % held out);
  - episodes: sales (conversation prefixes → noul);
  - `jaggedness` (40k; origin not documented — inferred: synthetic data modelled on TypeSafe's jaggedness list);
  - synthetic `longctx` / `hardctx`;
  - MCQ (ARC, CommonsenseQA, OpenBookQA) appears in eval_in.
  - `commercial_only false`. No Jev outputs are mentioned as targets. The classic typed-decisions adapter has a
    `calibration_teacher` block (`HF-C:adapters/typed_decisions.json`), so that dataset carries teacher
    distributions.
- **Classic 0.8B engine**: 1,204,888 train items, 45,368 calibration, 4,116 steps, **7.9 minutes**
  (`HF-C:vega_config.json` `training`).
- **Attaching a reader** (`G:vegaml/_attach.py:288-450`):
  1. The anchor (0.8B) reads the user's texts under 6 fixed probe questions covering all three types
     (`PROBES`, `:59-85`).
  2. The new reader reads the same prompts, and its features are standardised.
  3. A closed-form ridge per tapped layer (λ ∈ {1e-3…10} chosen by 10 % holdout cosine; options and chunks folded
     into the deepest layer, `:92-150`).
  4. 300 AdamW steps (lr 3e-4) with loss = CE to the engine's own anchor-path probabilities + MSE to the anchor
     features (`:372-415`).
  - No labels are used. The engine is copied byte-for-byte.
- **Adapter fine-tune** (`G:notebooks/vega_finetune.py`, `G:README.md:475-561`): cached features, then gated LoRA
  r32 + gate. The loss is CE + `gate_weight`·BCE (positives = task, negatives = `ag_news` replay)
  (`G:notebooks/vega_finetune.py:281-297`). Upstream ran it only as a 96-row smoke test (0.313 → 0.542 on 48 items).

---

## Evals

**vega-U** (`G:docs/vega-u.md:99-125`, `HF-U:*/eval_report.json`). "held-out" = `eval_in`, the test split **of the
training sources** (26 sources); "zero-shot" = `eval_zs`, 4 sources (natural_instructions held-out tasks,
emotion_zs, prompt_injections_zs, sst5_zs):

| reader | how | held-out acc (n) | held-out ECE | zero-shot acc (n) | **zero-shot ECE** | AURC |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | attached cold | 0.811 (21,008, capped 400/source) | 0.021 | 0.629 (1,300) | **0.134** | 0.053 |
| Qwen3.5-4B | attached cold | 0.791 | 0.017 | 0.568 | – | 0.063 |
| Qwen3.5-0.8B | joint | 0.765 (50,676) | 0.010 | 0.534 (5,862) | 0.056 | 0.073 |
| gemma-3-1b-it | joint | 0.738 | 0.007 | 0.504 | – | 0.093 |

- The zero-shot sample sizes differ by reader. For the 27B, `natural_instructions` is 100 items at 0.990, against
  1,200 items at 0.592 for the 0.8B. The 27B zero-shot figure is therefore not comparable.
- By type, held-out, 0.8B / 27B: choice 0.797 / 0.878, **score 0.620 / 0.617**, noul 0.795 / 0.832. The score
  ceiling holds across readers. Zero-shot boolean ECE is 0.309 (0.8B).
- Per source, 0.8B → 27B: longctx 0.831 → 0.941, hardctx 0.875 → 0.972, ARC 0.647 → 0.922, CSQA 0.490 → 0.792,
  ANLI 0.561 → 0.776, MNLI 0.836 → 0.919. Flat sources: bitext 0.99, enron 0.99, phishing 0.99, AG News 0.92-0.93,
  helpsteer2 0.59 → 0.55.
- "Cold-attached readers score higher" is real but confounded (inferred): they are larger, and their ports were
  fitted on 27-39k items drawn from the same distribution.

**Classic vega-08b vs Jev** (`HF-C:README.md:138-252`; one live run 2026-10-11 on 2×T4, 3,841 decisions per system,
zero-shot, `mode="engine"`):

| metric | Vega 0.8B | Vega 4B | Jev (hosted) |
|---|---|---|---|
| public menu, 43 tasks, macro acc | 0.556 | 0.717 | **0.836** |
| public menu ECE / NLL | 0.151 / 1.099 | 0.148 / **0.768** | **0.115** / 0.878 |
| JevBench 231 acc / hard 111 | 0.545 / 0.414 | 0.662 / 0.513 | **0.866 / 0.730** |
| JevBench paraphrase consistency | 0.694 | 0.722 | **0.972** |
| DecisionBench 1,075 acc | 0.509 | 0.633 | **0.752** |
| DecisionBench ECE / NLL | 0.103 / 1.628 | **0.098 / 1.420** | 0.122 / 3.196 |
| p50 ms per decision | **74** | 106 | 162 |

- Upstream's own control (`HF-C:README.md:228-232`): Qwen3.5-4B **prompted directly, no engine** scores 0.700 on the
  menu and 0.634 on DecisionBench, "within noise of the 4B engine". On that menu the engine adds probabilities,
  coverage and latency, not accuracy. The exception is wide label spaces (151 intents: 0.900 vs 0.100).
- `LocalLLaMA/typed-decisions` test with the in-distribution adapter (`HF-C:run_summary.json`, `HF-C:README.md:64-76`):
  0.8B acc 0.763 / ECE 0.026 / 22 ms; 4B 0.803 / 0.019 / 28 ms. Base engine without the adapter: 0.389.
- An older item-paired selection, chosen because Vega leads there (self-labelled as selected,
  `HF-C:README.md:256-331, 522-526`): phishing 75.4 vs 61.9; temporal_numeric 46.7 vs 20.0; pooled ECE 9.5 vs 9.3.
- Vision: RVL-CDIP-N 0.793 zero-shot (vs OCR + Jev 0.896); MMMU-Pro 0.240 (0.157 with the image withheld).
- Not on the Decision Index board, and no Kev-style frozen-suite comparison exists.

---

## Serving

- Python: `vegaml.load(name|path, backbone=, device=, max_len=, token=)` → `Vega.decide(state, questions, mode, alpha,
  views)`; `decide_image`, `fit`, `attach`, `calibrate` (`G:vegaml/__init__.py:224-426`).
  - Hub download is allow-listed to data files (config, `engine.safetensors`, adapters, port, `stats.npz` with
    `allow_pickle=False`). No remote code (`G:vegaml/hub.py:1-7, 95-107`, `G:vegaml/_universal.py:335-337`; test
    `tests/test_no_remote_code.py`).
- HTTP `vegaml-serve` (FastAPI): `POST /v1/decide`, `POST /v1/decide/batch` (≤ 256), `GET /health`, `GET /v1/model`.
  One global lock serialises requests (`G:vegaml/serve.py:1-88`).
  - **Not wire-compatible with `/v1/systemone`**: the type name is `boolean`, the fields are `probs`/`boolean`/`score`
    (int)/`expected_score`, and there is no `usage` block, only `n_tokens`.
- Other surfaces: CLI with JSONL; MCP stdio server; ONNX export of the engine only, the reader stays in torch
  (`G:vegaml/onnx_export.py`); an MLX port of the engine (`G:vegaml/mlx_engine.py`); Docker compose; LangChain gate and
  router; pydantic `structured`.
- Latency: README 22 ms (0.8B) / 28 ms (4B) per decision on the typed-decisions harness; 74-106 ms p50 on the public
  menu (T4); 267 ms T4 fp16 and 280 ms M-series fp32 in the older card. Load time 27 s (0.8B) and 61 s (4B)
  (`G:README.md:469-473`).

---

## Licence

- Code: Apache-2.0 (`G:LICENSE`, `G:pyproject.toml:11`).
- Weights: `cardData.license apache-2.0` on both HF repos (`HF-*/api.json`). These cover the engine, ports and adapters
  only. The readers keep their own licences: Qwen3.5/3.8 Apache-2.0 per hf-sweep; **gemma-3-1b-it is gated under the
  Gemma terms**, and one of the two jointly-trained readers (and so part of the engine's training signal) is Gemma.
- Data: public datasets with mixed licences (`commercial_only false`); not itemised.

---

## Changelog (from tags and docs; the shallow clone has one commit)

- v0.1.0 to v0.4 (2026-10-09 …): classic per-size checkpoints `vega-08b-public-intents` (0.8B + 4B), gated adapters,
  thermal calibration, conformal and abstain.
- 0.5.0: `noul` renamed `boolean`, with a remap for old conformal tables (`G:README.md:368-371`).
- 0.7.0: `mode="auto"` default (TTT head where it is worth using). 0.8.0: the image path works from PyPI, and
  per-adapter gate thresholds.
- 0.10.0-0.10.1: universal checkpoints load (`port`, `stats.npz`), and `backbone=` picks the reader dir. 0.11.0: vega-U
  is the default. 0.12.0: `vegaml.attach` / `vegaml.calibrate`. 0.12.1-0.12.3: symlink-safe calibration writes,
  attached dirs load by full repo id, and `auto` must beat the engine's accuracy (`G:docs/vega-u.md:127-158`).

---

## Notable design choices

### vs Jev
1. **The frozen reader is the main bet.** Jev is post-trained end to end (RLCD). Vega never touches the LM and puts all
   learning in a 14M head. Upstream's own data shows the cost: zero-shot accuracy sits far below Jev (menu 0.56/0.72 vs
   0.84; JevBench hard 0.41/0.51 vs 0.73), and a prompted 4B without the engine matches the engine's accuracy.
2. **Probabilities are the selling point, not accuracy.** Each answer carries a physics-conditioned temperature,
   split-conformal sets, an abstain threshold and an OOD "unbound" flag. Jev returns only probabilities and a
   closed-form confidence. Vega's `confidence` is max p, not Jev's normalised formula.
3. **Per-question passes rather than one shared state.** Jev ingests the state once for all questions. Vega does so
   only for states ≥ 4,096 tokens (KV/recurrent cache copy). Below that, every question re-reads the state, with the
   question placed in front of it (query-aware reading).
4. **Interface drift.** `boolean` instead of `noul`, `score` returns the argmax level (int) plus `expected_score` where
   Jev returns the expectation, and the HTTP route is `/v1/decide`. It is not a drop-in for the TypeSafe SDK. No `box`
   type exists, but images are accepted through the reader's vision tower (Jev 1.14 needs a header).
5. **Context and options.** 73,728 tokens nominally, but with silent truncation (contradicting the docs) and options cut
   to 32 tokens. There is no 255-option guarantee; training saw ≤ 20 options.

### vs kev
1. **Both use a frozen base and prefill only**, but kev adapts the backbone with LoRA (r16 → full FT at 27B). Vega
   trains nothing in the reader, so swapping readers is a linear port and not a retrain. That decoupling is unique among
   the clones.
2. **Readout.** kev uses a dot-product pointer: `<decide>` query against `</opt>` keys. Vega mean-pools spans (no
   special tokens) and scores options through a learned energy landscape plus a recurrent integrator. Both read options
   causally in order, and both have a final position that has seen all options (kev `<decide>`, Vega "Outcome:" last
   token).
3. **Isolation.** kev uses a block-causal packed mask (attention-only bases) or causal rows over a cached state (hybrid
   bases). Vega always uses rows, with the cache only above 4,096 state tokens and no packing.
4. **Calibration.** kev ships one temperature in `head.pt`, fitted on held-out *datasets* since U. Vega ships a per-type
   temperature that is a linear function of physics features, plus conformal, abstain and unbound thresholds. Those
   thresholds are fitted on a calibration split of the training distribution (plus half the zero-shot pool), which kev's
   upstream now explicitly refuses (`calibrate_checkpoint.py` rejects in-distribution rows).
5. **Training cost.** Cached features make a full engine run 8-10 minutes. kev's runs take hours of H100 time. The price
   is that the engine cannot fix reading errors of the reader, which is why kev's MMLU/date gaps close with backbone
   tuning and Vega's do not.
6. **Evaluation discipline.** kev uses frozen hashed suites, a locked test and Jev on the same items. Vega has one live
   Jev run with the full menu printed (to its credit) beside a selected-wins table. Its headline vega-U numbers are
   in-distribution splits, and the zero-shot pools differ per reader.

### Relevance for Zeiger (inferred)
- The **port idea** (a frozen engine in a shared space plus a per-reader ridge map, distilled against the engine's own
  answers) is a cheap way to move a head between backbones, and it could transfer to our head.
- Vega's results also argue against freezing the reader for browser grounding. Gains come from reading, and only a
  better reader delivers them.
- The **score rail** (ordinal centres on a learned 1-d line) is a small, transplantable inductive bias for `score`.
