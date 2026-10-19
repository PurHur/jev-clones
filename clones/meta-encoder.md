# MetaEncoder-30B (Meta, `facebook/meta-encoder`)

A **bi-encoder**, not a Jev clone in the strict sense. Meta fine-tunes its own multimodal decoder
`Muse-Glimmer-30B` with contrastive learning so that the request and each candidate become **one
L2-normalised last-token vector each**. A decision is `softmax(cos(q, c_k) / τ)` over the candidates. Because
the candidate vectors never see the request, they can be cached or put in an ANN index. The paper calls this a
"Multimodal System One Encoder" and benchmarks it against Jev-style models on JEVBench, ImaJEV and S1MB.

Source pins used for this profile:

| ref | what | where |
|---|---|---|
| `P` | arXiv 2610.11316**v1** (dated 2026-10-09, posted 2026-10-08), pdftotext, cited by line | `notes/hf-raw/_papers/arxiv_2610.11316v1_MetaEncoder.txt` (+ `.pdf`) |
| `HF` | model card README (visible although the repo is gated), API JSON at sha `1a78a4bd`, `tree.json` | `notes/hf-raw/facebook__meta-encoder/{files/README.md, api.json, api2.json, tree.json}` |
| `E` | `github.com/facebook/meta-encoder-eval` @ `ecc77922` (2026-10-09, "clean up"), shallow clone, read only, never run | `sources/meta-encoder/` |
| `B` | backbone config/README from the public mirror `unsloth/Muse-Glimmer-30B` | `notes/hf-raw/unsloth__Muse-Glimmer-30B/files/` |
| `S` | public Space `facebook/meta-encoder-space` (`app.py`, `data/gallery.json` with the model's own scores) | `notes/hf-raw/facebook__meta-encoder/space/` |
| `R` | S1MB results dataset `hotchpotch/s1mb-result`: runs `facebook__meta-encoder-cosine-t0.0184671662` (137 English tasks), `facebook__meta-encoder` (27 Japanese), `imguangyu__metaencoder_think` (137, CoT variant) | `notes/hf-raw/facebook__meta-encoder/s1mb/` |

**Not readable.** `modeling_metaencoder.py` (14,755 B), `vllm_metaencoder.py` (4,579 B), `USAGE.md`,
`NOTICE`, `config.json`, `chat_template.jinja` and the commit list all return **HTTP 401** anonymously
(`gated: "auto"`, logged in `notes/hf-raw/facebook__meta-encoder/_fetch_log.txt`). Everything about the
modelling code below therefore comes from the paper, the card, the eval repo (which calls the public
`MetaEncoder` class and quotes the training repo's collator) and the run settings in `R`. No weights were downloaded.

> **Main finding.** MetaEncoder is the **far "cacheable" end** of the design space. There is no decision head,
> no option markers, no mask engineering and no type-specific readout: one forward pass per *text* (request or
> candidate), last-token pooling, cosine, softmax with a temperature. It gets Jev-like closed-set accuracy by
> **listing the options inside the request prompt** (MMLU +18.1 pp, `P:366-367`), so the request vector "knows"
> the alternatives, and by rendering each candidate as **label + description** so it can act as a pointer back
> into that list (`P:396-445`). JEVBench orig 0.944 vs OpenJev-27B 0.986 (`P:339`). Calibration is a temperature
> fitted per eval, not a trained-for-calibration objective.

---

## Identity

| | |
|---|---|
| Repo (weights) | `facebook/meta-encoder`, created 2026-09-30, last modified 2026-10-13, 130 downloads / 27 likes (2026-10-19), **gated auto** (`HF:api.json`) |
| Name in paper/card | **MetaEncoder-30B**, "Multimodal System One Encoder" (`HF:README.md:16,23`) |
| Paper | *MetaEncoder: Exploring the Limit of Bi-Encoders for Multimodal System One Decision Making with Natural Language Interface*, Cheng, Sun, Singh, … , Xiao (14 authors, "Meta AI"), arXiv 2610.11316 (`P:1-7`) |
| Code | Only the evaluation harness is public: `facebook/meta-encoder-eval` (Apache-2.0). The training repo is referenced but not released (`E:metaencoder_eval/prompt.py:6-10`, `E:encoder.py:8-10`) |
| Demo | `facebook/meta-encoder-space` (gradio; it shows saved gallery scores, it does not run the model) (`S:app.py:1-12,143-145`) |
| Base | `meta-models/Muse-Glimmer-30B` (Meta Superintelligence Lab, released 2026-08, Apache-2.0), `base_model_relation: finetune` (`HF:README.md:3-4`, `B:README.md:36-37`) |
| Params | **29,776,626,688 BF16**, identical to the base (`HF:api.json safetensors`, `B:model.safetensors.index.json metadata`). Two shards, 49.9 GB + 9.65 GB (`HF:tree.json`) |
| Pipeline tag | `feature-extraction`; `transformersInfo.auto_model AutoModelForMultimodalLM`; `config.architectures ["MuseGlimmerForConditionalGeneration"]`, `model_type muse_glimmer` (`HF:api.json`) |
| Lineage | Meta's multimodal-embedding line (VLM2Vec-derived eval code, `E:NOTICE`; Think-then-Embed / TTE-flash, which share authors, `P:651-659`). It cites TypeSafe's Jev launch as the "recent trend" it answers (`P:35-36,766`) |

---

## Architecture

### Backbone (Muse-Glimmer-30B, from `B:config.json` and `B:README.md:52-77`)

| | |
|---|---|
| Type | dense causal transformer + perception encoder, `MuseGlimmerForConditionalGeneration` (`B:config.json:2-7`) |
| Text model | 52 layers, d = 6656, FFN 19,968 (SwiGLU, `silu`), 32 Q heads / **2 KV heads** (GQA 16:1), head_dim 128 (`B:config.json:17-21,132-134`) |
| Attention pattern | `[sliding, sliding, sliding, full]` × 13 → **39 sliding (window 2048) + 13 full** (`B:config.json:76-129,144`). "Gated attention: yes" (`B:README.md:64`) |
| Positions | RoPE θ = 500k on sliding layers only; `layer_rope_theta` is **0 on every full layer** (NoPE global layers) (`B:config.json:22-75`, `B:README.md:69`) |
| Other | `qk_scale_factor 3.87`, `output_multiplier 0.196`, `final_logit_softcapping 20.0`, `post_norm_eps 1e-8`, untied LM head (`B:config.json:16,135-138,145`) |
| Context | `max_position_embeddings 131,072` ("131,072+") (`B:config.json:130`, `B:README.md:74`) |
| Tokenizer | 202,048 = 200,000 BPE + 2,048 special; BOS `<|begin_of_text|>` 200000, EOS 200001, pad `<|finetune_right_pad|>` (`B:config.json:14-15,147`, `HF:api.json config.tokenizer_config`) |
| Vision | ~1.8B ViT-G/14 perception encoder, 50 layers (window + full every 4th, last layer full), width 1536, patch 14, temporal patch 2, merge 2; ≤ 4,096 tokens per image; video at 2 fps, `num_frames 96`, ≤ 144 tokens per frame (`B:config.json:151-221`, `B:processor_config.json`) |

**What is trained.** The paper says only that the decoder is "fine-tuned" into an encoder (`P:19,77-78,189`) and
the card says "contrastively fine-tuning" (`HF:README.md:36`). Neither says LoRA vs full weights, or whether
the vision tower was frozen. *Inferred*: the shipped checkpoint has exactly the base parameter count and an
index file of identical size (132,674 B in both repos), so it is a **full weight set with the same tensor
names, including `lm_head` and the vision tower**. That means either full fine-tuning or a merged adapter; the
LM head is carried along but unused for scoring. The base card describes its own perception encoder as frozen
(`B:README.md:242`).

### Input encoding (natural-language interface)

There is no schema, no special delimiter tokens and no option markers. Request and candidates are each
**rendered as free text** and wrapped in the same chat template (`P:819-828`):

```
system: Represent the user's input.             <- constant, on BOTH sides
user:   {instruction}{content}  (+ image / video content parts)
        chat template, add_generation_prompt=False -> sequence ends at the user turn's <|eot|>
        the hidden state of that final token is the embedding
```

- In the training repo's wire format an item is `[<|patch|> …][<|video|> ]{instruction}{content}`. The
  `` separator splits the instruction from the content, and `MetaEncoder` items carry them as
  `{"instruction", "text", "images"|"video"}` (`E:prompt.py:6-30`, `E:encoder.py:52-72`). Bare labels have no
  separator and are pure content (`E:prompt.py:15-16`).
- vLLM serving builds "the exact MetaEncoder prompt with a single BOS token" (`HF:README.md:211-214`).
- **Request**, Jev-style rows (JEVBench, ImaJEV), built by one shared function (`P:831-842`, `E:jevbench.py:17-37`):

```
{state}

Task: {instruction}
Criteria:
- {label}: {meaning of the label}
- {label}: {meaning of the label}
Options: a, b, c          <- the option list, only for closed sets
```

  - criteria may be a dict (`- key: value`) or a list (`- i: value`) (`E:jevbench.py:17-22`);
  - booleans are rendered as `yes: …` / `no: …` instead of True/False, "since a bare True is an opaque token" (`P:840-842`);
  - the S1MB adapter renders the state as key: value lines, with `query_prefix "Select the correct option."` and `state_serialization raw-state-json` (`P:843-847`, `R:*/settings`).
- **Candidates**: one encoding per candidate, with the same system prompt and only the candidate's own text (`P:828-829`). Three renderings were tested (`P:396-401`):

| | candidate text | MMLU | MMMU | MMMU-Pro | source |
|---|---|---|---|---|---|
| C0 | description only | 0.7456 | 0.5629 | 0.4916 | `P:405-408` |
| C1 | bare label `C` | 0.6512 | 0.4983 | 0.3803 | |
| C2 | label + description `C. red, oxidative, …` | **0.7767** | **0.5788** | **0.5042** | |

  The recommendation is C2 (`P:445`). JEVBench and ImaJEV, however, use **bare labels** (yes / no / option values / `unknown`), which "point back into the criteria block" (`E:jevbench.py:7-8`, `E:README.md:72-83,92-98`). MMLU/MMMU in the eval repo use letter + text (`E:tasks/closed_set.py:13-17,69-75`). Video uses `(A) text` (`P:882-883`).
- Options in the request: the closed-set prompt lists them; retrieval prompts do not (`P:308-312`). Training randomly **drops the option list for a fraction of closed-set examples**, so that the request embedding stays useful without it (`P:205-208`). The fraction is not given.
- Limits seen in the harnesses:
  - ImaJEV state is cut to **4,000 characters** (`E:imajev.py:45-48`);
  - S1MB runs use `max_input_tokens` 16,384 (English) / 8,192 (Japanese), `input_length_policy reject-overflow`, `padding_side left` (`R:*/clinc-choice-test-v1`, `R:facebook__meta-encoder/*`);
  - the vLLM example uses `max_model_len 8192` and ≤ 2 images per prompt (`HF:README.md:194-197`).

### Attention / isolation

- **The backbone's causal attention is kept unchanged.** There is no bidirectional conversion of the kind LLM2Vec uses, and no mask (`P:192-201`). The last token sees everything, including all visual tokens.
- **Request ↔ candidate: zero cross-attention.** That is the definition of a bi-encoder (`P:209-213`). Candidates never see the state, the instruction or each other. "The candidate embedding cannot see those criteria" (`P:135-137`) is exactly why the criteria and options are copied into the *request*.
- **Between questions: one request encoding per (state, question).** S1MB rows are emitted "per (case, decision) pair, because a single case may carry several decisions" (`P:846-847`). There is no shared-prefix packing, so questions are trivially isolated, and the state is re-encoded once per question. *Inferred*: a prefix KV cache over `system + state` would work with causal attention, but neither the paper nor the public code mentions one.

### Readout math

```
q   = h_last(request) / ||h_last(request)||_2          (P:195-197, eq. 1)
c_k = h_last(cand_k)  / ||h_last(cand_k)||_2
s_k = q . c_k                                         cosine, inner product of unit vectors (P:209)
p_k = exp(s_k / tau) / sum_j exp(s_j / tau)           over the question's own candidate set
```

- Embedding dimension: *inferred* d = 6656, the text hidden size, exposed as `model.embedding_dim` (`E:encoder.py:117`). `out_hidden_size 6144` in the config belongs to the vision projector. A cached candidate is ~26.6 KB in fp32.
- vLLM equivalent: `PoolerConfig(pooling_type="LAST", use_activation=True)`, "last token, L2-normalised" (`HF:README.md:198`).
- There is **no learned head of any kind**: no projection, no type embedding, no per-option bias. The decision rule is `argmax cos`, with ties going to the earlier candidate (`E:imajev.py:122`).
- The candidate-order note "gold first … only matters for exact ties" (`E:tasks/closed_set.py:17,76`) shows that scores are independent of candidate order **on the candidate side by construction**. Order inside the request's option list can still matter. Measured on MMLU over 10 permutations, the accuracy std is ≤ 0.003 for C0/C1/C2 (`P:449-457`, Fig. 1). That is a dataset-level accuracy spread, not a per-item probability shift.

```
               request side (encoded once per question)              candidate side (cacheable, request-independent)
 +-------------------------------------------------------------+     +-----------------------------------+
 | sys: Represent the user's input.                            |     | sys: Represent the user's input.  |
 | user: {state}\n\nTask: {instr}\nCriteria:\n- yes: ..\n- no: | x K | user: "yes"   (bare label, or     |
 |       ..\nOptions: yes, no   [+ image/video parts]   <|eot|>|     |        "C. text" for MMLU)  <|eot|>|
 +------------------------------------------------|------------+     +---------------------|-------------+
          Muse-Glimmer-30B, causal, 39 swa(2048)+13 full(NoPE)                same weights f_theta
                                                  v                                        v
                                     h_last -> L2 norm -> q (d~6656)          h_last -> L2 norm -> c_k
                                                  \_______________ q . c_k / tau ___________/
                                                                      |
                                                     softmax over the question's K candidates
                                         (open set: same scores against an ANN index of millions)
```

### Question types

There is no type field: every type is a candidate set.

| Jev type | MetaEncoder rendering | probability output | source |
|---|---|---|---|
| `noul` | 2 candidates, `yes` / `no` (bare). The request lists `yes: {desc}` / `no: {desc}`. ImaJEV adds a third `unknown` candidate for abstention. In S1MB the candidates are `Yes`/`No` with generic details ("the condition asked in the instruction holds under its stated rules") | softmax over 2 (or 3) cosines; P(true) = p_yes | `P:840-842`, `E:imajev.py:51-60`, `S:gallery.json s1mb_open_jev__*` |
| `choice` | one candidate per option (label, or label + description) | softmax over K | `P:831-838`, `E:jevbench.py:29-37,70-72` |
| `score` | one candidate per level, with the level's natural-language criterion. The numeric value is kept in a **sidecar** for graded credit and is never shown to the model | softmax over levels; the expected score comes from the sidecar values (S1MB: `normalized_expected_score_mae`) | `P:857-872`, `R:*score*.json.xz` |

There is no `box` output. `confidence` in Jev's sense is not produced; the harnesses compute their own metrics.

### Calibration

- **No trained calibration objective.** The loss is plain InfoNCE (below); the paper never mentions calibration, ECE or a temperature value (`P:209-238`).
- **The temperature is set by whoever evaluates.**
  - ImaJEV: one τ picked by NLL on the calibration split from the grid `{0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0}` (`E:imajev.py:40,77-85,195-198`). Its `confidence_source` is `softmax_cosine_calibration_split_temperature` (`E:imajev.py:134`).
  - S1MB English run: **τ = 0.0184671662** (logit scale ≈ 54.2), `probability_rule joint-cosine-softmax`. It was run on Meta-internal infra (`python 3.12.14+meta`, `torch 2.15.0a0+fb`, kernel `fbk`) on 2026-10-06 (`R:facebook__meta-encoder-cosine-t0.0184671662/*`).
  - S1MB Japanese run: τ = 0.03 (`R:facebook__meta-encoder/*`).
  - *Inferred*: 0.01847 is not a grid value, so it is either the training temperature or a continuous fit. Neither is documented. The τ used for the Space gallery probabilities is not stated.
- Effect visible in `R` (our unweighted means over the 59 English noul tasks): accuracy 0.822 at Brier skill 0.375. The CoT "think" variant reaches higher accuracy (0.868) but **lower Brier skill (0.324)**, because it emits one-hot probabilities (`probability_rule one-hot-verification-argmax`) (`R:imguangyu__metaencoder_think/*`). These are our aggregates, not board numbers.

### Context & limits

| | value | source |
|---|---|---|
| Backbone window | 131,072 | `B:config.json:130` |
| Tested request length | ≤ 16,384 tokens (S1MB English), 8,192 (vLLM example, S1MB Japanese); ImaJEV state ≤ 4,000 chars | `R`, `HF:README.md:195`, `E:imajev.py:47` |
| Candidates per question | unbounded on the scoring side. The paper frames closed sets as "< 256" (`P:20,246`), but the gallery has 1,000-way ImageNet/ChartQA, 894-way A-OKVQA and 174-way SSv2, and S1MB has 151-way CLINC. The real limit for *closed-set accuracy* is how many options fit in the request's option list | `P:20,246`, `S:gallery.json`, `R:clinc-choice-test-v1` |
| Open-set pool | "millions" via ANN; demo pools of 1k–37k (toolde 37,292) | `HF:README.md:33-34`, `S:gallery.json retrieval` |
| Media | ≤ 4,096 visual tokens per image; video 8–64 frames sampled from 64 extracted (eval); vLLM path has no video | `B:processor_config.json`, `P:466,893`, `HF:README.md:170-171` |
| Candidate cache in S1MB adapter | `candidate_cache_capacity 4096`, `candidate_batch_size 8` | `R:*/settings` |

---

## Training

Disclosed (`P:215-238`):

```
L = -(1/B) * sum_i log [ exp(s(q_i, c_i+)/tau) / sum_{c in N(q_i) ∪ {c_i+}} exp(s(q_i, c)/tau) ]      (eq. 2)
```

- **Unidirectional**: request → candidate only, with no symmetric candidate → request term (`P:216-222`).
- The negative set depends on the task (`P:232-238`):
  - **open set (retrieval)**: candidates gathered from the other requests of the **global batch** ("gathering across all devices", `P:80`) plus hard negatives where available. In-batch duplicates of the positive are removed;
  - **closed set (decision)**: **the request's own option set minus the gold**, so the training softmax is exactly the evaluation softmax.
- Option-list dropout on a random fraction of closed-set requests (`P:205-208`).

**Not disclosed:** data mix and size, τ (learned or fixed), lr, batch size, steps, hardware, LoRA vs full,
whether the vision tower is trained, and the number of hard negatives. The paper has no training-details
section. Clues from the eval repo, all **inferred** about the training mix:
- the "seven NanoBEIR subsets whose BEIR counterpart was in training" (MSMARCO, HotpotQA, FEVER, NFCorpus, FiQA2018, NQ, Quora) use the **e5-mistral** query instructions, which suggests e5-mistral/BEIR training data (`E:tasks/nanobeir_dataset.py:6-27`);
- video classification and retrieval prompts are "aligned to the trained … instruction" (HMDB51 form; MSR-VTT-style), which suggests MMEB/VLM2Vec-style video training sets (`E:tasks/video_classification_utils.py:20-28`, `E:tasks/vatex_dataset.py:15`);
- the parsers are adapted from VLM2Vec (`E:NOTICE`), which suggests the MMEB training recipe, plus decision data for the closed-set objective (sources unknown).

There is no Jev distillation claim and no proper-scoring-rule loss.

---

## Evals

**Headline** (`P:337-361` Table 2 = `HF:README.md:43-58`; per-task numbers in `E:reference_scores.json`):

| suite | MetaEncoder | OpenJev-27B | WeMM-Emb 9B | Qwen3-VL-Emb 8B |
|---|---|---|---|---|
| JEVBench orig (72) / easy (48) / hard (111) | 0.9444 / 1.0 / 0.7387 | **0.9861** / 1.0 / **0.7658** | 0.625 / 0.979 / 0.487 | 0.639 / 0.938 / 0.432 |
| "Jev Intelligence" | 81.4 | **84.5** | 45.8 | 42.1 |
| ImaJEV (dev, 173) | **0.8555** | – | 0.8117 | 0.7792 |
| MMLU (C2, options in prompt) | 0.7767 | **0.8601** (paper only; the card shows "–") | 0.7325 | 0.6544 |
| MMMU / VideoMMMU / NaturalBench / TempCompass | 0.579 / 0.59 / 0.813 / 0.735 | – | 0.566 / 0.485 / 0.802 / **0.753** | |
| NanoBEIR nDCG@10 | **0.6634** | – | 0.6175 | 0.6061 |
| MMEB-V3 image / video / visdoc | 0.790 / 0.649 / 0.815 | – | **0.810 / 0.743 / 0.833** | |

- "OpenJev (27B)" is `openjev/openjev` by Alex Wortega, a Qwen3.5-based cross-encoder (`P:775-776`). The JEVBench numbers above are tiny-n: 72 rows, so 0.944 vs 0.986 is a 3-row difference.
- ImaJEV calibration split accuracy is 0.9012 (n = 81). The test split is scored by the maintainers, and no test number is given (`E:README.md:11`, `E:reference_scores.json:15-25`).
- **Ablations** (`P:366-372, 405-445, 535-546`):
  - options in the request: MMLU 0.5645 → 0.7456 (+18.1 pp, 56/57 subjects better), MMMU +7.5 pp, MMMU-Pro(10) +7.3 pp;
  - the bare-letter candidate (C1) still reaches 0.651 on MMLU (chance 0.25), "a clean demonstration that the model resolves a pointer". On MMMU its deficit grows with K: +0.028 at 3 options, −0.079 at 4, −0.134 at 5, −0.173 at 10 (`P:438-444`);
  - **encoder vs the same backbone as a constrained one-token decoder** (logits restricted to A–Z): MMLU 0.7740 vs 0.7743 (tie); MMMU 0.5730 vs 0.4962; Video-MMMU 0.5415 vs 0.5267. The two agree on 72–82% of items (`P:535-546`);
  - **reasoning is the bottleneck**: unconstrained CoT decoding gives 0.7202 on MMMU. Prepending Muse-Glimmer's CoT to the encoder request gives **0.7155**, and 0.6871 with the final label stripped from the CoT (`P:572-587`). This is the basis of the `metaencoder_think` S1MB variant (`R:imguangyu__metaencoder_think` settings: reasoner = Muse-Glimmer-30B, `max_new_tokens 1024`, `candidate_verify` scoring, one-hot output, 59.5B total parameters counted).
- **S1MB** (Table 7, 110 tasks, accuracy, `P:963-1082`): the best are `open_jev__browser_control_v1` 1.0, `laya__enron_spam` 0.96, workflow controls 0.89–0.93 (these mirror TypeSafe's evals.typesafe.ai workflows). The worst are argument_quality 0.27, stepgame 0.28, esci 0.33, tic_tac_toe 0.38, drone_control 0.44, and `laya__typed_decisions` (score) 0.556. Our unweighted means of `baseline_adjusted_score` over the English run in `R`: noul 0.581, choice 0.615, score 0.310, overall 0.554 (think variant: 0.684 / 0.747 / 0.622 / 0.701). **`score` is the weakest type**, which is consistent with the bi-encoder having no ordinal structure.
- **Not on** the community Decision Index board snapshot (`notes/hf-raw/_spaces/dix_leaderboard.md`). No ECE is reported anywhere in the paper.
- Latency: not reported. S1MB English, 137 tasks / 26,269 decisions, took ~6,000 s total wall time at case batch size 1 on one GPU (`R`, `elapsed_seconds` summed). That is ~0.23 s per decision including candidate encoding, an upper bound and not tuned serving.

---

## Serving

- **HF / PyTorch**: `MetaEncoder.from_pretrained(path)` from the repo's own `modeling_metaencoder.py` (`snapshot_download` + `sys.path`), `transformers>=5.15`, `torchvision<0.27`. Pinned in the eval repo: `transformers==5.15.1`, `torch==2.9.1`, `torchvision==0.24.1` (BICUBIC vs LANCZOS resize changes image embeddings), optional `flash_attn` (`HF:README.md:66-79`, `E:requirements.txt`, `E:README.md:24-25`).
  - API: `model.match(task, candidates, top_k)` → `Match(index, score, candidate)`, best first; `model.encode(items)` → tensor; `match(task, index_tensor)` reuses a pre-encoded pool, "the scores are identical" (`HF:README.md:86-163`).
  - A full bf16 copy is ~62 GB, so 80 GB GPUs are needed; eval runs one copy per GPU × 8 (`E:README.md:27-28`).
- **vLLM 0.23.0** on the Transformers backend, `runner="pooling"`, `LAST` pooling with L2 norm, `enforce_eager=True` ("torch.compile does not handle this model's patched modules yet"), `FLASH_ATTN`, text + image only (`HF:README.md:168-209`).
  - `vllm_metaencoder.register()` exists because vLLM's generic backend "drops the weight-less norm inside MuseGlimmer's token embedding and resizes its other weight-less norms". `to_prompt()` builds the prompt with a single BOS (`HF:README.md:211-214`). The file itself is gated.
  - Parity: cosine 0.9998 on text and 0.993 on images; on 200 ImageNet images, top-1 85.5% (vLLM) vs 84.5%, with the same prediction on 97.5% of images. Multi-GPU and `vllm serve` are "not verified yet" (`HF:README.md:216-219`).
- There is **no `/v1/systemone` endpoint**, no typed-question API and no Jev wire compatibility. The caller renders the prompt and the candidate strings. The S1MB adapter shows one way to map Jev requests (`renderer text-choice-option-id-v1`, `R`).
- Batched bf16 is not bit-exact across batch compositions: tasks move by "a question or two" (`E:README.md:62-64`).

---

## Licence

- Weights: **Apache-2.0** (`HF:cardData.license`), "inherited from the base model", plus Muse-Glimmer's **Usage Policy** (`USAGE_POLICY.md`, gated) and a NOTICE of changes (`HF:README.md:221-225`). Access requires accepting the gate (automatic).
- Base Muse-Glimmer-30B: Apache-2.0 (`B:README.md:2,37,235`). Its training data includes "information from Meta's products and services" (`B:README.md:76`).
- Eval code: Apache-2.0, with parts adapted from VLM2Vec (Apache-2.0) (`E:LICENSE`, `E:NOTICE`).

---

## Notable design choices vs Jev and vs kev

| | Jev (reference) | kev | MetaEncoder |
|---|---|---|---|
| Interaction | state read once, questions in parallel, options read jointly as an ordered list (`jev-reference.md` §3) | one packed/row pass; options and `<decide>` in the same causal context; pointer `</opt>`·`<decide>` | **none between request and candidate**; options are visible to the request only as text in the prompt |
| Head | unknown (pointer or slot head suspected) | `PointerHead`, `W_q`, `W_k`, 256-d, trained | **no head**: cosine of two last-token vectors |
| Backbone use | unknown (decoder suspected) | frozen Qwen + LoRA (full FT at 27B), causal | Muse-Glimmer-30B, causal kept, fine-tuned (scope undisclosed) |
| Multi-question | many questions per call, isolated | block-causal mask or rows over a cached state | one request encoding per (state, question), no prefix sharing |
| Candidate cost | grows with options in context; ≤ 255 | ≤ 255, all in context | candidates **cacheable and indexable**, millions for retrieval |
| Types | noul/choice/score/box with distinct readouts | 3 types → one pointer softmax | 3 types → one cosine softmax; score via sidecar values |
| Calibration | RLCD, "optimised against outcomes" | one post-hoc T in `head.pt` | **τ chosen per evaluation** (0.0185 / 0.03 / grid fit); InfoNCE only |
| Multimodal | images in 1.14 (box) | no | image + video natively, on both sides |
| API | `/v1/systemone` | `/v1/systemone` clone | Python `match`/`encode`, vLLM pooling |

1. **It is the opposite design point to every decoder clone.** Hume's and kev's reconstruction of Jev puts state, question and options in one context and reads a pointer. MetaEncoder removes all request↔candidate attention and gets the "joint view" back through **prompt duplication**: options and criteria are written into the request text, and each candidate's label acts as a key back into that list (C1 ≈ pointer, `P:438-441`). That is cheaper than a cross-encoder and allows caching. It cannot model a candidate in the light of the state, which is why `score` and multi-step tasks suffer.
2. **"System One" is used loosely.** No typed contract, no noul/score semantics and no confidence formula. Probabilities exist only once someone picks a τ. For Jev-style use, the integration layer (the S1MB adapter, ImaJEV harness) does the typing, rendering and temperature work.
3. **Reasoning, not the scorer, is the stated bottleneck** (`P:572-587`): a single vector matches a constrained decoder on MMLU and beats it on MMMU, but both trail CoT by ~20 pp. Meta's answer is to prepend generated CoT (the "think" variant), which gives up the System One latency and calibration advantage.
4. **Order robustness is structural on the candidate side** (scores do not depend on the other candidates at all). This differs from kev's default (order-sensitive, with option isolation as an opt-in) and from Jev's documented first-option lean. Order in the request list still enters through the causal prompt, measured as only ≤ 0.003 accuracy std.
5. **Size and serving.** At 30B dense and ~62 GB bf16 it is heavier than Kev-27B and far heavier than the 4B clones. It has no prefix cache across questions, no `/v1/systemone`, and its modelling code is gated. As an export target or baseline for Zeiger it is the reference for **"cacheable candidate embeddings"** (useful for very large option sets, e.g. a browser element shortlist before a cross-attending head), not a drop-in decision model.
