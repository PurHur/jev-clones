# Zeiger next: architecture decisions worth trying

Written 2026-10-19 from the clone profiles in `clones/`. Generated from `data/zeiger_next.json` by `scripts/build_site.py`; edit the JSON. Builds on the laya-browser roadmap of 22 Sep (ids like R4, T3, A1 refer to it).

**Baseline.** Zeiger round 17: Qwen3-0.6B-Base, full fine-tune, causal backbone, one sequence per question, option markers + 2-layer set head, chunked 2k long context, temperature per option count. Short 0.900, whole pages 0.793, general 0.807 (Jev 0.850), sorting/relations 0.620, multi-step logic 0.248.

**Findings these attack** (from the laya-browser roadmap):

- **F1** Recipe plateau: rounds 12–18 trade skills instead of adding them (round 18 +1.7 whole pages, −2.7 short)
- **F3** Goal-level instructions on whole pages: 0.378 (right element in top 50 for 95%, top 1 for 38%)
- **F4** Options far apart on a page cannot be compared (org chart 1 hop 0.85 close vs 0.21 spread)
- **F5** Precondition shortcut: clicks the named button even when disabled (0% from depth 1)
- **F6** GPU time is the scarce resource: 2 rounds per night, 26% padding
- **FG** General decisions 0.807 vs Jev 0.850; calibration breaks on unfamiliar questions (logic ECE 0.37)

## Ranked list

| # | Decision | Attacks | Gain | Cost | Effort | Roadmap |
|---:|---|---|---|---|---|---|
| 1 | [KL anchor to the previous champion](#kl-anchor) | F1 | large | low | 1 round | new (complements R2/R3) |
| 2 | [Minimal edit pairs that flip the answer](#edit-pairs) | F5, F3 | large | low | data, 1 round | new (sharpens D5) |
| 3 | [Head that reads every chunk's tokens, not only the markers](#memory-head) | F4, F3 | large | medium | new head, 1–2 rounds | extends R4 |
| 4 | [Encode the page once, one row per question](#shared-page) | F6, F3 | medium | medium | engine + 1 round | new |
| 5 | [Order-free set module](#order-free) | F3 | small | low | head, 1 round | extends T3 |
| 6 | [Interpolate towards the base: scaled fine-tune delta](#wise-ft) | FG, F1 | medium | evaluation only | eval only | extends T1 |
| 7 | [Input-dependent temperature, certified thresholds](#calibration) | FG | medium | evaluation only | fit + eval | extends T6 |
| 8 | [Bidirectional full-attention layers in a hybrid](#noncausal-layers) | F3, F4 | medium | medium | 1 round on the 0.8B | extends A1 |
| 9 | [Decision-token query next to the markers](#decision-query) | F3 | small | low | head, 1 round | new |
| 10 | [Distil from permissive top clones, soft targets](#teachers) | FG | medium | medium | labels + rounds | extends D1 |
| 11 | [One head, a linear port per backbone](#head-port) | F6 | small | low | bake-off tooling | extends A2 |
| 12 | [Cacheable shortlist, then cross-attending rerank](#shortlist) | F3 | medium | low | inference + eval | extends T2 |
| 13 | [Looped depth: more compute for hard questions](#looped) | F5, F4 | medium | high | new backbone | new |
| 14 | [WebJev's step format and data](#web-format) | F3 | medium | low | data + serving | extends R6, M7 |
| 15 | [Start yes/no and score readouts from LM-head rows](#lmhead-init) | FG | small | low | head, 1 round | new |
| 16 | [Distilled small encoder for a CPU tier](#small-tier) | F6 | small | medium | new student | new |

## Which base models each decision works on

| Decision | Qwen3 0.6B / 1.7B (dense causal decoder (today)) | Qwen3.5 0.8B / 2B / 4B (DeltaNet + full-attention hybrid) | EuroBERT-610m / mmBERT / Ettin (bidirectional encoder) | Gemma 4 E2B / E4B (sliding + full attention) |
|---|---|---|---|---|
| 1. KL anchor to the previous champion | yes | yes | yes | yes |
| 2. Minimal edit pairs that flip the answer | yes | yes | yes | yes |
| 3. Head that reads every chunk's tokens, not only the markers | yes | yes | yes | yes |
| 4. Encode the page once, one row per question | yes | with changes | no | yes |
| 5. Order-free set module | yes | yes | yes | yes |
| 6. Interpolate towards the base: scaled fine-tune delta | yes | yes | yes | yes |
| 7. Input-dependent temperature, certified thresholds | yes | yes | yes | yes |
| 8. Bidirectional full-attention layers in a hybrid | with changes | yes | n/a | with changes |
| 9. Decision-token query next to the markers | yes | yes | yes | yes |
| 10. Distil from permissive top clones, soft targets | yes | yes | yes | yes |
| 11. One head, a linear port per backbone | yes | yes | yes | yes |
| 12. Cacheable shortlist, then cross-attending rerank | yes | yes | yes | yes |
| 13. Looped depth: more compute for hard questions | no | no | no | no |
| 14. WebJev's step format and data | yes | yes | yes | yes |
| 15. Start yes/no and score readouts from LM-head rows | yes | yes | with changes | yes |
| 16. Distilled small encoder for a CPU tier | n/a | n/a | yes | n/a |

<a id="kl-anchor"></a>
## 1. KL anchor to the previous champion

**From:** [Amazon Strands decider](clones/amazon-strands-decider.md)

**What they do.** Strands trains every release with a KL term (weight 0.3) pulling the student's option distribution towards a frozen earlier release (hobson-v19). The KL is over option slots, so it even works across tokenizers.

**Why it should help Zeiger.** Our rounds trade skills: each round improves one slice and forgets another. Anchoring on the champion's distributions makes the next round add instead of trade.

**What we would change.** Add KL(student ‖ champion) over the option logits at weight 0.1–0.3 to train.py. Champion logits can be precomputed once per round file.

Attacks F1 · expected gain large · cost low · 1 round · roadmap: new (complements R2/R3)

_Base models:_ Architecture-agnostic: it only needs option logits from both models.

<a id="edit-pairs"></a>
## 2. Minimal edit pairs that flip the answer

**From:** [Bespoke Nimble](clones/bespoke-nimble.md)

**What they do.** Nimble's main idea is in the data: pairs of examples where an edit of at most 8 words to one fact flips the label, so the model cannot answer from the surface.

**Why it should help Zeiger.** The precondition shortcut is exactly this failure: the model clicks the button the instruction names even when it is disabled. Pairs that differ only in 'disabled', 'required field empty' or 'checkbox unticked' teach the fact that matters.

**What we would change.** Generate pairs from crawled forms: same page and instruction, flip one precondition attribute in the element table, label moves from the target to the blocking element (or none).

Attacks F5, F3 · expected gain large · cost low · data, 1 round · roadmap: new (sharpens D5)

_Base models:_ Pure data; no model change.

<a id="memory-head"></a>
## 3. Head that reads every chunk's tokens, not only the markers

**From:** [Cloudflare Clef](clones/cloudflare-clef.md)

**What they do.** Clef's ~128M head first lets every option cross-attend to the whole prompt's hidden states (memory), then runs 4 layers over the options. Options get evidence from anywhere in the input, not only from their own position.

**Why it should help Zeiger.** Facts spread across chunks fail because options in different chunks only meet through one marker vector each. Cross-attention from each option to the token states of all chunks lets a count, a maximum or a 2-hop chain see the evidence directly.

**What we would change.** Keep the chunked backbone. In the head, add cross-attention from option vectors to all chunk token states (or to a few pooled summary tokens per chunk for memory), then the existing set layers. Keep Clef's question isolation problem out: one question per head pass.

Attacks F4, F3 · expected gain large · cost medium · new head, 1–2 rounds · roadmap: extends R4

_Base models:_ Head-only change; cost grows with options × memory tokens, not with page length squared.

<a id="shared-page"></a>
## 4. Encode the page once, one row per question

**From:** [Kev 1.0](clones/kev.md), [Liquid d1-3B](clones/liquid-d1.md), [Mapika decider](clones/mapika-decider.md), [WebJev](clones/webjev.md)

**What they do.** Kev 1.0, Strands, decider and d1 compute the state once, cache it, and continue each question as its own branch. Isolation is exact and the state is paid for once. d1 shows that conv layers need only the last 2 inputs; Kev 1.0 copies DeltaNet recurrent state per branch.

**Why it should help Zeiger.** jevpilot asks several questions about one page (which element, which operation, is the step done). Today each question re-encodes up to 32k tokens. WebJev's split into an operation question plus a target question only becomes cheap with a shared page.

**What we would change.** Put the page first, cache its KV once per chunk, then append each question's instruction + options as a branch. With pure attention (Qwen3) this is a KV fork or kev gen-1's block-causal mask.

Attacks F6, F3 · expected gain medium · cost medium · engine + 1 round · roadmap: new

_Base models:_ Hybrid: copy the DeltaNet recurrent and conv state per branch (Kev 1.0 shared_prefix.py). Encoder: impossible, the page would depend on the question.

<a id="order-free"></a>
## 5. Order-free set module

**From:** [OpenJev (ejhshen)](clones/openjev.md), [Standard One SH](clones/standardone.md), [Juspay Xor](clones/misc-company-clones.md)

**What they do.** ejhshen/OpenJev feeds option vectors to a set transformer with no positional information, then a bilinear scorer: option order cannot change the answer. Xor gets the same effect at 2× cost by scoring the original and reversed order and merging.

**Why it should help Zeiger.** Our set head is already a transformer over markers. Removing any position signal there is nearly free, and the causal backbone's left-to-right bias is then the only order effect left, measurable with Xor's two-order trick.

**What we would change.** Strip position from the head input (no learned position, shuffle option order in training), add a reversed-order eval to measure residual bias.

Attacks F3 · expected gain small · cost low · head, 1 round · roadmap: extends T3

_Base models:_ Free with an encoder backbone, where options already see each other both ways.

<a id="wise-ft"></a>
## 6. Interpolate towards the base: scaled fine-tune delta

**From:** [interfaze lev](clones/lev.md), [Kev-27B](clones/kev.md), [Strands seed soup](clones/amazon-strands-decider.md)

**What they do.** lev serves its LoRA at 0.25 of full strength and gains 0.689 → 0.743 on S1Bench (0.728 when tuned only on held-out families). Kev-27B averages its full fine-tune 0.85/0.15 with its LoRA version (WiSE-FT). Strands concatenates three seed LoRAs so the update is their exact mean.

**Why it should help Zeiger.** Our general and unfamiliar-question scores lag, and calibration collapses out of distribution. Pulling the fine-tuned weights part of the way back to the base usually trades a little in-domain accuracy for a lot of robustness.

**What we would change.** Evaluate θ = base + α·(champion − base) for α in 0.5…1.0 on the frozen and sealed tests. No training. The backbone mixes; the head stays at full strength.

Attacks FG, F1 · expected gain medium · cost evaluation only · eval only · roadmap: extends T1

_Base models:_ One evening of evaluation.

<a id="calibration"></a>
## 7. Input-dependent temperature, certified thresholds

**From:** [Von](clones/von.md), [Amazon Strands decider](clones/amazon-strands-decider.md), [basal-1.5](clones/group-language.md), [pplx-decider](clones/pplx-decider.md)

**What they do.** Von sets the temperature from entropy, state length and option count. Strands fits one temperature per question type against ECE rather than NLL. basal publishes confidence thresholds certified at 1% and 5% error. Perplexity only keeps a checkpoint whose dev ECE and Brier are within 0.01 of a reference.

**Why it should help Zeiger.** Our short ECE is 0.029, whole pages 0.047, logic 0.37. A temperature that sees page length, option count and entropy fixes the spread; a certified threshold is exactly what jevpilot's 'act or hand off' gate (T5) needs.

**What we would change.** Fit T(features) on val; publish per-type certified thresholds with the checkpoint; add an ECE/Brier guard to promotion.

Attacks FG · expected gain medium · cost evaluation only · fit + eval · roadmap: extends T6

_Base models:_ Post-hoc; ships in config.json next to temperature_by_options.

<a id="noncausal-layers"></a>
## 8. Bidirectional full-attention layers in a hybrid

**From:** [vLLM-SR d3](clones/vllm-sr.md), [pplx-decider v1.1+](clones/pplx-decider.md), [H2O lightning](clones/h2o-lightning.md)

**What they do.** d3 (#2 on the board) and pplx-decider v1.1+ switch only the full-attention layers of Qwen3.5 to noncausal (padding mask only) with a forward hook; DeltaNet layers stay causal. Options and the state then see each other both ways in a quarter of the layers.

**Why it should help Zeiger.** Ettin shows an encoder beats a decoder at equal size, but switching to a full encoder costs our chunking and long context. This keeps the Qwen3.5 hybrid (262k native, our 4B track) and still lets options see later options.

**What we would change.** On the Qwen3.5 track: mask hook on full-attention layers only. Costs prefix caching (decision 4) for the noncausal part, so pair it with one question per row.

Attacks F3, F4 · expected gain medium · cost medium · 1 round on the 0.8B · roadmap: extends A1

_Base models:_ Dense Qwen3: all layers would go bidirectional (A1, breaks our mask-free flash path). Encoder: already bidirectional.

<a id="decision-query"></a>
## 9. Decision-token query next to the markers

**From:** [kev](clones/kev.md), [Amazon Strands decider](clones/amazon-strands-decider.md), [vLLM-SR Decision 2.0](clones/vllm-sr.md)

**What they do.** kev and Strands put a decision token after all options; its hidden state, which has read every option, is the query of a pointer head against each option's end token. Decision 2.0 adds an MLP term to the bilinear score.

**Why it should help Zeiger.** In a causal backbone our first options never see later ones. A decision token placed after the options of each chunk sees them all; using it as the query (or as an extra input to the set head) gives every score a view of the whole option list.

**What we would change.** Append a decision token per chunk; feed its state into the set head and add a bilinear(query, marker) term to the scorer.

Attacks F3 · expected gain small · cost low · head, 1 round · roadmap: new

_Base models:_ Small change; most useful while the backbone stays causal.

<a id="teachers"></a>
## 10. Distil from permissive top clones, soft targets

**From:** [pplx-decider](clones/pplx-decider.md), [vLLM-SR d3](clones/vllm-sr.md), [Kev-27B](clones/kev.md), [Mapika decider](clones/mapika-decider.md)

**What they do.** pplx-decider (65.6) and d3 (66.3) weights are Apache-2.0, Kev-27B Apache-2.0, decider Apache-2.0 (some training data CC BY-SA / NC). Perplexity trains on soft annotator distributions; Laya found its RL term adds nothing over soft cross-entropy.

**Why it should help Zeiger.** General decisions are 4 points behind Jev, and we already distil Kev-8B. A 27B teacher at 65+ on the board is a much stronger general teacher, with licences that allow it, unlike distilling Jev.

**What we would change.** Label the general pool with d3-flash (9B) or pplx-decider (27B, fits 128 GB unified memory in bf16) by day; train on their full distributions.

Attacks FG · expected gain medium · cost medium · labels + rounds · roadmap: extends D1

_Base models:_ Check each teacher's data licences before publishing a student; never use Jev-distilled teachers (AutoTrust).

<a id="head-port"></a>
## 11. One head, a linear port per backbone

**From:** [vegaml](clones/vegaml.md)

**What they do.** vegaml's vega-U shares one 14.3M engine across readers from 0.8B to 27B; each backbone attaches through its own linear port, without retraining the engine.

**Why it should help Zeiger.** A backbone bake-off (A2) currently starts every candidate's head from scratch. Moving our trained set head through a learned port gives each new backbone a warm head, so 150-step bake-off runs measure the backbone, not head warm-up.

**What we would change.** Train a linear port (d_backbone → 1024) for each candidate with the head frozen for the first steps, then unfreeze.

Attacks F6 · expected gain small · cost low · bake-off tooling · roadmap: extends A2

_Base models:_ The enabler for trying the other base models quickly.

<a id="shortlist"></a>
## 12. Cacheable shortlist, then cross-attending rerank

**From:** [Meta MetaEncoder](clones/meta-encoder.md), [Contrastive-LM CLM](clones/clm.md), [Decima (small encoders)](clones/group-small-encoders.md)

**What they do.** MetaEncoder and CLM embed state and candidates separately and score by cosine, so candidate embeddings can be cached. Decima scores every option independently at linear cost on one CPU core.

**Why it should help Zeiger.** On goal-level whole pages the right element is in our top 50 for 95% of questions but first for 38%. A cheap first stage keeps 32–64 candidates; the full model then reads only those, with more context per option.

**What we would change.** Stage 1: Zeiger's own pass (or a small bi-encoder) keeps the top 32. Stage 2: re-ask on those with nearby text and section headings.

Attacks F3 · expected gain medium · cost low · inference + eval · roadmap: extends T2

_Base models:_ Stage 1 can be the current model; no new training needed to test it.

<a id="looped"></a>
## 13. Looped depth: more compute for hard questions

**From:** [OpenLAIR lev (looped)](clones/lev.md)

**What they do.** OpenLAIR's lev runs one checkpoint at loop depth 1 to 4 on Ouro looped backbones; Lev-2.6B goes from 63.2% to 76.2% on JevBench with more loops.

**Why it should help Zeiger.** Multi-step logic sits at chance (0.248) and goal-level questions need inference steps. Adaptive depth spends compute only on the hard questions.

**What we would change.** Experiment only: one round on Ouro-1.4B with our head, compare depth 1 vs 4 on the logic and goal-level slices.

Attacks F5, F4 · expected gain medium · cost high · new backbone · roadmap: new

_Base models:_ Needs a looped base (Ouro / Nanbeige); check the base licence first. Long-context behaviour unknown (2k–64k).

<a id="web-format"></a>
## 14. WebJev's step format and data

**From:** [WebJev-35B-A3B](clones/webjev.md)

**What they do.** WebJev asks a 'which operation' question (incl. DONE) and then one 'which target' question per allowed operation, with richer element fields; ~64k verified live-web decisions; speaks jev-ultrafast's format with a 125-task live benchmark.

**Why it should help Zeiger.** Zeiger answers only 'which element'. An explicit operation/DONE question covers jevpilot's 'is this step finished' need, and the 125-task benchmark measures task success, not just top-1.

**What we would change.** Add the operation question type to our data and serve.py; adopt the jev-ultrafast wire format; run their benchmark weekly (M7).

Attacks F3 · expected gain medium · cost low · data + serving · roadmap: extends R6, M7

_Base models:_ Check WebJev's dataset licence before training on it.

<a id="lmhead-init"></a>
## 15. Start yes/no and score readouts from LM-head rows

**From:** [pplx-decider](clones/pplx-decider.md), [AutoTrust](clones/autotrust-jev.md), [Cloudflare Clef](clones/cloudflare-clef.md)

**What they do.** pplx-decider and AutoTrust initialise their readout rows from the base model's LM-head rows for the code tokens; Clef adds a word-similarity prior from LM-head rows to its learned score.

**Why it should help Zeiger.** Our noul and score questions go through the same marker head as choices, starting from zero. An LM-head prior ('true', 'false', digits) gives them a sensible start; this matters most when switching backbones.

**What we would change.** Add a prior term: dot(option marker state, LM-head row of the option's label word), learned scale.

Attacks FG · expected gain small · cost low · head, 1 round · roadmap: new

_Base models:_ Encoders have an MLM head instead; the same trick works with its rows.

<a id="small-tier"></a>
## 16. Distilled small encoder for a CPU tier

**From:** [Crivo-150M](clones/group-small-encoders.md), [Von](clones/von.md)

**What they do.** Crivo-150M (from an Ettin 150M reranker) scores 22.2 on the board, above 0.6–0.8B decoders such as Kev-0.8B (13.7). Von runs ModernBERT-large at ~0.1 s on CPU.

**Why it should help Zeiger.** Zeiger on CPU takes 340 ms at 10 options and 4.9 s at 150. A distilled 150M encoder student of the champion could serve CPU-only users, with Zeiger as the fallback.

**What we would change.** Distil the champion's distributions into Ettin-150M / mmBERT-base with chunking; target short questions first.

Attacks F6 · expected gain small · cost medium · new student · roadmap: new

_Base models:_ A deployment tier, not a better Zeiger.
