# Jev (TypeSafe), the reference model

Reference profile of the hosted model that the open "Jev clones" try to replicate. Compiled 2026-10-19.

**Tags.** `[official]` means TypeSafe docs, site, blog, SDK source, or statements by TypeSafe staff.
`[reconstruction: X]` means a third-party inference or measurement by X. `[our measurement]` means our own
API usage. Where sources disagree, both are given.

**Key sources**

- Docs: <https://docs.typesafe.ai/llms.txt> (index). Mainly [/api](https://docs.typesafe.ai/api), [/models](https://docs.typesafe.ai/models), [/model-jaggedness/jev-1.13](https://docs.typesafe.ai/model-jaggedness/jev-1.13), [/confidence](https://docs.typesafe.ai/confidence), [/primitives](https://docs.typesafe.ai/primitives), [/introduction/machine-learning-primer](https://docs.typesafe.ai/introduction/machine-learning-primer), [/images](https://docs.typesafe.ai/images) and [/primitives/box](https://docs.typesafe.ai/primitives/box)
- Blog: [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (2026-09-15) and [Jev 1.14](https://typesafe.ai/blog/jev-1-14) (2026-10-14). Evals site: <https://evals.typesafe.ai>
- SDKs, source read from the tarballs and never executed: `typesafe-sdk` 0.7.5 on [PyPI](https://pypi.org/project/typesafe-sdk/) and `@typesafe-ai/sdk` 0.7.0 on [npm](https://www.npmjs.com/package/@typesafe-ai/sdk)
- Archer Hume, [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked) (2026-09-17). Based on about 10k API calls against jev-1.13.0. It is his only Jev post; his other post, "Move Fast and Kill People", is unrelated.
- [HN launch thread](https://news.ycombinator.com/item?id=49717558) (1989 points, 520 comments). The CEO posts as `CompleteSkeptic` and staff member Nathan as `zenlikethat`.
- [Latent Space podcast with Diogo Almeida](https://www.latent.space/p/jev) and [TechCrunch 2026-09-18](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/)
- kev (jaredpalmer/kev), local checkout `/home/ai/projects/kev` (README, PLAN.md, `runs/jev-*`). It measures Jev through Vercel AI Gateway.

---

## 1. Identity

| Item | Value | Tag |
|---|---|---|
| Vendor | TypeSafe AI ("Made in SF"). Founder and CEO: Diogo Almeida (ex-OpenAI, InstructGPT/RLHF co-author). Co-founders: Erik Gafni and Sasha Sheng. | [official] [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev); co-founder names from [TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/) |
| Funding | $40M seed, led by DCVC | [reconstruction: press, e.g. [winzheng](https://www.winzheng.com/en/article/typesafe-ai-jev-system-one-model-launch)]; not on typesafe.ai |
| Model class | "System One model", the first of its kind. Typed decisions plus probabilities, no generated text. | [official] [/concepts/system-one](https://docs.typesafe.ai/concepts/system-one) |
| Public launch | 2026-09-15 (blog post and HN). Early access with a waitlist. Came "after two years in stealth"; company founded about Aug 2024. | [official] blog; founding date from [ai.engineer bio](https://ai.engineer/speakers/diogo-almeida) |
| Versions | `jev-1.13.0` (launch version; the docs example responses report it). `jev-1.14.0`, available from 2026-10-14. | [official] [/models](https://docs.typesafe.ai/models) |
| Aliases | `jev-latest` → `jev-1.13.0` ("most recent stable", SDK default). `jev-preview` → `jev-1.14.0`. The 1.14 blog says `jev-latest` moves to 1.14.0 "in a week", so around 2026-10-21. 1.13.0 gets 60 days' deprecation notice; 1.14.0 gets 30. | [official] /models, [1.14 blog](https://typesafe.ai/blog/jev-1-14) |
| Version stability | "Jev models won't change once deployed." No LTS promise, though 1.13.0 might become one. | [official] CEO on [Latent Space](https://www.latent.space/p/jev) ~00:48 |
| Distribution | Hosted only, at `api.typesafe.ai`. Also on Vercel AI Gateway as `typesafe-ai/jev` (providers: typesafe-ai, digitalocean). Also listed on Cloudflare. Weights are not released. | [official] /api; [Vercel page](https://vercel.com/ai-gateway/models/jev) |
| SDKs | Python `typesafe-sdk`: 0.5.7 (2026-09-14) to 0.7.5 (2026-10-14). JS `@typesafe-ai/sdk`: 0.5.7 (2026-09-11) to 0.7.0 (2026-10-14). Repos: `github.com/typesafe-ai/typesafe-sdk-python`. | [official] changelogs ([py](https://docs.typesafe.ai/sdk/python/changelog), [js](https://docs.typesafe.ai/sdk/javascript/changelog)) |
| Customisation | No fine-tuning or LoRA on customer data. The same weights serve every account. Not trained on customer requests. | [official] /models |
| Language | English is primary and best. Other languages, including CJK, are handled "but not equally well". 1.14 has its largest gain in multilingual. | [official] /models, 1.14 blog |

## 2. API contract

### Endpoint
- `POST https://api.typesafe.ai/v1/systemone` with `Authorization: Bearer <key>`. `GET /v1/models` lists the aliases, with name, description, and release_date. [official] [/api](https://docs.typesafe.ai/api), [/models](https://docs.typesafe.ai/models)
- Body: UTF-8 JSON, at most 32 MiB both compressed and decompressed. Nesting at most 64 levels. Numbers must be finite. `Content-Encoding` may be gzip, x-gzip, or zstd (window ≤ 8 MiB). SDKs zstd-compress bodies of 1 KiB or more. [official] /api
- Errors: 400 (rule violation; `max_tokens_exceeded` means the input exceeds the context length), 401, 413, 415, 422 (validation), 429 (rate limit, measured in tokens/s and requests/s per org), 529 (overloaded). SDKs retry 429/5xx/529 with backoff and honour `retry-after` and `retry-after-ms`. [official] /api, /models, SDK `_core/retry.py`

### Request
```jsonc
{
  "model": "jev-latest",                    // or jev-1.13.0, jev-1.14.0, jev-preview
  "state": "<string | object | array>",     // shared content, encoded once
  "questions": {                            // map; keys are YOUR ids, "not sent to the model"
    "q1": {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},   // criteria optional
    "q2": {"type": "choice", "instructions": "...", "criteria": {"opt_a": "desc", "opt_b": null}},  // map option -> desc|null
    "q3": {"type": "score",  "instructions": "...", "criteria": ["level 0 desc", "level 1 desc", "..."]}, // ordered list
    "q4": {"type": "box",    "instructions": "the mug in `photo`"}          // 1.14 only, images header
  },
  "assets": {"<uuid>": {"type": "image", "format": "jpg", "data_base64": "..."}}  // optional, images header
}
```
- `instructions` and `criteria` values can be a string, an object, or an array. Structured JSON is encouraged; instructions refer to state fields with backticks, e.g. `` `order.charges` ``. [official] [/primitives/advanced](https://docs.typesafe.ai/primitives/advanced)
- Question ids are not sent to the model. [official] /primitives. Hume found the id length counts in billed tokens, which suggests the count is computed after inference. [reconstruction: Hume]

### Response
```jsonc
{
  "model": "jev-1.13.0",
  "answers": {
    "q1": {"type": "noul",   "noul": 0.95},                                     // no confidence field
    "q2": {"type": "choice", "choice": "opt_a", "probabilities": {"opt_a": 0.61, "opt_b": 0.39}, "confidence": 0.22},
    "q3": {"type": "score",  "score": 1.4, "legend": {"0": "...", "1": "...", "2": "..."},
           "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}, "confidence": 0.35},
    "q4": {"type": "box", "box": {"top": 0.25, "bottom": 0.82, "left": 0.44, "right": 0.84}, "confidence": 0.92}
  },
  "usage": {"input_tokens": 328, "output_tokens": 34}                          // output tokens are free
}
```
[official] /api, primitive pages, SDK `_schemas/models.py`

### Question types and limits
| Type | Output | Limits | Tag |
|---|---|---|---|
| `noul` (from "Bernoulli") | P(yes) in [0,1] | `criteria` optional (`true`/`false` descriptions) | [official]; the name's origin from the CEO on [HN](https://news.ycombinator.com/item?id=49718407) |
| `choice` | argmax option, `probabilities` (sum ≈ 1), `confidence` | 2–255 options. The launch post: "cardinality up to 255, with a two-stage process above that". Option text may be null, in which case the model reads the name alone. | [official] /primitives/choice, blog |
| `score` | probability-weighted expected level, per-level `probabilities`, `legend`, `confidence` | 2–10 levels (the SDK schema only enforces ≥ 1) | [official] /primitives/score, SDK |
| `box` | 4 edges as fractions of the image; "each coordinate is the expected value of its edge's predicted position"; confidence may be null | 1.14 only, requires header `TypeSafe-Danger-Zone-Images: true`, "dreamer-ready" (experimental) | [official] /primitives/box, SDK schema |
| Questions per request | **No limit on count.** Only the context limits apply. | Hume tested 1,500 questions in a few hundred ms, and a 23k-token state with 5,000 questions fitting. Our ~256 ceiling is our client batching cap, not an API limit. | [official] /primitives, /models; [reconstruction: Hume]; [our measurement] |
| Generation | Strings and sequential outputs are "not allowed at all", so that all outputs can be computed in parallel | | [official] CEO on [HN](https://news.ycombinator.com/item?id=49719122) |

### Context limits
- Default: **64k tokens per request** (state plus all questions, state counted once) and **32k tokens per question** (state plus one question). [official] /models. Hume measured about 65,536 and about 32,768. [reconstruction: Hume]. Vercel lists the context as 32,000. [official/partner: Vercel]
- Header `TypeSafe-Danger-Zone-Long-Context: true` raises both limits to **262k**. It is "slower and less accurate". [official] /models. "Jev supports longer context than production enables." [official] 1.14 blog

### Pricing and latency claims
- $0.042 per M input tokens ($42 per B). Output tokens are free. The launch post says pricing "may be subsidized". [official] /models, blog
- `output_tokens` is a billing figure, not a count of decoding steps. A 200-option question reported 1,911 output tokens; a 255-option one reported 2,714. [reconstruction: Hume]
- Latency: "70ms to 500ms" end to end, "40x–200x faster" than frontier LLMs [official, blog]. Home page: 0.114 s vs 8.566 s, "193.6x faster", "444.6x cheaper" [official, typesafe.ai]. Cookbook mean round trips: 111–114 ms [official, llms-full cookbooks]. Batching 13 questions into one call: "12.2x cheaper, 10.0x faster" on the cookbook page, but "11.5x / 9.6x" on /primitives; the docs contradict themselves. [official]
- Server-side times: 44–218 ms for 360–30k state tokens with one question; 50–610 ms for 1–1,500 questions on a short state; 200 options about as fast as 2. [reconstruction: Hume]
- 0.25–0.7 s per call round trip [our measurement]. kev measured a median of 218 ms and p95 of 371 ms via Vercel Gateway [reconstruction: kev `runs/transfer-jev-v1/report.json`].

### Calibration and confidence
- "Trained for calibrated decisions: probabilities optimized against outcomes." Calibration is a group property, not a per-answer guarantee. [official] /concepts/system-one, AI primer
- `confidence` is **not learned**. It is a closed-form function of the returned probabilities [official, /confidence]:
  - noul: none returned. The suggested equivalent is `|2p − 1|`.
  - choice: `(p_max − 1/n) / (1 − 1/n)`
  - score: `max(0, 1 − Σ p_i·|i − m| / MAD_unif)`, where m is the modal level and MAD_unif is the mean absolute deviation of a uniform distribution measured from the middle level.
- Hume's section title "Train the distribution, then calculate confidence" agrees with this. [reconstruction: Hume]

### Documented weaknesses of jev-1.13 (reviewed 2026-10-02)
[official] [/model-jaggedness/jev-1.13](https://docs.typesafe.ai/model-jaggedness/jev-1.13), in order:
1. Literal reading
2. Math and numbers (counting, hex/RGB, interpolating between score levels; "score levels are weak in numerical calibration")
3. Date and time comparison
4. Indirection and multi-hop reasoning
5. Large state full of irrelevant detail ("Jev suffers from context rot")
6. Adversarial content and prompt injection in state
7. Contradictory instructions and criteria
8. **Choice option order**: leans toward the first option
9. Not built for generation

The CEO's own claim that Jev is "by far the best at not degrading in long context" ([Latent Space](https://www.latent.space/p/jev) ~01:25) conflicts with item 5.

## 3. Architecture

Overall, TypeSafe discloses **almost nothing** about internals. "Architecture is close to the chest for now, but we have talked about writing a paper" [official, CEO on [HN](https://news.ycombinator.com/item?id=49718824)]. On Latent Space he "probably shouldn't talk too much about the insides of ML", and the RLCD paper is "not yet" out. The launch post says only that Jev has "a new model architecture, a parallel sampler, and a new training method (RLCD)" [official]. Everything else is reconstruction.

| Aspect | What is known or claimed | Tag | Confidence |
|---|---|---|---|
| Model family | "Transformer-based", "not a large language model" | [official, per TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/) | medium (press paraphrase) |
| Built on a pretrained LM? | The RLCD diagram shows "pretrained language models" branching into RLHF, RLVR, and RLCD, with RLCD presented as post-training. The CEO: "if you gave me a billion dollars, I wouldn't pre-train"; mid-training is "a cost-saving thing". Outside observers suspect an open-weight base. | [official] AI primer, [Latent Space](https://www.latent.space/p/jev) 01:48–01:49; [reconstruction: TechCrunch] | medium-high that a pretrained LM is the starting point; base unknown |
| Which base | Unknown. The tokenizer matches none of 192 public tokenizers (415 probes). Closest is Qwen (348/415); the vocabulary tracks o200k, but per-digit splitting rules o200k out. "Rules out an unchanged public tokenizer, not a public base model." | [reconstruction: Hume] | low |
| Size | Not disclosed. "Different sizes of Jev"? "Absolutely" (future). Hume suggests a **sparse MoE with ~10B active params**, because about 30k tokens in ~160 ms rules out a dense 70B on 8×H100. He labels this "the least certain part". An HN skeptic guessed about 1B; not credible. | [official] Latent Space 01:12; [reconstruction: Hume] | low |
| Encoder vs decoder | Hume assumes a **causal decoder** as "the likely choice" but says his experiments cannot tell a causal decoder from a bidirectional encoder. One HN commenter guessed "stripped-down text diffusion". | [reconstruction: Hume] | low-medium |
| Decoding | No autoregressive decoding. "Outputs all probabilities in parallel instead of autoregressively generating by token". Strings are banned so that every output can be computed in parallel. The meaning of "parallel sampler" is undefined. | [official] blog, CEO on HN | high (no decoding) / unknown (what the sampler is) |
| State encoding | State is "ingested once", and every question is evaluated against it in parallel. State counts once against both limits. Hume: consistent with a shared prefix KV cache plus per-question causal suffixes (citing Hydragen and DeFT as analogues, not confirmed). Question tokens cost about 2× state tokens. CEO: "pay for a long state once and ask many questions". | [official] /models, Latent Space 01:37; [reconstruction: Hume] | high (state shared once) / medium (prefix-KV mechanism) |
| Attention between questions | **Isolated.** "Every answer is independent. One question's answer is not hidden context for another. You can add or remove questions without changing the others' results." Hume: a secret placed in a sibling question gives 0.00, while the same secret in state gives 0.90–0.92. The exact mask is unknown; it is commonly reconstructed as block-causal (state prefix visible to all, branches mutually masked). | [official] /primitives; [reconstruction: Hume]; kev implements block-causal | high (behavioural isolation) / medium (mask shape) |
| Options within a Choice | Options are **read jointly as an ordered list**, not scored independently. Information placed after the options is usable (a "reference card" in the last slot gives 16/16; first 12/16; middle 11/16). Adding an irrelevant 5th option shifts the log-odds between two others by −0.28 (95% CI −0.36 to −0.19), which rejects fixed independent logits plus softmax. Order changes the probabilities (ticket reversal: 0.84–0.89 → 0.93–0.96). Docs: leans toward the first option. Option boundaries cannot be forged: injected fake options never displaced real ones across 7 delimiter formats. | [reconstruction: Hume]; [official] jaggedness #8 | high (joint/ordered reading, unforgeable boundaries) |
| Option scoring | Unknown. Hume's candidates are (a) a final-position slot or classifier head over numbered slots, mapped back to caller keys, or (b) a **pointer scorer**: the decision state dot-producted with each option's own hidden state. A 200-option copy test (correct answer 1.00 at every position) "suggests a pointer" but he calls it inconclusive. Reserved label tokens are possible, though the fake-option test weighs against plain-text markers. kev and most clones use a pointer over `</opt>` hidden states against a `<decide>` token placed after all options. | [reconstruction: Hume]; [reconstruction: kev README] | low-medium |
| Readout: noul | Hume's suggestion: one scalar plus a sigmoid. He reports no direct evidence for sigmoid vs 2-way softmax. The API returns only p, no confidence. | [reconstruction: Hume]; [official] /confidence | low |
| Readout: choice | Softmax over K = 2..255 options; K is set per request. | [reconstruction: Hume]; [official] limit | medium |
| Readout: score | Distribution over ordered levels (2–10); `score` is the expectation. Officially "weak in numerical calibration" between levels. Whether there is an ordinal loss or structure is unknown. | [official] /primitives/score, jaggedness; [reconstruction: Hume] | high (distribution plus expectation) / unknown (head) |
| Readout: box (1.14) | Each edge is "the expected value of its edge's predicted position", which implies a distribution over discretised positions per edge (4 heads or bins), not regression. Billed 21 output tokens in the doc example. | [official] SDK schema description | medium (inferred from the wording) |
| Multimodality | 1.14 docs: "Image assets require a **vision-language model**". The 1.14 blog says jev-1.13.0 "was multimodal all along". The /models and /concepts pages still say text-only. This is evidence of a VLM backbone, or at least a vision tower. | [official] SDK `_schemas/models.py`, 1.14 blog | medium-high |
| Context window | 64k per request / 32k per question by default; 262k with the danger-zone header ("supports longer context than production enables"). Suggests a backbone with ≥ 262k native or extended positions. | [official] /models, 1.14 blog | high (limits) / low (backbone implication) |
| Tokens | Input count: noul = 4 shared + 15 per answer + length of the id tokens; adding a question costs +8. Counts are additive across questions. Answer values don't change the count. Per-digit tokenisation. | [reconstruction: Hume] | medium |
| Determinism | Repeated identical calls vary slightly. CEO: determinism trades off against intelligence per dollar; "robustness > determinism". | [reconstruction: Hume]; [official] Latent Space | high |
| Calibration method | **RLCD** ("Reinforcement Learning for Calibrated Decisions"), a post-training method on a pretrained LM. Recipe unpublished. Probabilities are "optimized against outcomes". Hume guesses a proper scoring rule (log or Brier loss); which one is unknown. No post-hoc temperature is documented. | [official] AI primer; [reconstruction: Hume] | high (name) / unknown (recipe) |
| Training data | **All synthetic** ("Yep"). TypeSafe calls itself a "data lab", owning "statistically well-understood synthetic data". No customer data, partly because real data's power-law bias causes overfitting. "Data is probably far more interesting than architecture." | [official] Latent Space ~00:21, TechCrunch, CEO on [HN](https://news.ycombinator.com/item?id=49718824) | high (synthetic) / unknown (content) |
| Serving | GPU-constrained; "really early in our infra days". Over 1T tokens/day, mostly machine traffic (CEO claim). | [official] Latent Space | n/a |

## 4. Performance claims and public benchmarks

**Official**
- The CEO is "extremely anti-public benchmarks"; TypeSafe publishes no MMLU-style numbers and no calibration curves. [official] Latent Space ~00:17
- [evals.typesafe.ai](https://evals.typesafe.ai) has four workflows, scored against labels averaged from GPT-6 Astra and Fable 5.1 at high reasoning. Jev runs in workflow mode only.

  | Workflow | Jev accuracy | Cost per case | Time per case |
  |---|---|---|---|
  | Security incidents | 61.7% | $0.0001 | 0.3 s |
  | Agent-trace observability | 71.6% | $0.0003 | 0.5 s |
  | Invoice processing | 61.8% | $0.0011 | 0.5 s |
  | Customer service | 76.0% | $0.0001 | 0.4 s |
  | **Mean** | **67.8%** | **$0.0004** | **0.4 s** |

  For comparison: Sonnet 5 67.8%, Terra 67.9% at 10.1 s and $0.0304, Opus 5 73.1% at 37.8 s and $0.176, Sol 74.1% at 23.3 s and $0.084. TypeSafe notes the workflows were designed in-house and may be biased.
- "Zero hallucinations / never makes type errors" is not an empirical number; it follows from schema matching. [official] blog
- Cookbook: CLERC legal re-ranking top-1 accuracy 5% → 18% and top-10 38% → 62% over BM25 (40 queries). [official] [/cookbooks/rerank_typesafe](https://docs.typesafe.ai/cookbooks/rerank_typesafe)
- Partner anecdotes: Vercel's command-safety classifier was 5–18× faster than the OpenAI model it replaced, with better accuracy. In an email-classification test, Gemini was slightly more accurate but 10–20× more expensive. [reconstruction: TechCrunch]

**Third party**
- Calibration on 1,200 MMLU items: **ECE 0.031**. 990 predictions fall in the 0.9–1.0 bin (predicted 98.7%, observed 96.3%). [reconstruction: Hume]
- Fresh generated maths [reconstruction: Hume]:
  - 3-digit multiplication: 86.7% accuracy at mean p 0.83
  - 2-step word problems: 32% at 0.30
  - Modular exponentiation: 56% at 0.35
- Hume also cites MMLU-Pro 84.6% only as a size comparison; it is not a Jev score. [reconstruction: Hume]
- kev's frozen suites, run via Vercel Gateway [reconstruction: kev `runs/jev-*`, README]:
  - decision-v4 dev: acc **0.845**, Brier 0.237, ECE 0.067
  - transfer-v4 dev (out of domain for kev): acc **0.857**, Brier 0.211, ECE 0.049, confident-error rate 3.7%
  - Held-out policy-pair "both correct": 0.86
  - Per source on transfer-v4: QNLI 0.93, SciQ 0.99, TweetEval-offensive 0.81, PAWS 0.79, MMLU 0.90, Emotion 0.59, deadline date-arithmetic 0.93
  - Option permutation: 0 argmax flips out of 36–60, but mean max probability delta 0.016–0.024. kev's own warning: "zero argmax flips do not prove architectural invariance".
- Speed: 0.25–0.7 s per call round trip, with ~256 questions in one call working. [our measurement]

## 5. Open questions

1. **Backbone identity and size.** Is it an open-weight base (Qwen-family is suggested by the tokenizer proximity) with a modified tokenizer, or an in-house mid-trained model? Dense or MoE? Active and total parameter counts?
2. **Encoder vs decoder.** Causal or bidirectional attention inside the state and inside each branch? Nothing published separates the two.
3. **Exact mask.** Block-causal with a shared prefix KV, or separate forward passes per question batched over a cached prefix? Do option spans attend to each other bidirectionally? Hume's results show options interact and later options can inform the decision; the order bias suggests causal reading.
4. **What the "parallel sampler" is.** It might be only the batch scheduling of branches. Is there any iterative or refinement step (the diffusion-style hypothesis), or is it one forward pass?
5. **Option scoring head.** Pointer over option hidden states, slot or classifier head, or LM logits on reserved label tokens? How are 255 options and the "two-stage process above 255" handled?
6. **Readout per type.** Noul as a sigmoid vs a 2-way softmax over true/false criteria. Score as a K-way softmax vs an ordinal or cumulative-link head. Box binning resolution.
7. **RLCD.** Reward (log, Brier, or other proper score), RL vs supervised on outcome labels, and whether there is any post-hoc temperature or isotonic step. How the synthetic data is generated and from which teacher models.
8. **Context mechanics.** Why do the limits sit at 32k per question and 64k per request when 262k is unlocked? Is it a native long-context base or extended RoPE? How does the ~2× cost of question tokens relate to compute?
9. **Vision.** Which vision encoder is used, and was 1.13 really multimodal from the start, as the 1.14 blog claims? The docs say text-only.
10. **What changed in 1.14.** Weights, data, or head? No numbers have been published.
11. **Determinism source.** Non-batch-invariant kernels vs deliberate sampling noise.
12. **Inconsistencies to keep in mind:**
    - Batching speedups: 12.2×/10.0× vs 11.5×/9.6× in different doc pages.
    - "Text only" vs "multimodal all along".
    - "Context rot" (jaggedness page) vs "best at not degrading in long context" (CEO).
    - Vercel lists a 32,000 context vs TypeSafe's 64k per request / 32k per question.
