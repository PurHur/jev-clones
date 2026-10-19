# Zeiger (ours)

## Identity
- Code: https://github.com/PurHur/zeiger (Apache-2.0), engine only, 5 commits up to 3ce3973 (2026-09-22)
- Weights: https://huggingface.co/php-ai/zeiger-0.6b (CC-BY-SA-4.0, pre-alpha), holds training round 17
- Training: PurHur/laya-browser (private), `longctx` track of the night loop
- Scope: **browser grounding**. Given a page and a step instruction, pick an element or "none". General decision
  data (incl. structured reasoning) is mixed in, but the target task is a web page.

## Architecture
**Backbone**: `Qwen/Qwen3-0.6B-Base` (28 layers, hidden 1024), LM head removed, **all weights fine-tuned** (no
LoRA). The causal decoder is used as an encoder. `zeiger/model.py:15`

**Attention**: plain causal attention, mask-free (`is_causal=True` SDPA, no L×L mask is built, so SDPA stays on the
flash path). There is **no block mask between questions**: each question is its own sequence/row.
`zeiger/model.py:22-40`

**Input encoding** (`zeiger/encoding.py:118-140`), one question per sequence:

```
<state JSON ≤1536 tok> \n <kind> question: <instructions ≤512 tok> \n
 opt0 ≤200 tok <M>  opt1 <M>  …  optK-1 <M>  <eos>          <M> = <|object_ref_end|> (reserved Qwen token)
```

The marker's hidden state summarises its option. Because attention is causal, option *i* sees the state, the
question and options 0..i-1, but not the options after it.

**Chunked-prefix encoding** for long pages (`encoding.py:143-176`): if the question does not fit 2,048 tokens, every
chunk repeats `state + question` and carries one run of options. Attention cost is linear in page size, and the
window is the head's limit (`max_len` 131,072) rather than the backbone's. Trained up to 32,768 tokens.

**Readout head** (~18M params, `model.py:43-73`):
1. gather the marker hidden states from every chunk into one option set
2. LayerNorm + **question-type embedding** (choice / score / noul)
3. **2-layer bidirectional TransformerEncoder over the options** (16 heads, FFN 2×width, pre-norm), so options are
   compared with each other in both directions. Cost O(K²) in the number of options, never O(L²) in page length
4. MLP scorer, one logit per option, then softmax over the options

**Question types**: all three become option sets. `choice` uses its options as given (`"none"` is a regular
option), `score` uses `level i: <criterion>`, and `noul` becomes two options, `false: …` / `true: …`
(`encoding.py:36-44`). One shared head serves all three.

**Limits**: 40 to 1,500 options tested. Options that do not fit are refused (`"options do not fit the window"`),
never truncated.

**Calibration**: temperature per option-count bucket stored in the checkpoint (`temperature_by_options`, e.g.
`choice:11+ → 1.234`), refit on val.

## Training (laya-browser)
- Loss: CE over option markers − 0.5·spherical score; later rounds add a decision-cost reward (0.3) and Brier (0.05)
- AdamW 1e-5 (encoder) / 1e-4 (head), warm-up + cosine, about 400–500 steps × 32k tokens per round on a single Strix Halo
  (ROCm, gfx1151), about 1,300 tok/s
- Data: Common Crawl element tables (jevpilot's static analyzer), a polite live crawl, Mind2Web/MT-Mind2Web, judge-verified
  jevpilot steps, whole-page "long" questions, general decision data plus structured-reasoning families
  (sort/graph/relation, tables/joins/lists/timelines)
- Model soups were used (round 12 was rebuilt as `2·soup(12,15) − round15`)

## Evals
- Promotion score = 0.4 short + 0.4 long + 0.2 general + 0.1 structured
- Round 17 on Jev's random general sample: **0.824** vs Jev's 0.850 (our own type-stratified general test differs)
- Latency on GPU: about 140 ms per short question, 1.5 s per whole page; on CPU (bf16, 16 cores) 340 ms at 10 options

## Serving
`python serve.py` → `POST /decide` with `{state, questions}`, which resembles System One but is **not** wire-compatible.
Supports ROCm, CUDA and CPU. llama.cpp cannot run the architecture.

## Notable choices vs Jev
- One question per sequence instead of many questions packed into one pass (no block-causal mask), which costs
  re-encoding the state for each question
- Options are compared in a separate set transformer instead of only in the backbone
- Chunked long context targets whole web pages (Jev caps choice at 255 options; we go to 1,500)
- Domain-specialised (browser) instead of a general decision model
