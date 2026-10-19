---
license: apache-2.0
library_name: vega
pipeline_tag: text-classification
tags:
  - typed-decisions
  - decision-model
  - system-one
  - calibration
  - conformal-prediction
  - multimodal
base_model: Qwen/Qwen3.5-0.8B
---

<img src="assets/vega-icon.svg" width="68" height="68" alt="Vega" />

# Vega 0.8B: public intents

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZxS_Ee4-oyfo8ePKiAgbehv5I1Rh3F7O?usp=sharing)

Inference on a single T4, end to end, in the browser.


**800 million or 4 billion parameters. A 73,728-token context. Images. Running on your own
hardware.**

That combination is the point: a decision model small enough to sit inside your own process, that can
read a whole contract in one pass and look at a picture.

![Where Vega comes out ahead of a hosted reference](assets/vega_benchmark_wins.png)

<sub>Every place Vega beat a hosted reference in one run on 2026-10-11, two Tesla T4s, 3,841
decisions per system. The win list is derived from the run's own JSON, so a row that stops being a
win disappears by itself. [The rest of the menu](#the-rest-of-the-menu) goes the other way and is
on this card too.</sub>

A new architecture with its own training and calibration method. The language model inside it is
**frozen and never trained**; everything learned lives in a small physics engine on top of it.

| | |
|---|---|
| sizes in this repo | **0.8B** at the root (default) and **4B** under `4b/` |
| backbone | `Qwen/Qwen3.5-0.8B` or `Qwen/Qwen3.5-4B`, frozen, SHA-256 pinned |
| trained parameters | engine 57 MB (0.8B) or 124 MB (4B), plus gated adapters |
| **context** | **73,728 tokens** |
| **images** | yes, the frozen backbone's vision encoder, via `vega_vision.py` |
| answer types | `choice`, `boolean` (probability a statement is true), `score` |
| latency | **0.267 s/decision** (T4, fp16) · **0.28 s** (Apple M-series, fp32) |

---

## The two sizes

Both are in this repository and take the same code path. The 0.8B sits at the root because that is
where it has always been; the 4B is under `4b/`.

```python
import vegaml

v = vegaml.load()        # 0.8B, the default
v = vegaml.load("4b")    # 4B, same interface
```

| | 0.8B | 4B |
|---|---|---|
| frozen backbone | Qwen3.5-0.8B, 24 layers, d 1024 | Qwen3.5-4B, 32 layers, d 2560 |
| trained engine | 14,342,512 params, 57 MB | 30,879,088 params, 124 MB |
| task adapter | 835,937 params | 1,431,905 params |
| with the task adapter | 0.763 | **0.803** |
| soft accuracy | 0.680 | **0.725** |
| Brier | 0.314 | **0.270** |
| calibration error | 0.026 | **0.019** |
| per decision | **22 ms** | 28 ms |

`LocalLLaMA/typed-decisions` test split, 2,050 decisions, each checkpoint's own run of the same
harness, full tables in `eval/typed_decisions.md` and `4b/eval/typed_decisions.md`.

Per workflow with the adapter, the 4B leads on five of six:

| workflow | 0.8B | 4B |
|---|---|---|
| apibank | 0.895 | **0.965** |
| when2call | 0.904 | **0.946** |
| appliances | 0.851 | **0.891** |
| banking77 | 0.714 | **0.749** |
| esci | 0.564 | **0.644** |
| clinc150 | **0.880** | 0.867 |

The 4B is the better model on nearly everything measured here. The 0.8B is still the default,
because 22 ms against 28 ms and 57 MB against 124 MB settles more deployments than four points of
accuracy does, and because every Jev comparison on this card was run on the 0.8B.

## 73,728-token context

`max_len` is **73,728 tokens**, and the architecture is built for it rather than merely permitting it:

* **Read-once prefix caching.** A state of at least 4,096 tokens is run **once** with a KV cache; every
  question about that state then continues from a copy of it. Twelve questions about one 73k-token
  contract cost roughly one read, not twelve.
* **A memory reader.** Long states are cut into 16 equal pieces, mean-pooled at the deepest feature
  layer, and fed to the engine alongside the pooled spans, so evidence late in a long document still
  reaches the decision.
* **No silent truncation.** An input that would not fit is **refused**, never quietly shortened. A
  decision made on a truncated contract is worse than no decision.

The frozen backbone carries 262,144 rotary positions, so 73k sits well inside its range.

## Images

The backbone is multimodal and its vision encoder is frozen with the rest, so the same typed
interface takes a picture. Needs the `vision` extra, which carries Pillow and torchvision; the
backbone's processor loads torchvision on the way in and refuses to run without it.

```python
# pip install "vegaml[vision]"
import vegaml
from PIL import Image

v = vegaml.load()                       # or load("4b")
v.decide_image(Image.open("invoice.png"), {
    "kind": {"type": "choice", "instructions": "What kind of document is this?",
             "criteria": {"invoice": "a bill", "receipt": "proof of payment",
                          "contract": "an agreement", "other": "none of these"}}})
```

Images are resized to fit `max_pixels` (default 1024·32·32). An image answer carries the same fields
as a text answer: the calibrated probabilities, the conformal set and the abstain flag.

**Requires `vegaml` 0.8.0 or later.** Before that release the packaged image path raised
`ModuleNotFoundError` on the first call, because it reached for a module that ships only in the
training tree. The figures below were measured there, where that module exists; they are unchanged,
but 0.7.1 and earlier cannot reproduce them from PyPI.

The figures below are all text; the vision path is functional but not covered by them.

---

## Where Vega surpasses the hosted model

The figure at the top of this card, as text. Every row of the 2026-10-11 run where a local model
beat the hosted one, and nothing else.

| benchmark | Vega 0.8B | Vega 4B | Jev |
| --- | ---: | ---: | ---: |
| **Public tasks it wins, of the 43 on the menu** | | | |
| unfair_tos, 57 rows, +12 | 0.737 | **0.877** | 0.667 |
| tweet_stance_abortion, 55 rows, +6 | 0.364 | **0.545** | 0.436 |
| prompt_injections, 55 rows, +5 | 0.746 | **0.855** | 0.764 |
| tweet_sentiment, 59 rows, +4 | 0.424 | **0.593** | 0.525 |
| tweet_offensive, 59 rows, +3 | **0.780** | **0.780** | 0.729 |
| enron_spam, 60 rows, +3 | **1.000** | **1.000** | 0.950 |
| jailbreak_cls, 58 rows, +2 | 0.983 | **1.000** | 0.966 |
| glue_rte, 60 rows, +2 | 0.633 | **0.883** | 0.850 |
| emotion, 60 rows, +2 | 0.550 | **0.650** | 0.617 |
| toxic_chat_toxicity, 60 rows, +1 | 0.950 | **1.000** | 0.983 |
| superglue_cb, 45 rows, +1 | 0.667 | **0.822** | 0.800 |
| clinc150_oos, 60 rows, +1 | 0.683 | **0.900** | 0.883 |
| **Slices it wins** | | | |
| JevBench family:temporal_numeric, 15 items | **0.467** | 0.400 | 0.200 |
| DecisionBench domain:privacy, 25 items | 0.880 | **1.000** | 0.960 |
| DecisionBench domain:public_sector, 25 items | 0.760 | **0.960** | 0.880 |
| **Calibration and cost, whole suites** | | | |
| Public menu, negative log likelihood | 1.099 | **0.768** | 0.878 |
| Public menu, p50 ms per decision | **74** | 106 | 162 |
| DecisionBench, calibration error | 0.103 | **0.098** | 0.122 |
| DecisionBench, negative log likelihood | 1.628 | **1.420** | 3.196 |
| DecisionBench, KL to the gold distribution | 1.012 | **0.854** | 6.099 |
| DecisionBench, rows answered | **1.000** | **1.000** | 0.997 |
| DecisionBench, p50 ms per decision | **85** | 245 | 163 |

Three kinds of win are in there and they do not carry the same weight.

**The calibration block is the one to trust.** It is measured on all 1,075 DecisionBench rows and
the margins are wide: the likelihood is better by a factor of two, the distance to the gold
distribution by a factor of seven. The hosted model is confident at 0.874 on rows it gets right
0.752 of the time, which is the expensive kind of wrong. Vega also answers every row, and the 0.8B
answers in 74 ms on a T4 with nothing leaving the machine.

**The task wins are thin.** A public task is at most 60 rows, so the `+n` in each label is the
whole margin in rows. Eight of the twelve are three rows or fewer. `unfair_tos` sits only 2 rows
above its own majority class while the hosted model sits 10 below it, and `tweet_stance_abortion`
is 7 rows below its majority class, so on those two neither model beats answering the same thing
every time. The one that survives a careful look is `prompt_injections`: 0.855 against 0.764, 15
rows above a 0.582 majority class.

**The slices are hints.** `temporal_numeric` is 4 net rows of 15 and both DecisionBench domains
are one and two rows. Worth chasing at a larger n, not worth quoting.

---

## The rest of the menu

The other side of the same run, so the figure at the top of this card has a frame. 43 public tasks
at up to 60 rows each, the JevBench public tiers, and DecisionBench: 3,841 decisions per system,
with the hosted reference called live over the same rows. Zero-shot, `mode="engine"`, no head
fitted, nothing tuned on evaluation data.

| metric | Vega 0.8B | Vega 4B | Jev |
| --- | ---: | ---: | ---: |
| **Public menu, 43 tasks, macro average** | | | |
| accuracy | 0.556 | 0.717 | **0.836** |
| skill, chance corrected | 0.388 | 0.594 | **0.755** |
| macro F1 | 0.486 | 0.649 | **0.796** |
| expected calibration error | 0.151 | 0.148 | **0.115** |
| negative log likelihood | 1.099 | **0.768** | 0.878 |
| area under the risk coverage curve | 0.362 | 0.169 | **0.077** |
| latency per decision, p50 ms | **74** | 106 | 162 |
| **JevBench, 231 items** | | | |
| accuracy | 0.545 | 0.662 | **0.866** |
| accuracy, hard tier, 111 items | 0.414 | 0.513 | **0.730** |
| paraphrase consistency, 36 groups | 0.694 | 0.722 | **0.972** |
| expected calibration error | 0.178 | 0.169 | **0.067** |
| **DecisionBench, 1,075 items** | | | |
| accuracy | 0.509 | 0.633 | **0.752** |
| expected calibration error | 0.103 | **0.098** | 0.122 |
| negative log likelihood | 1.628 | **1.420** | 3.196 |
| KL to the gold distribution, 10 rows | 1.012 | **0.854** | 6.099 |
| coverage, rows answered | **1.000** | **1.000** | 0.997 |
| latency per decision, p50 ms | **85** | 245 | 163 |

The hosted model is more accurate across the board, and the gap is widest where wording changes:
it answers a paraphrase the same way 0.972 of the time against 0.722 for the 4B. What the local
models win is honest probabilities. On DecisionBench the calibration error, the likelihood and the
distance to the gold distribution all go to Vega, because the hosted model is confident at 0.874
on rows it gets right 0.752 of the time. The local models also answer every row, and the 0.8B
answers in 74 ms on a T4 with nothing leaving the machine.

Both backbones were measured on the same rows with no engine in the path. `Qwen3.5-4B` prompted
directly reaches 0.700 on the menu and 0.634 on DecisionBench, within noise of the 4B engine, so
on this menu the engine's contribution is coverage, probabilities and latency rather than raw
accuracy. Where it is accuracy too is the wide label spaces: 0.900 against 0.100 on a 151-intent
task, 0.650 against 0.183 on 100-way contract clauses.

### Reproducing it

The run is one Kaggle notebook,
[`notebooks/vega_benchmark_2xT4.ipynb`](https://github.com/NandhaKishorM/vegaml/blob/main/notebooks/vega_benchmark_2xT4.ipynb),
and it is self-contained: it writes out its own harness modules, so all it needs is a session with
**GPU T4 x2**, **Internet on**, and two secrets under *Add-ons -> Secrets*:

| secret | what it is for | without it |
|---|---|---|
| `JEV_API` | the key for the hosted reference at `api.typesafe.ai`, read with `UserSecretsClient().get_secret("JEV_API")` and never printed | the `jev/*` column is skipped with a printed reason, everything local still runs |
| `HF_TOKEN` | a Hugging Face read token for the checkpoints and the datasets | the public checkpoints still load anonymously, a few datasets may rate limit |

About four and a half hours on 2x T4, or five minutes with `CFG["quick"] = True`. Compare
`summary.public_macro`, `results.jevbench.all` and `results.decisionbench.all` in the JSON it
writes against the table above: the keys are the same strings. Local readouts at the same
`CFG["seed"]` and the same library versions reproduce to the row; the hosted column will not,
because its weights can change under you.
[`bench/RESULTS.md`](https://github.com/NandhaKishorM/vegaml/blob/main/bench/RESULTS.md) is the
long form, with per task accuracy for all 43 tasks.

---

## An earlier, item paired run against Jev 1.13.0

![Where Vega 0.8B beats Jev 1.13.0](assets/vega_wins.png)

A different and narrower measurement, from before the menu above existed: our own runs of public
harnesses with Jev 1.13.0 called live over identical items, no tuning on the evaluation data, and
the tasks chosen because Vega leads there. Kept because the phishing result is not on the menu and
is the sharpest of them. `enron-spam`, `ag-news` and `temporal_numeric` are on the menu too, and
they came out the same way there.

### Phishing screening: the clearest win

800 emails, a perfect 50/50 split:

| | accuracy | precision | recall | caught |
|---|---|---|---|---|
| **Vega 0.8B + adapter** | **75.38** | 0.837 | **0.630** | **252 / 400** |
| Jev 1.13.0 | 61.88 | 0.961 | 0.247 | 99 / 400 |

**+13.50 accuracy, and 2.5× the phishing caught** (McNemar p = 3e-11). Jev is almost never wrong when it
calls phishing, but it misses three quarters of it. For a screening task that is the wrong operating
point, and Vega sits on the right one.

### Date and quantity reasoning

JevBench v1.4.2, `temporal_numeric`: dates and quantities judged against policy thresholds:

| | score |
|---|---|
| **Vega 0.8B + adapter** | **46.7** |
| Jev 1.13.0 | 20.0 |

**+26.7**, more than double.

### S1MB: benchmarks where Vega wins outright

From the public S1MB evaluator (137 benchmarks, 26,269 decisions), scored by the harness itself:

| benchmark | task | Vega | Jev |
|---|---|---|---|
| enron-spam | boolean | **1.000** | 0.920 |
| ag-news | choice | **0.955** | 0.806 |
| typed-decisions-score | score | **0.438** | 0.395 |
| go-emotions | boolean | **0.488** | 0.485 |
| drone-control | choice | **0.079** | 0.000 |

### Calibration: at parity with a hosted model

Expected calibration error over 5,096 item-paired decisions:

| | ECE |
|---|---|
| Vega 0.8B + adapter | **9.5** |
| Jev 1.13.0 | 9.3 |

And **better than Jev** on individual benchmarks:

| benchmark | Vega ECE | Jev ECE |
|---|---|---|
| product relevance | **6.0** | 22.0 |
| banking intents | **5.0** | 9.7 |
| appliance routing | **5.0** | 5.7 |

Calibrated probability is the hard part of a decision model. A 0.8B engine matching a hosted one on it,
and beating it by 16 points of ECE on one benchmark, is the result worth noting.

### Speed

| | median per decision |
|---|---|
| **Vega 0.8B + adapter** | **280 ms** (local, Apple M-series, fp32) |
| | **267 ms** (T4, fp16) |
| Jev 1.13.0 | 591 ms (hosted API) |

**2.1× faster**, running entirely on your own hardware, no API call, no per-decision fee, no data
leaving the machine.

---

### Images: what the vision path actually scores

The hosted baseline takes no image input, so there is no like-for-like comparison. On document
pages it was given Apple Vision OCR of the same images and read that text, which makes its row a
two-model pipeline rather than a model that sees.

**RVL-CDIP-N**, 1,002 document pages, the 12 of 16 RVL-CDIP categories the set contains:

| system | reads | accuracy | ECE | median latency |
|---|---|---|---|---|
| Apple Vision OCR + Jev 1.13.0 | text | **0.896** | **0.045** | **1,049 ms** |
| **Vega 0.8B**, zero-shot | the image | 0.793 | 0.207 | 1,989 ms |
| DiT, best published on this set | the image | 0.786 | n/a | n/a |

DiT was trained on RVL-CDIP and is the strongest published result on this out-of-distribution set.
Vega has never seen the dataset and lands level with it. Not quite like-for-like: Vega chose among
the 12 categories present, while a classifier trained on RVL-CDIP chooses among all 16. The
pipeline latency includes the 649 ms Apple Vision takes per page, without which the API call alone
is 400 ms.

**MMMU-Pro**, 300 items over 30 strata, chance 0.120:

| system | reads | accuracy | ECE |
|---|---|---|---|
| Gemini 3.1 Pro, best reported Oct 2026 | the image | **0.839** | n/a |
| Jev 1.13.0 | text only, no image input | 0.337 | 0.160 |
| **Vega 0.8B** | the image | 0.240 | **0.052** |
| Vega 0.8B, same questions | image withheld | 0.157 | 0.062 |

An 800M model is nowhere near a frontier one on university-level reasoning. The result worth
keeping is the last two rows: withholding the image costs 8.3 points, so the vision path carries
real signal rather than the text answering alone. Calibration error is three times better than the
hosted model's on the same items.

## Architecture

Two halves with a hard boundary between them.

**Perception (frozen).** The causal LM is read, never trained. Hidden states are tapped at layers 13 and
19 and pooled over three spans (state, question, final position), plus one pooled vector per answer
option. The model is cut off after the deepest layer read, so a decision never pays for the layers above
it, and the LM head is never run. No text is generated anywhere in the path.

**The physics engine (trained, 55 MB).** The pooled features become a world latent, then a foresight
latent, then a probe and an impulse that place a particle `(z₀, p₀)` in a 64-dimensional decision space.
Each candidate answer is a Gaussian potential well:

```
U(z) = ½κ‖z‖²  −  Σ_k a_k · exp( −‖z − c_k‖² / 2σ_k² )
```

The quadratic term keeps the particle bounded; each well pulls it toward one answer with its own depth
`a_k`, centre `c_k` and width `σ_k`, all produced from that option's own features. The particle rolls
under damped Hamiltonian dynamics (symplectic Euler with friction and a state-space thermostat), and the
answer is read from where it settles:

```
E_k = ‖z_T − c_k‖² / 2σ_k²  −  log a_k        P = softmax(−E / τ)
```

A decision is a short physical simulation with a fixed step budget, not a sampling loop: constant cost,
deterministic result.

**Gated adapters (5.3 MB).** Two low-rank adapters (rank 32, α 32) attach to seven engine projections. A
per-question gate (a small sigmoid head over the world latent) decides for **each question
independently** whether an adapter contributes. The gate value and the routed adapter are returned with
every answer, so routing is auditable rather than implicit.

## Related work

A frozen language model feeding a trainable module described as a differentiable physics engine in a
learned latent space was proposed for physical reasoning in **CWMI**, [Inducing Causal World Models in
LLMs for Zero-Shot Physical Reasoning](https://arxiv.org/abs/2507.19855) (Sharma et al., 2025). The
separation of a frozen reader from a trained dynamics module is the same idea, and the resemblance is
worth stating plainly.

What differs is what the dynamics are *for*. CWMI learns physics as subject matter: its latent state
is meant to carry position, velocity, mass and material, it is supervised against ground-truth final
states from video and a simulator, and it answers physical-reasoning questions (PIQA, 89.4%). Its
module is a 12-layer Transformer decoder of roughly 256M trainable parameters trained on 8 H100s.

Vega uses dynamics as the inference mechanism rather than as a model of the world. Its 64-dimensional
space has no physical referent; the wells are built from the candidate answers of whatever question
was asked, so the landscape is rebuilt per question rather than learned once. The engine is 14.3M
parameters with no attention anywhere, and the motion is an explicit damped Hamiltonian integrated
with symplectic Euler against a written-down potential, not a learned state transition. The answer is
the lowest-energy well, and the residual kinetic energy at the end is what sets the temperature, which
is the part that makes the probabilities mean something. CWMI reports accuracy; it has no calibration,
conformal or abstain machinery, which is most of what is claimed here.

## Calibration

The readout temperature is not a constant. It is predicted per decision from the **physical state** the
particle ended in:

```
log τ = b + w · [ log(1 + residual kinetic energy),
                  log(1 + distance to the nearest well bottom),
                  fraction of the step budget used,
                  log (number of options) ]
```

A particle still in motion, or stopped far from any well, is an uncertain decision and gets a hotter
temperature. Each adapter route carries its own calibration vector. On top of this, every answer carries
a **conformal set** at a chosen risk level α, plus **abstain** and **unbound** flags for cases the engine
will not stand behind.

---

## Usage

```python
from vega_api import Vega

v = Vega("vega-08b-public-intents", backbone="Qwen/Qwen3.5-0.8B", device="cuda")

out = v.decide(
    {"from": "billing@acme.com", "subject": "Invoice overdue", "body": "..."},
    {"team":  {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "payments and invoices",
                            "technical": "product faults",
                            "sales": "new business"}},
     "churn": {"type": "boolean", "instructions": "Is this customer at risk of leaving?",
               "criteria": {"true": "shows intent to cancel", "false": "no such signal"}}},
    alpha="0.1")

print(out["answers"]["team"])   # choice, probabilities, conformal set, abstain, gate, adapter
```

Both questions are answered from **one** read of the state.

* `alpha` sets the conformal risk level.
* `views=2` or `3` re-reads the state with options reversed and in a second surface form, averaging the
  probabilities, steadier answers at roughly `views ×` the cost.
* `adapters="off"` runs the base engine alone; the adapters are load-bearing, so expect a drop.

### Which readout answers

`vegaml` 0.7.0 and later default to `mode="auto"`: a question is answered by a head fitted on your
own labelled examples where it has one worth using, and by the physics engine everywhere else. With
nothing fitted, `"auto"` **is** the engine, conformal sets and abstain flags included, so it is what
every figure on this card measures. Each answer carries `readout`, either `"engine"` or `"ttt"`, so
a mixed result is never ambiguous.

```python
import vegaml

v = vegaml.load()                                # 0.8B, mode="auto"
out = v.decide(state, questions)                 # engine, since nothing is fitted
print(out["answers"]["team"]["readout"])         # 'engine'

v.fit(examples, questions)                       # 20 or so labelled examples
print(v.decide(state, questions)["answers"]["team"]["readout"])   # 'ttt'
```

`fit` cross-validates each head and discards any that fails to beat chance, and `"auto"` uses only
heads whose accuracy is known to beat it. A fitted head is sharper than the engine on the label
space it was fitted for and carries no calibration and no abstain flag, which is the trade `readout`
makes visible. Pass `mode="engine"` to reproduce this card's numbers regardless of what is fitted.

## Files

| file | |
|---|---|
| `engine.safetensors` | the trained physics engine |
| `adapters/` | two gated low-rank adapters with their calibration and conformal tables |
| `vega_config.json` | taps, limits, step budget, memory-reader settings |
| `vega_api.py`, `vega_common.py` | the inference path; no other runtime needed |
| `vega_vision.py` | the image read path |

The backbone is fetched from the Hub and verified against a pinned SHA-256: a mismatched or silently
updated base model is refused rather than used.

## Scope of these figures

Three kinds of figure sit on this card and they should be read differently.

**"Where Vega surpasses the hosted model"**, which is also the picture at the top, is a selected
set, but it is selected out of a full evaluation that is printed in full right underneath it. Each
row carries its item count and its margin in rows, so a thin win reads as a thin win.

**"The rest of the menu"** is that full evaluation: every task reported, win or lose, and the
losses are the larger part. The hosted model is more accurate on 31 of the 43 public tasks against
the 4B and on 38 against the 0.8B, and it leads on both other suites. It is one run, so a per task
margin of two or three rows is noise, and the notebook that produced it is linked so you can run
it yourself.

**"An earlier, item paired run"** is a selected set with no such frame around it: measured and
reproducible, but chosen because Vega leads there, and the complete per-benchmark breakdown behind
it is not on this card. Jev leads on most of the benchmarks in that run too, particularly
reranking and multi-step reasoning over long documents. Ask for the breakdown if you need it for a
deployment decision.
