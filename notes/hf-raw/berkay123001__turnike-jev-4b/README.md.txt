---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
base_model_relation: adapter
pipeline_tag: text-classification
language:
- tr
- en
library_name: peft
tags:
- text-classification
- decision
- decision-model
- typed-decisions
- calibrated-probabilities
- system-one
- jev
- jevbench
- noul
- choice
- score
- turkish
- lora
- multiple-choice
---

![Turnike-JEV-4B](assets/banner_turnike.png)

# Turnike-JEV-4B — Open-Weight Decision Model

**Turnike-JEV-4B: an open, bilingual (English + Turkish) 4B Jev-class decision model — text plus closed questions in, calibrated option probabilities out, ~100 ms, fully offline.**

**Turnike-JEV-4B: 4B parametreli, iki dilli (İngilizce + Türkçe), açık ağırlıklı, Jev sınıfı bir karar modeli — metin ve kapalı sorular girer, kalibre seçenek olasılıkları çıkar; ~100 ms, tamamen çevrim dışı.**

- **English: most accurate in its class on JevBench.** Macro accuracy **0.774** on the full JevBench test (22 configs, 22,773 rows, official `jevify-run report`). The best 4B entry on the public jev-bench leaderboard has 0.751 and the top entry overall 0.770 (a 35B MoE). Our result has not been submitted to the leaderboard and the 0.004 lead is small. Jev 1.13.0 scores 0.733 on the same set.
- **Turkish: best on human-labelled decisions.** tr_insan (1,379 human-labelled Turkish decisions): **0.748**, against 0.727 for Jev, 0.714 for Microsoft-Decision-1 (9B, API; paired McNemar p = 0.003), 0.708 for Kev-4B and 0.675 for Intern-Decision-4B.
- **Calibrated without a recipe.** Macro ECE on full JevBench is **0.084** from raw probabilities, against 0.113 for Jev 1.13.0. Leaderboard models that use a fitted recipe get to about 0.05, so fit a temperature if you need tighter calibration.
- **Fast.** One decision takes **~109 ms** in bf16 and **~82 ms** in FP8. Eight questions on one text, in one pass, take ~156 / ~126 ms. Measured on one RTX 5060 Ti 16 GB (8.7 GB used), with an earlier run of the same architecture and size.
- **Many questions, one pass.** Up to 16 questions about one text are answered in a single forward pass.
- **Runs on your hardware, offline.** You need a ~130 MB LoRA adapter (or the merged bf16 weights in `berkay123001/turnike-jev-4b-merged`, or the torchao FP8 copy in `berkay123001/turnike-jev-4b-fp8`: 5.5 GB on disk, about 4.5 GB of GPU memory, same accuracy as bf16 within noise) and one consumer GPU. There is no API and the text never leaves the machine. CPU works, but slowly.
- **Cheap.** On your own hardware there is no per-call fee. For comparison, our logs show Jev's API at about 1.8e-5 $ per one-question request and 3.6e-5 $ per eight-question request, and Microsoft-Decision-1 at 0.7–1.5e-5 $ per request.
- **Three sizes, one recipe.** Turnike-JEV-0.8B, Turnike-JEV-4B and Turnike-JEV-9B share the data, the draw and the hyperparameters; nothing was tuned per size. Full JevBench macro 0.681 / 0.774 / 0.783; the 9B fits a 16 GB GPU in FP8 and is ahead of Microsoft-Decision-1 — Microsoft's post-trained Qwen3.5-9B — on human-labelled Turkish (0.763 vs 0.714, p = 2e-6). See [Model family](#model-family).
- **Honest.** We found and fixed two training-data shortcuts (see [Training](#training)). "None of the above" is tested on a balanced set whose option count does not give the answer away. All test sets used to pick this run are listed in [Honesty notes](#honesty-notes).

*"JevBench" on this card means the [Praveenrajus/jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench) dataset (22 configs) scored with the `jevify-run report` of [uspraveen/Jevify](https://github.com/uspraveen/Jevify). It is not the [JevBench board by Benchmark Heaven](https://benchmarkheaven.com/jev-models), a different benchmark with the same name, which we have not entered yet.*

![Five decision models on the same items](assets/fig_compare.png)

```python
# pip install torch "transformers>=5.17" peft accelerate ; karar.py: hf download berkay123001/turnike-jev-4b karar.py --local-dir .
from karar import Karar
k = Karar("berkay123001/turnike-jev-4b")

# TR
k.sor("Kulaklık iki günde bozuldu, paramı geri istiyorum.",
      {"Yazar para iadesi istiyor mu?": {"type": "noul"},
       "Mesajın tonu nedir?": ["olumlu", "olumsuz", "nötr"]})
# -> {"Yazar para iadesi istiyor mu?": {"false": p, "true": p}, "Mesajın tonu nedir?": {"olumlu": p, "olumsuz": p, "nötr": p}}

# EN
k.sor("The production database has been down since 9am. Please escalate now.",
      {"Does this message require immediate action?": {"type": "noul"},
       "Which team should handle it?": {"billing": "payments, refunds", "infra": "servers, outages", "sales": "pricing"}})
```

> **Declarations / Beyanlar**
> - Trained on public Turkish and English datasets, LLM-labelled public texts and synthetic data; the datasets are not released. / Herkese açık Türkçe ve İngilizce veri setleri, LLM ile etiketlenmiş açık metinler ve sentetik veriyle eğitildi; veri setleri yayınlanmıyor.
> - Not distilled from any commercial decision API. / Herhangi bir ticari karar API'sinden damıtılmadı.
> - AI assistance: the card and code were written with Claude coding agents under the author's direction ([details](#ai-assistance)). / Kart ve kod, yazarın yönlendirmesiyle Claude kodlama ajanlarıyla yazıldı.
> - No web/browser data in this release; the browser version will be published separately. / Bu sürümde web verisi yok; tarayıcı sürümü ayrı yayınlanacak.

Turnike-JEV-4B is a LoRA adapter (r=16, about 130 MB) on [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). It does not generate text. The text and the questions go into one chat prompt, and the assistant turn is a JSON skeleton with one placeholder per question. Every option of a question is mapped to a single-token symbol (A, B, C, ...). At the token just before each placeholder, the model's final hidden state is multiplied with the output rows of that question's symbols only. A softmax over those symbols gives the option probabilities.

- Author: Berkay Hasret (independent researcher)
- Languages: English and Turkish
- Base model: Qwen/Qwen3.5-4B (Apache-2.0, Copyright 2026 Alibaba Cloud), frozen during training; only the adapter is trained
- License of the adapter, the merged weights, `karar.py` and this card: Apache-2.0 (see [License](#license))
- Merged bf16 weights (no PEFT needed) are published as a separate repo: `berkay123001/turnike-jev-4b-merged`

<a id="model-family"></a>
## Model family

Three sizes, one recipe: the same data, the same draw (seed 3) and the same hyperparameters (LoRA r=16, α=32, lr 1e-4, 2,831 steps, effective batch 32) for every size. Nothing was tuned per size, so the smallest and the largest model are probably somewhat under-tuned. Each size is its own repo; the row in bold is the model of this card.

| model | base | trainable params | full JevBench (22,773) | JevBench EN-300 (EN) | held-6 / other-16 | tr_insan, 1,379 human-labelled (TR) | practical TR, 2,966 | LegalBench rules, 885 (EN) | weights | trained on |
|---|---|---|---|---|---|---|---|---|---|---|
| [Turnike-JEV-0.8B](https://huggingface.co/berkay123001/turnike-jev-0.8b) | Qwen/Qwen3.5-0.8B | 10.8M | **0.681** | 0.676 | 0.682 / 0.674 | 0.699 | 0.840 | 0.724 | adapter (43 MB) | RTX 5060 Ti 16 GB, batch 4 × 8 |
| **Turnike-JEV-4B** (this card) | Qwen/Qwen3.5-4B | 32.5M | **0.774** | 0.769 | 0.826 / 0.748 | 0.748 | 0.891 | 0.832 | adapter (130 MB) · [merged bf16](https://huggingface.co/berkay123001/turnike-jev-4b-merged) · [FP8](https://huggingface.co/berkay123001/turnike-jev-4b-fp8) | RTX 5060 Ti 16 GB, batch 4 × 8 |
| [Turnike-JEV-9B](https://huggingface.co/berkay123001/turnike-jev-9b) | Qwen/Qwen3.5-9B | 43.3M | **0.783** | 0.780 | 0.839 / 0.758 | 0.763 | 0.889 | 0.890 | adapter (173 MB) · [merged bf16](https://huggingface.co/berkay123001/turnike-jev-9b-merged) · [FP8](https://huggingface.co/berkay123001/turnike-jev-9b-fp8) | Colab RTX PRO 6000 96 GB, batch 16 × 2 |

<sub>Same protocol as the tables below: single runs, the same items for every model, no calibration recipe; full JevBench = official `jevify-run report` on all 22,773 test rows. Differences below about 0.01 are within run-to-run noise. The machine-translated TR-50 is in Table 5.</sub>

## Quick start

```bash
hf download berkay123001/turnike-jev-4b karar.py requirements.txt --local-dir .
pip install -r requirements.txt     # torch, transformers 5.x, peft, safetensors, accelerate
# optional: pip install torchao (FP8), causal-conv1d and flash-linear-attention (faster linear-attention layers)
```

```python
from karar import Karar                                   # karar.py from this repo
k = Karar("berkay123001/turnike-jev-4b")                     # base model is downloaded automatically
print(k.sor(
    "Hocam merhaba, BIL204 proje ödevi bu cuma teslim ama geçen hafta ameliyat oldum, raporum ekte. "
    "Teslimi salıya uzatabilir misiniz?",
    {"Bu e-posta hangi kutuya gitmeli?": ["öğrenci-ders", "idari-yazışma", "araştırma-yayın", "tanıtım-reklam"],
     "Gönderen bir cevap bekliyor mu?": {"type": "noul"},
     "Mesajın tonu nedir?": {"olumlu": "genel olarak olumlu", "olumsuz": "genel olarak olumsuz",
                             "nötr": "duygu belirtmiyor, yalnızca bilgi aktarıyor"}}))
```

`sor(text, questions)` returns `{question: {label: probability}}`, and the probabilities of each question sum to 1. A question can be given in any of these forms:

| form | meaning |
|---|---|
| `["a", "b", ...]` | multiple choice, the option text is the label |
| `{"label": "description", ...}` | multiple choice with a short description per option (usually better) |
| `{"type": "noul"}` | yes/no; returns `{"false": p, "true": p}` |
| `{"type": "choice" \| "score" \| "noul", "instructions": "...", "criteria": {...}}` | full form, as used in our evaluations |

If every key is a short identifier (`"urgent"`, `"iade_istiyor"`), it becomes the field name in the prompt and the question text comes from `instructions`. Otherwise the fields are named `q` / `q1..qN` and the key is the question text. More than 16 questions are split into balanced passes. A question with more than 62 options is solved by two-stage elimination; options that do not reach the final get probability 0.

Command line:

```bash
python karar.py --adapter berkay123001/turnike-jev-4b --metin "..." \
    --soru "Mesajın tonu nedir?|olumlu|olumsuz|nötr" --soru "Cevap bekliyor mu?|evet|hayır" [--fp8]
```

Merged weights (no PEFT): `Karar(None, taban="berkay123001/turnike-jev-4b-merged")`. `birlestir.py` produces them from this adapter with the same bf16 merge that `karar.py` does at load time.

`karar.py` is a single self-contained file (about 400 lines; imports only torch, transformers, peft and optionally torchao). `test_karar.py` has 9 tests:

- input validation and independence from any other code;
- 9 Turkish/English examples and the CLI;
- long text, many options and many questions;
- FP8;
- when the evaluation code is available, token-level prompt identity and probability identity with the code behind all numbers below.

Measured (bf16, RTX 5060 Ti): max |Δp| 0.00000 against that code on 33 questions, including an 18-question case and a 71-option case. FP8 against bf16: max |Δp| 0.071 on the 9 examples, with the same argmax.

Tips:
- Options work best as full candidate answers or with a short description ("olumsuz": "genel olarak olumsuz"), not bare codes.
- The question wording matters; an explicit rule with examples beats a vague question.
- Yes/no questions: either `{"type": "noul"}` (the form used in training for most yes/no data) or two descriptive options.

## Method

![Architecture: (a) one call, (b) training, (c) symbol readout](assets/arch.png)

**Readout.** For question $q$ with options $O_q$, let $s(o)$ be the symbol of option $o$, $h_q$ the final hidden state at the token just before $q$'s placeholder, and $w_s$ the output-head row of symbol $s$. Then

$$
p_q(o) \;=\; \frac{\exp\!\big(w_{s(o)}^{\top} h_q\big)}{\sum_{o' \in O_q} \exp\!\big(w_{s(o')}^{\top} h_q\big)}, \qquad |O_q| \le 62 .
$$

Only the $|O_q|$ symbol rows are multiplied (in fp32). The 248,320-token vocabulary is never scored, and no temperature is applied ($T = 1$).

**Packing.** One prompt holds $Q \le 16$ questions about the same text, and one forward pass yields all $h_{q_1}, \dots, h_{q_Q}$. Attention is causal, so a question's readout sees the text, the full decision schema and the `?` placeholders of the earlier fields. It does not see their answers.

**Training loss.** Each question has a target distribution $y_q$: the gold option smoothed to 0.9, or the crowd-label distribution where the source has one. Over all questions $\mathcal{B}$ in a mini-batch:

$$
\mathcal{L} \;=\; \frac{1}{|\mathcal{B}|} \sum_{q \in \mathcal{B}} \mathrm{KL}\!\left(y_q \,\middle\|\, p_q\right)
\;=\; \frac{1}{|\mathcal{B}|} \sum_{q \in \mathcal{B}} \sum_{o \in O_q} y_q(o) \big[\log y_q(o) - \log p_q(o)\big].
$$

There is no per-question weight in the loss. Group shares, class balancing (α = 0.5) and the 0.08 share of broad-template questions act on how often an item is drawn, not on $\mathcal{L}$.

```text
Algorithm 1   One Turnike-JEV-4B call: sor(text, questions)
Input : text x; questions q_1..q_N, each with options O_q (labels, optional descriptions)
Output: for every q, a probability for every option in O_q

 1  validate the inputs; write every question in the full form {type, instructions, criteria}
 2  questions with |O_q| > 62 go to line 10; split the rest into ceil(N/16) balanced packs
 3  for each pack P:
 4      prompt <- system message
               + "## State" + x as a JSON string
               + "## Decision schema": per q, "field: question" then "A = label: description", ...
               + assistant skeleton  { "q1": "?", "q2": "?", ... }    (x truncated if > 4,096 tokens)
 5      run ONE forward pass of base + LoRA (no generation)
 6      for each q in P:
 7          h_q <- final hidden state at the token just before q's "?"
 8          z_q <- W_sym[symbols of q] . h_q                (<= 62 rows, fp32)
 9          p_q <- softmax(z_q)                              (T = 1; any temperature is up to the caller)
10  for each q with |O_q| > 62:
        k <- ceil(|O_q| / 62); split O_q into k balanced chunks; one pass per chunk;
        keep the top floor(62/k) options of every chunk; one final pass over the finalists;
        options that are not finalists get probability 0
11  return {q: {label: p_q(label)}}
```

## Prompt format

The prompt is the Qwen3.5 chat template with an empty thinking block. The system message and the user-message layout are fixed and must not be changed:

```
system: You are a careful decision assistant. Use the state and decision schema in the user message to make the
        requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) ...
user:   Return one answer for every field using the supplied answer symbols.

        ## State
        "<text as a JSON string>"
        ## Decision schema
        q1: <question>
            A = <label>: <description>
            B = <label>: <description>
        q2: ...
assistant: {
            "q1": "?",      <- probabilities are read at the token before each "?" placeholder
            "q2": "?"
        }
```

Yes/no questions are written as `A = no: The answer is no (...)`, `B = yes: The answer is yes (...)`. If the prompt exceeds `MAKS_ISTEM_TOKEN = 4096` tokens (training used 1536), the text is truncated from the end. Questions and options are never truncated.

## Results

*"JevBench" on this card means the [Praveenrajus/jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench) dataset (22 configs) scored with the `jevify-run report` of [uspraveen/Jevify](https://github.com/uspraveen/Jevify). It is not the [JevBench board by Benchmark Heaven](https://benchmarkheaven.com/jev-models), a different benchmark with the same name, which we have not entered yet.*

All numbers below are our own measurements unless marked as reported by someone else. Every model saw the same items and was scored by the same code. Probabilities are raw, with no calibration recipe. Each result is a single run.

| model | JevBench EN-300 (EN) | LegalBench rules, 885 (EN) | tr_insan, 1,379 human-labelled (TR) | practical TR, 2,966 (TR) |
|---|---|---|---|---|
| **Turnike-JEV-4B** | 0.769 | 0.832 | 0.748 | 0.891 |
| **Turnike-JEV-9B** | 0.780 | 0.890 | 0.763 | 0.889 |
| **Turnike-JEV-0.8B** | 0.676 | 0.724 | 0.699 | 0.840 |
| Microsoft-Decision-1 (9B, closed, API) | 0.775 | – | 0.714 | 0.857 |
| Jev 1.13.0 (API) | – | 0.924 † | 0.727 | 0.872 |
| Kev-4B | 0.706 | 0.786 | 0.708 | 0.797 |
| jevify Qwen3.5-4B readout LoRA + coherence | – | – | 0.642 | 0.847 |
| internlm/Intern-Decision-4B | 0.687 | 0.656 | 0.675 | 0.843 |
| OpenJev 4B (qwen3.5-4b-nli-v5) | – | – | 0.645 | 0.738 |
| Qwen3.5-4B, zero-shot (our prompt) | – | – | 0.612 | 0.745 |
| Kev-0.8B | 0.570 | 0.496 | 0.583 | 0.697 |

† Reported by the jevify author on the same 885 items (jevify README, finding 14); not our run. – = not measured. Run-to-run noise: retraining an earlier run of the same recipe with another seed moved tr_insan by +0.001 and EN-300 by +0.002. The machine-translated JevBench TR-50 is a supplementary set (Table 5); the human-labelled tr_insan is the primary Turkish measure.

### English

#### Table 1. Full JevBench test, next to the public leaderboard

| model | source | macro acc | macro ECE ↓ | macro Brier ↓ | sel@90 | choice / score / noul acc |
|---|---|---|---|---|---|---|
| **Turnike-JEV-4B** | our run, not submitted | **0.774** | 0.084 | 0.293 | **0.805** | 0.791 / 0.599 / 0.902 |
| **Turnike-JEV-9B** | our run, not submitted | **0.783** | 0.076 | 0.280 | **0.817** | 0.806 / 0.597 / 0.913 |
| **Turnike-JEV-0.8B** | our run, not submitted | **0.681** | 0.078 | 0.385 | **0.711** | 0.665 / 0.539 / 0.822 |
| Qwen/Qwen-AgentWorld-35B-A3B (readout LoRA + coherence) | reported by the leaderboard | 0.770 | 0.055 | 0.292 | 0.799 | 0.792 / 0.566 / 0.917 |
| Qwen/Qwen3.5-9B (readout LoRA + coherence) | reported by the leaderboard | 0.763 | 0.056 | 0.299 | 0.791 | 0.791 / 0.554 / 0.906 |
| Qwen/Qwen3.5-9B (readout LoRA) | reported by the leaderboard | 0.759 | 0.052 | 0.302 | 0.787 | 0.790 / 0.541 / 0.905 |
| Qwen/Qwen3.5-4B (readout LoRA + coherence) | reported by the leaderboard | 0.751 | 0.058 | 0.316 | 0.779 | 0.771 / 0.555 / 0.893 |
| perplexity-ai/pplx-decider-v1-27b (hosted API) | reported by the leaderboard | 0.748 | 0.080 | 0.318 | 0.774 | 0.775 / 0.519 / 0.911 |
| Qwen/Qwen3.5-4B-Base (readout LoRA + coherence) | reported by the leaderboard | 0.747 | 0.058 | 0.318 | 0.777 | 0.771 / 0.535 / 0.897 |
| Qwen/Qwen3.5-4B (Tier 2 residual) | reported by the leaderboard | 0.747 | 0.110 | 0.342 | 0.774 | 0.770 / 0.529 / 0.903 |
| Qwen/Qwen3.5-4B (readout LoRA) | reported by the leaderboard | 0.743 | 0.059 | 0.321 | 0.772 | 0.766 / 0.530 / 0.896 |
| google/gemma-4-E4B-it (readout LoRA + coherence) | reported by the leaderboard | 0.742 | 0.053 | 0.322 | 0.772 | 0.759 / 0.546 / 0.889 |
| Qwen/Qwen3.5-4B-Base (readout LoRA) | reported by the leaderboard | 0.741 | 0.064 | 0.324 | 0.770 | 0.768 / 0.522 / 0.893 |
| google/gemma-4-E4B-it (readout LoRA) | reported by the leaderboard | 0.737 | 0.057 | 0.330 | 0.766 | 0.754 / 0.531 / 0.892 |
| Qwen/Qwen3.5-4B (Tier 2 residual, lr 3e-05) | reported by the leaderboard | 0.734 | 0.096 | 0.342 | 0.762 | 0.761 / 0.518 / 0.884 |
| Jev 1.13.0 (TypeSafe API) | reported by the leaderboard | 0.733 | 0.113 | 0.349 | 0.760 | 0.770 / 0.503 / 0.881 |
| google/gemma-4-E4B (readout LoRA) | reported by the leaderboard | 0.708 | 0.061 | 0.359 | 0.736 | 0.717 / 0.502 / 0.874 |
| togethercomputer/Tev1-4B-experimental (Tev1 prompt) | reported by the leaderboard | 0.703 | 0.086 | 0.361 | 0.732 | 0.717 / 0.512 / 0.850 |
| Qwen/Qwen3.5-4B (Tier 1 residual) | reported by the leaderboard | 0.698 | 0.089 | 0.378 | 0.729 | 0.699 / 0.507 / 0.862 |
| Qwen/Qwen3.5-4B (Tier 0, no training) | reported by the leaderboard | 0.662 | 0.090 | 0.401 | 0.689 | 0.687 / 0.468 / 0.796 |

<sub>Leaderboard rows: the public jev-bench leaderboard (Praveenrajus/jev-bench dataset card, data v0.1.1, 46 rows), values as published: the top 10, the other 4B-class entries with macro ≥ 0.66, and the Jev reference row. Per-config values for these rows are not available to us. Ours: full test split (22,773 rows), official `jevify-run report`, argmax and yes/no at p ≥ 0.5, raw probabilities, one deterministic run. The lead over the top row is small and has not been checked by the leaderboard maintainers; leaderboard models report lower ECE (≈0.05).</sub>

![Full JevBench, per config: Turnike-JEV-4B − Jev 1.13.0](assets/fig_jevbench_delta.png)

#### Table 2. Full JevBench, per config, against Jev 1.13.0

| config | type | n | Turnike-JEV-4B acc | Turnike-JEV-4B ECE | Turnike-JEV-9B acc | Turnike-JEV-9B ECE | Turnike-JEV-0.8B acc | Turnike-JEV-0.8B ECE | Jev 1.13.0 acc | Jev 1.13.0 ECE | Turnike-JEV-4B − Jev | Turnike-JEV-9B − Jev | Turnike-JEV-0.8B − Jev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *held-out 6 (not in training)* | | | | | | | | | | | | | |
| `arc_challenge` | choice | 1,000 | 0.932 | 0.087 | 0.949 | 0.097 | 0.701 | 0.052 | 0.979 | 0.010 | −0.047 | −0.030 | −0.278 |
| `clinc150` | choice | 1,000 | 0.856 | 0.162 | 0.863 | 0.123 | 0.710 | 0.120 | 0.893 | 0.033 | −0.037 | −0.030 | −0.183 |
| `fever_evidence` | noul | 1,000 | 0.952 | 0.066 | 0.961 | 0.079 | 0.865 | 0.088 | 0.972 | 0.028 | −0.020 | −0.011 | −0.107 |
| `measuring_hate_speech` | score | 1,000 | 0.631 | 0.070 | 0.635 | 0.060 | 0.599 | 0.055 | 0.527 | 0.237 | +0.104 | +0.108 | +0.072 |
| `strategyqa_grounded` | noul | 687 | 0.943 | 0.118 | 0.946 | 0.109 | 0.766 | 0.088 | 0.956 | 0.059 | −0.013 | −0.010 | −0.190 |
| `yelp5` | score | 1,000 | 0.668 | 0.061 | 0.695 | 0.070 | 0.521 | 0.055 | 0.685 | 0.174 | −0.017 | +0.010 | −0.164 |
| *held-out 6 (not in training), macro* | | 5,687 | **0.830** | 0.094 | **0.842** | 0.090 | **0.694** | 0.077 | 0.835 | 0.090 | −0.005 | +0.006 | −0.142 |
| *other 16 (train splits in training)* | | | | | | | | | | | | | |
| `banking77` | choice | 1,000 | 0.862 | 0.100 | 0.862 | 0.093 | 0.701 | 0.099 | 0.796 | 0.095 | +0.066 | +0.066 | −0.095 |
| `boolq` | noul | 1,000 | 0.904 | 0.085 | 0.921 | 0.079 | 0.812 | 0.043 | 0.917 | 0.021 | −0.013 | +0.004 | −0.105 |
| `chaosnli` | choice | 1,599 | 0.631 | 0.108 | 0.672 | 0.071 | 0.578 | 0.142 | 0.615 | 0.222 | +0.016 | +0.057 | −0.037 |
| `civil_comments` | noul | 2,000 | 0.919 | 0.086 | 0.912 | 0.074 | 0.921 | 0.074 | 0.729 | 0.045 | +0.191 | +0.183 | +0.193 |
| `go_emotions` | choice | 1,000 | 0.569 | 0.032 | 0.577 | 0.043 | 0.559 | 0.034 | 0.282 | 0.384 | +0.287 | +0.295 | +0.277 |
| `helpsteer2_helpfulness` | score | 1,000 | 0.446 | 0.025 | 0.441 | 0.032 | 0.410 | 0.034 | 0.363 | 0.232 | +0.083 | +0.078 | +0.047 |
| `helpsteer2_verbosity` | score | 1,000 | 0.679 | 0.160 | 0.637 | 0.111 | 0.678 | 0.184 | 0.341 | 0.231 | +0.338 | +0.296 | +0.337 |
| `ledgar` | choice | 1,000 | 0.775 | 0.099 | 0.790 | 0.090 | 0.686 | 0.064 | 0.751 | 0.117 | +0.024 | +0.039 | −0.065 |
| `massive` | choice | 1,000 | 0.857 | 0.099 | 0.849 | 0.093 | 0.739 | 0.114 | 0.808 | 0.090 | +0.049 | +0.041 | −0.069 |
| `mmlu` | choice | 1,000 | 0.729 | 0.075 | 0.795 | 0.064 | 0.497 | 0.111 | 0.923 | 0.027 | −0.194 | −0.128 | −0.426 |
| `mnli` | choice | 1,000 | 0.905 | 0.085 | 0.897 | 0.087 | 0.814 | 0.055 | 0.883 | 0.032 | +0.022 | +0.014 | −0.069 |
| `paws` | noul | 1,000 | 0.909 | 0.079 | 0.913 | 0.084 | 0.865 | 0.064 | 0.846 | 0.040 | +0.063 | +0.067 | +0.019 |
| `sms_spam` | noul | 800 | 0.983 | 0.093 | 0.980 | 0.089 | 0.976 | 0.089 | 0.965 | 0.068 | +0.018 | +0.015 | +0.011 |
| `sst5` | score | 1,000 | 0.582 | 0.059 | 0.593 | 0.042 | 0.525 | 0.037 | 0.565 | 0.190 | +0.017 | +0.028 | −0.040 |
| `strategyqa_closed` | noul | 687 | 0.703 | 0.055 | 0.761 | 0.037 | 0.550 | 0.063 | 0.785 | 0.042 | −0.082 | −0.024 | −0.235 |
| `stsb` | score | 1,000 | 0.585 | 0.048 | 0.581 | 0.047 | 0.501 | 0.047 | 0.538 | 0.119 | +0.047 | +0.043 | −0.037 |
| *other 16 (train splits in training), macro* | | 17,086 | **0.752** | 0.081 | **0.761** | 0.071 | **0.676** | 0.078 | 0.694 | 0.122 | +0.058 | +0.067 | −0.018 |
| **all 22, macro** | | 22,773 | **0.774** | **0.084** | **0.783** | **0.076** | **0.681** | **0.078** | 0.733 | 0.113 | +0.041 | +0.051 | −0.052 |

<sub>Jev values: the Jev 1.13.0 per-config table on the jev-bench dataset card (TypeSafe API, every test record); our project records hold the same accuracies for all 22 configs. Turnike-JEV-4B is ahead of Jev on 14 of 22 configs and by +0.041 macro. On the six held-out sources Turnike-JEV-4B (0.830) is above the leaderboard author's best open model (0.828, as stated on the card) and behind Jev (0.835) and the hosted pplx-decider-v1-27b (0.840). Jev also leads on the knowledge configs (mmlu, strategyqa_closed, arc_challenge).</sub>

#### Table 3. JevBench EN-300: the same 6,600 rows for every model

| model | EN-300 macro | held-6 / other-16 | McNemar vs Turnike-JEV-4B (only-Turnike-JEV-4B / only-other) |
|---|---|---|---|
| **Turnike-JEV-4B** | 0.769 | 0.826 / 0.748 | – |
| **Turnike-JEV-9B** | 0.780 | 0.839 / 0.758 | 291 / 366, p = 0.004 |
| **Turnike-JEV-0.8B** | 0.676 | 0.682 / 0.674 | 993 / 378, p = 6e-64 |
| Microsoft-Decision-1 (9B, closed, API) | 0.775 | 0.843 / 0.750 | 426 / 465, p = 0.20 |
| Kev-4B | 0.706 | 0.772 / 0.682 | 795 / 381, p = 4e-34 |
| internlm/Intern-Decision-4B | 0.687 | 0.783 / 0.651 | 901 / 362, p = 2e-53 |
| Kev-0.8B | 0.570 | 0.586 / 0.564 | 1666 / 349, p = 4e-205 |

<sub>Protocol: 300 rows per config from the JevBench test split (seed 7), 22 configs, the same 6,600 rows for every model; exact two-sided McNemar on paired rows. Not measured on this subset: Jev 1.13.0 (API), jevify Qwen3.5-4B readout LoRA + coherence, OpenJev 4B (qwen3.5-4b-nli-v5), Qwen3.5-4B, zero-shot (our prompt) (Jev: see Tables 1–2). We do not know whether Microsoft-Decision-1 was trained on JevBench. Intern-Decision-4B used its own prompt and calibration temperature (for more than 62 options we added the same elimination step), Kev its own HTTP server. EN-300 macro ECE: Turnike-JEV-4B 0.087, Microsoft-Decision-1 0.079, Intern-Decision-4B 0.115, Kev-4B 0.135.</sub>

### Turkish

#### Table 4. Turkish human gold (tr_insan), per task

| model | topic (301) | sentiment 3-way (300) | NLI (300) | film stars 1–5 (178) | product stars 1–5 (300) | all (1,379) | McNemar vs Turnike-JEV-4B |
|---|---|---|---|---|---|---|---|
| **Turnike-JEV-4B** | 0.924 | 0.910 | 0.790 | 0.562 | 0.480 | **0.748** | – |
| **Turnike-JEV-9B** | 0.934 | 0.910 | 0.830 | 0.584 | 0.483 | **0.763** | 62 / 82, p = 0.11 |
| **Turnike-JEV-0.8B** | 0.884 | 0.870 | 0.727 | 0.472 | 0.450 | **0.699** | 152 / 84, p = 1e-5 |
| Microsoft-Decision-1 (9B, closed, API) | 0.887 | 0.893 | 0.727 | 0.545 | 0.450 | **0.714** | 141 / 94, p = 0.003 |
| Jev 1.13.0 (API) | 0.904 | 0.893 | 0.777 | 0.517 | 0.457 | **0.727** | 131 / 101, p = 0.06 |
| Kev-4B | 0.887 | 0.860 | 0.737 | 0.528 | 0.453 | **0.708** | 126 / 70, p = 8e-5 |
| jevify Qwen3.5-4B readout LoRA + coherence | 0.844 | 0.823 | 0.747 | 0.337 | 0.333 | **0.642** | 222 / 75, p = 5e-18 |
| internlm/Intern-Decision-4B | 0.867 | 0.860 | 0.707 | 0.455 | 0.397 | **0.675** | 172 / 71, p = 8e-11 |
| OpenJev 4B (qwen3.5-4b-nli-v5) | 0.804 | 0.837 | 0.683 | 0.455 | 0.370 | **0.645** | 253 / 111, p = 7e-14 |
| Qwen3.5-4B, zero-shot (our prompt) | 0.850 | 0.850 | 0.540 | 0.326 | 0.377 | **0.612** | 280 / 92, p = 3e-23 |
| Kev-0.8B | 0.841 | 0.693 | 0.630 | 0.371 | 0.293 | **0.583** | 309 / 81, p = 2e-32 |

<sub>Protocol: human gold labels from five public Turkish datasets, test texts only. All models were scored by the same script against the same gold file; no model answered "unsure". Stars are exact match on a 5-point scale (within ±1 star Turnike-JEV-4B gets 0.91 / 0.90). Our leakage scan flagged 19 items as overlapping with training texts; without them Turnike-JEV-4B gets 0.746. Microsoft-Decision-1 and Jev were called through their APIs, the others ran locally.</sub>

#### Table 5. JevBench TR-50 (supplementary): the same 1,100 machine-translated rows for every model

| model | TR-50 macro | held-6 / other-16 | McNemar vs Turnike-JEV-4B (only-Turnike-JEV-4B / only-other) |
|---|---|---|---|
| **Turnike-JEV-4B** | 0.700 | 0.750 / 0.681 | – |
| **Turnike-JEV-9B** | 0.721 | 0.793 / 0.694 | 77 / 100, p = 0.10 |
| **Turnike-JEV-0.8B** | 0.560 | 0.583 / 0.551 | 239 / 85, p = 4e-18 |
| Microsoft-Decision-1 (9B, closed, API) | 0.745 | 0.840 / 0.709 | 72 / 121, p = 5e-4 |
| Kev-4B | 0.655 | 0.720 / 0.630 | 127 / 77, p = 6e-4 |
| jevify Qwen3.5-4B readout LoRA + coherence | 0.687 | 0.743 / 0.666 | 106 / 92, p = 0.36 |
| internlm/Intern-Decision-4B | 0.624 | 0.700 / 0.595 | 143 / 59, p = 3e-9 |
| OpenJev 4B (qwen3.5-4b-nli-v5) | 0.621 | 0.717 / 0.585 | 172 / 85, p = 6e-8 |
| Kev-0.8B | 0.508 | 0.580 / 0.481 | 295 / 84, p = 1e-28 |

<sub>Protocol: 50 rows per config (seed 7) from a machine-translated Turkish JevBench test, 22 configs, 1,100 rows. The set is small and noisy: it moved by 0.02 between two seeds of the same recipe. Exact two-sided McNemar on paired rows; Microsoft-Decision-1 rescored from its raw API answers with the same decision rule (macro 0.7445, as originally reported). We treat this set as supplementary: 50 rows per config and machine translation; the human-labelled tr_insan (Table 4) is the primary Turkish measure.</sub>

### Behaviour tests (Table 6)

| test | **Turnike-JEV-4B** | **Turnike-JEV-9B** | **Turnike-JEV-0.8B** | Jev 1.13.0 (API) | Kev-4B | internlm/Intern-Decision-4B | Kev-0.8B |
|---|---|---|---|---|---|---|---|
| planted instruction: accuracy clean / planted (800 × 2) | 0.927 / 0.941 | 0.932 / 0.936 | 0.885 / 0.892 | – | 0.897 / 0.816 | 0.877 / 0.599 | 0.826 / 0.568 |
| hijack rate ↓ | −0.014 | −0.004 | −0.008 | – | +0.081 | +0.279 | +0.259 |
| LegalBench rule application (885, 10 tasks, none trained on) | 0.832 | 0.890 | 0.724 | 0.924 † | 0.786 | 0.656 | 0.496 |

<sub>Protocol: one run each, same items and scoring code. Planted instruction: the same 800 items with a sentence inserted into the text that names a wrong answer; hijack rate = how much more often the model gives the named answer than on the clean copy (0 = ignored). † Reported by the jevify author on the same 885 items; not our run.</sub>

Measured for the Turnike models only (the competitors were not run on these two tests):

- **Turnike-JEV-4B, "None of the above", balanced** (908 MMLU/ARC items × 2 variants, half with the right option removed; the option count no longer gives the answer away): accuracy 0.768 (drop variant) / 0.742 (add variant). It picks "none" in 0.809 / 0.798 of the items where the right option is missing, and wrongly in 0.223 / 0.252 where it is present. In the "add" variant the added option is off-topic only in items without the right option, which can itself hint "none"; the "drop" variant has no such hint.
- **Turnike-JEV-4B, KVKK sufficiency** (internal Turkish personal-data check; two yes/no questions: is a legal basis stated, is the data controller named; 1,172 decisions): 2 errors. On the real policy texts it is 1,160/1,160 correct (named controller 9/9, legal basis 10/10, must-be-insufficient 1,141/1,141); all 2 errors are on the 12 short synthetic passages (titles, headings, one-sentence statements) of the controller question (10/12). Gold sets are tiny; two seeds of the 4B recipe gave 11 and 16 errors.

- **Turnike-JEV-9B, "None of the above", balanced** (908 MMLU/ARC items × 2 variants, half with the right option removed; the option count no longer gives the answer away): accuracy 0.801 (drop variant) / 0.792 (add variant). It picks "none" in 0.813 / 0.778 of the items where the right option is missing, and wrongly in 0.168 / 0.139 where it is present. In the "add" variant the added option is off-topic only in items without the right option, which can itself hint "none"; the "drop" variant has no such hint.
- **Turnike-JEV-9B, KVKK sufficiency** (internal Turkish personal-data check; two yes/no questions: is a legal basis stated, is the data controller named; 1,172 decisions): 2 errors. On the real policy texts it is 1,158/1,160 correct (named controller 9/9, legal basis 9/10, must-be-insufficient 1,140/1,141); errors by set: A-legal 1/10, C2 1/899. Gold sets are tiny; two seeds of the 4B recipe gave 11 and 16 errors.

- **Turnike-JEV-0.8B, "None of the above", balanced** (908 MMLU/ARC items × 2 variants, half with the right option removed; the option count no longer gives the answer away): accuracy 0.558 (drop variant) / 0.532 (add variant). It picks "none" in 0.666 / 0.653 of the items where the right option is missing, and wrongly in 0.402 / 0.428 where it is present. In the "add" variant the added option is off-topic only in items without the right option, which can itself hint "none"; the "drop" variant has no such hint.
- **Turnike-JEV-0.8B, KVKK sufficiency** (internal Turkish personal-data check; two yes/no questions: is a legal basis stated, is the data controller named; 1,172 decisions): 6 errors. On the real policy texts it is 1,158/1,160 correct (named controller 7/9, legal basis 10/10, must-be-insufficient 1,141/1,141); errors by set: A-controller 2/9, S 4/12. Gold sets are tiny; two seeds of the 4B recipe gave 11 and 16 errors.

### Latency and cost (Table 7)

| model | where | 1 question, p50 | 8 questions on one text, p50 | GPU memory | cost per call |
|---|---|---|---|---|---|
| **Turnike-JEV-4B**, bf16, optional kernels | RTX 5060 Ti 16 GB | **109 ms** | **156 ms** | 8.7 GB | none (own hardware) |
| **Turnike-JEV-4B**, FP8 (`--fp8`), optional kernels | RTX 5060 Ti 16 GB | **82 ms** | **126 ms** | 8.8 GB | none |
| **Turnike-JEV-4B**, FP8 (stored weights, `karar.py`) | RTX 5060 Ti 16 GB | **81 ms** | **123 ms** | 4.2 GiB | none |
| Turnike-JEV-4B architecture, bf16, no optional kernels | RTX 5060 Ti 16 GB | 120 ms | 173 ms | 8.7 GB | none |
| **Turnike-JEV-9B**, FP8 (stored weights, `karar.py`) | RTX 5060 Ti 16 GB | **139 ms** | **206 ms** | 8.4 GiB | none |
| **Turnike-JEV-0.8B**, bf16 (`karar.py`) | RTX 5060 Ti 16 GB | **24 ms** | **32 ms** | 1.4 GiB | none |
| Microsoft-Decision-1 | API, network included | 913 ms (tr_insan run, n = 1,379) | – | – | ≈0.7e-5–1.5e-5 $ (0.042 $/M input tokens) |
| Jev 1.13.0 | API, network included | 592 ms (n = 42) | 386 ms (n = 44) | – | ≈1.8e-5 $ (1 q) / ≈3.6e-5 $ (8 q) |
| Kev-4B, its own HTTP server | RTX 5060 Ti 16 GB | – | 282 ms | – | none |
| Intern-Decision-4B, bf16, no optional kernels | RTX 5060 Ti 16 GB | 121 ms | 173 ms | 8.7 GB | none |
| Qwen3.5-4B zero-shot, bf16, no optional kernels | RTX 5060 Ti 16 GB | 120 ms | 173 ms | 8.7 GB | none |

![Accuracy on Turkish human gold vs latency](assets/fig_speed.png)

<sub>Protocol: local rows are wall-clock `decide()` time, CUDA-synchronised, 2 warm-up calls dropped. "1 question" is a 7-option topic question over 60 tr_insan texts; "8 questions" is 15 messages × 8 support-desk questions in one call. The Turnike-JEV-4B kernel and FP8 rows were measured with an earlier run of the same architecture and size, with `causal-conv1d` and `flash-linear-attention` installed. The rows marked "no optional kernels" come from one session without them. Kev-4B: 8-question p50 through its own HTTP server (our earlier run). API latencies come from separate runs and include the network round trip; Jev from our demo logs, Microsoft-Decision-1 with 4 parallel requests, cost = our spend / requests. The stored-weights FP8 row is this model ([berkay123001/turnike-jev-4b-fp8](https://huggingface.co/berkay123001/turnike-jev-4b-fp8), measured through `karar.py`). FP8 kept accuracy on tr_insan (0.748 → 0.740, McNemar p = 0.07) and EN-300 (0.769 → 0.767, p = 0.42), but about 2–2.5% of individual decisions flip; use bf16 if you threshold on probabilities.</sub>

## Training

- **Objective:** KL divergence between the target distribution and the model's option distribution, per question, averaged over the questions in a batch (see [Method](#method)). Hard labels are smoothed to 0.9 on the gold option; crowd-label distributions are used where the source has them.
- **Items:** each training item is one text with 8–16 questions packed into one prompt, exactly the inference format. Items are drawn from the pool with per-group shares (about 47% English), max length 1,536 tokens, seed 3. If two questions have option lists that reveal each other's answer, they never share a prompt.
- **Adapter:** LoRA r=16, alpha=32, dropout 0.05 on all attention, MLP and linear-attention projections of the language model; the vision tower is untouched and not used. lr 1e-4, warm-up 3%, effective batch 32, 2,831 steps on one RTX 5060 Ti 16 GB (batch 4 × accumulation 8).
- **Shortcut history:** we found and removed two training-data shortcuts.
  - In an earlier run, the model could read a "none of the above" answer from a sibling question in the same packed item. Fix: such questions are now split into separate prompts.
  - The next run inferred "none" from the option count, because items with the right option removed had one option fewer. Fix in this release: the option count and any added off-topic option are independent of the answer (7,585 rows rebuilt).
- **Family:** Turnike-JEV-0.8B (Qwen3.5-0.8B) and Turnike-JEV-9B (Qwen3.5-9B) were trained with exactly this recipe and this draw, with no per-size tuning; the 9B ran on a Colab RTX PRO 6000 96 GB with batch 16 × accumulation 2 (the same effective batch of 32).
- An earlier internal model distilled from a commercial API was not released; Turnike-JEV-4B is trained without such labels.

<a id="training-data"></a>
## Training data

Trained on public Turkish and English datasets, LLM-labelled public texts and synthetic data; the datasets are not released. / Herkese açık Türkçe ve İngilizce veri setleri, LLM ile etiketlenmiş açık metinler ve sentetik veriyle eğitildi; veri setleri yayınlanmıyor.

Not distilled from any commercial decision API. / Herhangi bir ticari karar API'sinden damıtılmadı.

This release contains no web/browser data; the browser version will be published separately. / Bu sürümde web verisi yok; tarayıcı sürümü ayrı yayınlanacak.

## Limitations

- **"None of the above" is used somewhat too often.** On the balanced test the model picks "none" in 0.80–0.81 of the items where the right option is missing, but also in 0.22–0.25 of the items where it is present. This is a decision-rule choice, not only a model property: `karar.secim(probs, hicbiri="oos", esik=0.6)` picks the catch-all option only when its probability is at least the threshold. On clinc150 (151 intents, EN-300) that lifts the 9B from 0.850 to 0.873 and the 4B from 0.830 to 0.840 at 0.5; on the balanced "none" test the same threshold costs 3–4 points, so keep plain argmax for gates and audits where flagging the uncertain case is useful.
- **Not the best everywhere.** Microsoft-Decision-1 (9B) is ahead on JevBench TR-50 (0.745 vs 0.700, p = 5e-4) and slightly ahead on EN-300 (n.s.). Jev is ahead on the six held-out JevBench sources, on knowledge questions (mmlu 0.923 vs 0.729) and on LegalBench rule application (0.924 vs 0.832).
- **KVKK thresholds are noisy.** The gold sets are tiny (9, 10, 12 items) and a few items sit near p = 0.5. Do not threshold yes/no probabilities near 0.5 without your own validation set.
- **go_emotions, hate speech, star ratings:** on full JevBench the model scores 0.569 on go_emotions and 0.631 on measuring_hate_speech. The boundary between offensive language and group-targeted hate is the weakest point, together with 5-level star ratings.
- **Mostly LLM labels.** Much of the training data is LLM-labelled, so the model inherits those judgements. tr_insan is the only human-labelled Turkish test.
- **Not trained on source code.** On a small internal security code-review test (32 snippets with access-control bugs, deliberately fuzzy options, 128 decisions) the 4B scores 0.523 (0.8B 0.352, 9B 0.594), against 0.805 for Jev 1.13.0 and 0.695 for Microsoft-Decision-1; the other local 4B decision models we ran (Qwen3.5-4B zero-shot, openjev, Intern-Decision-4B, jevify readout) land between 0.414 and 0.547. Treat code review as out of scope for this release.
- **Knowledge questions** come from the backbone. The model is meant to judge a given text, not to recall facts.
- **Calibration:** no temperature or recipe is applied (macro ECE 0.084 on full JevBench). Fit a temperature on your own validation data if you need calibrated probabilities.
- Prompts longer than 4,096 tokens are truncated (text only); training used 1,536 tokens. Questions with more than 62 options use an elimination step (two or more passes).

<a id="honesty-notes"></a>
## Honesty notes

- **Test sets were used for selection.** We compared training runs on tr_insan, practical TR, EN-300, TR-50, the LegalBench rule test, the balanced "none of the above" test and KVKK. The acceptance rule for this run was set on them too: tr_insan ≥ 0.74, EN-300 ≥ 0.743, balanced "none of the above" ≥ 0.62 overall and ≥ 0.50 "none" found, KVKK errors ≤ 12; the run passed 5/5. None of these numbers is a fully untouched estimate. The full JevBench run was made after selection.
- **In-distribution parts.** The train splits of 16 JevBench configs and of the five tr_insan sources are in training (test texts excluded). Only the six held-out JevBench sources, the LegalBench rule test and the behaviour tests are unseen tasks.
- **Single runs.** Each number comes from one training run. Differences below about 0.02 on TR-50 and 0.01 elsewhere are within run-to-run noise.
- **Earlier runs.** Two earlier clean runs (temiz3 / temiz4) scored 0.753 / 0.740 on tr_insan and 0.764 / 0.767 on EN-300. They were rejected because of the shortcuts described above.
- **Competitors.** Microsoft-Decision-1 and Jev were run through their APIs; whether they were trained on JevBench is unknown. Intern-Decision-4B was run locally with its own prompt format and calibration temperature.

## License

The base model Qwen/Qwen3.5-4B is licensed under Apache-2.0 (the `LICENSE` file in its repository, revision 851bf6e8, "Copyright 2026 Alibaba Cloud"). The adapter weights, the merged weights, `karar.py`, `birlestir.py`, the tests and this card are released under Apache-2.0 (`LICENSE`).

<a id="ai-assistance"></a>
## AI assistance

This model card, its figures, `karar.py`, `birlestir.py`, the tests and most of the training and evaluation code were written with Claude coding agents (Claude Code, Anthropic) under the author's direction. The author is responsible for the content and verified the reported numbers.

## Citation

```bibtex
@misc{turnike_jev_4b_2026,
  title        = "Turnike-JEV-4B: a bilingual (English + Turkish) decision model on Qwen3.5-4B",
  author       = "Hasret, Berkay",
  year         = "2026",
  howpublished = "https://huggingface.co/berkay123001/turnike-jev-4b"
}
```

## Türkçe özet

Turnike-JEV-4B, İngilizce ve Türkçe için Qwen3.5-4B üzerine kurulu, yaklaşık 130 MB'lık bir LoRA adaptörüdür. Metin üretmez: bir metin ve bir ya da birkaç kapalı soru verildiğinde her seçeneğe bir olasılık döndürür. Bir metnin 16 sorusuna kadar tek ileri geçişte cevap verir. RTX 5060 Ti'de tek karar bf16 ≈ 109 ms, FP8 ≈ 82 ms; 8 soru bf16 ≈ 156 ms, FP8 ≈ 126 ms sürer. Her sorunun seçenekleri tek-token sembollere (A, B, C, ...) eşlenir. Olasılık, her yer tutucudan hemen önceki konumda yalnız o sorunun sembol satırlarından okunur.

Herkese açık Türkçe ve İngilizce veri setleri, LLM ile etiketlenmiş açık metinler ve sentetik veriyle eğitildi; veri setleri yayınlanmıyor. Herhangi bir ticari karar API'sinden damıtılmadı. Eğitimde kaynak kodu yoktur; kod incelemesi bu sürümün kapsamı dışındadır. Bu sürümde web verisi yok; tarayıcı sürümü ayrı yayınlanacak.

Tam JevBench'te (22.773 satır, resmi `jevify-run report`, 22 config eşit ağırlıklı makro; buradaki JevBench, Praveenrajus/jev-bench veri setidir, Benchmark Heaven'ın aynı adlı JevBench tablosu değildir) Turnike-JEV-4B 0,774 alır: makro ECE 0,084, held-6 0,830, diğer-16 0,752. Aynı sette Jev 1.13.0 0,733 alır. Herkese açık jev-bench liderlik tablosunda (46 satır) 0,774, en üstteki girdinin (35B MoE, 0,770) önünde birinci sıraya denk gelirdi; fark küçük ve sonuç tabloya gönderilmedi. İnsan etiketli Türkçe testte (tr_insan, 1.379 karar) Turnike-JEV-4B 0,748, Microsoft-Decision-1 0,714 (p = 0,003), Jev 0,727, Intern-Decision-4B 0,675 alır. Çeviri JevBench TR-50'de Microsoft-Decision-1 önde: 0,745'e 0,700. Gömülü talimata kanma −0,014 (Intern +0,279). Dengeli "hiçbiri" testinde Turnike-JEV-4B 0,742–0,768. Bu test setlerinin hepsi koşular arasında seçim yaparken kullanıldı; tamamen bağımsız tahmin değiller.

Sınırlamalar: "Yukarıdakilerden hiçbiri" biraz fazla kullanılıyor; doğru şık varken de 0,22–0,25 oranında seçiliyor. KVKK eşikleri gürültülü (2 hata, ama tohumlar arasında 11 ile 16 arasında oynuyor). go_emotions, nefret/saldırgan dil sınırı ve yıldız puanı zayıf. Etiketlerin çoğu LLM'den geliyor.

Lisans: taban model Apache-2.0; adaptör, birleştirilmiş ağırlıklar ve kod da Apache-2.0. Kullanım için `karar.py` içindeki `Karar(...).sor(metin, sorular)` yeterli. Bu kart ve çıkarım kodu, yazarın yönlendirmesiyle Claude (Anthropic) kodlama ajanları tarafından yazıldı; sayılar yazar tarafından deney kayıtlarıyla doğrulandı.
