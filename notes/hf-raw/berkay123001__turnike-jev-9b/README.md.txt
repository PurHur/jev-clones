---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
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

![Turnike-JEV-9B](assets/banner_turnike.png)

# Turnike-JEV-9B — Open-Weight Decision Model

**Turnike-JEV-9B: the 9B member of the Turnike family — an open, bilingual (English + Turkish) Jev-class decision model; text plus closed questions in, option probabilities out, fully offline.**

**Turnike-JEV-9B: Turnike ailesinin 9B üyesi — iki dilli (İngilizce + Türkçe), açık ağırlıklı, Jev sınıfı bir karar modeli; metin ve kapalı sorular girer, seçenek olasılıkları çıkar; tamamen çevrim dışı.**

- **The highest full-JevBench score we have seen.** Macro accuracy **0.783** on the official report, above the top public leaderboard entry (0.770, a 35B MoE), the best 9B entry (0.763) and Jev 1.13.0 (0.733 on the same report). On the six held-out sources **0.842**, ahead of Jev (0.835) and pplx-decider-v1-27b (0.840). Not submitted to the leaderboard yet.
- **Same base model as Microsoft-Decision-1; a 173 MB adapter, 2,831 steps on one GPU.** Microsoft-Decision-1 is Microsoft's post-trained Qwen3.5-9B (closed weights, API only). On 1,379 human-labelled Turkish decisions Turnike-JEV-9B is ahead (**0.763** vs 0.714, paired McNemar p = 2e-6); on JevBench EN-300 the two are level (0.780 vs 0.775, n.s.); on machine-translated TR-50 Microsoft is ahead (0.745 vs 0.721).
- **Fits a 16 GB consumer GPU in FP8** with the same decisions as bf16 on our test prompts.
- **Same recipe as [Turnike-JEV-4B](https://huggingface.co/berkay123001/turnike-jev-4b).** The same data, the same draw and the same hyperparameters; nothing was tuned for this size. The 4B card carries the full benchmark tables, the method figure and the honesty notes; this card gives the numbers for the 9B model.
- **Full JevBench test** (22 configs, 22,773 rows, official `jevify-run report`, raw probabilities): macro accuracy **0.783**, macro ECE 0.076, six held-out sources 0.842, other 16 0.761; Turnike-JEV-4B 0.774. Single run.
- **English:** JevBench EN-300 macro accuracy **0.780** (six held-out sources 0.839, other 16 0.758); Turnike-JEV-4B: 0.769. Other models on the same 6,600 rows: Microsoft-Decision-1 (9B, closed, API) 0.775, Kev-4B 0.706, internlm/Intern-Decision-4B 0.687, Kev-0.8B 0.570.
- **Turkish:** tr_insan, 1,379 human-labelled Turkish decisions: **0.763**; Turnike-JEV-4B: 0.748. Others on the same items: Jev 1.13.0 (API) 0.727, Microsoft-Decision-1 (9B, closed, API) 0.714, Kev-4B 0.708, internlm/Intern-Decision-4B 0.675.
- **Also:** practical Turkish questions (2,966) 0.889; LegalBench rule application (885 items, tasks not trained on) 0.890; machine-translated JevBench TR-50 (supplementary, 50 rows per config) 0.721.
- **Runs on your hardware, offline.** A LoRA adapter of about 173 MB on Qwen/Qwen3.5-9B; the text never leaves the machine. Up to 16 questions about one text are answered in a single forward pass. Merged bf16 weights (no PEFT) are in `berkay123001/turnike-jev-9b-merged` and a torchao FP8 copy in `berkay123001/turnike-jev-9b-fp8`: `Karar(None, taban="berkay123001/turnike-jev-9b-merged", fp8=True)` quantizes layer by layer while loading and `Karar(None, taban="berkay123001/turnike-jev-9b-fp8")` loads the stored FP8 weights; both fit a 16 GB GPU (about 9 GB in steady state after the unused vision tower and output head are dropped, ~12 GB peak while loading; measured on an RTX PRO 6000 and checked on an RTX 5060 Ti 16 GB with the stored FP8 weights: 11.9 GB peak, 9.0 GB steady, 19 s to load, same decisions on the test prompts), with the same argmax decisions as bf16 on our test prompts.
- **Latency.** FP8 (stored weights): **~139 ms** per decision (1 question) and ~206 ms for 8 questions on one text, 8.4 GiB in steady state; measured on an RTX 5060 Ti 16 GB through `karar.py`, p50, CUDA-synchronised, 2 warm-up calls dropped (1 question: a 7-option topic question over 60 tr_insan texts; 8 questions: 15 support-desk messages). The 4B under the same protocol: ~108 / ~156 ms in bf16, ~82 / ~127 ms in FP8.

*"JevBench" on this card means the [Praveenrajus/jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench) dataset (22 configs) scored with the `jevify-run report` of [uspraveen/Jevify](https://github.com/uspraveen/Jevify). It is not the [JevBench board by Benchmark Heaven](https://benchmarkheaven.com/jev-models), a different benchmark with the same name, which we have not entered yet.*

> **Declarations / Beyanlar**
> - Trained on public Turkish and English datasets, LLM-labelled public texts and synthetic data; the datasets are not released. / Herkese açık Türkçe ve İngilizce veri setleri, LLM ile etiketlenmiş açık metinler ve sentetik veriyle eğitildi; veri setleri yayınlanmıyor.
> - Not distilled from any commercial decision API. / Herhangi bir ticari karar API'sinden damıtılmadı.
> - AI assistance: the card and code were written with Claude coding agents under the author's direction ([details](#ai-assistance)). / Kart ve kod, yazarın yönlendirmesiyle Claude kodlama ajanlarıyla yazıldı.
> - No web/browser data in this release; the browser version will be published separately. / Bu sürümde web verisi yok; tarayıcı sürümü ayrı yayınlanacak.

Turnike-JEV-9B is a LoRA adapter (r=16, 43.3M trainable parameters, about 173 MB) on [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). It does not generate text. The text and the questions go into one chat prompt, and the assistant turn is a JSON skeleton with one placeholder per question. Every option of a question is mapped to a single-token symbol (A, B, C, ...). At the token just before each placeholder, the model's final hidden state is multiplied with the output rows of that question's symbols only. A softmax over those symbols gives the option probabilities.

- Author: Berkay Hasret (independent researcher)
- Languages: English and Turkish
- Base model: Qwen/Qwen3.5-9B (Apache-2.0, Copyright 2026 Alibaba Cloud), frozen during training; only the adapter is trained
- License of the adapter, `karar.py` and this card: Apache-2.0 (see [License](#license))
- Trained on: Colab RTX PRO 6000 96 GB, batch 16 × 2; 2,831 steps, effective batch 32, max 1,536 tokens, seed 3

<a id="model-family"></a>
## Model family

Three sizes, one recipe: the same data, the same draw (seed 3) and the same hyperparameters (LoRA r=16, α=32, lr 1e-4, 2,831 steps, effective batch 32) for every size. Nothing was tuned per size, so the smallest and the largest model are probably somewhat under-tuned. Each size is its own repo; the row in bold is the model of this card.

| model | base | trainable params | full JevBench (22,773) | JevBench EN-300 (EN) | held-6 / other-16 | tr_insan, 1,379 human-labelled (TR) | practical TR, 2,966 | LegalBench rules, 885 (EN) | weights | trained on |
|---|---|---|---|---|---|---|---|---|---|---|
| [Turnike-JEV-0.8B](https://huggingface.co/berkay123001/turnike-jev-0.8b) | Qwen/Qwen3.5-0.8B | 10.8M | **0.681** | 0.676 | 0.682 / 0.674 | 0.699 | 0.840 | 0.724 | adapter (43 MB) | RTX 5060 Ti 16 GB, batch 4 × 8 |
| [Turnike-JEV-4B](https://huggingface.co/berkay123001/turnike-jev-4b) | Qwen/Qwen3.5-4B | 32.5M | **0.774** | 0.769 | 0.826 / 0.748 | 0.748 | 0.891 | 0.832 | adapter (130 MB) · [merged bf16](https://huggingface.co/berkay123001/turnike-jev-4b-merged) · [FP8](https://huggingface.co/berkay123001/turnike-jev-4b-fp8) (accuracy check pending) | RTX 5060 Ti 16 GB, batch 4 × 8 |
| **Turnike-JEV-9B** (this card) | Qwen/Qwen3.5-9B | 43.3M | **0.783** | 0.780 | 0.839 / 0.758 | 0.763 | 0.889 | 0.890 | adapter (173 MB) · [merged bf16](https://huggingface.co/berkay123001/turnike-jev-9b-merged) · [FP8](https://huggingface.co/berkay123001/turnike-jev-9b-fp8) | Colab RTX PRO 6000 96 GB, batch 16 × 2 |

<sub>Same protocol as the tables below: single runs, the same items for every model, no calibration recipe; full JevBench = official `jevify-run report` on all 22,773 test rows. Differences below about 0.01 are within run-to-run noise. The machine-translated TR-50 is in Table 5.</sub>

## Quick start

```bash
pip install torch "transformers>=5.17" peft accelerate
hf download berkay123001/turnike-jev-9b karar.py requirements.txt --local-dir .
# optional: pip install torchao (FP8), causal-conv1d and flash-linear-attention (faster linear-attention layers)
```

```python
from karar import Karar
k = Karar("berkay123001/turnike-jev-9b")   # the base model (Qwen/Qwen3.5-9B) is read from adapter_config.json and downloaded automatically

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

`karar.py` is the same file for every Turnike size; the question formats (`noul` yes/no, `choice` with a list or a dict of option descriptions, `score`) and the command line (`python karar.py --help`) are documented on the [Turnike-JEV-4B card](https://huggingface.co/berkay123001/turnike-jev-4b#quick-start). `Karar(..., fp8=True)` enables FP8 on CUDA.

## Results

All numbers are our own measurements with the same scripts and the same items as on the 4B card; single runs, raw probabilities, no calibration recipe. Turnike rows are in bold.

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

The detailed tables (per-config JevBench against Jev 1.13.0, paired McNemar tests, the "none of the above" and planted-instruction behaviour tests, KVKK, latency) are on the [Turnike-JEV-4B card](https://huggingface.co/berkay123001/turnike-jev-4b#results). Turnike-JEV-4B is the reference model of the family; this size was measured afterwards with the same code.

## Method and training

One call, one forward pass: the text and up to 16 questions are packed into a single chat prompt whose assistant turn is a JSON skeleton with one placeholder per question; each option is mapped to a single-token symbol and the option probabilities are read from a softmax over those symbols only (see the [Method](https://huggingface.co/berkay123001/turnike-jev-4b#method) and [Prompt format](https://huggingface.co/berkay123001/turnike-jev-4b#prompt-format) sections of the 4B card, with the algorithm and the figure).

Training: KL divergence between the target distribution and the model's option distribution per question; LoRA r=16, α=32, dropout 0.05 on all attention, MLP and linear-attention projections of the language model (the vision tower is untouched and unused); lr 1e-4, warm-up 3%, effective batch 32, 2,831 steps, max length 1,536 tokens, the same 80,000 packed items (seed 3) as the 4B. Two training-data shortcuts found during the 4B work ("none of the above" read from a sibling question, and from the option count) were fixed before this run; see [Training](https://huggingface.co/berkay123001/turnike-jev-4b#training) on the 4B card.

## Limitations

- **No per-size tuning.** The learning rate, the number of steps and the LoRA rank were chosen for the 4B; this size may be under- or over-trained. Single run; differences below about 0.01 (0.02 on TR-50) are within run-to-run noise.
- **Test sets were used for selection** of the recipe (on the 4B); none of the numbers is a fully untouched estimate. The train splits of 16 JevBench configs and of the five tr_insan sources are in training (test texts excluded).
- **Mostly LLM labels.** Much of the training data is LLM-labelled, so the model inherits those judgements. tr_insan is the only human-labelled Turkish test.
- **Calibration:** no temperature or recipe is applied. Fit a temperature on your own validation data if you need calibrated probabilities, and do not threshold yes/no probabilities near 0.5 without a validation set.
- **Not trained on source code.** On a small internal security code-review test (32 snippets with access-control bugs, deliberately fuzzy options, 128 decisions) this size scores 0.594, against 0.805 for Jev 1.13.0 and 0.695 for Microsoft-Decision-1; the other local 4B decision models we ran (Qwen3.5-4B zero-shot, openjev, Intern-Decision-4B, jevify readout) land between 0.414 and 0.547. Treat code review as out of scope for this release.
- **FP8** (`Karar(..., fp8=True)`) was validated on the 4B only. On the 9B, FP8 kept accuracy on EN-300 (0.780 → 0.782, McNemar p = 0.42) and on tr_insan (0.763 → 0.759, p = 0.38), with about 2% of individual decisions flipping; all 14 test-prompt decisions kept their argmax (max |Δp| 0.08). Measured on an RTX PRO 6000.
- Long prompts are truncated (text only; `maks_istem_token` in `karar.py`); training used 1,536 tokens. Questions with more than 62 options use an elimination step.

## License

The base model Qwen/Qwen3.5-9B is licensed under Apache-2.0 (Copyright 2026 Alibaba Cloud). The adapter weights, `karar.py`, the tests and this card are released under Apache-2.0 (`LICENSE`).

<a id="ai-assistance"></a>
## AI assistance

This model card, `karar.py`, the tests and most of the training and evaluation code were written with Claude coding agents (Claude Code, Anthropic) under the author's direction. The author is responsible for the content and verified the reported numbers.

## Citation

```bibtex
@misc{turnike_jev_9b_2026,
  title        = "Turnike-JEV-9B: a bilingual (English + Turkish) decision model on Qwen/Qwen3.5-9B",
  author       = "Hasret, Berkay",
  year         = "2026",
  howpublished = "https://huggingface.co/berkay123001/turnike-jev-9b"
}
```

## Türkçe özet

Turnike-JEV-9B, Turnike ailesinin 9B üyesidir: Qwen/Qwen3.5-9B üzerine yaklaşık 173 MB'lık bir LoRA adaptörü. Metin üretmez; bir metin ve bir ya da birkaç kapalı soru verildiğinde her seçeneğe bir olasılık döndürür, bir metnin 16 sorusuna kadar tek ileri geçişte cevap verir. RTX 5060 Ti 16 GB üzerinde FP8 ile tek karar ≈ 139 ms, 8 soru ≈ 206 ms sürer (8,4 GiB bellek). Eğitim verisi, çekiliş ve bütün ayarlar Turnike-JEV-4B ile birebir aynıdır; bu boyut için ayrı ayar yapılmadı.

Sonuçlar (aynı betikler, aynı maddeler, tek koşu): tam JevBench (22.773 satır; Praveenrajus/jev-bench veri seti, Benchmark Heaven'ın aynı adlı JevBench tablosu değil) makro 0.783 (Turnike-JEV-4B: 0.774), JevBench EN-300 makro 0.780 (Turnike-JEV-4B: 0.769), insan etiketli Türkçe test tr_insan 0.763 (Turnike-JEV-4B: 0.748), pratik Türkçe sorular 0.889, LegalBench kural uygulama 0.890, çeviri JevBench TR-50 0.721. Ayrıntılı tablolar, davranış testleri ve dürüstlük notları Turnike-JEV-4B kartındadır.

Herkese açık Türkçe ve İngilizce veri setleri, LLM ile etiketlenmiş açık metinler ve sentetik veriyle eğitildi; veri setleri yayınlanmıyor. Herhangi bir ticari karar API'sinden damıtılmadı. Eğitimde kaynak kodu yoktur; kod incelemesi bu sürümün kapsamı dışındadır. Lisans: taban model ve adaptör Apache-2.0. Kullanım için `karar.py` içindeki `Karar("berkay123001/turnike-jev-9b").sor(metin, sorular)` yeterlidir. Bu kart ve çıkarım kodu, yazarın yönlendirmesiyle Claude (Anthropic) kodlama ajanları tarafından yazıldı; sayılar yazar tarafından doğrulandı.
