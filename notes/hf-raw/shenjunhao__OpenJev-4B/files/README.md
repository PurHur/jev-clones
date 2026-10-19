---
license: mit
base_model: Qwen/Qwen3.5-4B
language:
- en
- zh
tags:
- openjev
- decision-making
- open-vocabulary
- reinforcement-learning
- safetensors
---

<div align="center">
<h1>OpenJev-4B</h1>
<p style="color:#627f92;font-size:18px"><strong>Learning Open-Vocabulary Probabilistic Decision Models from Pretrained Language Models</strong></p>
<p>
  <a href="https://github.com/ejhshen/OpenJev"><img src="https://img.shields.io/badge/GitHub-OpenJev-181717.svg?logo=github&amp;logoColor=white" alt="GitHub repository"></a>
</p>
</div>

## 1. Introduction

**OpenJev-4B** is a decision model built from the text backbone of [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). Given a **state**, a **question**, and a request-defined set of **natural-language options**, it directly predicts a probability distribution over those options. It supports selection, binary judgments, and rubric-based scoring through the same probabilistic interface.

The model combines a language-model backbone with a lightweight **Option Set Interactor** and a shared decision head. Its output dimension follows the candidate set supplied with each request: option names and optional criteria are inputs, rather than classes fixed in the model parameters. Decision probabilities are computed without autoregressive answer generation.

OpenJev-4B is post-trained on [OpenJevData-140k](https://huggingface.co/datasets/shenjunhao/OpenJevData-140k) in two stages: **supervised fine-tuning (SFT)** followed by **REINFORCE-Analysis (REINFORCE-A)**. SFT establishes decision behavior using predominantly hard labels. RL increases the share of soft-target contexts and refines probability quality through sampled outcome feedback.

## 2. Model Summary

![OpenJev architecture: independent option branches, a two-layer Option Set Interactor, and question-option compatibility scoring](assets/openjev-architecture.png)

OpenJev-4B brings together the following architecture and training features:

- **Open-Vocabulary Decisions**: Accepts request-defined options and criteria, returning explicit probabilities for selection, binary judgments and rubric-based scoring.
- **Permutation Equivariance**: Shared option encoders and an order-independent decision head make predictions follow the options when they are reordered, up to numerical precision.
- **Cross-Option Understanding**: A lightweight **Option Set Interactor** models semantic relationships across candidates, including overlap, competition and references to other options.
- **Direct, Parallel Inference**: Computes decision probabilities without autoregressive answer generation, with parallel option branches and shared-prefix reuse for attention caches and recurrent states.
- **Two-Stage Post-Training**: Full-parameter **SFT → REINFORCE-Analysis** combines predominantly hard-label supervision with a larger share of soft-target contexts during RL to refine probability quality.
- **Efficient Outcome-Based RL**: **REINFORCE-Analysis** samples 16 candidate actions with 10% uniform exploration and learns from binary outcome feedback. Analytic computation, importance correction and a leave-one-out baseline reduce reliance on sampled estimates.
- **Replay and Reference Regularization**: A fixed 25% of SFT contexts are replayed with the same RL objective. A frozen SFT reference provides KL regularization to help retain learned behavior.

Training data: [OpenJevData-140k](https://huggingface.co/datasets/shenjunhao/OpenJevData-140k).

## 3. Evaluation Results

1. **Accuracy (↑):** the percentage of hard-label examples for which the highest-probability option matches the gold answer. Higher is better; unsupported or failed predictions count as incorrect.
2. **Brier (↓):** the average sum of squared differences between predicted and gold option probabilities. Lower is better, with **0** indicating a perfect match. For hard labels, the gold distribution is one-hot; for soft labels, it is the provided target distribution.

We compare **OpenJev-4B** with **Jev, Decider 4B, JevK5 4B, and Intern-Decision-4B** on [MMDM](https://huggingface.co/datasets/shenjunhao/mmdm) and [JevBench Public](https://github.com/fstandhartinger/jevbench).

<style>
.openjev-eval th,.openjev-eval td{box-sizing:border-box}
.openjev-eval td{font-variant-numeric:tabular-nums}
.openjev-eval sup{font-size:9px;line-height:0;vertical-align:super;margin-left:2px}
.openjev-eval a{color:inherit;text-decoration:none}
.openjev-eval a:hover{text-decoration:underline}
.openjev-eval .oj-muted{color:#738895;font-size:11px;font-weight:400}
.openjev-eval .oj-separator{color:#9baeb9;padding:0 2px}
@media(prefers-color-scheme:dark){
.openjev-eval{color:#dce5eb!important}
.openjev-eval th{color:#bfd1de!important;border-color:#7894a5!important}
.openjev-eval .oj-band{background:#344650!important;color:#d9e5ed!important}
.openjev-eval .oj-soft{background:#3b4756!important;color:#d8e0eb!important}
.openjev-eval .oj-own{background:rgba(137,163,184,.13)!important}
.openjev-eval .oj-total{background:rgba(137,163,184,.16)!important}
.openjev-eval .oj-muted{color:#a3b4c0!important}
}
</style>
<div style="overflow-x:auto;margin:18px 0 10px">
<table class="openjev-eval" style="display:table!important;width:100%;min-width:700px;table-layout:fixed;border-collapse:collapse;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;font-size:12px;color:#374f60;line-height:1.4">
<thead><tr>
<th style="width:16%;padding:10px 6px;overflow-wrap:break-word;text-align:left;border-bottom:2px solid #7894a5;color:#486779;font-weight:600">Task type</th>
<th class="oj-own" style="width:16.8%;padding:10px 4px;text-align:center;border-bottom:2px solid #7894a5;color:#486779;background:rgba(137,163,184,.11);"><a href="https://huggingface.co/shenjunhao/OpenJev-4B">OpenJev-4B</a></th>
<th class="" style="width:16.8%;padding:10px 4px;text-align:center;border-bottom:2px solid #7894a5;color:#486779;"><a href="https://docs.typesafe.ai/introduction/quickstart">Jev</a><sup><a href="#openjev-note-1">1</a></sup></th>
<th class="" style="width:16.8%;padding:10px 4px;text-align:center;border-bottom:2px solid #7894a5;color:#486779;"><a href="https://huggingface.co/Mapika/decider-4b">Decider 4B</a><sup><a href="#openjev-note-3">3</a></sup></th>
<th class="" style="width:16.8%;padding:10px 4px;text-align:center;border-bottom:2px solid #7894a5;color:#486779;"><a href="https://huggingface.co/alibiserikbay/JevK5">JevK5 4B</a><sup><a href="#openjev-note-3">3</a></sup></th>
<th class="" style="width:16.8%;padding:10px 4px;text-align:center;border-bottom:2px solid #7894a5;color:#486779;"><a href="https://huggingface.co/internlm/Intern-Decision-4B">Intern-<wbr>Decision-4B</a><sup><a href="#openjev-note-2">2</a></sup></th>
</tr></thead>
<tbody>
<tr><td colspan="6" class="oj-band " style="padding:9px 6px;background:#dfe8ed;color:#486779;font-weight:600;letter-spacing:.3px">MMDM · Hard · Accuracy (%) ↑ / Brier ↓<sup><a href="#openjev-note-4">4</a></sup></td></tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Answer preference<br><span class="oj-muted">n = 185</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>90.81</strong><span class="oj-separator">/</span><strong>0.1236</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">88.11<span class="oj-separator">/</span>0.1794</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">85.95<span class="oj-separator">/</span>0.2263</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">77.84<span class="oj-separator">/</span>0.3276</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">77.84<span class="oj-separator">/</span>0.3142</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Code / database<br><span class="oj-muted">n = 122</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>84.43</strong><span class="oj-separator">/</span><strong>0.2128</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">83.61<span class="oj-separator">/</span>0.2189</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">77.05<span class="oj-separator">/</span>0.3293</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">72.95<span class="oj-separator">/</span>0.3546</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">68.03<span class="oj-separator">/</span>0.3364</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Commonsense context<br><span class="oj-muted">n = 355</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);">93.80<span class="oj-separator">/</span>0.0854</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>96.90</strong><span class="oj-separator">/</span><strong>0.0409</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">95.21<span class="oj-separator">/</span>0.0687</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">93.80<span class="oj-separator">/</span>0.1073</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">87.04<span class="oj-separator">/</span>0.2096</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Constraint planning<br><span class="oj-muted">n = 334</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>76.05</strong><span class="oj-separator">/</span>0.2944</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">75.75<span class="oj-separator">/</span><strong>0.2859</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">71.26<span class="oj-separator">/</span>0.3685</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">64.37<span class="oj-separator">/</span>0.4816</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">58.38<span class="oj-separator">/</span>0.5222</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Deductive reasoning<br><span class="oj-muted">n = 619</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);">90.47<span class="oj-separator">/</span>0.1389</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>94.51</strong><span class="oj-separator">/</span><strong>0.0781</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">86.27<span class="oj-separator">/</span>0.1861</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">79.97<span class="oj-separator">/</span>0.2809</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">68.66<span class="oj-separator">/</span>0.3848</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Emotion recognition<br><span class="oj-muted">n = 9</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>100.00</strong><span class="oj-separator">/</span><strong>0.0243</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">77.78<span class="oj-separator">/</span>0.2353</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>100.00</strong><span class="oj-separator">/</span>0.0684</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">77.78<span class="oj-separator">/</span>0.3633</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">77.78<span class="oj-separator">/</span>0.3690</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Evidence validation<br><span class="oj-muted">n = 177</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>91.53</strong><span class="oj-separator">/</span>0.1221</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">90.96<span class="oj-separator">/</span><strong>0.1211</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">85.31<span class="oj-separator">/</span>0.2103</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">88.14<span class="oj-separator">/</span>0.1960</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">84.18<span class="oj-separator">/</span>0.2301</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Intent routing<br><span class="oj-muted">n = 253</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>94.07</strong><span class="oj-separator">/</span><strong>0.1045</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">93.28<span class="oj-separator">/</span>0.1047</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">92.49<span class="oj-separator">/</span>0.1223</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">87.75<span class="oj-separator">/</span>0.1688</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">64.82<span class="oj-separator">/</span>0.2352</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Knowledge QA<br><span class="oj-muted">n = 572</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);">95.63<span class="oj-separator">/</span>0.0693</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>98.95</strong><span class="oj-separator">/</span><strong>0.0202</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">96.85<span class="oj-separator">/</span>0.0527</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">94.76<span class="oj-separator">/</span>0.0847</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">88.99<span class="oj-separator">/</span>0.1722</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Multi-hop reasoning<br><span class="oj-muted">n = 211</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>92.42</strong><span class="oj-separator">/</span><strong>0.0992</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">89.57<span class="oj-separator">/</span>0.1274</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">73.93<span class="oj-separator">/</span>0.3978</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">74.41<span class="oj-separator">/</span>0.3046</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">64.45<span class="oj-separator">/</span>0.4399</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Option logic<br><span class="oj-muted">n = 287</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>51.92</strong><span class="oj-separator">/</span><strong>0.5694</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">41.11<span class="oj-separator">/</span>0.6551</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">34.84<span class="oj-separator">/</span>0.7715</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">35.19<span class="oj-separator">/</span>0.7176</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">28.22<span class="oj-separator">/</span>0.7979</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Policy exceptions<br><span class="oj-muted">n = 576</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>95.31</strong><span class="oj-separator">/</span><strong>0.0688</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">90.97<span class="oj-separator">/</span>0.1301</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">82.81<span class="oj-separator">/</span>0.2620</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">72.74<span class="oj-separator">/</span>0.3858</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">63.89<span class="oj-separator">/</span>0.4630</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Quantitative reasoning<br><span class="oj-muted">n = 330</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);">69.09<span class="oj-separator">/</span>0.3752</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>77.88</strong><span class="oj-separator">/</span><strong>0.2841</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">51.21<span class="oj-separator">/</span>0.6098</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">46.36<span class="oj-separator">/</span>0.6636</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">37.58<span class="oj-separator">/</span>0.7533</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Reading comprehension<br><span class="oj-muted">n = 48</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);">95.83<span class="oj-separator">/</span>0.0523</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>100.00</strong><span class="oj-separator">/</span><strong>0.0091</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">95.83<span class="oj-separator">/</span>0.0640</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">91.67<span class="oj-separator">/</span>0.1074</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">89.58<span class="oj-separator">/</span>0.1868</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Score rubrics<br><span class="oj-muted">n = 135</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);">78.52<span class="oj-separator">/</span>0.2939</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>85.93</strong><span class="oj-separator">/</span><strong>0.2055</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">72.59<span class="oj-separator">/</span>0.3487</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">73.33<span class="oj-separator">/</span>0.3190</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">69.63<span class="oj-separator">/</span>0.3827</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Temporal / numeric<br><span class="oj-muted">n = 138</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);">55.07<span class="oj-separator">/</span>0.4695</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>64.49</strong><span class="oj-separator">/</span><strong>0.4235</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">54.35<span class="oj-separator">/</span>0.5197</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">53.62<span class="oj-separator">/</span>0.5379</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">59.42<span class="oj-separator">/</span>0.5422</td>
</tr>
<tr class="oj-total" style="background:#eaf0f3;border-top:2px solid #afc0cb"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:2px solid #afc0cb;font-weight:650">Total<br><span class="oj-muted">n = 4,351</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;background:rgba(137,163,184,.11);">85.57<span class="oj-separator">/</span>0.1864</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;"><strong>86.37</strong><span class="oj-separator">/</span><strong>0.1733</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;">78.88<span class="oj-separator">/</span>0.2832</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;">74.70<span class="oj-separator">/</span>0.3330</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;">66.95<span class="oj-separator">/</span>0.4069</td>
</tr>
<tr><td colspan="6" class="oj-band oj-soft" style="padding:9px 6px;background:#e4e8f0;color:#486779;font-weight:600;letter-spacing:.3px">MMDM · Soft · Brier ↓<sup><a href="#openjev-note-4">4</a></sup></td></tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Constraint planning<br><span class="oj-muted">n = 1</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>0.0070</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1412</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1338</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.0251</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.0258</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Evidence validation<br><span class="oj-muted">n = 22</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);">0.1066</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>0.0493</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1517</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.0773</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.0914</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Intent routing<br><span class="oj-muted">n = 22</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>0.0681</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1570</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1366</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1304</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.0838</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Policy exceptions<br><span class="oj-muted">n = 18</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>0.0423</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.0606</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1128</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.0916</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.0674</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Probabilistic reasoning<br><span class="oj-muted">n = 505</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>0.0278</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1164</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1387</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1092</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1242</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Score rubrics<br><span class="oj-muted">n = 30</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>0.0864</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.0895</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1653</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1242</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1746</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Temporal / numeric<br><span class="oj-muted">n = 12</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);">0.2168</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>0.0822</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1100</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1283</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">0.1747</td>
</tr>
<tr class="oj-total" style="background:#eaf0f3;border-top:2px solid #afc0cb"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:2px solid #afc0cb;font-weight:650">Total<br><span class="oj-muted">n = 610</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;background:rgba(137,163,184,.11);"><strong>0.0391</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;">0.1119</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;">0.1391</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;">0.1092</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;">0.1232</td>
</tr>
<tr><td colspan="6" class="oj-band " style="padding:9px 6px;background:#dfe8ed;color:#486779;font-weight:600;letter-spacing:.3px">JevBench · Public · Accuracy (%) ↑ / Brier ↓<sup><a href="#openjev-note-5">5</a></sup></td></tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Original<br><span class="oj-muted">n = 72</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>100.00</strong><span class="oj-separator">/</span>0.0312</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">98.61<span class="oj-separator">/</span><strong>0.0283</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">98.61<span class="oj-separator">/</span>0.0574</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">94.44<span class="oj-separator">/</span>0.1075</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">98.61<span class="oj-separator">/</span>0.0340</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Easy<br><span class="oj-muted">n = 48</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);"><strong>100.00</strong><span class="oj-separator">/</span><strong>0.000011</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>100.00</strong><span class="oj-separator">/</span>0.000196</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>100.00</strong><span class="oj-separator">/</span>0.000045</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>100.00</strong><span class="oj-separator">/</span>0.002509</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>100.00</strong><span class="oj-separator">/</span>0.001119</td>
</tr>
<tr class="" style="background:transparent;border-top:none"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:1px solid rgba(118,147,164,.20);font-weight:500">Hard<br><span class="oj-muted">n = 111</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);background:rgba(137,163,184,.11);">75.68<span class="oj-separator">/</span>0.3469</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">72.07<span class="oj-separator">/</span>0.3594</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">64.86<span class="oj-separator">/</span>0.4686</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);"><strong>77.48</strong><span class="oj-separator">/</span><strong>0.3118</strong></td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:1px solid rgba(118,147,164,.20);">73.87<span class="oj-separator">/</span>0.3439</td>
</tr>
<tr class="oj-total" style="background:#eaf0f3;border-top:2px solid #afc0cb"><td style="padding:8px 6px;overflow-wrap:break-word;border-bottom:2px solid #afc0cb;font-weight:650">Total<br><span class="oj-muted">n = 231</span></td>
<td class="oj-own" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;background:rgba(137,163,184,.11);"><strong>88.31</strong><span class="oj-separator">/</span>0.1764</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;">86.15<span class="oj-separator">/</span>0.1816</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;">82.68<span class="oj-separator">/</span>0.2431</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;">87.45<span class="oj-separator">/</span>0.1839</td>
<td class="" style="padding:8px 3px;text-align:center;white-space:nowrap;border-bottom:2px solid #afc0cb;">87.01<span class="oj-separator">/</span><strong>0.1761</strong></td>
</tr>
</tbody>
</table>
</div>

<ol style="font-size:12px;line-height:1.65;color:#627f92">
<li id="openjev-note-1"><strong>Jev:</strong> official API, evaluated as Jev 1.13.0.</li>
<li id="openjev-note-2"><strong>Intern-Decision-4B:</strong> official inference code and temperature 1.99241824, BF16, SDPA, no input truncation, in the existing Torch 2.11.0 / Transformers 5.17.0 environment. Its option-count limit prevents scoring 57 MMDM hard examples; these count as incorrect in accuracy and are excluded from the common probability-metric cohort.</li>
<li id="openjev-note-3"><strong>External model versions:</strong> Decider 4B v2.1 and JevK5 4B v0.3. Their returned decision distributions are used for probability metrics. OpenJev-4B is our final release model; earlier OpenJev checkpoints are omitted.</li>
<li id="openjev-note-4"><strong>MMDM:</strong> the published test split contains 4,961 decisions: 4,351 hard and 610 soft. Accuracy uses all hard examples. Brier uses the same 4,294 common valid hard examples across models (196 of 253 for intent routing), and all 610 soft examples, matching the dataset card. All models' totals use the corresponding scoring counts. Soft rows show Brier only. Bold marks the best value for each metric; zero probabilities are not clipped.</li>
<li id="openjev-note-5"><strong>JevBench Public:</strong> local evaluations of the v1.3.0 public suite: Original (72), Easy (48), Hard (111). Accuracy and Brier follow its gold-label scoring protocol; all five models return valid predictions for all 231 examples. “Hard” names a difficulty cohort here, rather than a target type. These results are separate from the official private leaderboard. Easy Brier is shown to six decimal places to retain the small differences.</li>
<li id="openjev-note-6"><strong>Evaluation context:</strong> the published MMDM split combines the previously separate development and test sets, so these numbers are not a new blind evaluation. OpenJev training includes newly constructed scenarios informed by public decision-task patterns. The tables report performance on these fixed evaluation collections.</li>
</ol>

Machine-readable values and evaluation counts are included in [evaluation.json](evaluation.json).

## 4. Model Weights and Code

The complete inference and two-stage training implementation is available at [**ejhshen/OpenJev**](https://github.com/ejhshen/OpenJev). This model repository contains the text backbone, decision head, tokenizer, model configuration and calibration settings. Load them together using the OpenJev runtime.

### Run a decision

Install the [GitHub repository](https://github.com/ejhshen/OpenJev#4-inference) in a compatible CUDA environment, then run:

```python
from openjev.runtime.engine import DecisionEngine

engine = DecisionEngine("shenjunhao/OpenJev-4B", execution_mode="expanded")
result = engine.predict({
    "state": "A customer recognizes a purchase but reports being charged twice.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "Duplicate payment": "The same purchase was charged more than once.",
                "Unrecognized payment": "The customer does not recognize the purchase."
            }
        }
    }
})
print(result["answers"]["route"]["choice"])
print(result["answers"]["route"]["probabilities"])
```

The model accepts a shared `state` and a `questions` object. For `choice`, each question supplies instructions and a mapping from candidate names to criteria; candidate names are defined by the request. It returns the selected candidate and its full probability distribution. The same interface supports `noul` for binary judgments and `score` for expected rubric-level scores. See the [input/output format and HTTP example](https://github.com/ejhshen/OpenJev/blob/main/docs/INFERENCE.md) for details.

The runtime downloads the complete artifact from this repository or accepts a local directory. The release uses **temperature 1.0**, matching the evaluation results. Decision outputs are computed directly, without generating explanation text.

### Train SFT and REINFORCE-A

Training data is available at [**shenjunhao/OpenJevData-140k**](https://huggingface.co/datasets/shenjunhao/OpenJevData-140k). The repository includes data preparation, the optimized distributed training engine, sample replay, resumable caches, checkpointing and model export. To fine-tune OpenJev-4B, set up the official verl/SGLang environment described in the [training guide](https://github.com/ejhshen/OpenJev/blob/main/docs/TRAINING.md), then load the complete released model:

```bash
hf download shenjunhao/OpenJev-4B --local-dir work/base-model
python -m openjev.cli validate-artifact work/base-model
python -m openjev.training.prepare_data \
  --dataset shenjunhao/OpenJevData-140k \
  --mandatory-replay training/mandatory-replay.json --output work/data
NPROC=8 bash training/run_two_stage.sh
```

The pipeline runs SFT, exports the SFT artifact, caches frozen reference probabilities, runs REINFORCE-A and exports the final decision model. Both stages use global batch 1,024 by default; their step counts are calculated from the prepared sample counts. Individual stage launchers and resume instructions are included in the training guide.
