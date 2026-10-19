---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: adapter
library_name: peft
datasets:
- ZefanCai/Open-Jev-v1.1
tags:
- open-jev
- qwen3.8
- lora
- non-generative
- typed-decisions
---

# Open-Jev-27B-v1.1

**A trained LoRA adapter and scalar decision head for the text backbone of Qwen/Qwen3.8-27B.** The package requires the exact upstream model and tokenizer revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and the [Open-Jev loader](https://github.com/Zefan-Cai/Open-Jev). It contains 15,471,617 trained parameters; upstream base-model weights are obtained separately.

**Evaluation complete:** all 127,787 original-mixture Test/OOD rows were evaluated and independently audited, with no missing, duplicate or failed predictions. The same checkpoint scored **197/231 (85.28%)** on the public JevBench subset and **80/111 (72.07%)** on its Hard tier. Full results, exact denominators and separate evaluation protocols are below.

Open-Jev scores caller-supplied candidates directly and returns typed decisions:

- **Choice:** a probability distribution over the supplied candidates.
- **Noul:** a yes/no probability.
- **Score:** probabilities over supplied ordinal levels and their expected value.

One LoRA adapter and scalar head serve these tasks. Each candidate receives a scalar score, so the number and meaning of candidates can change between requests and domains. The probabilities come from the decision head and saved temperature, without autoregressive answer generation. The server returns decisions; it does not execute actions.

The package uses a text backbone. Gameplay and browser integrations supply structured observations or text through their task adapters. This release does not establish image-only perception or a full autonomous workflow for every domain.

## Download and run

Use an environment with sufficient GPU memory for the upstream model and selected input sizes. These commands use the conservative uncached path:

```bash
git clone https://github.com/Zefan-Cai/Open-Jev.git
cd Open-Jev
python -m pip install -e '.[train]'
hf download ZefanCai/Open-Jev-27B-v1.1 --local-dir ./checkpoints/open-jev-27b-v1.1
python -m jev.server \
  --checkpoint ./checkpoints/open-jev-27b-v1.1/package/checkpoint \
  --device cuda:0 --max-length 4096 --batch-size 1 --no-prefix-cache \
  --host 127.0.0.1 --port 8791
```

The `train` extra also supplies inference dependencies: Transformers 5.10.2, PEFT 0.19.1, Accelerate 1.13.0, Torch and safetensors. Pin the downloaded model repository to a commit with `hf download --revision <commit>` for reproducible deployments.

The Open-Jev loader reads the exact upstream revision from `package/checkpoint/model.json`. The unchanged PEFT adapter configuration has an empty `base_model_name_or_path`; a generic `AutoPeftModel` text-generation call is not the loading interface for this package. It would also omit the separately saved scalar head and temperature.

Example request, provided as an input example rather than a recorded prediction:

```bash
curl http://127.0.0.1:8791/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice and want a refund.","questions":{"intent":{"type":"choice","instructions":"Choose the customer intent.","criteria":{"billing":"A payment or refund issue","technical":"A malfunction or setup issue","other":"Another request"}},"refund_requested":{"type":"noul","instructions":"Does the customer explicitly request a refund?"}}}'
```

The saved maximum input length is **4,096 tokens per candidate**. Overlength inputs are rejected rather than silently truncated. Larger experimental evaluation limits do not change the training length or establish equivalent quality at those lengths. Prefix-cache or cross-question batching experiments require their own checkpoint and hardware validation; this card makes no proprietary-kernel equivalence or GPU speedup claim.

## Training

This checkpoint completed a predeclared pass over the frozen `community-hard-mix-v2-final` training split:

| Setting | Value |
| --- | --- |
| Unique training rows | 148,639 |
| Optimizer steps for this mixture | 37,160 |
| Global batch | 4 |
| Consumed rows | 148,640; one row wraps at the final step |
| Trainable parameters | 15,471,617 |
| LoRA | Rank 8, alpha 16, dropout 0 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `out_proj` |
| LoRA / head learning rates | `2e-5` / `5e-5` |
| Objective | Candidate negative log likelihood plus Brier loss, weight `0.1` |
| Sampling / seed | Shuffled / `20260921` |
| Maximum training input length | 4,096 tokens |
| Saved temperature | `2.5343690298472983`, fitted on 512 calibration rows |
| Final continuation code commit | `7ac6bab261bd97efc3d446152999e94ae432e5b0` |

The run **warm-started from an earlier Open-Jev 27B checkpoint at step 26,500**, trained on the browser/drone expansion mixture. The new mixture started with fresh optimizer state, random streams and step cursor. Thus the 148,639 rows above describe this additional training stage, not all data ever used by the adapter. The source snapshot manifest hash is `28aca3c8977191f976b0b961bd9924e4ba4ea511c6b627a887ae7cad937f41a5`.

The run later continued through a state-preserving migration to four-rank FSDP2 for the frozen BF16 backbone, with replicated FP32 LoRA/head parameters. This records execution provenance; it does not claim bitwise equality to uninterrupted DDP. The training-time baseline is the warm-start checkpoint, not raw Qwen.

## Data and provenance

The final training stage combines **82,045 WANLI NLI rows**, **30,720 new synthetic community-task rows**, and **35,874 replay rows**. The new community tasks cover support, browser tools, games, information retrieval, RAG, contracts, shell-history tasks and rubric judgments. Replay adds the earlier extraction, email, citation, workflow, navigation, painting and control tasks. The original training manifest names 74 source groups; these are not 74 independently validated real-world domains.

The original frozen mixture contains 328,672 rows across all splits:

| Split | Original mixture | Public redistributable projection |
| --- | ---: | ---: |
| Train | 148,639 | 147,139 |
| Calibration | 26,764 | 26,675 |
| Validation | 25,482 | 25,413 |
| Test | 43,301 | 43,125 |
| OOD | 84,486 | 84,267 |

The [v1.1 dataset](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1), configuration `community-hard-mix-v2-redistributable`, omits 2,053 original `wikispeedia-v1` rows, including 1,500 training rows, because redistribution permission for the source graph/path archive has not been confirmed. **The public projection is not byte-identical to the data used for these weights.** Internal evaluation of the original mixture includes those Wiki rows. The dataset's reconstruction metadata can restore the original mixture when the omitted files are separately supplied and verified.

The original manifest SHA-256 is `7b26f948d2ae11f20d7a18be437d1ada616fc479e1596ae94be7596787fe4e54`; the original training-file SHA-256 is `959dac64adc1717d92e9bb80371d4b940b2b5994f9e36e469b3eaed85883fdff`. [The full manifest](provenance/original-training-mixture-manifest.json) and [provenance record](package/provenance.json) preserve split and component hashes. No dataset rows are bundled with this model.

WANLI is credited to [alisawuffles/WANLI](https://github.com/alisawuffles/WANLI) at revision `61c95318fd71c55b6ba355d76253254615f387ec`, under CC-BY-4.0. Its source premise/hypothesis text and published labels were converted into the common decision format. New synthetic community data and the listed original synthetic control sources carry their own CC0-1.0 declarations. The browser/drone aggregate retains per-component terms, including the Wikispeedia limitation above. Dataset licenses do not relicense model weights.

The saved lexical overlap screen found no matches against 531 frozen visible benchmark requests. This checks exact or lexical overlap only; it does not prove semantic independence, exclude paraphrases, or assess upstream pretraining exposure. Later community v3/v4 mixtures are not part of this completed training stage.

## Evaluation results and limits

The full internal evaluation covers **43,301 Test + 84,486 OOD = 127,787 rows** of the original frozen mixture. All rows have valid predictions; the independent audit found **zero missing, duplicate or failed records** and verified the exact checkpoint, dataset, code, saved-temperature probabilities, successful controller exits and cleanup. These are complete splits, without sampling.

| Checkpoint | Old Test | Old OOD | Expanded Test | Expanded OOD |
| --- | ---: | ---: | ---: | ---: |
| Open-Jev-27B-v1.1 | **9,876 / 10,046 (98.31%)** | **14,825 / 15,446 (95.98%)** | **41,357 / 42,789 (96.65%)** | **80,934 / 83,924 (96.44%)** |

Cells show correct hard labels / all hard labels. The complete raw panels contain 10,532 / 15,920 / 43,301 / 84,486 rows respectively. The expanded panels include **1,074 soft-label rows**; these remain in the probability metrics but are excluded from hard-label accuracy. Old Test/OOD are content-identical subsets of Expanded Test/OOD, so the panels overlap. Expanded scores cover the original complete mixture, including the Wiki rows omitted from the public data projection. The earlier 512-row training diagnostics are not used for any cell.

The [complete internal audit report](verification/internal-full.json) includes hard/soft denominators, NLL, Brier, expected accuracy, ECE and evidence hashes. It was produced with a BF16 base, FP32 head, maximum length **4,096**, no truncation, no prefix cache, one row per rank, and the checkpoint's saved temperature **2.5343690298472983**; no temperature was fitted on the evaluation data. The report SHA-256 is `19a154c1a6ed3935bac266ab04b8a395bc180739540127b4b3802190e3c34f97`.

### JevBench public subset

The separately audited JevBench run uses **231 public tasks**: 72 original, 48 easy and 111 Hard. This is **not the complete 534-task benchmark**; unavailable private/judge tiers were not evaluated and no full-benchmark composite is reported.

| Model | Public 231 | Hard 111 |
| --- | ---: | ---: |
| Released Open-Jev 2B | 150 / 231 (64.94%) | 46 / 111 (41.44%) |
| Released Open-Jev 9B | 179 / 231 (77.49%) | 66 / 111 (59.46%) |
| **Open-Jev-27B-v1.1** | **197 / 231 (85.28%)** | **80 / 111 (72.07%)** |
| Jev 1.13.0 | 200 / 231 (86.58%) | 81 / 111 (72.97%) |

27B v1.1 remains **three tasks behind Jev overall and one task behind on Hard**. This run used the identical adapter/head/checkpoint hash and saved temperature, with a **16,384-token** JevBench limit and no prefix cache or retries; it is a separate protocol from the 4,096-token internal evaluation. [The JevBench report](verification/jevbench.json) records task-family results, baseline counts and protocol details. New 2B training was stopped and new 9B training has not started; the smaller-model rows above refer to the previously released checkpoints.

No latency or efficiency comparison is reported in this package. Shared-node four-rank collective execution timings and single-GPU HTTP or hosted API timings measure different execution conditions and cannot establish a direct speedup ratio.

Typed-decision accuracy is not game win rate or end-to-end workflow completion. Synthetic holdouts, a single NLI corpus and lexical screening do not establish broad operational reliability or calibrated probabilities on arbitrary new tasks. The model is an independent Qwen-based Open-Jev implementation, not a release of Jev's private weights or sampler.

## Artifact verification

The package preserves the complete six-file checkpoint tree byte for byte, including the original generated adapter README:

| Artifact | SHA-256 |
| --- | --- |
| LoRA adapter | `1c857224bd3609c6a71eacf7f71dd021115fcc0f791936b1fc332e915b548a81` |
| Scalar head | `76e382f122abfa4e0c467d860a8d142d2fb6d2a98dc0ef9e19870bfc6eb296b4` |
| Original evaluated checkpoint directory | `c49994563c3c4f04a99d9130203c4e526f4ae5086c84deec57698d18cb652e71` |
| Packaged checkpoint directory | `c49994563c3c4f04a99d9130203c4e526f4ae5086c84deec57698d18cb652e71` |

The source and packaged checkpoint directory hashes are identical, so the downloaded checkpoint retains the identity used by the fixed-checkpoint evaluator. The unchanged `adapter/README.md` is a generated PEFT template preserved for byte identity; this root card provides the model documentation. [Checkpoint verification](verification/checkpoint.json) records all six file hashes.

[CPU validation](verification/cpu.json) checked 320 LoRA tensors plus the scalar head: all are finite FP32, and every adapter tensor shape matches the exact pinned Qwen text architecture instantiated on the meta device. It also verified calibration IDs and targets against the original calibration split. No base weights were materialized and no model inference was rerun during packaging. These checks establish structure and artifact identity, not a fresh 27B inference result.

## License and attribution

The trained adapter and head are distributed under **Apache-2.0** with the complete pinned Qwen/Alibaba Cloud license and attribution in [LICENSE](LICENSE) and [UPSTREAM.md](UPSTREAM.md). Open-Jev source code is **MIT**, preserved separately in [LICENSE-CODE](LICENSE-CODE). The upstream model/tokenizer must be obtained under their own terms. No endorsement by Alibaba Cloud, the Qwen Team or TypeSafe is implied.
