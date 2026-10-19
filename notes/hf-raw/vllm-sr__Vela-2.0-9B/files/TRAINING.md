# Training: Vela 2.0 9B

[Model card](README.md) · [Usage](USAGE.md) · [Evaluation](EVALUATION.md)

## Training stages

This model starts from [Decision-2.0-Lux-9B](https://huggingface.co/vllm-sr/Decision-2.0-Lux-9B), built on Qwen3.5-9B. Three stages add the Vela routing recipe and span capabilities:

| Stage | Steps | Updated parameters | Tasks and recipe |
| --- | ---: | --- | --- |
| 1. Vela routing recipe | 4,000 | Full fine-tune | Safety, prompt attacks, hallucination and PII spans; domain, modality, fact-check and feedback; general decisions from Decision 1.0 source families in our own wording; synthetic router-style questions in 21 languages |
| 2. Router span head | 3,000 | New word-by-label span head; backbone frozen | PII, hallucination and toxic spans |
| 3. Broad span head | 3,000 | Second span head; everything else frozen | Initialised from router head; 90% open-label extraction steps and 10% replay of stage-2 span data |

Stage 1 replays Decision 2.0 training rows on 7.5% of steps, with a KL term to the base to retain part of its general decisions. Stage 3 leaves the backbone, candidate head and router span head unchanged: router outputs were checked to be bitwise identical to the model without the broad head on 700 rows.

## Data mixtures

| data | sources | share of steps (stage 1) |
|---|---|---|
| safety / attack | AEGIS 2.0, PolyGuardMix, Nemotron-Safety-Guard v3, LLMail-Inject, Salad-Data | 12.2% |
| hallucination spans | LettuceDetect prose + code | 11.3% |
| PII | generated long-document PII, 17 types | 27.8% |
| router tasks | Global-MMLU, Aya, DiffusionDB, WildFeedback, router PII recipe, Presidio replay | 3.8% |
| general decisions | SNLI, BoolQ, ARC, MASSIVE, HelpSteer2, KLUE, JGLUE, SocialIQA, CosmosQA, TyDi QA, OASST1, AfriSenti, Super-NI, SentiMix | 12.5% |
| synthetic router-style decisions | generated with Qwen3-30B-A3B-Instruct-2507, kept only when a blind re-label agrees | 23.1% |
| Decision 2.0 replay | Decision 2.0 training rows (not redistributed) | 7.5% |
| small in-house set | | ~2% |

| data | sources (licence) | share of steps (stage 3) |
|---|---|---|
| NER, open labels | NuNER (MIT), pile-mistral (Apache-2.0), GLiNER multi-task synthetic (Apache-2.0), FiNER-139 (CC-BY-SA-4.0), Re-DocRED (MIT) | 22.5% |
| NER, MultiCoNER v2 | MultiCoNER v2 (CC-BY-4.0) | 9.0% |
| NER, Few-NERD | Few-NERD (CC-BY-SA-4.0) | 6.7% |
| NER, multilingual | Universal NER (CC-BY-SA-4.0) | 6.7% |
| relations (entity-anchored) | GLiNER multi-task synthetic (Apache-2.0), Re-DocRED (MIT), SynthIE (MIT) | 15.8% |
| entity-linking mentions | SynthIE (MIT), Re-DocRED (MIT) | 6.7% |
| RAG / extractive evidence | GLiNER multi-task synthetic (Apache-2.0), SQuAD v2 train (CC-BY-SA-4.0), HotpotQA train (CC-BY-SA-4.0), Natural Questions train (CC-BY-SA-3.0), KRLabsOrg verbatim-spans (RAGBench part) (Apache-2.0, over RAGBench CC-BY-4.0), KRLabsOrg tool-output extraction (Apache-2.0) | 22.5% |
| router span replay | the router span data of stage 2 (PII, hallucination, toxic) | 10.0% |

## Held-out data and provenance

- Broad-head zero-shot dev excludes 16 MultiCoNER / Few-NERD entity types and 79 hashed open labels from training.
- It also holds out Serbian Universal NER, hashed relation types (15% of Wikidata relations), and whole datasets: AskNews-NER, NCBI disease and ACL-Verbatim.
- Data are deduplicated against all benchmark texts: CrossNER, MIT Movie / Restaurant, the four relation sets, SQuAD v2 dev and ACL-Verbatim test. Training and evaluation rows are deduplicated against every test set.
- Entity descriptions for some open labels were written with Qwen2.5-7B-Instruct.
- NuNER, pile-mistral, GLiNER multi-task, SynthIE and verbatim-spans contain LLM-generated labels.

## Release files

| file | what |
|---|---|
| `model-*.safetensors` | backbone + decision heads and the router span head |
| `broad_head.safetensors` | the broad span head |
| `modeling_vela2.py`, `configuration_vela2.py` | `AutoModel` remote code |
| `vela2_inference.py`, `vela2_serve.py` | the engine (SystemOne, `predict()`, shortcuts, span-head dispatch) and an optional HTTP server |
| `calibration.json`, `config.json` | temperatures, thresholds, PII rule, broad-head threshold, optional noul calibration; geometry |
| `PARITY.md` | export parity against the research scorer |

## Licences and modifications

Weights, code and documentation are Apache-2.0. Base-model licences and notices are retained in `LICENSE`, `NOTICE`, `ATTRIBUTIONS.md` and `LICENSES/`; see [MODIFICATIONS.md](MODIFICATIONS.md) for changes against Decision 2.0. Training data retain their own licences, including the CC-BY-SA sources listed above.
