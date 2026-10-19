# Training data

Kai's own `decision_finetune`, as the Kai repository shipped it until its 2026-09-27 model-only release, in two runs, one per question path, since the Choice and Noul paths share no trainable weight. Tasks inside a path are mixed with temperature 2 and no task above 1.5 times an equal share: 160,000 Choice and 316,403 Noul rows per epoch, logical batch 64, encoder LR 2.5e-5, head LR 1e-4, weight decay 0.01, clip norm 5, seed 20260926. Choice trains for 3 epochs on options written as `KEY: description`, the form the Decision runtime in [semantic-router#4086](https://github.com/vllm-project/semantic-router/pull/4086) sends; Noul trains for 1 epoch. The checkpoint is picked by dev NLL: step 5,000 of 7,500 for Choice and the end of the Noul epoch, step 4,944. The two paths are then merged, with no interpolation back to Kai.

The data is the training split of the suite in [semantic-router#4305](https://github.com/vllm-project/semantic-router/issues/4305) with two additions found while tuning against the dev split: for jailbreak, opening messages of attack chats from the SaTML 2024 LLM CTF and, as benign rows, its utility-chat openings and persona prompts from prompts.chat; for PII, Few-NERD sentences with and without a person name. 366,323 distinct rows over ten tasks. The in-distribution test split was not trained on. CoCoNot is included: its dataset card names ODC-BY in its licensing section and the AI2 ImpACT Low Risk licence in its summary. Dev is the suite's dev split and only picks the checkpoint, the temperatures and the thresholds. No held-out file or source test below was trained on, except where [evaluation/RESULTS.md](evaluation/RESULTS.md) marks a slice of a training corpus.

The training files, their lock files and manifests, and the per-row predictions behind every table on this card are in [vllm-sr/router-signal-suite](https://huggingface.co/datasets/vllm-sr/router-signal-suite), with the commands that rebuild them.

## Sources

The builder generates no rows, but many corpora were themselves generated or translated by a model for their dataset, and many labels come from a model rather than from people. [semantic-router#4305](https://github.com/vllm-project/semantic-router/issues/4305) lists the origin of every corpus's text and labels and how each corpus's labels map to the task. Sources under CC BY-SA (Dolly, EXAMS, Stack Exchange, Schema-Guided Dialogue, Few-NERD) are named here for attribution. WildGuardMix and WildJailbreak are gated behind the AI2 Responsible Use Guidelines.

| source | revision | licence | rows used (task) |
|---|---|---|---|
| [Agent-Ark/Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) | 0df3cf37f2ab | Apache-2.0 | 6,862 (tool need) |
| [allenai/coconot](https://huggingface.co/datasets/allenai/coconot) | 2cbe16aabf90 | ODC-BY in the card's licensing section; the card summary names the AI2 ImpACT Low Risk licence | 2,143 (safety) |
| [allenai/WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) | c827c6df8fcf | ODC-BY | 2,519 (jailbreak), 13,264 (modality) |
| [allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix) | d29c47f41c8b | ODC-BY, gated (AI2 Responsible Use Guidelines) | 12,346 (safety) |
| [allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) | 5ddc12a7894f | ODC-BY, gated (AI2 Responsible Use Guidelines) | 17,734 (safety) |
| [argilla/databricks-dolly-15k-curated-multilingual](https://huggingface.co/datasets/argilla/databricks-dolly-15k-curated-multilingual) | 5f466e5af11f | CC BY-SA 3.0 | 1,776 (fact check), 7,111 (modality) |
| [bench-llm/or-bench](https://huggingface.co/datasets/bench-llm/or-bench) | e36d8b80e818 | CC BY 4.0 | 644 (safety) |
| [CohereLabs/aya_redteaming](https://huggingface.co/datasets/CohereLabs/aya_redteaming) | 5a16fee03c19 | Apache-2.0 | 4,237 (hazard) |
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | bdd27f4d94b9 | CC BY-SA 3.0 | 852 (domain) |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | 4f61ecb038e9 | Apache-2.0 | 227 (jailbreak) |
| [DFKI-SLT/few-nerd](https://huggingface.co/datasets/DFKI-SLT/few-nerd) | 205f3e9c9f35 | CC BY-SA 4.0 | 3,804 (pii) |
| [ethz-spylab/ctf-satml24](https://huggingface.co/datasets/ethz-spylab/ctf-satml24) | 11426babe61e | MIT | 1,561 (jailbreak) |
| [fka/prompts.chat](https://huggingface.co/datasets/fka/prompts.chat) | fbea17f2045d | CC0 1.0 | 1,281 (jailbreak) |
| [FreedomIntelligence/ShareGPT-4o-Image](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image)<br>[WINDop/OpenGPT-4o-Image](https://huggingface.co/datasets/WINDop/OpenGPT-4o-Image) | f9bcb9494e56<br>67d8e3f87f6a | Apache-2.0 | 11,788 (modality) |
| [galileo-ai/ragbench](https://huggingface.co/datasets/galileo-ai/ragbench) | 97808f3e5fd1 | CC BY 4.0 | 2,417 (hallucination) |
| [github/amazon-science/RefChecker](https://github.com/amazon-science/RefChecker) | 1df1b25cee79 | CC BY 4.0 annotations over Dolly texts (CC BY-SA 3.0) | 58 (hallucination) |
| [github/asappresearch/abcd](https://github.com/asappresearch/abcd) | 6b8700ce67c6 | MIT | 4,482 (pii) |
| [github/dataminr-ai/BUMP](https://github.com/dataminr-ai/BUMP) | 86327f14b9e0 | MIT | 164 (hallucination) |
| [github/google-research-datasets/dstc8-schema-guided-dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | e852981ae349 | CC BY-SA 4.0 | 13,557 (feedback) |
| [github/Lurunchik/NF-CATS](https://github.com/Lurunchik/NF-CATS) | 1b8c3b83337d | MIT | 482 (fact check) |
| [github/mtbench101/mt-bench-101](https://github.com/mtbench101/mt-bench-101) | bc18b3e2c18c | Apache-2.0 | 401 (feedback) |
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | e7f4b6456019 | Apache-2.0 | 2,768 (tool need) |
| [gretelai/synthetic_pii_finance_multilingual](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) | 7b844d167385 | Apache-2.0 | 12,952 (pii) |
| [hackaprompt/hackaprompt-dataset](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset) | 25b87fbedfb8 | MIT, gated | 5,072 (jailbreak) |
| [HuggingFaceGECLM/StackExchange_Mar2023](https://huggingface.co/datasets/HuggingFaceGECLM/StackExchange_Mar2023) | d331d671d414 | CC BY-SA (Stack Exchange) | 28,358 (domain), 4,992 (modality) |
| [ildpil/text-anonymization-benchmark](https://huggingface.co/datasets/ildpil/text-anonymization-benchmark) | 1f22ce098996 | MIT | 7,138 (pii) |
| [Itaykhealth/K-QA](https://huggingface.co/datasets/Itaykhealth/K-QA) | 99f624dfd517 | MIT | 689 (domain) |
| [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | 2f2ceeb39658 | Apache-2.0 | 494 (jailbreak) |
| [JiayinWang/URS](https://huggingface.co/datasets/JiayinWang/URS) | 056cdc756e60 | Apache-2.0 | 292 (fact check) |
| [joelniklaus/mapa](https://huggingface.co/datasets/joelniklaus/mapa) | bbb2a0157b76 | CC BY 4.0 | 4,262 (pii) |
| [kth8/user_prompt_domain_classification-500000x](https://huggingface.co/datasets/kth8/user_prompt_domain_classification-500000x) | 8cf823b608ad | Apache-2.0 | 16,032 (domain) |
| [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | 04737b65e90a | MIT | 788 (jailbreak) |
| [launch/open_question_type](https://huggingface.co/datasets/launch/open_question_type) | 1cf33ab60b18 | CC BY 4.0 | 920 (fact check) |
| [lmarena-ai/search-arena-24k](https://huggingface.co/datasets/lmarena-ai/search-arena-24k) | fac8dcf86146 | CC BY 4.0 (prompts) | 1,782 (fact check) |
| [lytang/C2D-and-D2C-MiniCheck](https://huggingface.co/datasets/lytang/C2D-and-D2C-MiniCheck) | 1f698000d2f0 | MIT | 4,158 (hallucination) |
| [MadeAgents/HammerBench](https://huggingface.co/datasets/MadeAgents/HammerBench) | 18b4f4ea47e8 | Apache-2.0 | 3,446 (tool need) |
| [mhardalov/exams](https://huggingface.co/datasets/mhardalov/exams) | 4ff10804abb3 | CC BY-SA 4.0 | 2,916 (domain) |
| [microsoft/WildFeedback](https://huggingface.co/datasets/microsoft/WildFeedback) | 8b1a3e530b94 | ODC-BY | 20,816 (feedback) |
| [nvidia/Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | d86bb8bedff5 | CC BY 4.0 | 12,574 (hazard), 7,485 (safety) |
| [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | b70ffaf5ff39 | CC BY 4.0 | 6,062 (pii) |
| [nvidia/Nemotron-Safety-Guard-Dataset-v3](https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3) | a3f7ecb3433d | CC BY 4.0 | 16,628 (hazard), 16,224 (safety) |
| [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) | 0582f7749df6 | CC BY 4.0 | 2,070 (tool need) |
| [opencompass/anah](https://huggingface.co/datasets/opencompass/anah) | 12238ed44444 | Apache-2.0 | 24 (hallucination) |
| [OpenSafetyLab/Salad-Data](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data) | d21a325e276a | Apache-2.0 | 3,523 (jailbreak) |
| [osunlp/AttributionBench](https://huggingface.co/datasets/osunlp/AttributionBench) | 62569e644f41 | Apache-2.0 | 7,504 (hallucination) |
| [peter-sushko/RealEdit](https://huggingface.co/datasets/peter-sushko/RealEdit) | e3a1da4e7a31 | CC BY 4.0 | 10,076 (modality) |
| [poloclub/diffusiondb](https://huggingface.co/datasets/poloclub/diffusiondb)<br>[succinctly/midjourney-prompts](https://huggingface.co/datasets/succinctly/midjourney-prompts) | fb620fbe49fa<br>e670508f77f2 | CC0 1.0 / Apache-2.0 | 4,659 (modality) |
| [s-nlp/PsiloQA](https://huggingface.co/datasets/s-nlp/PsiloQA) | 375c3321b833 | CC BY 4.0 | 7,914 (hallucination) |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 6bda777c88d2 | Apache-2.0 | 2,356 (tool need) |
| [TIGER-Lab/WebInstruct-verified](https://huggingface.co/datasets/TIGER-Lab/WebInstruct-verified) | 3e8a350b3a93 | Apache-2.0 | 16,309 (domain) |
| [ToxicityPrompts/PolyGuardMix](https://huggingface.co/datasets/ToxicityPrompts/PolyGuardMix) | 5b7d93e9e9a6 | CC BY 4.0 | 11,995 (safety) |
| [TrustAIRLab/in-the-wild-jailbreak-prompts](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts) | a10aab8eff1c | MIT | 903 (jailbreak) |
| [wandb/RAGTruth-processed](https://huggingface.co/datasets/wandb/RAGTruth-processed) | eb4f4b9d1b68 | MIT | 1,506 (hallucination) |
| [Wismut/nym-pii-multilingual-data](https://huggingface.co/datasets/Wismut/nym-pii-multilingual-data) | abe23bf08c30 | MIT | 9,036 (pii) |
| [yupp-ai/yupp-svg-20251204](https://huggingface.co/datasets/yupp-ai/yupp-svg-20251204) | 562316157648 | CC BY 4.0 | 1,890 (modality) |
