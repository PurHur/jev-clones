---
license: other
pipeline_tag: text-generation
library_name: transformers
base_model: Qwen/Qwen3-8B
tags:
  - context-management
  - tool-use
  - agent
---

# ContextPilot-8B

**ContextPilot: Teaching Agents for Proactive Context Management via Fine-grained RL**

<p align="center">
  <a href="https://github.com/Tencent/ContextPilot">
    <img
      src="https://img.shields.io/badge/ContextPilot-GitHub-blue?logo=github"
      alt="GitHub Repo"
    />
  </a>
  <a href="https://tencent.github.io/ContextPilot/">
    <img
      src="https://img.shields.io/badge/ContextPilot-Live_Demo-ff725e?logo=githubpages&logoColor=white"
      alt="ContextPilot Live Demo"
    />
  </a>
  <a href="https://arxiv.org/abs/2608.28476">
    <img
      src="https://img.shields.io/badge/ContextPilot-Paper-red?logo=arxiv&logoColor=red"
      alt="Paper"
    />
  </a>
  <a href="https://huggingface.co/collections/panzs19/contextpilot">
    <img
      src="https://img.shields.io/badge/ContextPilot-Models-yellow?logo=huggingface&logoColor=yellow"
      alt="Hugging Face Models"
    />
  </a>
</p>

ContextPilot-8B is the Qwen3-8B checkpoint of **ContextPilot**, a proactive context-management framework for long-horizon language-model agents. It teaches agents to plan, maintain long-term memory, and offload less useful context while they continue reasoning and using tools. For more details, see our [paper](https://arxiv.org/abs/2608.28476) and [code repository](https://github.com/Tencent/ContextPilot).

![ContextPilot overview](assets/contextpilot_overview.png)

## Overview

ContextPilot combines three main components:

- an extended context-management toolset with planning, structured memory, retrieval, and soft context offloading;
- context-aware partial rollout that focuses exploration on sensitive context-editing decisions; and
- fine-grained credit assignment that trains intermediate snapshots using the outcomes of their downstream branches.

The resulting agents are evaluated on long-context question answering and deep-search tasks; see the [evaluation instructions](https://github.com/Tencent/ContextPilot/tree/main/infer#evaluation) for details.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tencent/ContextPilot-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
```

Note that loading the checkpoint alone does not execute context-management tools; the tool definitions, agent runtime, and evaluation pipeline are provided in the [ContextPilot repository](https://github.com/Tencent/ContextPilot). See the [inference guide](https://github.com/Tencent/ContextPilot/blob/main/infer/README.md) for the full setup.

## Intended Use

This checkpoint is intended for research on proactive context management, long-horizon agents, long-context QA, and deep search.

## License

[LICENSE](https://huggingface.co/tencent/ContextPilot-8B/blob/main/LICENSE).

## Citation

```
@inproceedings{pan-etal-2026-contextpilot,
    title = "ContextPilot: Teaching Agents for Proactive Context Management via Fine-grained RL",
    author = "Pan, Zhuoshi  and
      Pei, Qizhi  and
      Lu, Junru  and
      Lin, Honglin  and
      Zhao, H. Vicky  and
      Yin, Di  and
      Sun, Xing",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    abstract = "Long-horizon agentic tasks require large language models (LLMs) to iteratively retrieve, integrate, and maintain dispersed information across multi-turn interactions, but preserving all interaction histories leads to a continuously growing working context. Recent proactive context management methods allow models to edit their own working context with specialized tools, yet they still face three key limitations: (1) a limited toolset restricted to search, deletion, and summarization, with no support for global planning, long-term memory, and adaptive compression; (2) inefficient exploration that treats context management actions uniformly despite their heterogeneous impacts on final outcomes; and (3) coarse-grained credit assignment that assigns the final trajectory-level reward to all intermediate context editing actions during RL. To bridge these gaps, we introduce ContextPilot, a proactive context management framework for long-horizon agentic reasoning. Our approach systematically augments the toolset with planning, long-term memory, and soft context offloading tools. We further propose an RL method tailored for context management, which uses context and entropy variation to identify critical editing decisions for branch sampling and estimates action-level advantages from all branched trajectories that pass through the corresponding context editing action. Experiments on long-context QA and deep search tasks show that ContextPilot achieves stronger performance with a more compact working context, consistently outperforming existing baselines across various base models and benchmarks. Code is available at \url{https://github.com/Tencent/ContextPilot}."
}
```
