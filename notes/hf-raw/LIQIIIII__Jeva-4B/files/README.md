---
license: apache-2.0
language:
  - en
tags:
  - jeva
  - system-one
  - decision-making
  - candidate-selection
  - scoring
  - decision-model
---

<div align="center">

**Jeva: System One Models from First Principles**

**Qi Li · Xinchao Wang**  
National University of Singapore

[Project Page](https://liqiiiii.github.io/Jeva/) · [GitHub](https://github.com/LiQiiiii/Jeva)

</div>

## Overview

Jeva-4B is a System One model for fast, direct decision-making over text. It selects among alternatives, assesses whether a statement is true, and assigns scores according to a rubric—without generating lengthy explanations.

Jeva is built around a simple idea: a decision model should respect the structure of the decision itself. The order of the options should not determine the answer, more options should not require a different model, and choosing nothing should be possible when nothing fits. Familiar options should also remain useful across new requests.

## Four Defining Properties

- **Null-aware.** Supports an explicit “none of the above” outcome when no candidate applies.
- **Permutation-invariant.** Designed to preserve decisions when the same candidates are reordered.
- **Option-scalable.** Accommodates both small and large candidate sets within the same model.
- **Cacheable.** Reuses previously processed options across new requests, including new combinations of familiar options.

**Reuse the options. Recompute the decision.**

## What You Can Do with Jeva

| Capability | Purpose | Example question |
| :--- | :--- | :--- |
| **Choice** | Select a candidate, with an optional “none of the above” outcome. | Which team should handle this request? |
| **Noul** | Estimate the probability that a statement is true. | Does this report describe an ongoing service outage? |
| **Score** | Produce a numerical assessment using a supplied rubric. | How severe is the reported customer impact? |

These capabilities support request routing, document selection, relevance assessment, and rubric-based evaluation. The same familiar alternatives can be used across different questions without reusing an earlier answer.

## Explore Jeva

- **[Project Page](https://liqiiiii.github.io/Jeva/):** Discover the work, illustrations, and research findings.
- **[GitHub Repository](https://github.com/LiQiiiii/Jeva):** Find usage examples and guidance for running Jeva with your own tasks.
- **[Questions and Feedback](https://github.com/LiQiiiii/Jeva/issues):** Share questions, report issues, or suggest improvements.

## Intended Use

Jeva is intended for research and decision-support applications. Evaluate its suitability for your task and retain human oversight for consequential decisions.

## License

Jeva-4B model weights, including the decision head (`head.pt`) and adapter weights (`backbone_adapter/`), are released under the [Apache License 2.0](LICENSE).

The bundled Qwen3.5-4B backbone retains its upstream [Apache-2.0 license](backbone/LICENSE). See [NOTICE](NOTICE) for attribution.

## Acknowledgment

If Jeva contributes to your work, please cite our work:

```
@article{li2026jeva,
  title={Jeva: System One Models from First Principles},
  author={Li, Qi and Wang, Xinchao},
  journal={arXiv preprint arXiv:2610.18773},
  year={2026}
}
```
