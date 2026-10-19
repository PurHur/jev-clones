---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3.5-9B
library_name: peft
---

# Bespoke-Nimble-9B-v3

Bespoke-Nimble-9B-v3 is a LoRA adapter for [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), made by [Bespoke Labs](https://bespokelabs.ai). Given a context and typed questions (a choice among options, or yes or no), it returns a probability for each option.

Decision Index 0.2.1: 56.88. The full run, the inference code and instructions are in [bespokelabs/bespoke-nimble-9b-v3-decision-index](https://huggingface.co/datasets/bespokelabs/bespoke-nimble-9b-v3-decision-index).

## License

Bespoke-Nimble-9B-v3 is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), so it may not be used for commercial purposes. The base model, Qwen/Qwen3.5-9B, is licensed separately under Apache 2.0.

## Citation

```bibtex
@misc{bespokelabs2026nimble9bv3,
  title        = {Bespoke-Nimble-9B-v3},
  author       = {Shreyas Pimpalgaonkar and Sid Devic and Mrinal Kumar and Pratik Dalal and Mahesh Sathiamoorthy},
  year         = {2026},
  publisher    = {Bespoke Labs},
  howpublished = {\url{https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3}}
}
```
