---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3.8-27B
library_name: transformers
---

# Nimble-v4-max

Nimble-v4-max is a decision model made by [Bespoke Labs](https://bespokelabs.ai), a fine-tune of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Given a context and typed questions (a choice among options, a yes or no, or a score on ordered levels), it returns a probability for each option.

Decision Index 0.3, public suite: 68.46. The full run, the inference code and instructions are in [bespokelabs/nimble-v4-max-decision-index](https://huggingface.co/datasets/bespokelabs/nimble-v4-max-decision-index).

## How to run it

Use the inference code in [bespokelabs/nimble-v4-max-decision-index](https://huggingface.co/datasets/bespokelabs/nimble-v4-max-decision-index) (`code/nimble/`). It builds the prompts that `schema_config.json` pins by hash, reads each request with the attention pattern the model was trained with, and scores one token per option. Loading the weights as a plain chat model, or generating text, does not reproduce its answers.

The model needs about 55 GB in BF16, so it runs on one 80 GB or larger GPU.

## License

Nimble-v4-max is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), so it may not be used for commercial purposes. The base model, Qwen/Qwen3.8-27B, is licensed separately under Apache 2.0; its license text is in `LICENSE-Qwen3.8-27B`.

## Citation

```bibtex
@misc{bespokelabs2026nimblev4max,
  title        = {Nimble-v4-max},
  author       = {Shreyas Pimpalgaonkar and Sid Devic and Mrinal Kumar and Pratik Dalal and Mahesh Sathiamoorthy},
  year         = {2026},
  publisher    = {Bespoke Labs},
  howpublished = {\url{https://huggingface.co/bespokelabs/Nimble-v4-max}}
}
```
