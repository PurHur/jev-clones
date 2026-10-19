---
license: cc-by-nc-4.0
---

# Nimble-v4-max on the Decision Index

A full public-suite Decision Index 0.3 run of [Nimble-v4-max](https://huggingface.co/bespokelabs/Nimble-v4-max): 68.46.

- `runs/nimble-v4-max/` holds the results and scores in the layout the Decision Index kit uploads.
- `code/nimble/` is the engine that produced them. With the kit ([apolinario/decision-index](https://github.com/apolinario/decision-index)) installed:

```
hf download bespokelabs/Nimble-v4-max --local-dir Nimble-v4-max
PYTHONPATH=code python -m decision_index run --engine nimble.evaluation.decision_index_engine:NimbleEngine \
    --option adapter=Nimble-v4-max --option max_tails=48 --rows <suite rows> --out <run dir>
```

It needs a CUDA GPU with BF16, torch 2.8, transformers 5.17.0, accelerate, flash-linear-attention 0.5.2 and causal-conv1d 1.7.0. The model needs about 55 GB in BF16 and runs on one RTX PRO 6000. `shared_prefix` is on by default: a request's shared part runs once, then each question's short tail. Requests longer than the model's 32,768-token context are refused as unsupported, and nothing is truncated.

The engine checks the model's `schema_config.json` before it runs: if the prompt files or the option-code tokens differ from the ones the model was trained with, it refuses to load.

Released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), like the model.

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
