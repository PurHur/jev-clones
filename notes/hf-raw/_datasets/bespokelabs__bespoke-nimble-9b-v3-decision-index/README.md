---
license: cc-by-nc-4.0
---

# Bespoke-Nimble-9B-v3 on the Decision Index

A full-suite Decision Index 0.2.1 run of [Bespoke-Nimble-9B-v3](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3): 56.88.

- `runs/bespoke-nimble-9b-v3/` holds the results and scores in the layout the Decision Index kit uploads.
- `code/nimble/` is the engine that produced them. With the kit ([apolinario/decision-index](https://github.com/apolinario/decision-index) at commit 87d4650) installed:

```
hf download bespokelabs/Bespoke-Nimble-9B-v3 --local-dir Bespoke-Nimble-9B-v3
PYTHONPATH=code python -m decision_index run --engine nimble.evaluation.decision_index_engine:NimbleEngine \
    --option adapter=Bespoke-Nimble-9B-v3 --rows <suite rows> --out <run dir>
```

It needs a CUDA GPU with BF16, torch 2.8, transformers 5.17.0, peft 0.21.0, flash-linear-attention 0.5.2 and causal-conv1d 1.7.0. The model needs about 20 GB in bf16. Requests too long for the model are refused as unsupported, and nothing is truncated.

Released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), like the model.

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
