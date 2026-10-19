---
license: cc-by-nc-4.0
---

# Bespoke-Nimble-27B-v3 on the Decision Index

A full public-suite Decision Index 0.3 run of [Bespoke-Nimble-27B-v3](https://huggingface.co/bespokelabs/Bespoke-Nimble-27B-v3): 65.33.

- `runs/bespoke-nimble-27b-v3/` holds the results and scores in the layout the Decision Index kit uploads.
- The run used the kit's own `http` engine ([apolinario/decision-index](https://github.com/apolinario/decision-index) at commit 9eb2dbe, edition 0.3, unmodified runner) against Bespoke Labs' hosted `/v1/systemone` endpoint, named in `environment.json`. The model's temperature is set on the server.
- The weights are not released. The Decision Index maintainers can get an API key for the endpoint to run the private parts and measure latency.

```
DECISION_INDEX_API_KEY=<key> python -m decision_index run --engine http \
    --option base_url=<endpoint in environment.json> --option model=nimble-27b --edition 0.3 --out <run dir>
```

Requests longer than the model's 32,768-token context are refused as unsupported, and nothing is truncated.

Released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), like the model.

## Citation

```bibtex
@misc{bespokelabs2026nimble27bv3,
  title        = {Bespoke-Nimble-27B-v3},
  author       = {Shreyas Pimpalgaonkar and Sid Devic and Mrinal Kumar and Pratik Dalal and Mahesh Sathiamoorthy},
  year         = {2026},
  publisher    = {Bespoke Labs},
  howpublished = {\url{https://huggingface.co/bespokelabs/Bespoke-Nimble-27B-v3}}
}
```
