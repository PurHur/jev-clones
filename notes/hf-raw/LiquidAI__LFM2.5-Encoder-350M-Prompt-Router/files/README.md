---
language:
- en
- de
- es
- fr
- it
- nl
- pl
- pt
- ar
- hi
- ja
- ru
- tr
- vi
- zh
tags:
- liquid
- lfm2
- lfm2.5
- bidirectional
- masked-lm
- encoder
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
pipeline_tag: text-classification
base_model:
    - LiquidAI/LFM2.5-Encoder-350M
---

<div align="center">
  <img 
    src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/2b08LKpev0DNEk6DlnWkY.png" 
    alt="Liquid AI" 
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em; margin-bottom: 1em;">
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> • 
    <a href="https://docs.liquid.ai/lfm/getting-started/welcome"><strong>Docs</strong></a> • 
    <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a> • 
    <a href="https://discord.com/invite/liquid-ai"><strong>Discord</strong></a>
  </div>
</div>

# LFM2.5-Encoder-350-Prompt-Router

A full fine-tune of [LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M) with a zero-shot routing head that scores a prompt against user-defined routing lanes in a single encoder pass.

Find more details about our encoders in our [blog post](https://www.liquid.ai/blog/lfm2-5-encoders).

> [!NOTE]
> 💻 **Demos**: Try this fine-tuned model running in a CPU-only Hugging Face space:
> **[Zero-shot prompt routing](https://huggingface.co/spaces/LiquidAI/prompt-routing)** — define your own routing lanes as free text. The model scores the whole prompt against every lane in one pass.

## Usage

> ⚠️ Loads custom code via `trust_remote_code=True` (the model wraps a `trust_remote_code` encoder).

Install the required packages:

```bash
pip install torch transformers
```

Run zero-shot prompt routing:

```python
from transformers import AutoModel, AutoTokenizer

model_id = "LiquidAI/LFM2.5-Encoder-350-Prompt-Router"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()

routes = ["Coding", "Sales", "Creative writing", "General knowledge"]
prompt = "Can you help me debug a failing Python unit test?"

print(model.route(prompt, routes, tokenizer=tokenizer))
```

## 📬 Contact

- Got questions or want to connect? [Join our Discord community](https://discord.com/invite/liquid-ai)
- If you are interested in custom solutions with edge deployment, please contact [our sales team](https://www.liquid.ai/contact).

## Citation

```bibtex
@article{liquidAI2026Encoders,
  author = {Liquid AI},
  title = {LFM2.5-Encoders: Fast at Long Context, Even on CPU},
  journal = {Liquid AI Blog},
  year = {2026},
  note = {www.liquid.ai/blog/lfm2-5-encoders},
}
```
