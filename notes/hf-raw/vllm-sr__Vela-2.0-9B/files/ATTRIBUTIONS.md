# Source credits

- [vllm-sr/Decision-2.0-Lux-9B](https://huggingface.co/vllm-sr/Decision-2.0-Lux-9B) at `f3122c7c8abd302326c22220aac4095eb1799f37`: direct weight origin of the backbone and decision head (Apache-2.0, `LICENSE`). Its own training data reach this model through its weights; see its model card.
- Qwen3.5-9B (Qwen team): upstream text backbone and tokenizer of Decision-2.0-Lux-9B (Apache-2.0, `LICENSES/Qwen3.5-9B-LICENSE.txt`).
- Decision 2.0 training rows used as replay with a KL term to Decision-2.0-Lux-9B's own probabilities (7.5 % of stage-1 steps); not redistributed.
- Training data of this fine-tune (not redistributed here; each keeps its own licence, some CC BY-SA share-alike): listed with their licences in the model card, section Training (router recipe, router span head, broad span head).
