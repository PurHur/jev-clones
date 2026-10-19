# Source credits

- [vllm-sr/Decision-2.0-Nox-4B](https://huggingface.co/vllm-sr/Decision-2.0-Nox-4B) at `d55528d1635fc474061ec59e31a7c722d3e7ab95`: direct weight origin of the backbone and decision head (Apache-2.0, `LICENSE`). Its own training data reach this model through its weights; see its model card.
- Qwen3.5-4B-Base (Qwen team): upstream text backbone and tokenizer of Decision-2.0-Nox-4B (Apache-2.0, `LICENSES/Qwen3.5-4B-Base-LICENSE.txt`).
- Decision 2.0 training rows used as replay with a KL term to Decision-2.0-Nox-4B's own probabilities (7.5 % of stage-1 steps); not redistributed.
- Training data of this fine-tune (not redistributed here; each keeps its own licence, some CC BY-SA share-alike): listed with their licences in the model card, section Training (router recipe, router span head, broad span head).
