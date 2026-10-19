# Modifications

This repository is derived from `vllm-sr/Decision-1.0-Kai-0.6B` at revision `7185f514` and `vllm-sr/Vela-1.0-Encoder-307M` at revision `fe9ccc074b78`.

- **Encoder weights.** The encoder was initialised from Kai's Choice-path trunk, extracted into the `Vela-1.0-Encoder-307M` layout, and then trained further (101,000 steps) as the Vela 2.0 schema encoder. Kai's Noul and Score paths and its decision heads are not included.
- **New parameters.** Eight marker-token embedding rows (ids 256,000 to 256,007), the choice / set / score readout (projections, MLP, temperature) and the word-level span readout were added and trained. They are defined in `modeling_vela2.py`.
- **Tokenizer files.** `tokenizer.json`, `tokenizer_config.json` and `special_tokens_map.json` are the Vela base tokenizer with eight added special tokens: `[Q]`, `[O]`, `[ABS]`, `[SEP_SCHEMA]`, `[SEG_user]`, `[SEG_answer]`, `[E]`, `[SEG_context]` (ids 256,000 to 256,007). The vocabulary, merges, normalizer, pre-tokenizer, post-processor and decoder are unchanged. `vela2_inference.py` removes the eight added tokens before encoding text and inserts the markers by id.
- **Licence files.** `LICENSE`, `NOTICE`, `NOTICE_SOURCES.json`, `DISTRIBUTION_TERMS.md`, `LICENSING_STATUS.md` and `LICENSES/` are Kai's, passed on unchanged. They describe Kai; they apply here to the material inherited from Kai and, through it, from Vela, mmBERT and the Gemma-origin tokenizer.
