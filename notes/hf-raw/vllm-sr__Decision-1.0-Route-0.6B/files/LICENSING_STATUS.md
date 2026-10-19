# Decision-1.0-Kai licensing

**Decision's model-weight contributions and documentation are licensed under Apache 2.0.** The repository is public and ungated: no account approval or acceptance form is required to download it.

| Material | License and attribution |
|---|---|
| Decision contributions | [Apache License 2.0](LICENSE) |
| Upstream Vela/mmBERT contributions | Retained [MIT terms](LICENSES/Upstream-MIT.txt) and author/source attribution in [NOTICE](NOTICE) |
| Historical adapted ModernBERT/Transformers runtime code (no longer bundled) | Retained [Apache 2.0 terms](LICENSES/Transformers-Apache-2.0.txt) and modification notices |
| Inherited Gemma-origin tokenizer material | Retained [tokenizer terms](LICENSES/gemma/TOKENIZER_TERMS.md), agreement, use policy and required Notice |

The Apache-2.0 project license does not replace third-party terms. In particular, the tokenizer inherited through Vela/mmBERT retains its upstream conditions; this does not describe Decision's independently trained encoder weights as Gemma weights. [Distribution terms](DISTRIBUTION_TERMS.md) preserve that component's requirements without a Hugging Face access gate.

This model-only package retains the original weight and tokenizer objects but no longer bundles executable runtime code. Upstream sources and pinned revisions remain recorded in [NOTICE_SOURCES.json](NOTICE_SOURCES.json). Training/evaluation data, Laya weights and Laya SDK source are not redistributed; separately installed dependencies retain their own licenses.
