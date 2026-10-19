# Upstream attribution and modifications

Copyright 2026 Alibaba Cloud.

Open-Jev-27B-v1.1 uses the text backbone of **Qwen/Qwen3.8-27B**, pinned to
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

- [Pinned upstream model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md)
- [Pinned upstream license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE)

The included `LICENSE` preserves those exact Apache License 2.0 bytes and
the Alibaba Cloud copyright notice. SHA-256:
`bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a`.
The pinned model card and Hugging Face revision metadata separately identify
Apache-2.0. The inspected upstream file inventory contains no `NOTICE` file.

**Modifications by Open-Jev:** `package/checkpoint/adapter/` contains trained
LoRA weights for the Qwen text backbone. `package/checkpoint/head.pt` is a
trained scalar decision head whose lineage starts from the upstream
Yes-minus-No output-embedding readout. `model.json` selects the pinned base
and the Open-Jev decision interface; `temperature.json` stores calibration
fitted on a separate split. These files are modified training products and
are not unmodified Qwen weights. Their training-stage and initialization
provenance is recorded in `package/provenance.json`.

The inference weight package is Apache-2.0. Original Open-Jev source code is
MIT; its separate license is included as `LICENSE-CODE`. Base weights,
tokenizer files, third-party runtime libraries and training dataset rows
are not bundled here.

The pinned Qwen model card requests citation of: Qwen Team,
*Qwen3.8-Max: A New Bar for Coding and Cowork*, August 2026,
<https://qwen.ai/blog?id=qwen3.8>. This citation title comes from that model
card and does not identify this 27B checkpoint as the hosted Qwen3.8-Max model.

No endorsement by Alibaba Cloud, the Qwen Team or TypeSafe is implied.
