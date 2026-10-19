---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
library_name: granite-library
base_model:
- ibm-granite/granite-4.0-micro
- ibm-granite/granite-4.1-3b
- ibm-granite/granite-4.1-8b
- ibm-granite/granite-4.1-30b
tags:
- granite
- uncertainty
- context-attribution
- requirement-checking
- decision
- classification
- lora
- alora
- peft
---

# Granite Core Library

The Granite Core Library includes four families of LoRA adapters, each developed for a specific task that enhances the base model's capabilities for explainability, calibration, constraint verification, and decision making. We provide adapters for: 
 * [ibm-granite/granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro)
 * [ibm-granite/granite-4.1-3b](https://huggingface.co/ibm-granite/granite-4.1-3b)
 * [ibm-granite/granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b)
 * [ibm-granite/granite-4.1-30b](https://huggingface.co/ibm-granite/granite-4.1-30b)

We give a brief overview of the functionality of each capability below; details can be found in each individual adapter readme.

## Capabilities implemented as LoRA adapters

The four capabilities that have been implemented as LoRA adapters and made available in this HF repository are:

**Context Attribution:** A LoRA adapter that predicts which sentences in the context (including documents and previous conversation turns) were most important for the base model to generate each sentence in its response. The adapter provides contributive attribution, identifying the context sentences that actually influenced the model's response, and ranks them by importance. This is a post-generation intrinsic designed to explain the base model's behavior. Details can be found in the [context-attribution readme](context-attribution/README.md).

**Decision:** LoRA and aLoRA adapters that turn the base model into a decision engine: given some state (a ticket, a conversation, a document, any text or JSON) and a question with a fixed set of options, the adapter picks the best option and returns a probability over all of them. The answer is read directly from the next-token logits in a single forward pass, so each decision costs one decode step and always lands inside the supplied option set. Supported question types are `choice` (best of N), `noul` (yes/no), and `score` (ordinal 0..N). Available for granite-4.1-8b and granite-4.1-30b. Details can be found in the [decision readme](decision/README.md).

**Requirement Check:** A LoRA adapter that judges whether an assistant's generation satisfies a set of user-specified requirements, outputting a binary yes/no assessment. It is designed for evaluating constraint satisfaction in instruction-following scenarios, including formatting, content, structural requirements, and quality assurance pipelines. Details can be found in the [requirement-check readme](requirement-check/README.md).

**Uncertainty:** A LoRA adapter that provides calibrated certainty scores for the base model's responses. Given a conversation ending with an assistant response, it calculates a certainty percentage reflecting how confident the model is about its answer. Certainty scores are calibrated such that among answers assigned a score of X%, approximately X% are correct. Details can be found in the [uncertainty readme](uncertainty/README.md).



## Recommended Use

The recommended way to call all adapters is through the [Mellea](https://mellea.ai) framework. For code snippets demonstrating how to use them please refer to the [Mellea adapter examples](https://github.com/generative-computing/mellea/tree/main/docs/examples/intrinsics) (note: within Mellea, adapters are referred to as "intrinsics").


## Model Signing

All adapter artifacts in this repository are signed to ensure integrity and provenance. Each adapter includes a `model.sig` signature file in its `lora/` directory that covers all artifacts in that directory (`adapter_config.json`, `adapter_model.safetensors`, `io.yaml`).

| Adapter | Signature File Path (example) | Signing Identity |
|-----------|-------------------------------|-----------------|
| Context Attribution | `context-attribution/granite-4.1-3b/lora/model.sig` | `Granite-sign@ibm.com` |
| Decision | `decision/granite-4.1-8b/lora/model.sig` | Certificate-based, see [below](#verifying-model-signatures) |
| Requirement Check | `requirement-check/granite-4.1-3b/lora/model.sig` | `Granite-sign@ibm.com` |
| Uncertainty | `uncertainty/granite-4.1-3b/lora/model.sig` | `Granite-sign@ibm.com` |

The same pattern applies to all model variants (`granite-4.0-micro`, `granite-4.1-3b`, `granite-4.1-8b`, `granite-4.1-30b`) and to `alora/` directories where available. The Decision adapters are provided for `granite-4.1-8b` and `granite-4.1-30b` only.

### Verifying Model Signatures

> **Note:** The Decision adapters use a separate, certificate-based signing process, so the commands below don't apply to them. See [Verifying the signature](decision/README.md#verifying-the-signature) in the decision readme.

To verify the integrity of a downloaded adapter, use the [model-signing](https://github.com/sigstore/model-transparency) tool:

```bash
# Install the model signing verification tool
pip install model-signing

# Verify all artifacts in an adapter's lora/ directory
model_signing verify sigstore \
  --signature <adapter>/<model-variant>/lora/model.sig \
  --identity Granite-sign@ibm.com \
  --identity_provider https://sigstore.verify.ibm.com/oauth2 \
  <adapter>/<model-variant>/lora/
```

For example, to verify the context-attribution adapter for granite-4.1-3b:

```bash
model_signing verify sigstore \
  --signature context-attribution/granite-4.1-3b/lora/model.sig \
  --identity Granite-sign@ibm.com \
  --identity_provider https://sigstore.verify.ibm.com/oauth2 \
  context-attribution/granite-4.1-3b/lora/
```

Each `model.sig` file contains a signature over all adapter artifacts in the corresponding `lora/` directory, signed with the identity `Granite-sign@ibm.com`. This allows users to confirm that the adapter has not been tampered with after release.


## Resources

- Learn about the latest updates with Granite: https://www.ibm.com/granite
- Get started with tutorials, best practices, and prompt engineering advice: https://www.ibm.com/granite/docs/
- Learn about the latest Granite learning resources: https://research.ibm.com/blog/granite-4-1-ai-foundation-models