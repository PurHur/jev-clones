---
license: apache-2.0
language:
- en
library_name: peft
base_model:
- ibm-granite/granite-4.1-8b
- ibm-granite/granite-4.1-30b
tags:
- granite
- decision
- classification
- lora
- alora
- peft
---

# Decision

**Decision** turns a Granite 4.1 model into a *decision engine*: given some **state** (a ticket,
a conversation, a document, any text or JSON) and a **question** with a fixed set of options, it
picks the best option and returns a probability over all of them. The answer is read directly from
the model's next-token logits in a **single forward pass**, so each decision costs
one decode step and always lands inside the option set you supply.

These are small **LoRA / aLoRA** adapters (PEFT) for **Granite-4.1-8B** and **Granite-4.1-30B**. They
attach to the frozen public checkpoints at inference time.

## Summary

| | |
|---|---|
| **Backbone** | [granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b) · [granite-4.1-30b](https://huggingface.co/ibm-granite/granite-4.1-30b) (frozen) |
| **Design** | LoRA (active over the whole prompt) and aLoRA (activated LoRA, switches on at a marker) |
| **Weights** | `adapter_model.safetensors` per variant — `r=64`, `alpha=128`, dropout `0.05` |
| **Adapted modules** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **Task** | pick one option from a caller-supplied set, with a probability per option |
| **Question types** | `choice` (best of N) · `noul` (yes/no) · `score` (ordinal 0..N) |
| **Readout** | softmax over single-token option-label logits at the first assistant position |
| **Decode cost** | one forward pass per question (`max_completion_tokens: 1`, `temperature: 0.0`) |
| **aLoRA marker** | `### DECISION` → ids `[14711, 43917, 25189]` |
| **License** | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) · IBM Research · [Granite Libraries](https://huggingface.co/collections/ibm-granite/granite-libraries) |

## How it works

To get the decision score, the adapter is asked to produce the *single label* of the chosen
option, and we read the distribution over those labels from one forward pass:

- **Prompt** — the `state`, `question`, and options as lettered lines (`A: …`, `B: …`), closed with
  `add_generation_prompt=True` so the next token *is* the label.
- **Labels** — options map to **single-token** labels (`A`–`Z`, then digits / letter-pairs for wide
  questions), one vocab id each, making the one-pass readout possible.
- **Readout** — run the prompt once, take the logits at the final position, gather only the per-option
  label logits, softmax. No sampling, no `.generate()`, no second token.
- **Answer** — the probabilities are reduced per question `type` (below). A request may carry several
  questions; each is one independent pass over the same state.

| `type`   | Options you give                        | What you get back |
|----------|-----------------------------------------|-------------------|
| `choice` | ≥ 2 named options                       | top option key + `probabilities` over all keys |
| `noul`   | none (optional false/true descriptions) | `noul` = `P(true)` + `{false, true}` probabilities |
| `score`  | 2–10 ordinal level descriptions         | expected level `Σ i·p(i)` + per-level probabilities |

**aLoRA** differs only in *where* the adapter is active: it renders the state as a first user turn
that runs on the **frozen base model** (so its KV cache can be computed once and re-used across many
questions), and switches on only at the `### DECISION` marker opening the question turn. `peft` reads
`alora_invocation_tokens` from the config and computes the activation offset automatically.

It does **not** generate free text, cannot choose outside your option set, is never merged into the
base, and is not a chat model — the base handles conversation, this adapter only decides.

## Use

```bash
pip install "transformers>=4.44" "peft>=0.18" torch   # peft >= 0.18 required for aLoRA
```

One self-contained snippet covers both flavours. Attach the adapter (never merged), render
state + question + lettered options, read the answer from one forward pass. `VARIANT="lora"` makes
the adapter active over the whole prompt, while `VARIANT="alora"` runs the state on the frozen base and
switches the adapter on at the `### DECISION` marker.

```python
import json, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE    = "ibm-granite/granite-4.1-30b"              # or granite-4.1-8b
VARIANT = "lora"                                     # "lora" or "alora"
ADAPTER = f"granite-4.1-30b/{VARIANT}"

tok   = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)    # attach; aLoRA reads alora_invocation_tokens, no merge
model.eval()

state    = "My payouts have failed three times. The bank says everything is fine. Can someone fix this?"
question = "Which team should handle this ticket?"
options  = {"payments": "Payout failures and payment processing",
            "account":  "Login and account access",
            "other":    "Something else"}

keys    = list(options)
labels  = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"][:len(keys)]        # single-token labels, one id per option
lab_ids = [tok(l, add_special_tokens=False)["input_ids"][0] for l in labels]
opts    = [f"{l}: {options[k]}" for l, k in zip(labels, keys)]
tail    = ["Question:", question, "", "Options:", *opts, "", "Reply with exactly one option label."]

if VARIANT == "alora":   # state on frozen base (turn 1), adapter activates at the marker (turn 2)
    msgs = [{"role": "user", "content": "State:\n" + state},
            {"role": "user", "content": "### DECISION\n" + "\n".join(tail)}]
else:                    # whole-prompt LoRA: one user turn
    msgs = [{"role": "user", "content": "\n".join(["State:", state, ""] + tail)}]

prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)  # next token = the label
ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)

if VARIANT == "alora":   # confirm the adapter actually engages
    inv, seq = json.load(open(f"{ADAPTER}/adapter_config.json"))["alora_invocation_tokens"], ids[0].tolist()
    assert sum(seq[i:i+len(inv)] == inv for i in range(len(seq)-len(inv)+1)) == 1, "marker must appear once"

with torch.no_grad():
    logits = model(input_ids=ids).logits[0, -1].float()               # one forward pass
probs = F.softmax(logits[torch.tensor(lab_ids)], dim=-1)              # softmax over option labels only
print({k: round(p.item(), 4) for k, p in zip(keys, probs)}, "-> choice:", keys[int(probs.argmax())])
# -> {'payments': 0.91, 'account': 0.03, 'other': 0.06} -> choice: payments
```

**Notes.** In a serving pipeline, call these through [Mellea](https://mellea.ai) as *intrinsics* driven
by the per-variant `io.yaml` ([examples](https://github.com/generative-computing/mellea/tree/main/docs/examples/intrinsics)).
Decoding is greedy and deterministic (`temperature=0.0`, `max_completion_tokens=1`).

## Results

Evaluation results on the Jev Decision Index v0.3 public set.

| Model | Knowledge | Language | Retrieval | Tools | Arts | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Decision LoRA — Granite-4.1-30B  | 34.8 | 61.3 | 64.4 | 69.3 | 29.2 | **53.30** |
| Decision aLoRA — Granite-4.1-30B | 34.3 | 54.0 | 60.8 | 70.3 | 30.7 | **50.91** |
| Decision LoRA — Granite-4.1-8B   | 28.3 | 52.1 | 60.5 | 62.7 | 16.5 | **46.00** |
| Decision aLoRA — Granite-4.1-8B  | 27.7 | 43.5 | 56.2 | 58.6 | 22.2 | **42.56** |


## Intended use & limitations

For routing/triage (`choice`), gating/guards (`noul`), scoring (`score`), and general single-label
classification, anywhere a *distribution over a known set of choices* is needed rather than text
generation. The adapter selects only among the options it is given; decision quality depends on the
clarity of the state, question, and option descriptions.

## Files

```
decision/
├── granite-4.1-8b/   { lora, alora }/  → adapter_config.json · adapter_model.safetensors · io.yaml · model.sig
└── granite-4.1-30b/  { lora, alora }/  → adapter_config.json · adapter_model.safetensors · io.yaml · model.sig
```

- `adapter_config.json` — PEFT config (rank, alpha, target modules, and aLoRA `alora_invocation_tokens`)
- `adapter_model.safetensors` — the trained LoRA weights (never merged)
- `io.yaml` — Mellea intrinsic spec (name, parameters, prompt-layout notes)
- `model.sig` — detached signature over the three files above (see [Verifying the signature](#verifying-the-signature))

Inference code for both flavours is in [Use](#use) above; the directory ships the adapters, their
configs, and a per-variant signature.

## Verifying the signature

Each variant ships a detached `model.sig` ([`model_signing`](https://github.com/sigstore/model-transparency),
chaining to the public **DigiCert Trusted Root G4** CA). To check a variant is authentic and unmodified:

```bash
pip install model-signing==1.1.1
curl -O https://cacerts.digicert.com/DigiCertTrustedRootG4.crt.pem   # public root; not shipped here

model_signing verify certificate \
  --signature granite-4.1-30b/lora/model.sig \
  --log_fingerprints \
  --certificate_chain DigiCertTrustedRootG4.crt.pem \
  granite-4.1-30b/lora
```

Expected output — `Verification succeeded` is per-variant (it confirms that folder's files), while the
three SHA-256 fingerprints are **identical for all four variants** (the signer; only the signed content
differs). Confirm both match before trusting the model:

```text
[  init  ] sha256 Fingerprint: 55:2F:7B:DC:F1:A7:AF:9E:6C:E6:72:01:7F:4F:12:AB:F7:72:40:C7:8E:76:1A:C2:03:D1:D9:D2:0A:C8:99:88
[ verify ] sha256 Fingerprint: 46:01:1E:DE:1C:14:7E:B2:BC:73:1A:53:9B:7C:04:7B:7E:E9:3E:48:B9:D3:C3:BA:71:0C:E1:32:BB:DF:AC:6B
[ verify ] sha256 Fingerprint: F2:9F:3F:0F:7E:95:41:60:14:69:D8:E6:44:9C:93:54:43:A0:BA:91:89:57:EE:49:CA:FC:64:AD:6E:1D:96:02
Verification succeeded
```

- `[ init ]` — the trust anchor you supplied, **DigiCert Trusted Root G4** (matches
  `openssl x509 -in DigiCertTrustedRootG4.crt.pem -noout -fingerprint -sha256`).
- `[ verify ]` ×2 — the IBM signing certificate chain carried in the signature; matching these pins
  the signature to the expected signer (`Verification succeeded` alone only proves the chain reaches
  the public root).

Repeat for each variant: `granite-4.1-8b/{lora,alora}`, `granite-4.1-30b/{lora,alora}`.

## License

Adapters: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). Base-model use is additionally
governed by the Granite model terms (see the base cards linked above).
