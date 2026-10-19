"""Backbone (Ministral-3-8B, Mistral3ForConditionalGeneration) + LoRA + joint schema head (DESIGN.md 2.1, 3.1).

- The inner model (`.model`, Mistral3Model) is called; its `last_hidden_state` is the post-norm final state.
  `lm_head` is never called and is deleted after load (131,072 x 4,096 bf16 = 1.07e9 B).
- Extra taps: outputs of decoder layers by index, captured with forward hooks.
- LoRA: language-model linear layers only (q/k/v/o/gate/up/down), vision tower + projector frozen, alpha = 2r.
  Same leaf discovery and vision-exclusion rule as the Standard One LoRA trainer.
"""
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from head import HeadConfig, JointSchemaHead
from render import token_roles

VISION_EXCLUDE_REGEX = r".*\.(vision_tower|multi_modal_projector)\..*"


def lora_leaves(model):
    """Leaf names of the language-model attention + MLP linears (as the trainer's `attention_mlp` targets)."""
    attention, mlp = set(), set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or ".language_model." not in name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if ".self_attn." in name:
            attention.add(leaf)
        elif ".mlp." in name and leaf in {"gate_proj", "up_proj", "down_proj"}:
            mlp.add(leaf)
    if not attention:
        raise ValueError("no attention projections found")
    return sorted(attention | mlp)


def decoder_layers(model):
    lm = model.model.language_model
    return lm.layers


def tiny_config(vocab_size=131072, layers=3, hidden=64, heads=4, kv_heads=2, intermediate=128):
    """A tiny, randomly initialised Mistral3 (Ministral3 text + Pixtral vision) config for CPU smoke tests."""
    from transformers import Mistral3Config
    text = {"model_type": "ministral3", "hidden_size": hidden, "intermediate_size": intermediate,
            "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv_heads,
            "head_dim": hidden // heads, "vocab_size": vocab_size, "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-5, "hidden_act": "silu", "tie_word_embeddings": False, "sliding_window": None,
            "rope_parameters": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 16.0, "llama_4_scaling_beta": 0.1,
                                "mscale": 1.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 16384,
                                "rope_theta": 1000000.0, "rope_type": "yarn", "type": "yarn"}}
    vision = {"model_type": "pixtral", "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
              "num_attention_heads": 2, "head_dim": 16, "image_size": 1540, "patch_size": 14, "num_channels": 3,
              "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "rope_theta": 10000.0}
    return Mistral3Config(text_config=text, vision_config=vision, image_token_index=10, spatial_merge_size=2,
                          multimodal_projector_bias=False, projector_hidden_act="gelu", vision_feature_layer=-1,
                          tie_word_embeddings=False)


class SchemaModel(nn.Module):
    """Backbone (optionally LoRA-wrapped) + head. `forward_micro` runs one micro-batch of encoded requests."""

    def __init__(self, backbone, head, taps, peft_model=None):
        super().__init__()
        self.backbone = backbone          # Mistral3ForConditionalGeneration (LoRA layers injected in place)
        self.peft_model = peft_model      # the PeftModel wrapper (for save_pretrained) or None
        self.head = head
        self.taps = list(taps)
        self._captured = {}
        self._capturing = False   # hooks record only during our own forward, not during checkpoint recompute
        self._hooks = []
        layers = decoder_layers(backbone)
        for t in self.taps:
            if t == "final":
                continue
            idx = int(t)
            if not 0 <= idx < len(layers):
                raise ValueError(f"tap layer {idx} out of range (0..{len(layers) - 1})")
            self._hooks.append(layers[idx].register_forward_hook(self._make_hook(idx)))

    def _make_hook(self, idx):
        def hook(module, inputs, output):
            if self._capturing:
                self._captured[idx] = output[0] if isinstance(output, (tuple, list)) else output
        return hook

    def device(self):
        return next(self.head.parameters()).device

    def backbone_states(self, batch):
        """Run the inner model on a padded batch; returns list of [B, T, H] tensors in tap order."""
        self._captured = {}
        dev = batch["input_ids"].device
        kwargs = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "position_ids",
                                                           "pixel_values", "image_sizes")}
        use_bf16 = dev.type == "cuda"
        self._capturing = True
        try:
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = self.backbone.model(**kwargs, use_cache=False, return_dict=True)
        finally:
            self._capturing = False
        states = []
        for t in self.taps:
            states.append(out.last_hidden_state if t == "final" else self._captured[int(t)])
        self._captured = {}
        return states

    def forward_micro(self, encs, batch):
        """encs: list of encoded requests (render.Encoder.encode outputs, right-padded in `batch`).
        Returns list (per request) of lists (per question) of fp32 logit vectors (canonical option order)."""
        states = self.backbone_states(batch)
        outputs = []
        dev = states[0].device
        with torch.autocast(device_type=dev.type, enabled=False):
            for i, enc in enumerate(encs):
                n = enc["n_tokens"]
                taps = [s[i, :n] for s in states]   # bf16 on GPU; the head casts inside its checkpointed projection
                roles = torch.tensor(enc.get("_roles") or token_roles(enc), device=dev)
                outputs.append(self.head.forward_one(taps, enc, roles))
        return outputs

    def trainable_named(self):
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]


def pad_pixel_values(chunks):
    """Join per-row pixel tensors [n_i, C, H_i, W_i] into one [sum n_i, C, maxH, maxW] tensor, zero-padded at the
    bottom/right. Every row went through the processor on its own, so H/W differ between rows; this is exactly what
    PixtralImageProcessor._pad_for_batching does for a multi-image call, and the vision tower crops each image back
    to its `image_sizes` entry (patch_embeds[..., :h // patch, :w // patch]) so the padding never reaches the model."""
    h = max(int(c.shape[-2]) for c in chunks)
    w = max(int(c.shape[-1]) for c in chunks)
    return torch.cat([torch.nn.functional.pad(c, (0, w - int(c.shape[-1]), 0, h - int(c.shape[-2])))
                      for c in chunks], 0)


def make_batch(encs, pad_id, device):
    """Right-padded text batch (position ids restart per row); image micro-batches carry pixel values."""
    longest = max(e["n_tokens"] for e in encs)
    ids = torch.full((len(encs), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(encs), longest), dtype=torch.long)
    for i, e in enumerate(encs):
        ids[i, :e["n_tokens"]] = torch.tensor(e["input_ids"])
        mask[i, :e["n_tokens"]] = 1
    pos = torch.arange(longest).unsqueeze(0).expand(len(encs), -1).contiguous()
    batch = {"input_ids": ids.to(device), "attention_mask": mask.to(device), "position_ids": pos.to(device)}
    pix = [e.get("_pixel") for e in encs]
    if any(p is not None for p in pix):
        if not all(p is not None for p in pix):
            raise ValueError("image micro-batches must be homogeneous")
        batch["pixel_values"] = pad_pixel_values([torch.as_tensor(p["pixel_values"]) for p in pix]).to(device)
        batch["image_sizes"] = torch.cat([torch.as_tensor(p["image_sizes"]) for p in pix], 0).to(device)
        if device.type == "cuda":
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    return batch


def build_model(snapshot=None, tiny=None, lora_rank=64, lora_alpha=None, lora_dropout=0.0, taps=("final", 25),
                head_overrides=None, device="cpu", grad_checkpointing=True, init_adapter=None, attn="sdpa",
                seed=0):
    """Load (or create tiny) backbone, delete lm_head, freeze, add LoRA (rank 0 = none), attach the head.

    Returns (SchemaModel, info dict)."""
    import peft
    import transformers
    torch.manual_seed(seed)
    if tiny is not None:
        torch.manual_seed(1234)   # fixed: the tiny random "base" must be identical in training and serving
        cfg = tiny if not isinstance(tiny, dict) else tiny_config(**tiny)
        backbone = transformers.Mistral3ForConditionalGeneration._from_config(cfg, attn_implementation=attn)
        backbone = backbone.to(device)
    else:
        backbone = transformers.Mistral3ForConditionalGeneration.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
            attn_implementation=attn, device_map={"": device} if str(device) != "cpu" else None)
        if str(device) == "cpu":
            backbone = backbone.to(device)
    torch.manual_seed(seed)
    backbone.config.use_cache = False
    lm_head_bytes = backbone.lm_head.weight.numel() * backbone.lm_head.weight.element_size()
    del backbone.lm_head
    backbone.lm_head = nn.Identity()  # never called; keeps attribute access working
    for p in backbone.parameters():
        p.requires_grad_(False)
    if grad_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        vt = getattr(backbone.model, "vision_tower", None)
        if vt is not None and hasattr(vt, "gradient_checkpointing_disable"):
            vt.gradient_checkpointing_disable()
    leaves = lora_leaves(backbone)
    peft_model = None
    if init_adapter:
        peft_model = peft.PeftModel.from_pretrained(backbone, str(init_adapter), is_trainable=True)
    elif lora_rank:
        alpha = lora_alpha if lora_alpha is not None else 2 * lora_rank
        peft_model = peft.get_peft_model(backbone, peft.LoraConfig(
            r=lora_rank, lora_alpha=alpha, lora_dropout=lora_dropout, target_modules=leaves,
            exclude_modules=VISION_EXCLUDE_REGEX, bias="none"))
    leaked = 0
    for name, p in backbone.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            if p.requires_grad:
                leaked += 1
            p.requires_grad_(False)
        elif p.requires_grad:
            p.data = p.data.float()  # LoRA weights fp32 (as the trainer)
    text_cfg = backbone.config.text_config
    hcfg = HeadConfig(hidden_size=text_cfg.hidden_size, n_taps=len(taps), taps=list(taps),
                      **(head_overrides or {}))
    head = JointSchemaHead(hcfg).to(device=device, dtype=torch.float32)
    model = SchemaModel(backbone, head, taps, peft_model)
    info = {"lora_leaves": leaves, "lora_rank": lora_rank,
            "lora_alpha": (lora_alpha if lora_alpha is not None else 2 * lora_rank) if lora_rank else None,
            "lora_params": sum(p.numel() for n, p in backbone.named_parameters() if p.requires_grad),
            "head_params": sum(p.numel() for p in head.parameters()), "lm_head_bytes_freed": lm_head_bytes,
            "vision_lora_leaked_frozen": leaked, "taps": list(taps), "head_config": hcfg.to_dict(),
            "layers": len(decoder_layers(backbone)), "hidden_size": text_cfg.hidden_size}
    return model, info


def save_head(head, folder, extra=None):
    from safetensors.torch import save_file
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous().cpu() for k, v in head.state_dict().items()},
              str(folder / "schema_head.safetensors"))
    cfg = {"head": head.cfg.to_dict(), **(extra or {})}
    tmp = folder / "schema_head_config.json.tmp"
    tmp.write_text(json.dumps(cfg, indent=2) + "\n")
    os.replace(tmp, folder / "schema_head_config.json")


def load_head_weights(head, folder):
    from safetensors.torch import load_file
    state = load_file(str(Path(folder) / "schema_head.safetensors"))
    missing, unexpected = head.load_state_dict(state, strict=True), None
    return missing
