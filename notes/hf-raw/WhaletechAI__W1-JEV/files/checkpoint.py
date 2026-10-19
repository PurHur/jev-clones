"""Strict model-state loading for w1-jev inference."""
import torch


def load_model_state(model, saved, model_config):
    """Strict TE -> baseline PyTorch mapping; never load optimizer/EMA as weights."""
    mapping = {}
    for key, value in saved.items():
        clean = key
        while any(clean.startswith(p) for p in ("module.", "model.", "_orig_mod.")):
            clean = clean.split(".", 1)[1]
        if clean in mapping:
            raise ValueError(f"Duplicate normalized checkpoint key: {clean}")
        mapping[clean] = value
    auxiliary = sorted(k for k in mapping if k.endswith("._extra_state"))
    for key in auxiliary:
        del mapping[key]
    rope_key = "rope.rope.inv_freq"
    rope_verified = rope_key in mapping
    if rope_verified:
        saved_freq = mapping.pop(rope_key).float()
        expected = 1.0 / (model_config["rope_theta"] **
                          (torch.arange(0, model_config["head_dim"], 2).float() / model_config["head_dim"]))
        if saved_freq.shape != expected.shape or not torch.allclose(saved_freq, expected, atol=1e-6, rtol=1e-5):
            raise ValueError("Saved RoPE frequencies disagree with training configuration")
    target = model.state_dict()
    if set(target) != set(mapping):
        raise ValueError(f"Strict checkpoint mismatch: missing={sorted(set(target)-set(mapping))}, unexpected={sorted(set(mapping)-set(target))}")
    with torch.no_grad():
        for key, tensor in target.items():
            original = mapping[key]
            if not isinstance(original, torch.Tensor) or tensor.shape != original.shape:
                raise ValueError(f"Checkpoint tensor shape mismatch: {key}")
            if not torch.isfinite(original).all():
                raise ValueError(f"Non-finite checkpoint tensor: {key}")
            tensor.copy_(original)
    return {"strict_keys": True, "loaded_tensors": len(target),
            "ignored_te_auxiliary_state": auxiliary, "saved_rope_frequency_verified": rope_verified}
