"""Build strands-decider-2B ONNX files from onnx-community/Qwen3.5-2B-ONNX-OPT's text graphs.

strands-decider-2B-hobson-v19 is Qwen3.5-2B-Base with a rank-16 LoRA adapter and a
pointer head on the final hidden states. The base graphs were exported from Qwen3.5-2B
(instruct), which has the same architecture and parameter names, so we keep the graphs
and put the decider's weights into them:

- weights: Qwen3.5-2B-Base at the revision the adapter was trained on, with the LoRA
  merged in (W + alpha/r * B @ A), mapped onto each initializer by name with the same
  transform as in OneJev's value-checked mapping (transpose, 1 + weight for zero-centred
  RMSNorm), plus the small linear-attention tensors that mapping skipped
  (A_neg_exp = -exp(A_log), dt_bias, the gated norm's weight).
- graph: the lm_head is removed; the final norm's output, after the num_logits_to_keep
  slice, becomes the `last_hidden_state` output. The decider never reads logits.
- q4f16: the same, re-quantized with ONNX Runtime's 4-bit block-32 kernel.
- the pointer head goes to head.json for the browser (2 x 256 x 2048 fp32 + norm).
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from huggingface_hub import snapshot_download
from onnx import helper, numpy_helper
from onnxruntime.capi._pybind_state import quantize_matmul_4bits, quantize_matmul_8bits
from safetensors import safe_open

BASE = Path(os.environ.get("BASE_DIR", "base_onnx"))
OUT = Path(os.environ.get("OUT", "out"))
# Value-checked map from the Qwen3.5 OPT graph initializers to HF parameters (made for
# OneJev-0.8B by matching values; same names in the 2B graphs).
ONEJEV_MAPPING = Path(os.environ.get("MAPPING", "qwen3_5_opt_mapping.json"))
DECIDER = Path(snapshot_download("StrandsAgents/strands-decider-2B-hobson-v19", ignore_patterns=["eval/*", "training/*"]))
TORSO = Path(snapshot_download("Qwen/Qwen3.5-2B-Base", revision="b1485b2fa6dfa1287294f269f5fb618e03d52d7c"))
PREFIX = "model.language_model."


def decider_params():
    """Base text weights in fp32 with the LoRA adapter merged."""
    params = {}
    for f in sorted(TORSO.glob("*.safetensors")):
        with safe_open(str(f), "pt") as st:
            for k in st.keys():
                if k.startswith(PREFIX):
                    params[k] = st.get_tensor(k).to(torch.float32)
    cfg = json.loads((DECIDER / "lora/adapter_config.json").read_text())
    assert not cfg.get("use_dora") and not cfg.get("use_rslora")
    scale = cfg["lora_alpha"] / cfg["r"]
    merged = 0
    with safe_open(str(DECIDER / "lora/adapter_model.safetensors"), "pt") as st:
        for k in st.keys():
            if ".lora_A." not in k:
                continue
            target = PREFIX + k.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
            a = st.get_tensor(k).to(torch.float32)
            b = st.get_tensor(k.replace(".lora_A.", ".lora_B.")).to(torch.float32)
            params[target] += scale * (b @ a)
            merged += 1
    print("merged", merged, "LoRA pairs, scale", scale, file=sys.stderr)
    return {k: v.numpy() for k, v in params.items()}


def build_mapping(graph_model):
    """OneJev's decoder mapping by name (same names for 0.8B and 2B), with this graph's
    shapes, the lm_head dropped and the small linear-attention tensors added."""
    onejev = json.loads(ONEJEV_MAPPING.read_text())["mapping"]["decoder_model_merged"]
    spec = {k: dict(v) for k, v in onejev.items() if not k.startswith("lm_head")}
    for init in graph_model.graph.initializer:
        m = re.fullmatch(r"model\.layers\.(\d+)\.gdn\.(A_neg_exp|dt_bias|norm\.weight)", init.name)
        if m:
            param = f"{PREFIX}layers.{m[1]}.linear_attn." + {"A_neg_exp": "A_log", "dt_bias": "dt_bias", "norm.weight": "norm.weight"}[m[2]]
            spec[init.name] = {"param": param, "transform": "neg_exp" if m[2] == "A_neg_exp" else "identity"}
    return spec


def transformed(params, spec, shape):
    p = params[spec["param"]]
    how = spec["transform"]
    if how == "transpose":
        p = p.T
    elif how == "plus_one":
        p = p + 1.0
    elif how == "neg_exp":
        p = -np.exp(p)
    return np.ascontiguousarray(p.reshape(shape))


def quant4(w):
    """[K, N] float -> MatMulNBits (packed [N, K/32, 16], fp16 scales [N, K/32], zp [N, K/64])."""
    K, N = w.shape
    kb = (K + 31) // 32
    packed = np.zeros((N, kb, 16), "uint8")
    zp = np.zeros((N, (kb + 1) // 2), "uint8")
    scales = np.zeros((N, kb), np.float16)
    quantize_matmul_4bits(packed, np.ascontiguousarray(w.astype(np.float16)), scales, zp, 32, N, K, False)
    return packed, scales, zp


def quant8(w):
    """[K, N] float -> 8-bit MatMulNBits (packed [N, K/32, 32], fp16 scales [N, K/32], zp [N, K/32])."""
    K, N = w.shape
    kb = (K + 31) // 32
    packed = np.zeros((N, kb, 32), "uint8")
    zp = np.zeros((N, kb), "uint8")
    scales = np.zeros((N, kb), np.float16)
    quantize_matmul_8bits(packed, np.ascontiguousarray(w.astype(np.float16)), scales, zp, 32, N, K, False)
    return packed, scales, zp


BITS = int(os.environ.get("BITS", "4"))
ONLY_QUANT = os.environ.get("ONLY_QUANT") == "1"
QNAME = os.environ.get("QNAME", "q4f16")


def set_init(init, array):
    init.CopyFrom(numpy_helper.from_array(array.astype(numpy_helper.to_array(init).dtype, copy=False), init.name))


def expose_hidden_state(model):
    """Replace the lm_head and the `logits` output with `last_hidden_state`."""
    g = model.graph
    head = [n for n in g.node if n.name.startswith("/lm_head/MatMul")]
    assert len(head) == 1, [n.name for n in head]
    sliced = head[0].input[0]
    g.node.remove(head[0])
    g.node.append(helper.make_node("Identity", [sliced], ["last_hidden_state"], name="/last_hidden_state"))
    logits = next(o for o in g.output if o.name == "logits")
    elem = logits.type.tensor_type.elem_type
    idx = list(g.output).index(logits)
    g.output.remove(logits)
    g.output.insert(idx, helper.make_tensor_value_info("last_hidden_state", elem, ["batch_size", "sequence_length", 2048]))
    used = {i for n in g.node for i in n.input}
    for init in [i for i in g.initializer if i.name not in used]:
        g.initializer.remove(init)


def check_against_base(model, spec, params):
    """Each swapped weight should still look like the instruct weight it replaces:
    instruct and base differ by fine-tuning, so cosine similarity stays high, and a
    wrong transform or mismatched parameter shows up as ~0."""
    worst = []
    for init in model.graph.initializer:
        if init.name in spec:
            old = numpy_helper.to_array(init).astype(np.float32).reshape(-1)
            new = transformed(params, spec[init.name], list(init.dims)).reshape(-1)
            cos = float(old @ new / (np.linalg.norm(old) * np.linalg.norm(new) + 1e-12))
            worst.append((cos, init.name))
    worst.sort()
    print("lowest cosine to the instruct weights:", [(round(c, 3), n) for c, n in worst[:5]], file=sys.stderr)
    assert worst[0][0] > 0.5, worst[:5]


def save(model, name):
    path = OUT / "onnx" / f"{name}.onnx"
    for stale in path.parent.glob(f"{name}.onnx_data*"):
        stale.unlink()
    onnx.save(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
              location=f"{name}.onnx_data", size_threshold=1024)
    print("wrote", path, file=sys.stderr)


def main():
    (OUT / "onnx").mkdir(parents=True, exist_ok=True)
    params = decider_params()
    embed = PREFIX + "embed_tokens.weight"

    # fp32 decoder
    model = onnx.load(str(BASE / "onnx/decoder_model_merged.onnx"), load_external_data=not ONLY_QUANT)
    spec = build_mapping(model)
    missing = [i.name for i in model.graph.initializer if i.name not in spec and (i.external_data or len(i.raw_data) >= 1024) and not i.name.startswith("lm_head")]
    assert not missing, missing
    if ONLY_QUANT:
        model = None
    if model is not None:
        expose_hidden_state(model)
        check_against_base(model, spec, params)
        done = 0
        for init in model.graph.initializer:
            if init.name in spec:
                set_init(init, transformed(params, spec[init.name], list(init.dims)))
                done += 1
        print("decoder fp32: replaced", done, file=sys.stderr)
        save(model, "decoder_model_merged")
        del model

    # q4f16 decoder: re-quantize MatMulNBits weights, swap the rest
    model = onnx.load(str(BASE / "onnx/decoder_model_merged_q4f16.onnx"), load_external_data=True)
    expose_hidden_state(model)
    by_flat = {name.replace(".", "_"): name for name in spec}
    inits = {i.name: i for i in model.graph.initializer}
    done = 0
    for name, init in inits.items():
        if name.endswith("_quant"):
            stem = name[: -len("_quant")]
            s = spec[by_flat[stem]]
            assert s["transform"] == "transpose", (stem, s)  # every MatMulNBits weight is a Linear's W.T [K, N]
            w = transformed(params, s, list(params[s["param"]].T.shape))
            if BITS == 8:
                packed, scales, zp = quant8(w)
                for i, arr in ((init, packed), (inits[stem + "_scales"], scales), (inits[stem + "_zp"], zp)):
                    i.CopyFrom(numpy_helper.from_array(arr, i.name))
            else:
                packed, scales, zp = quant4(w)
                set_init(init, packed.reshape(list(init.dims)))
                set_init(inits[stem + "_scales"], scales.reshape(list(inits[stem + "_scales"].dims)))
                set_init(inits[stem + "_zp"], zp.reshape(list(inits[stem + "_zp"].dims)))
            done += 1
        elif name in spec:
            set_init(init, transformed(params, spec[name], list(init.dims)))
            done += 1
    if BITS == 8:
        for n in model.graph.node:
            if n.op_type == "MatMulNBits":
                for a in n.attribute:
                    if a.name == "bits":
                        a.i = 8
    print(f"decoder {QNAME}: replaced", done, file=sys.stderr)
    save(model, f"decoder_model_merged_{QNAME}")
    del model
    if ONLY_QUANT:
        return

    # embeddings: fp32 identity, q4f16 GatherBlockQuantized (blocks along each row)
    model = onnx.load(str(BASE / "onnx/embed_tokens.onnx"), load_external_data=True)
    (init,) = [i for i in model.graph.initializer if i.name == "model.embed_tokens.weight"]
    set_init(init, params[embed])
    save(model, "embed_tokens")
    model = onnx.load(str(BASE / "onnx/embed_tokens_q4f16.onnx"), load_external_data=True)
    inits = {i.name: i for i in model.graph.initializer}
    (stem,) = [n[: -len("_quant")] for n in inits if n.endswith("_quant")]
    packed, scales, zp = quant4(params[embed].T)
    set_init(inits[stem + "_quant"], packed.reshape(list(inits[stem + "_quant"].dims)))
    set_init(inits[stem + "_scales"], scales.reshape(list(inits[stem + "_scales"].dims)))
    set_init(inits[stem + "_zp"], zp.reshape(list(inits[stem + "_zp"].dims)))
    save(model, "embed_tokens_q4f16")

    # pointer head + calibration, as plain JSON for the browser
    cfg = json.loads((DECIDER / "hobson_config.json").read_text())
    with safe_open(str(DECIDER / "head.safetensors"), "pt") as st:
        head = {k: st.get_tensor(k).to(torch.float32).numpy() for k in st.keys()}
    np.save(OUT / "head.npy", np.concatenate([head[k].reshape(-1) for k in ("norm.weight", "norm.bias", "q.weight", "q.bias", "k.weight", "k.bias")]))
    (OUT / "decider_config.json").write_text(json.dumps({
        "hidden_size": 2048, "pointer_dim": cfg["pointer_dim"], "layer_norm_eps": 1e-5,
        "head_layout": ["norm.weight[2048]", "norm.bias[2048]", "q.weight[256x2048]", "q.bias[256]", "k.weight[256x2048]", "k.bias[256]"],
        "temperature": cfg["temperature"], "temperature_by_kind": cfg["temperature_by_kind"],
        "ordinal_smoothing": cfg["ordinal_smoothing"], "max_length": cfg["max_length"],
    }, indent=2) + "\n")
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
