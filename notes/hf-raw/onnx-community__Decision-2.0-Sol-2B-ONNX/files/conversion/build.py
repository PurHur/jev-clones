"""Build one fused ONNX graph per precision for a Qwen3.5 Decision 2.0 model.

Decision 2.0 (vllm-sr) models are full fine-tunes of Qwen3.5 text models with a
`CandidateHead` on the final hidden states. The base graphs are onnx-community's
Qwen3.5 *-ONNX-OPT text graphs, whose initializers map onto the HF parameters by
name (qwen3_5_opt_mapping.json, value-checked for OneJev and strands-decider). We
put the Decision 2.0 weights in, drop the LM head, and fuse embeddings + decoder
(empty caches and position ids built in-graph) + the head into one graph:

    input_ids [B, L], attention_mask [B, L], answer_pos [B], option_pos [B, K]
      -> logits [B, K]   (before temperature and score offsets)

The head (decision2/_vendor/dev2model/decision_model.py, CandidateHead), in fp32:
    c = LN_c(h[option]), q = LN_q(h[answer])
    logit = (key(c) . query(q)) / sqrt(256) + scalar(gelu(candidate_mlp(c) + query_mlp(q)))

Precisions: q8 (8-bit block-32 MatMulNBits decoder, 4-bit embeddings; ships as
model_quantized.onnx) and q4f16 (4-bit everywhere; model_q4f16.onnx).

    MODEL=Eos-0.8B BASE=base/qwen3_5-0.8b python build.py
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
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.capi._pybind_state import quantize_matmul_4bits, quantize_matmul_8bits
from safetensors import safe_open

MODEL = os.environ["MODEL"]
BASE = Path(os.environ["BASE"])
OUT = Path(os.environ.get("OUT", f"out/{MODEL}"))
SRC = Path(snapshot_download(f"vllm-sr/Decision-2.0-{MODEL}", ignore_patterns=["assets/*"]))
MAPPING = json.loads(Path("qwen3_5_opt_mapping.json").read_text())["mapping"]
PREFIX = "model.language_model."  # the mapping's HF names


def params():
    """Backbone weights in fp32, renamed to the mapping's HF names."""
    out = {}
    for f in sorted((SRC / "backbone").glob("*.safetensors")):
        with safe_open(str(f), "pt") as st:
            for k in st.keys():
                name = re.sub(r"^(model\.language_model\.|model\.|language_model\.)", "", k)
                out[PREFIX + name] = st.get_tensor(k).to(torch.float32).numpy()
    return out


def spec_for(graph):
    spec = {k: dict(v) for k, v in MAPPING["decoder_model_merged"].items() if not k.startswith("lm_head")}
    for init in graph.initializer:
        m = re.fullmatch(r"model\.layers\.(\d+)\.gdn\.(A_neg_exp|dt_bias|norm\.weight)", init.name)
        if m:
            param = f"{PREFIX}layers.{m[1]}.linear_attn." + {"A_neg_exp": "A_log", "dt_bias": "dt_bias", "norm.weight": "norm.weight"}[m[2]]
            spec[init.name] = {"param": param, "transform": "neg_exp" if m[2] == "A_neg_exp" else "identity"}
    return spec


def transformed(p, spec, shape):
    a = p[spec["param"]]
    t = spec["transform"]
    if t == "transpose":
        a = a.T
    elif t == "plus_one":
        a = a + 1.0
    elif t == "neg_exp":
        a = -np.exp(a)
    return np.ascontiguousarray(a.reshape(shape))


def quant(w, bits):
    """[K, N] -> MatMulNBits packed weights, fp16 scales and zero points (block 32)."""
    K, N = w.shape
    kb = (K + 31) // 32
    scales = np.zeros((N, kb), np.float16)
    w16 = np.ascontiguousarray(w.astype(np.float16))
    if bits == 8:
        packed, zp = np.zeros((N, kb, 32), "uint8"), np.zeros((N, kb), "uint8")
        quantize_matmul_8bits(packed, w16, scales, zp, 32, N, K, False)
    else:
        packed, zp = np.zeros((N, kb, 16), "uint8"), np.zeros((N, (kb + 1) // 2), "uint8")
        quantize_matmul_4bits(packed, w16, scales, zp, 32, N, K, False)
    return packed, scales, zp


def put(init, array, keep_dtype=True):
    dtype = numpy_helper.to_array(init).dtype if keep_dtype else array.dtype
    init.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(array).astype(dtype, copy=False), init.name))


def decoder(p, bits):
    model = onnx.load(str(BASE / "onnx/decoder_model_merged_q4f16.onnx"), load_external_data=True)
    spec = spec_for(model.graph)
    by_flat = {k.replace(".", "_"): k for k in spec}
    inits = {i.name: i for i in model.graph.initializer}
    done = 0
    for name, init in inits.items():
        if name.endswith("_quant"):
            stem = name[: -len("_quant")]
            if stem.startswith("lm_head"):
                continue
            s = spec[by_flat[stem]]
            assert s["transform"] == "transpose", (stem, s)
            packed, scales, zp = quant(transformed(p, s, list(p[s["param"]].T.shape)), bits)
            if bits == 8:
                for i, a in ((init, packed), (inits[stem + "_scales"], scales), (inits[stem + "_zp"], zp)):
                    put(i, a, keep_dtype=False)
            else:
                put(init, packed.reshape(list(init.dims)))
                put(inits[stem + "_scales"], scales.reshape(list(inits[stem + "_scales"].dims)))
                put(inits[stem + "_zp"], zp.reshape(list(inits[stem + "_zp"].dims)))
            done += 1
        elif name in spec:
            put(init, transformed(p, spec[name], list(init.dims)))
            done += 1
    if bits == 8:
        for n in model.graph.node:
            if n.op_type == "MatMulNBits":
                for a in n.attribute:
                    if a.name == "bits":
                        a.i = 8
    print(f"decoder {bits}-bit: replaced {done}", file=sys.stderr)
    return model


def embeddings(p):
    model = onnx.load(str(BASE / "onnx/embed_tokens_q4f16.onnx"), load_external_data=True)
    inits = {i.name: i for i in model.graph.initializer}
    (stem,) = [n[: -len("_quant")] for n in inits if n.endswith("_quant")]
    packed, scales, zp = quant(p[PREFIX + "embed_tokens.weight"].T, 4)
    for suffix, a in (("_quant", packed), ("_scales", scales), ("_zp", zp)):
        put(inits[stem + suffix], a.reshape(list(inits[stem + suffix].dims)))
    return model


def fuse(model, embed, head, hidden):
    g = model.graph
    for init in embed.graph.initializer:
        g.initializer.append(init)
    pre, post = [], []
    c = lambda name, a: g.initializer.append(numpy_helper.from_array(np.asarray(a), name))

    def node(bucket, op, ins, outs, **kw):
        bucket.append(helper.make_node(op, ins, outs, name=f"/fuse/{outs[0]}", **kw))

    # lm_head out, last_hidden_state in
    lm = [n for n in g.node if n.name.startswith("/lm_head/MatMul")]
    assert len(lm) == 1
    hidden_name = lm[0].input[0]
    g.node.remove(lm[0])

    c("fz", np.array([0], np.int64)); c("fo", np.array([1], np.int64))
    c("fz_s", np.array(0, np.int64)); c("fo_s", np.array(1, np.int64)); c("f3", np.array([3], np.int64))
    c("flast", np.array([-1], np.int64)); c("num_logits_to_keep", np.array(0, np.int64))
    node(pre, "Shape", ["input_ids"], ["ids_shape"])
    node(pre, "Gather", ["ids_shape", "fz"], ["batch"], axis=0)
    node(pre, "Gather", ["ids_shape", "fo_s"], ["length_s"], axis=0)
    node(pre, "Range", ["fz_s", "length_s", "fo_s"], ["arange"])
    node(pre, "Unsqueeze", ["arange", "fz"], ["arange_1"])
    node(pre, "Unsqueeze", ["arange_1", "fz"], ["arange_11"])
    node(pre, "Unsqueeze", ["length_s", "fz"], ["length"])
    node(pre, "Concat", ["f3", "batch", "length"], ["pos_shape"], axis=0)
    node(pre, "Expand", ["arange_11", "pos_shape"], ["position_ids"])
    drop = []
    for inp in g.input:
        if not inp.name.startswith("past_"):
            continue
        dims = inp.type.tensor_type.shape.dim
        c(f"{inp.name}_tail", np.array([0 if d.dim_param else d.dim_value for d in dims[1:]], np.int64))
        node(pre, "Concat", ["batch", f"{inp.name}_tail"], [f"{inp.name}_shape"], axis=0)
        pre.append(helper.make_node("ConstantOfShape", [f"{inp.name}_shape"], [inp.name], name=f"/fuse/{inp.name}",
                                    value=helper.make_tensor(f"{inp.name}_zero", inp.type.tensor_type.elem_type, [1], [0])))
        drop.append(inp)

    for k in ("candidate_norm.weight", "candidate_norm.bias", "query_norm.weight", "query_norm.bias", "candidate_mlp.bias"):
        c("h_" + k, head[k])
    for k in ("key", "query", "candidate_mlp", "query_mlp"):
        c(f"h_{k}_wt", head[f"{k}.weight"].T.copy())
    c("h_scalar_wt", head["scalar.weight"].T.copy())  # [256, 1]
    c("h_scale", np.array(256 ** -0.5, np.float32))
    node(post, "Cast", [hidden_name], ["h32"], to=TensorProto.FLOAT)
    node(post, "Unsqueeze", ["answer_pos", "flast"], ["q_idx"])
    node(post, "GatherND", ["h32", "q_idx"], ["q_h"], batch_dims=1)          # [B, H]
    node(post, "Unsqueeze", ["option_pos", "flast"], ["c_idx"])
    node(post, "GatherND", ["h32", "c_idx"], ["c_h"], batch_dims=1)          # [B, K, H]
    node(post, "LayerNormalization", ["c_h", "h_candidate_norm.weight", "h_candidate_norm.bias"], ["c_n"], axis=-1, epsilon=1e-5)
    node(post, "LayerNormalization", ["q_h", "h_query_norm.weight", "h_query_norm.bias"], ["q_n"], axis=-1, epsilon=1e-5)
    node(post, "MatMul", ["c_n", "h_key_wt"], ["c_key"])                      # [B, K, 256]
    node(post, "MatMul", ["q_n", "h_query_wt"], ["q_query"])                  # [B, 256]
    node(post, "Unsqueeze", ["q_query", "flast"], ["q_col"])                  # [B, 256, 1]
    node(post, "MatMul", ["c_key", "q_col"], ["bil3"])
    node(post, "Squeeze", ["bil3", "flast"], ["bil_raw"])
    node(post, "Mul", ["bil_raw", "h_scale"], ["bilinear"])
    node(post, "MatMul", ["c_n", "h_candidate_mlp_wt"], ["c_mlp0"])
    node(post, "Add", ["c_mlp0", "h_candidate_mlp.bias"], ["c_mlp"])
    node(post, "MatMul", ["q_n", "h_query_mlp_wt"], ["q_mlp"])
    node(post, "Unsqueeze", ["q_mlp", "fo"], ["q_mlp_row"])                   # [B, 1, 256]
    node(post, "Add", ["c_mlp", "q_mlp_row"], ["pre_gelu"])
    node(post, "Gelu", ["pre_gelu"], ["act"])                                 # exact (erf) GELU, opset 20+
    node(post, "MatMul", ["act", "h_scalar_wt"], ["nl3"])
    node(post, "Squeeze", ["nl3", "flast"], ["nonlinear"])
    node(post, "Add", ["bilinear", "nonlinear"], ["logits"])

    for inp in drop + [i for i in g.input if i.name in ("inputs_embeds", "position_ids", "num_logits_to_keep", "attention_mask")]:
        g.input.remove(inp)
    g.input.insert(0, helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]))
    g.input.insert(1, helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "sequence_length"]))
    g.input.append(helper.make_tensor_value_info("answer_pos", TensorProto.INT64, ["batch_size"]))
    g.input.append(helper.make_tensor_value_info("option_pos", TensorProto.INT64, ["batch_size", "num_options"]))
    while len(g.output):
        g.output.pop()
    g.output.append(helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", "num_options"]))
    nodes = list(embed.graph.node) + pre + list(g.node) + post
    del g.node[:]
    g.node.extend(nodes)
    needed, keep = {"logits"}, []
    for n in reversed(list(g.node)):
        if any(o in needed for o in n.output):
            keep.append(n)
            needed.update(i for i in n.input if i)
    del g.node[:]
    g.node.extend(reversed(keep))
    used = {i for n in g.node for i in n.input}
    for init in [i for i in g.initializer if i.name not in used]:
        g.initializer.remove(init)
    del g.value_info[:]
    if not any(o.domain in ("", "ai.onnx") and o.version >= 20 for o in model.opset_import):
        for o in model.opset_import:
            if o.domain in ("", "ai.onnx"):
                o.version = 21
    return model


def save(model, name):
    path = OUT / "onnx" / f"{name}.onnx"
    for stale in path.parent.glob(f"{name}.onnx_data*"):
        stale.unlink()
    onnx.save(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
              location=f"{name}.onnx_data", size_threshold=1024)
    print("wrote", path, file=sys.stderr)


def main():
    (OUT / "onnx").mkdir(parents=True, exist_ok=True)
    p = params()
    with safe_open(str(SRC / "decision_head.safetensors"), "pt") as st:
        head = {k: st.get_tensor(k).to(torch.float32).numpy() for k in st.keys()}
    hidden = head["key.weight"].shape[1]
    for bits, name in ((8, "model_quantized"), (4, "model_q4f16")):
        save(fuse(decoder(p, bits), embeddings(p), head, hidden), name)


if __name__ == "__main__":
    main()
