"""Fuse embed_tokens + decoder_model_merged + the pointer head into one graph.

    input_ids [B, L], attention_mask [B, L], answer_pos [B], option_pos [B, K]
      -> logits [B, K]   (before the per-question-type temperature)

The decider answers in one pass with no cache, so the cache inputs become in-graph
zeros sized by the batch, position_ids become arange(L) on all three rope axes, and
the cache outputs are dropped. With right padding this is exact: the full attention
and the linear attention are both causal, so a row's real tokens never see its pads.

The head is PointerHead from strands_decider/modeling.py, in fp32:
    h = LayerNorm(hidden[pos]);  logits = (k(h_options) . q(h_answer)) / sqrt(256)
One graph lets open-jev load it like Kev, through PreTrainedModel's single-session path.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(os.environ.get("OUT", "out"))
PRECISION = os.environ.get("PRECISION", "q4f16")
suffix = "" if PRECISION == "fp32" else f"_{PRECISION}"
cfg = json.loads((OUT / "decider_config.json").read_text())
H, D = cfg["hidden_size"], cfg["pointer_dim"]

embed = onnx.load(str(OUT / f"onnx/embed_tokens{os.environ.get('EMBED_SUFFIX', suffix)}.onnx"), load_external_data=True)
model = onnx.load(str(OUT / f"onnx/decoder_model_merged{suffix}.onnx"), load_external_data=True)
g = model.graph
nodes = []
c = lambda name, array: g.initializer.append(numpy_helper.from_array(np.asarray(array), name))
node = lambda op, ins, outs, **kw: nodes.append(helper.make_node(op, ins, outs, name=f"/fuse/{outs[0]}", **kw))

# 1. embeddings: the embed graph's nodes and weights, feeding inputs_embeds
for init in embed.graph.initializer:
    g.initializer.append(init)
embed_nodes = list(embed.graph.node)

# 2. shapes: B and L from input_ids
c("fuse_zero", np.array([0], np.int64))
c("fuse_one", np.array([1], np.int64))
c("fuse_zero_s", np.array(0, np.int64))
c("fuse_one_s", np.array(1, np.int64))
node("Shape", ["input_ids"], ["ids_shape"])
node("Gather", ["ids_shape", "fuse_zero"], ["batch"], axis=0)       # [1]
node("Gather", ["ids_shape", "fuse_one_s"], ["length_s"], axis=0)   # scalar

# position_ids [3, B, L] = arange(L)
c("fuse_three_b", np.array([3], np.int64))
node("Range", ["fuse_zero_s", "length_s", "fuse_one_s"], ["arange"])
node("Unsqueeze", ["arange", "fuse_zero"], ["arange_1l"])
node("Unsqueeze", ["arange_1l", "fuse_zero"], ["arange_11l"])
node("Unsqueeze", ["length_s", "fuse_zero"], ["length"])
node("Concat", ["fuse_three_b", "batch", "length"], ["pos_shape"], axis=0)
node("Expand", ["arange_11l", "pos_shape"], ["position_ids"])
c("num_logits_to_keep", np.array(0, np.int64))

# 3. empty caches: zeros of [B, *fixed dims], or [B, heads, 0, head_dim] for the KV cache
drop = []
for inp in g.input:
    if not inp.name.startswith("past_"):
        continue
    dims = inp.type.tensor_type.shape.dim
    elem = inp.type.tensor_type.elem_type
    tail = [0 if d.dim_param else d.dim_value for d in dims[1:]]
    c(f"{inp.name}_tail", np.array(tail, np.int64))
    node("Concat", ["batch", f"{inp.name}_tail"], [f"{inp.name}_shape"], axis=0)
    zero = helper.make_tensor(f"{inp.name}_zero", elem, [1], [0])
    nodes.append(helper.make_node("ConstantOfShape", [f"{inp.name}_shape"], [inp.name], name=f"/fuse/{inp.name}", value=zero))
    drop.append(inp)

# 4. pointer head on the hidden states
prelude = nodes
nodes = []
flat = np.load(OUT / "head.npy").astype(np.float32)
norm_w, norm_b, q_w, q_b, k_w, k_b = np.split(flat, np.cumsum([H, H, D * H, D, D * H, D])[:-1])
c("head_norm_w", norm_w); c("head_norm_b", norm_b)
c("head_q_wt", q_w.reshape(D, H).T.copy()); c("head_q_b", q_b)
c("head_k_wt", k_w.reshape(D, H).T.copy()); c("head_k_b", k_b)
c("head_scale", np.array(D ** -0.5, np.float32))
c("fuse_last", np.array([-1], np.int64))
node("Cast", ["last_hidden_state"], ["hidden32"], to=TensorProto.FLOAT)
node("Unsqueeze", ["answer_pos", "fuse_last"], ["answer_idx"])            # [B, 1]
node("GatherND", ["hidden32", "answer_idx"], ["answer_h"], batch_dims=1)  # [B, H]
node("Unsqueeze", ["option_pos", "fuse_last"], ["option_idx"])            # [B, K, 1]
node("GatherND", ["hidden32", "option_idx"], ["option_h"], batch_dims=1)  # [B, K, H]
node("LayerNormalization", ["answer_h", "head_norm_w", "head_norm_b"], ["answer_n"], axis=-1, epsilon=cfg["layer_norm_eps"])
node("LayerNormalization", ["option_h", "head_norm_w", "head_norm_b"], ["option_n"], axis=-1, epsilon=cfg["layer_norm_eps"])
node("MatMul", ["answer_n", "head_q_wt"], ["q_mm"]); node("Add", ["q_mm", "head_q_b"], ["q"])      # [B, D]
node("MatMul", ["option_n", "head_k_wt"], ["k_mm"]); node("Add", ["k_mm", "head_k_b"], ["k"])      # [B, K, D]
node("Unsqueeze", ["q", "fuse_last"], ["q_col"])                                                   # [B, D, 1]
node("MatMul", ["k", "q_col"], ["dot"])                                                            # [B, K, 1]
node("Squeeze", ["dot", "fuse_last"], ["dot_bk"])
node("Mul", ["dot_bk", "head_scale"], ["logits"])

# 5. rewire inputs and outputs
for inp in drop + [i for i in g.input if i.name in ("inputs_embeds", "position_ids", "num_logits_to_keep")]:
    g.input.remove(inp)
g.input.insert(0, helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]))
for i, inp in enumerate(g.input):
    if inp.name == "attention_mask":
        g.input.remove(inp)
        g.input.insert(1, helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "sequence_length"]))
        break
g.input.append(helper.make_tensor_value_info("answer_pos", TensorProto.INT64, ["batch_size"]))
g.input.append(helper.make_tensor_value_info("option_pos", TensorProto.INT64, ["batch_size", "num_options"]))
while len(g.output):
    g.output.pop()
g.output.append(helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", "num_options"]))
all_nodes = embed_nodes + prelude + list(g.node) + nodes
del g.node[:]
g.node.extend(all_nodes)

# drop nodes that no longer reach the output (the cache outputs' producers stay if shared)
needed = {"logits"}
keep = []
for n in reversed(list(g.node)):
    if any(o in needed for o in n.output):
        keep.append(n)
        needed.update(i for i in n.input if i)
keep.reverse()
del g.node[:]
g.node.extend(keep)
used = {i for n in g.node for i in n.input}
for init in [i for i in g.initializer if i.name not in used]:
    g.initializer.remove(init)
for vi in list(g.value_info):
    g.value_info.remove(vi)

name = os.environ.get("NAME", f"model{suffix}")  # q8f16 ships as model_quantized (Transformers.js "q8")
path = OUT / f"onnx/{name}.onnx"
for stale in path.parent.glob(f"{name}.onnx_data*"):
    stale.unlink()
onnx.save(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
          location=f"{name}.onnx_data", size_threshold=1024)
# (onnx.checker rejects ORT ops such as SimplifiedLayerNormalization; parity_fused.py checks it in ORT.)
print("wrote", path, len(g.node), "nodes", file=sys.stderr)
