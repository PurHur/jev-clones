"""Export a Qwen3 Decision 2.0 model (Kai-0.6B) to one ONNX graph with torch.onnx.

    input_ids [B, L], attention_mask [B, L], answer_pos [B], option_pos [B, K]
      -> logits [B, K]   (before temperature and score offsets)

Qwen3 is a plain transformer, so the backbone and CandidateHead export directly
(eager attention, causal + padding mask). Then MatMul weights are quantized to
8-bit (model_quantized) and 4-bit (model_q4f16, fp16 activations) MatMulNBits.

    MODEL=Kai-0.6B python export_qwen3.py
"""
import json
import math
import os
import sys
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import Qwen3Model

MODEL = os.environ.get("MODEL", "Kai-0.6B")
OUT = Path(os.environ.get("OUT", f"out/{MODEL}"))
SRC = Path(snapshot_download(f"vllm-sr/Decision-2.0-{MODEL}", ignore_patterns=["assets/*"]))


class Decision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Qwen3Model.from_pretrained(SRC / "backbone", dtype=torch.float32, attn_implementation="eager").eval()
        h = load_file(str(SRC / "decision_head.safetensors"))
        H = h["key.weight"].shape[1]
        self.candidate_norm = torch.nn.LayerNorm(H)
        self.query_norm = torch.nn.LayerNorm(H)
        self.key = torch.nn.Linear(H, 256, bias=False)
        self.query = torch.nn.Linear(H, 256, bias=False)
        self.candidate_mlp = torch.nn.Linear(H, 256)
        self.query_mlp = torch.nn.Linear(H, 256, bias=False)
        self.scalar = torch.nn.Linear(256, 1, bias=False)
        self.load_state_dict({k: v.float() for k, v in h.items()}, strict=False)

    def forward(self, input_ids, attention_mask, answer_pos, option_pos):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        b = torch.arange(hidden.shape[0])
        c = self.candidate_norm(hidden[b[:, None], option_pos])
        q = self.query_norm(hidden[b, answer_pos])
        bilinear = (self.key(c) * self.query(q)[:, None, :]).sum(-1) / math.sqrt(256)
        nonlinear = self.scalar(F.gelu(self.candidate_mlp(c) + self.query_mlp(q)[:, None, :])).squeeze(-1)
        return bilinear + nonlinear


def main():
    (OUT / "onnx").mkdir(parents=True, exist_ok=True)
    model = Decision().eval()
    missing = set(model.state_dict()) - {f"backbone.{k}" for k in model.backbone.state_dict()} - set(load_file(str(SRC / "decision_head.safetensors")))
    assert not missing, missing
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 0, 0]])
    mask = torch.tensor([[1] * 8, [1] * 6 + [0, 0]])
    args = (ids, mask, torch.tensor([7, 5]), torch.tensor([[2, 4, 6], [2, 4, 4]]))
    path = OUT / "onnx/model_fp32.onnx"
    torch.onnx.export(
        model, args, str(path), dynamo=False, opset_version=21,
        input_names=["input_ids", "attention_mask", "answer_pos", "option_pos"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, "attention_mask": {0: "batch_size", 1: "sequence_length"},
                      "answer_pos": {0: "batch_size"}, "option_pos": {0: "batch_size", 1: "num_options"}, "logits": {0: "batch_size", 1: "num_options"}},
    )
    m = onnx.load(str(path), load_external_data=True)
    onnx.save(m, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location="model_fp32.onnx_data", size_threshold=1024)
    print("wrote", path, file=sys.stderr)


if __name__ == "__main__":
    main()
