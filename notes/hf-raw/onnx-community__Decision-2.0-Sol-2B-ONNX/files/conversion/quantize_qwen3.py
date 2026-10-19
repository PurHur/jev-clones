"""8-bit (model_quantized) and 4-bit (model_q4) block-32 MatMulNBits from the fp32 export,
with the embedding Gather in 4-bit for both. Activations stay fp32."""
import os, sys
from pathlib import Path
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig

MODEL = os.environ.get("MODEL", "Kai-0.6B")
d = Path(f"out/{MODEL}/onnx")
for bits, name in ((8, "model_quantized"), (4, "model_q4")):
    m = onnx.load(str(d / "model_fp32.onnx"), load_external_data=True)
    # The embedding Gather only quantizes to 4-bit; then the MatMuls to `bits`.
    for op, b in (("Gather", 4), ("MatMul", bits)):
        cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=False, bits=b, op_types_to_quantize=(op,), quant_axes=((op, 1 if op == "Gather" else 0),))
        q = MatMulNBitsQuantizer(m, block_size=32, is_symmetric=False, bits=b, algo_config=cfg, op_types_to_quantize=(op,), quant_axes=((op, 1 if op == "Gather" else 0),))
        q.process()
        m = q.model.model
    for stale in d.glob(f"{name}.onnx_data*"):
        stale.unlink()
    onnx.save(m, str(d / f"{name}.onnx"), save_as_external_data=True, all_tensors_to_one_file=True, location=f"{name}.onnx_data", size_threshold=1024)
    print("wrote", name, file=sys.stderr)
