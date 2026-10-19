#!/bin/bash
#SBATCH --job-name=27b-scratch-glmfiltered-full-ddp8
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1500G
#SBATCH --time=2-00:00:00
#SBATCH --output=/traindata/bo/autojev-27b-scratch-glmfiltered-full-20261004/train-%j.log
set -euo pipefail
root=/traindata/bo/autojev-27b-scratch-glmfiltered-full-20261004
python_bin=/traindata/bo/autojev-monorepo-import/.venv/bin/python
test_python=/traindata/bo/autojev-lora-head-20261002/env/bin/python
cd "$root/source"
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HOME=/traindata/bo/autojev-attention-ablation-20261002/hf-cache
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export AUTOJEV_EVENTS="$root/run/events.jsonl"
unset LD_LIBRARY_PATH
"$python_bin" "$root/verify_inputs.py"
AUTOJEV_TEST_DEVICE=cuda "$test_python" -m pytest -p no:cacheprovider -c /dev/null tests/test_distributed_training.py tests/test_warmup.py -q > "$root/gpu-preflight.log" 2>&1
cat "$root/gpu-preflight.log"
"$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 --module autojev.train \
  --train "$root/data/train.jsonl" --temperature "$root/data/temperature.jsonl" \
  --development "$root/data/dev.jsonl" --reference "$root/data/reference.jsonl" \
  --public "$root/data/public.jsonl" --run "$root/run" --output "$root/checkpoints" \
  --base-model Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --epochs 1 --seed 20260920 --lr 2e-6 --warmup-ratio 0.15 --weight-decay 0.01 \
  --batch-size 32 --effective-batch-size 256 --token-budget 8192 --max-length 8192 --cpu-threads 32 \
  --eval-every 50 --public-eval-every 50 --resume-every 50 \
  --attention-mode noncausal_full_attention --pooling last --brier-weight 0 --save-every 500
