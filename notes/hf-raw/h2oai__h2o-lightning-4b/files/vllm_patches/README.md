# vLLM 0.30.0 source patches for serving H2O-Lightning-4B at high concurrency

`VLLM_PATCHES=1 bash serve.sh` applies both (exactly, with no fuzz) to the vLLM 0.30.0 in the active environment (see the model card, "Serving at
scale"). By hand, from the environment's `site-packages` directory: `patch -p1 -F0 < vllm030_<name>.patch`. Check
`sha256sum -c SHA256SUMS` first. They apply to vLLM 0.30.0 only.

| Patch | What it changes | Answers |
|---|---|---|
| `vllm030_scheduler_skip_same_step_mamba.patch` | `vllm/v1/core/sched/scheduler.py`, `kv_cache_manager.py`, `single_type_kv_cache_manager.py`: a waiting request that cannot reuse a recurrent-state block written in the current step is skipped for this step and re-queued at the front, instead of ending the scan of the waiting queue | scheduling only: unchanged |
| `vllm030_gdn_initial_state_no_sync.patch` | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`: the recurrent state of sequences without a cached state is zeroed with `torch.where` instead of a boolean-mask assignment, which synchronised the GPU in each linear-attention layer | identical values |
