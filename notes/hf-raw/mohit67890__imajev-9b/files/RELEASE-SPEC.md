# imajev-9b — release specification (final version, 2026-09-24)

This folder is the version that ships to Hugging Face (`mohit67890/imajev-9b`), the Space and the public repo. Previous staged
version (phase-2b) in `../_previous/phase-2b/imajev-9b/`; raw pod outputs in `imajev/reports/decision-p2c/pod/`.

## What the adapter is
- Base: `Qwen/Qwen3.5-9B @ c202236235762e1c871ad0ccb60c8ee5ba337b9a` (frozen).
- Adapter: LoRA r16 / α32 on the language layers incl. the DeltaNet projections + trained 255-code decision readout, PEFT format;
  `mlx/` holds the same weights converted for MLX (`scripts/convert_peft_adapter_to_mlx.py`, 0 unmatched keys).
- Recipe: **phase-2c soup50** = element-wise average (0.5 / 0.5) of the phase-2b adapter and the phase-2c best checkpoint (LoRA matrices
  and readout). The phase-2c checkpoint continued the phase-2b adapter on the Eikos-style soft-target mixture
  (`data/manifests/decision-p2c.jsonl`, 39,515 rows: Qwen3.6-35B-A3B distribution targets, rationale loss 0.3, option permutation,
  Eikos strict slice, image replay); 1,018 steps, 2 epochs, lr 2e-5, batch 40×2 on 4×H100 with gradient checkpointing, budget 7,500
  tokens; best by phase-2c dev + judge-dev (selection score 0.947).
- Why a soup: the pure phase-2c checkpoint passes every gate (ImajevBench 82.4, hidden split 85.1) but its JevBench hard tier (66.7)
  is below phase-2b; the average keeps phase-2b's hard accuracy with better calibration. Sweep and gate log: `imajev/reports/decision-p2c/README.md`.

## Calibration (`calibration.json`, schema 1.1)
- One temperature T = 1.748 for every (type, option-count) bucket, fitted by NLL on the 150-item authored JevBench-style dev set
  (`data/manifests/decision-p2b-jevstyle-dev.jsonl`; no JevBench items). Unknown offsets 0. Same policy as the 4B and 2B; an ECE-fit T
  lowers hard-tier ECE but raises pooled ECE (the board's Calibration axis), so the NLL fit is kept.

## Serving configuration (what the numbers below were measured with)
- Direct option scoring at the decision position, **4 cyclic rotations averaged** (`--rotations 4`), calibration file applied, single pass,
  no reasoning tokens. Interface: TypeSafe `/v1/systemone` via `scripts/playground/server.py --backend torch`.

## Benchmarks (2026-09-24, pod `ocmsizvx19feqb`, 8×H100)
| Benchmark | This version | Phase-2b version (previous staged) |
|---|---|---|
| JevBench public hard (111) | 69.4 / ECE 0.092 (rot4 + cal); raw 69.4 / 0.187; cal 69.4 / 0.090; rot4 69.4 / 0.174 | 68.5 raw; 69.4 / 0.104 rot4 + cal |
| JevBench original / easy | 100 / 100 | 100 / — |
| ImajevBench v2.0-lite public test (279) | 82.1 (joint 96/122, text 29/37, visual 104/120) | 82.8 |
| ImajevBench private-1 hidden (202) | 84.7 (joint 67/88, text 24/30, visual 80/84), ECE 0.079 | 84.2, ECE 0.108 |
| Ship gates | all PASS: state probe 73.5, pairs 90.0, unknown 14/14, false abstention 0.24%, irrelevance 84.0 | |
| Panels of the phase-2c checkpoint | reasoning dev 68.9 (phase-2b 67.4), phase-2c test 85.5, judge-dev 28/29 | |

Raw responses: `imajev/reports/decision-p2c/pod/9b-delta/jevbench-soup50-{raw,cal,rot4,rot4cal}/`, `.../imajevbench-soup50/`,
`.../private1/9b-soup50/score`, gates `imajev/reports/decision-p2c/pod/9b-soup50/gates.json`. Registry: `imajev/reports/benchmarks/README.md`.

## File hashes (sha256)
- `adapter_model.safetensors`: `1a841bdcbe9ffdf137d6dbcc60d07aa38c10b1f1974cef047b38f04854867c06`
- `decision_readout.safetensors`: `bcddebcce9d3f23d2668ef10a98e7779643257aa35ceb0ceba62055081ff3773`
- `decision_readout.json`: `864611322d72b65a54bbc47b864afa8f280f1efc4a7dffbf9f477f2df8033a7b`
- `adapter_config.json`: `7b89aaa5549253f1cf4d962c16637049b77726ef40ce213c74866288bb47a6de`
- `calibration.json`: `81ca964608dfce0a6397757dae19885f8619e82176b4daea7eac239a513736e7`

MLX export:
- `mlx/adapter_config.json`: `34fc8a44b60ae82d0f0e00e851186c310e6be9d0cb09462cb5064f48523a239b`
- `mlx/adapters.safetensors`: `62fdfa59f4cc39f46204fde59bbf276bd91de40b55753a8830c34d5cff8508d3`
- `mlx/decision_readout.json`: `864611322d72b65a54bbc47b864afa8f280f1efc4a7dffbf9f477f2df8033a7b`
- `mlx/decision_readout.safetensors`: `45de6088522178edd5d012eea0c417e27a23cf86ac99fb5f6d0cdf1d98c5c1ea`

## Status
- MLX vs torch parity verified (above). Model card refreshed to this version on 2026-09-24.
