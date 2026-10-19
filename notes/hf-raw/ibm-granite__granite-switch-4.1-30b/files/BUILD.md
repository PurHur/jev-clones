# Granite Switch Composed Model

## Base Model

- Identifier: ibm-granite/granite-4.1-30b
- Model type: granite
- Architectures: GraniteForCausalLM
- Hidden size: 4096
- Hidden layers: 64
- Attention heads: 32
- Vocab size: 100352

## Embedded Adapters

Total adapters: **2**

| # | Name | Technology | Control Token | Token ID | Rank | Alpha | Target Modules | Source |
|---|------|------------|---------------|----------|------|-------|----------------|--------|
| 1 | guardian-core | alora | `<\|guardian-core\|>` | 100352 | 16 | 64 | k_proj, o_proj, q_proj, v_proj | ibm-granite/granitelib-guardian-r1.0 |
| 2 | decision | alora | `<\|decision\|>` | 100353 | 64 | 128 | down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj | alora |

## Composition Details

- base_param_count: 28,865,728,512
- composed_param_count: 29,989,814,272
- Param delta: +3.89%
- compose_settings:
  - include_adapters:
    - "guardian-core"
    - "decision"
  - adapter_substitute_token_ids:
    - 27
    - 14711
  - target_model: "granite-4.1-30b"
- adapter_sources:
  - "ibm-granite/granitelib-guardian-r1.0": "1a1a488a51d0d5307e30e5b3c1e66b4a5237d18a"
