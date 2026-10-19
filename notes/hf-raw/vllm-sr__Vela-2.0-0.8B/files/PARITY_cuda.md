# Vela 2.0 decoder export parity (GPU): stage2/ckpt-003000

Export `vela2_0.8B_nb_20261003`; reference = the research decoder scorer (dec_scorer_d3_20261001.D3Scorer: render_row(train=False), tree forward, DEV2.0 head, span head v2, windowed c), device cuda. Rows: choice:domain_dev 15, choice:factcheck_dev 10, choice:modality_dev 10, choice:feedback_dev 10, choice:oasst1 8, choice:boolq 8, choice:snli 8, choice:arc 6, choice:massive 6, score:helpsteer2 10, score:jsts 8, score:klue_sts 6, set:shield 40, span:pii_dev 30, span:pii_longdev_4096 4, span:pii_longdev_8192 3, span:halu 20, span:toxic 10, span:fewnerd 6, span:multiconer 6, span:tydiqa_goldp 6 = 230 real dev rows.

Probabilities are the calibrated ones the module returns (choice / score softmax over the options at T_type, set sigmoid at T_set, span sigmoid per word x label at T_span). Decisions: argmax (choice / score), set options above the calibrated threshold, decoded span sets (label, start, end) at the calibrated threshold (PII: length rule + sparse gate).

Arms: `ref_bf16` = research scorer, bf16 autocast (the evaluation setting); `ref_fp32_r` = research scorer in FP32 with its backbone Linear / conv1d weights rounded to bf16 (= the stored release weights); `ref_fp32` = research scorer in FP32, unrounded weights; `exp_*` = this export (vela2_inference + modeling_vela2).

## Rendering

| rows | sequences (incl. windows) | mismatches | parts ids | blocks | word positions / offsets | window count |
|---|---|---|---|---|---|---|
| 230 | 260 | 0 | 0 | 0 | 0 | 0 |

## Outputs

| comparison | type | n | max abs prob diff | max abs logit diff | argmax / decision diffs |
|---|---|---|---|---|---|
| ref_bf16_vs_exp_bf16 | choice | 169 | 0 | 0 | 0 |
| ref_bf16_vs_exp_bf16 | score | 101 | 0 | 0 | 0 |
| ref_bf16_vs_exp_bf16 | set | 26 | 0 | 0 | 0 |
| ref_bf16_vs_exp_bf16 | span (36360 word units) | 85 | 0 | 0 | 0 span-set diffs (495 spans), 0 unit-offset mismatches, PII gate fired on 0 rows |
| ref_fp32_r_vs_exp_fp32 | choice | 169 | 0 | 0 | 0 |
| ref_fp32_r_vs_exp_fp32 | score | 101 | 0 | 0 | 0 |
| ref_fp32_r_vs_exp_fp32 | set | 26 | 0 | 0 | 0 |
| ref_fp32_r_vs_exp_fp32 | span (36360 word units) | 85 | 0 | 0 | 0 span-set diffs (495 spans), 0 unit-offset mismatches, PII gate fired on 0 rows |
| ref_fp32_vs_exp_fp32 | choice | 169 | 0.00312 | 0.0382 | 0 |
| ref_fp32_vs_exp_fp32 | score | 101 | 0.00213 | 0.0213 | 0 |
| ref_fp32_vs_exp_fp32 | set | 26 | 0.00379 | 0.0271 | 0 |
| ref_fp32_vs_exp_fp32 | span (36360 word units) | 85 | 0.0332 | 0.168 | 5 span-set diffs (495 spans), 0 unit-offset mismatches, PII gate fired on 0 rows |

Decoder cross-check (export decoder vs `span_decode_r4` on the same reference probabilities): 0 row(s) differ.

## Bundled vs single (export, FP32)

60 multi-question rows (choice:oasst1 8, choice:massive 6, score:helpsteer2 10, set:shield 26, constructed:domain+4signals+pii 10): every question run alone (its own sequence) vs bundled. Choice / score / set: 251 questions, 0 bit-identical, max |logit diff| 0.0034. Span: 10 rows, 0 bit-identical, max |logit diff| 0.0283. Noise-floor control (the same bundled rows, one row per batch instead of batched with others): max |logit diff| 0.00472 (span 0.0251); decision flips bundled vs single: 0. Differences of this size come from floating-point batch shapes (padding, GEMM sizes), not from one question block seeing another.

## SystemOne vs predict()

6 SystemOne requests (choice / noul / score; span + choice over a 3-field JSON state; set): max |diff| between the SystemOne answers and the predict() probabilities they map from 0, 0 decision mismatches. HTTP route function (`system_one_request`, what `vela2_serve.py` calls): ok=True, other model name rejected=True.

## Arm run times

ref_bf16 36.9 s, ref_fp32_r 54.3 s, ref_fp32 9.6 s, exp_bf16 5.8 s, exp_fp32 9.7 s

## Notes

* Bar: rendering identical; with `ref_bf16` vs `exp_bf16` (same kernels, same weights as autocast sees them, same batches) the outputs should be identical up to kernel nondeterminism; `ref_fp32_r` vs `exp_fp32` checks the FP32 path on identical weights.
* `ref_fp32` vs `exp_fp32` is what storing the backbone matmul weights in bf16 changes when running in FP32 (no effect under bf16 autocast).
* Choice / score probabilities are over the real options (the convention the temperatures were fitted on); [ABS] is reported separately (abstain_probability). The research scorer slices [ABS] off, so its calibration is unfitted.
* Generic span threshold `span:*` = 0.5 and the GLiNER2-style default question texts for open-label classify / extract are export defaults, not dev-fitted.
