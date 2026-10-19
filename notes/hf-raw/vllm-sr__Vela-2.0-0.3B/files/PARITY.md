# Vela 2.0 export parity: R4b_s44 (v2 PII calibration)

This export (checkpoint R4b_s44, PII calibration v2); reference = the research scorer (r4 word units and r4 span decoder), fp32 (bf16 autocast off), on the GPU. Rows: choice:domain_dev 15, choice:factcheck_dev 10, choice:modality_dev 10, choice:feedback_dev 10, choice:oasst1 8, choice:boolq 8, choice:snli 8, choice:arc 6, choice:massive 6, score:helpsteer2 10, score:jsts 8, score:klue_sts 6, set:shield 40, span:pii_dev 30, span:pii_longdev_4096 4, span:pii_longdev_8192 3, span:halu 20, span:toxic 10, span:fewnerd 6, span:multiconer 6, span:tydiqa_goldp 6 = 230 real dev rows (4K PII rows = one 4,094-token part; 8K PII rows are windowed, 2 windows each).

Probabilities are the calibrated ones the module returns (choice / score softmax over the options at T_type, set sigmoid at T_set, span sigmoid per word x label at T_span). Decisions: argmax (choice / score), set options above the calibrated threshold, decoded span sets (label, start, end) at the calibrated threshold.

## Assembly

| rows | sequences | token-id mismatches | marker / pool position mismatches | word-unit mismatches |
|---|---|---|---|---|
| 230 | 233 | 0 | 0 | 0 |

## Outputs

| comparison | type | n | max abs prob diff | max abs logit diff | argmax / decision diffs |
|---|---|---|---|---|---|
| ref_vs_exp_torch_gpu | choice | 169 | 0.0000 | 0.0000 | 0 |
| ref_vs_exp_torch_gpu | score | 101 | 0.0000 | 0.0000 | 0 |
| ref_vs_exp_torch_gpu | set | 26 | 0.0000 | 0.0000 | 0 |
| ref_vs_exp_torch_gpu | span (36360 word units) | 85 | 1.060e-06 | 9.060e-06 | 0 span-set diffs (306 spans), 0 unit-offset mismatches, PII gate fired on 6 rows |
| ref_vs_exp_torch_cpu | choice | 169 | 2.331e-06 | 5.817e-05 | 0 |
| ref_vs_exp_torch_cpu | score | 101 | 1.283e-06 | 7.629e-06 | 0 |
| ref_vs_exp_torch_cpu | set | 26 | 2.253e-06 | 1.621e-05 | 0 |
| ref_vs_exp_torch_cpu | span (36360 word units) | 85 | 2.007e-05 | 7.620e-04 | 0 span-set diffs (306 spans), 0 unit-offset mismatches, PII gate fired on 6 rows |
| ref_vs_exp_onnx_cpu | choice | 169 | 2.154e-06 | 4.339e-05 | 0 |
| ref_vs_exp_onnx_cpu | score | 101 | 1.167e-06 | 1.431e-05 | 0 |
| ref_vs_exp_onnx_cpu | set | 26 | 2.069e-06 | 1.812e-05 | 0 |
| ref_vs_exp_onnx_cpu | span (36360 word units) | 85 | 2.708e-05 | 8.402e-04 | 0 span-set diffs (306 spans), 0 unit-offset mismatches, PII gate fired on 6 rows |
| ref_vs_exp_onnx_fp16_cpu | choice | 169 | 0.0060 | 0.0435 | 0 |
| ref_vs_exp_onnx_fp16_cpu | score | 101 | 0.0016 | 0.0268 | 0 |
| ref_vs_exp_onnx_fp16_cpu | set | 26 | 0.0060 | 0.0269 | 0 |
| ref_vs_exp_onnx_fp16_cpu | span (36360 word units) | 85 | 0.0497 | 0.3825 | 0 span-set diffs (306 spans), 0 unit-offset mismatches, PII gate fired on 6 rows |
| exp_torch_cpu_vs_exp_onnx_cpu | choice | 169 | 1.544e-06 | 2.098e-05 | 0 |
| exp_torch_cpu_vs_exp_onnx_cpu | score | 101 | 9.275e-07 | 1.431e-05 | 0 |
| exp_torch_cpu_vs_exp_onnx_cpu | set | 26 | 1.295e-06 | 1.621e-05 | 0 |
| exp_torch_cpu_vs_exp_onnx_cpu | span (36360 word units) | 85 | 9.537e-06 | 1.073e-04 | 0 span-set diffs (306 spans), 0 unit-offset mismatches, PII gate fired on 6 rows |

Decoder cross-check (export decoder vs original `span_decode_r4` on the same reference probabilities): 0 row(s) differ.

[ABS] probability, torch CPU vs ONNX CPU: max abs diff 4.486e-07 (the reference scorer does not expose [ABS]).

Arm run times: exp_torch_gpu 2.4 s, exp_torch_cpu 128.1 s, exp_onnx_cpu 63.0 s, exp_onnx_fp16_cpu 104.1 s

## Dev / test metrics re-scored through the export (torch, GPU)

| metric | n | threshold | eval (stored) | export bf16 | export fp32 |
|---|---|---|---|---|---|
| pii_dev_short (f1) | 888 | [0.1, 0.10124993574690892, 0.1117664050611105, 0.11898058065298875, 0.1242313865036034, 0.142542446831692, 0.15739884677257684, 0.1750180542386916, 0.18509570254007415, 0.221951325957207, 0.2452177088448381, 0.2728377294947921, 0.34654603000376644, 0.37877663950613477, 0.3968088648197345, 0.4163138536553291, 0.43746543383688835, 0.46046470810296286, 0.48554547047416285, 0.5129809408070596, 0.5430922019561052, 0.5762588561210075, 0.612932602015978, 0.6536546968898178, 0.6990786445040931, 0.75] | 0.9906 | 0.9894 | 0.9888 |
| pii_dev_4096 (f1) | 300 | [0.0007499000000000002, 0.1] | 0.9669 | 0.9726 | 0.9734 |
| pii_dev_8192 (f1) | 300 | [0.00031620000000000026, 0.1] | 0.9239 | 0.9391 | 0.9384 |
| pii_test_short (f1) | 1704 | [0.1, 0.10525098439904285, 0.10735110950631876, 0.11898058065298875, 0.1359660192824094, 0.13918971984824613, 0.14603153759065693, 0.15739884677257684, 0.17993971347102147, 0.18509570254007415, 0.19617568619126172, 0.221951325957207, 0.2452177088448381, 0.2538878383268606, 0.3060417963825519, 0.34654603000376644, 0.3968088648197345, 0.4163138536553291, 0.43746543383688835, 0.46046470810296286, 0.48554547047416285, 0.5129809408070596, 0.5430922019561052, 0.5762588561210075, 0.612932602015978, 0.6536546968898178, 0.6990786445040931, 0.75] | 0.9953 | 0.9953 | 0.9947 |
| halu_dev_carve (example_f1) | 2874 | [0.55] | 0.8898 | 0.8898 | 0.8897 |
| set:p_cats (micro_f1) | 575 | 0.3 | 0.5592 | 0.5592 | 0.5586 |
| set:r_cats (micro_f1) | 152 | 0.5 | 0.7397 | 0.7397 | 0.7397 |
| domain:test (accuracy) | 2573 |  | 0.8232 | 0.8232 | 0.8243 |
| domain:test (macro_f1) | 2573 | | 0.8246 | 0.8246 | 0.8256 |
| feedback:test (accuracy) | 2438 |  | 0.9914 | 0.9914 | 0.9918 |
| feedback:test (macro_f1) | 2438 | | 0.9888 | 0.9888 | 0.9893 |
| modality:test (accuracy) | 1320 |  | 0.9652 | 0.9652 | 0.9652 |
| modality:test (macro_f1) | 1320 | | 0.9608 | 0.9608 | 0.9608 |
| factcheck:test (accuracy) | 720 |  | 0.9972 | 0.9972 | 0.9972 |
| factcheck:test (macro_f1) | 720 | | 0.9972 | 0.9972 | 0.9972 |

Stored numbers: the research scorer's evaluation outputs for this checkpoint (PII dev per bucket, PII test short at the fine threshold, halu dev; set:p_cats / set:r_cats threshold metric on the Shield dev-carve sample; Tier-0 choice tests). The research scorer ran bf16 autocast on GPU; the bf16 column is the like-for-like re-score.

## PII calibration: v1 (pii_buckets) vs v2 (pii_length_rule + pii_sparse_gate)

Export, torch GPU, bf16 autocast (the eval setting); one forward per set, decoded under each calibration. Exact (label, start, end) micro F1. packed = packed-synthetic docs; sparse = needle-stress + negative-stress docs. Tier-0 short has neither kind (single sentences), so only `all`. Test is report-only (never read by the fit). Thresholds are the export's continuous values (no grid snapping).

Variants: `v1_buckets` anchors pii_buckets; `v2_length` anchors [(21, 0.75), (126, 0.05), (254, 0.03504), (510, 0.01205), (1022, 0.005919), (2046, 0.002037), (3070, 0.001427), (4094, 0.0007499), (8190, 0.0003162)]; `v2_length_gate` anchors [(21, 0.75), (126, 0.05), (254, 0.03504), (510, 0.01205), (1022, 0.005919), (2046, 0.002037), (3070, 0.001427), (4094, 0.0007499), (8190, 0.0003162)], gate K=3 T=0.1; `floor1e-3_length` anchors [(21, 0.75), (126, 0.05), (254, 0.03504), (510, 0.01205), (1022, 0.005919), (2046, 0.002037), (3070, 0.001427), (4094, 0.001), (8190, 0.001)]; `floor1e-3_length_gate` anchors [(21, 0.75), (126, 0.05), (254, 0.03504), (510, 0.01205), (1022, 0.005919), (2046, 0.002037), (3070, 0.001427), (4094, 0.001), (8190, 0.001)], gate K=3 T=0.1

| set | docs | v1_buckets | v2_length | v2_length_gate | floor1e-3_length | floor1e-3_length_gate | gate fired (v2) |
|---|---|---|---|---|---|---|---|
| dev:short all | 888 | 0.9906 | 0.9894 | 0.9894 | 0.9894 | 0.9894 | 19 |
| dev:short sparse | 48 | no gold, fp 0 | no gold, fp 0 | no gold, fp 0 | no gold, fp 0 | no gold, fp 0 | 19 |
| dev:128 all | 150 | 0.9530 | 0.9838 | 0.9817 | 0.9838 | 0.9817 | 120 |
| dev:128 packed | 30 | 0.9487 | 0.9778 | 0.9778 | 0.9778 | 0.9778 | 0 |
| dev:128 sparse | 120 | 0.9605 | 0.9944 | 0.9888 | 0.9944 | 0.9888 | 120 |
| dev:256 all | 150 | 0.9670 | 0.9856 | 0.9843 | 0.9856 | 0.9843 | 120 |
| dev:256 packed | 30 | 0.9736 | 0.9861 | 0.9861 | 0.9861 | 0.9861 | 0 |
| dev:256 sparse | 120 | 0.9432 | 0.9836 | 0.9778 | 0.9836 | 0.9778 | 120 |
| dev:512 all | 150 | 0.9579 | 0.9921 | 0.9907 | 0.9921 | 0.9907 | 120 |
| dev:512 packed | 30 | 0.9575 | 0.9955 | 0.9955 | 0.9955 | 0.9955 | 0 |
| dev:512 sparse | 120 | 0.9609 | 0.9677 | 0.9560 | 0.9677 | 0.9560 | 120 |
| dev:1024 all | 150 | 0.9245 | 0.9875 | 0.9909 | 0.9875 | 0.9909 | 120 |
| dev:1024 packed | 30 | 0.9228 | 0.9944 | 0.9944 | 0.9944 | 0.9944 | 0 |
| dev:1024 sparse | 120 | 0.9497 | 0.8923 | 0.9399 | 0.8923 | 0.9399 | 120 |
| dev:2048 all | 150 | 0.9799 | 0.9811 | 0.9839 | 0.9811 | 0.9839 | 120 |
| dev:2048 packed | 30 | 0.9860 | 0.9857 | 0.9857 | 0.9857 | 0.9857 | 0 |
| dev:2048 sparse | 120 | 0.8263 | 0.8585 | 0.9290 | 0.8585 | 0.9290 | 120 |
| dev:3072 all | 150 | 0.9693 | 0.9710 | 0.9746 | 0.9710 | 0.9746 | 120 |
| dev:3072 packed | 30 | 0.9747 | 0.9753 | 0.9753 | 0.9753 | 0.9753 | 0 |
| dev:3072 sparse | 120 | 0.7733 | 0.8056 | 0.9392 | 0.8056 | 0.9392 | 120 |
| dev:4096 all | 300 | 0.9669 | 0.9679 | 0.9726 | 0.9669 | 0.9711 | 240 |
| dev:4096 packed | 60 | 0.9720 | 0.9735 | 0.9735 | 0.9720 | 0.9720 | 0 |
| dev:4096 sparse | 240 | 0.7184 | 0.7013 | 0.9155 | 0.7184 | 0.9155 | 240 |
| dev:8192 all | 300 | 0.9239 | 0.9346 | 0.9391 | 0.9239 | 0.9258 | 240 |
| dev:8192 packed | 60 | 0.9258 | 0.9392 | 0.9392 | 0.9258 | 0.9258 | 0 |
| dev:8192 sparse | 240 | 0.7411 | 0.5678 | 0.9235 | 0.7411 | 0.9235 | 240 |
| test:short all | 1704 | 0.9953 | 0.9953 | 0.9953 | 0.9953 | 0.9953 | 57 |
| test:short sparse | 96 | no gold, fp 0 | no gold, fp 0 | no gold, fp 0 | no gold, fp 0 | no gold, fp 0 | 57 |
| test:4096 all | 30 | 0.9870 | 0.9884 | 0.9912 | 0.9870 | 0.9898 | 24 |
| test:4096 packed | 6 | 0.9896 | 0.9910 | 0.9910 | 0.9896 | 0.9896 | 0 |
| test:4096 sparse | 24 | 0.8421 | 0.8421 | 1.0000 | 0.8421 | 1.0000 | 24 |
| test:8192 all | 30 | 0.8944 | 0.9268 | 0.9291 | 0.8944 | 0.8958 | 24 |
| test:8192 packed | 6 | 0.8949 | 0.9285 | 0.9285 | 0.8949 | 0.8949 | 0 |
| test:8192 sparse | 24 | 0.8421 | 0.7500 | 1.0000 | 0.8421 | 1.0000 | 24 |

## Notes and open issues

* fp32 bar (max abs prob diff < 1e-4, identical decisions) applies to the fp32 arms; the fp16 ONNX arm (encoder fp16, heads fp32) is reported for information.
* The stored eval numbers were produced with bf16 autocast on GPU; the export reproduces them exactly in the same setting. In fp32 the same dev metrics move by up to ~1e-3 (table above); the thresholds were picked on the bf16 scores.
* PII thresholds (v2): `pii_length_rule` (log thr piecewise-linear in log n_tokens between per-bucket dev picks at 21, 126, 254, 510, 1,022, 2,046, 3,070, 4,094 and 8,190 tokens; grid floor 1e-4) then `pii_sparse_gate` (a doc with <= K spans at 0.5 is decoded at max(rule, T)); both fitted on dev only (full report in `pii_calibration_fit.json`). `pii_buckets` is kept only as the fallback when those keys are absent. Mid-length (128..3,072) evidence is dev-only: the PII test sets exist only at short / 4,094 / 8,190 tokens. The span parity checks above use the full v2 rule.
* Choice / score probabilities are over the real options (the convention the temperatures were fitted on); [ABS] is reported as a separate abstain probability. The research scorer never exposed [ABS], so its calibration is unfitted.
* Generic span threshold `span:*` = 0.5 and the GLiNER2-style default question texts for open-label classify / extract are export defaults, not dev-fitted.

