# Model details

Lux adapts the text backbone and tokenizer of [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a). The deployed decision model has **7,940,895,744 parameters**: 7,936,684,544 in the text backbone and 4,211,200 in the shared candidate head. The upstream vision tower, language-generation head and auxiliary prediction components are excluded.

The backbone contains 32 causal layers: 24 Gated DeltaNet layers and eight full-attention layers. Hidden width is 4,096 and feed-forward width is 12,288. Full attention uses 16 query heads and four key/value heads; Gated DeltaNet uses 16 key heads and 32 value heads with dimension 128.

For each question, the model encodes complete evidence, instructions and candidate descriptions. The decision head combines each contextual candidate-endpoint vector with the final global-query vector through a shared 256-dimensional bilinear/MLP readout. It scores all supplied candidates in one forward pass per question, with a BF16 backbone and FP32 head. Dynamic labels come from the current request.

The model uses full-parameter adaptation on a 24,000-example mixture of decision tasks and human-annotated natural-language judgments. The selected model continues hard-label cross-entropy training with the established mixture. Matched runs and their shared initialization were compared on 3,419 separate selection examples, using balanced decision and natural-reasoning task-family averages. A single positive temperature was then fitted on 1,814 independent calibration examples. Evaluation labels were not used to choose the checkpoint or fit its temperature. The alternative reading-source mixture did not win this comparison; the reported gain cannot be attributed to that replacement. [Data attribution](../ATTRIBUTIONS.md).

The exported bundle was reloaded through the default public API in a container that could not access the original training checkpoint, base model or cache. All 1,814 calibration raw-logit vectors matched the production selection/calibration runtime exactly. Mixed Choice/Noul/Score requests and rejection of complete-input overflow were also verified.

The decision benchmark reports observed regression evidence across multiple task families. Exclusion from this custom training does not establish absence from upstream pretraining. Accuracy, semantic consistency and probability calibration answer different questions and are reported separately in [evaluation methods](EVALUATION.md).
