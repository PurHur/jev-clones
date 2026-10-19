# VideoJev-4B usage

Follow the [quick start](README.md#quick-start) to install dependencies and load `model` and `processor`.

## Task settings

Pass questions for one video per call. Use `task="grounding"` with a `query`, or `task="choice"` with a `query` and `options`; see the [input example](README.md#quick-start).

```python
# Temporal grounding
grounding_questions = [q for q in questions if q["task"] == "grounding"]
results = model.predict_video(
    processor, "/path/to/video.mp4", grounding_questions,
    fps=4.0, max_frames=2048, context=147456,
)

# Multiple-choice video QA
qa_questions = [q for q in questions if q["task"] == "choice"]
results = model.predict_video(
    processor, "/path/to/video.mp4", qa_questions,
    fps=2.0, max_frames=2048, context=262144,
)
```

Grounding defaults to threshold 0.25 and smoothing 1. Pass `threshold=0.4` to `predict_video` or `predict_features` for a per-call override; set `model.config.decoder["threshold"] = 0.4` to change the loaded model's default. QA uses frames without subtitles or audio. For evaluation with this checkpoint:

```bash
python scripts/evaluate_hf.py --checkpoint ZhuYuhan/VideoJev-4B \
  --manifest /path/to/eval.jsonl --out runs/eval
```

Use the [evaluation code and JSONL format](https://github.com/MCG-NJU/VideoJev/blob/main/docs/EVALUATION.md). QA evaluation uses a fixed 256K context.

## Output format

One dictionary per question, in input order:

- `id`, `task`: copied from the input.
- `probabilities`: grounding uses independent sigmoid probabilities in time order; QA uses softmax in option order.
- `interval`: grounding start/end in seconds.
- `answer_index`, `answer`: QA's zero-based option index and exact option text.

See the [real-video example and actual output](README.md#quick-start).

