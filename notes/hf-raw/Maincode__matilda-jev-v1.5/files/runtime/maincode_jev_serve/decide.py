"""One decision request -> probability distributions (same scoring as the kev benchmark engine).

Every question of a request is scored against the same state in token-budgeted, length-sorted
batches, one forward pass per batch. Nothing is truncated: a request that does not fit (too long,
too many options, or out of GPU memory) raises CapacityError, which the server maps to HTTP 413.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch

from maincode_jev_serve.model import MAX_OPTIONS, DecisionModel, options
from maincode_jev_serve.types import DecisionInput, ImageInput


class CapacityError(ValueError):
    pass


def length_estimate(row: Mapping[str, Any]) -> int:
    """Cheap prompt-length estimate used only to order and pack batches (same as kev.train)."""
    measured = row.get("source", {}).get("input_tokens")
    if isinstance(measured, int) and not isinstance(measured, bool):
        return measured + 16
    return len(json.dumps([row["state"], row["question"]], ensure_ascii=False)) // 3 + 192 + 512 * len(row.get("images", []))


def microbatches(rows: Sequence[dict[str, Any]], batch_size: int, token_budget: int) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []
    longest = 0
    for row in rows:
        length = length_estimate(row)
        if pending and (len(pending) == batch_size or max(longest, length) * (len(pending) + 1) > token_budget):
            result.append(pending)
            pending, longest = [], 0
        pending.append(row)
        longest = max(longest, length)
    if pending:
        result.append(pending)
    return result


def decide(model: DecisionModel, state: object, questions: Mapping[str, Mapping[str, Any]], *, temperature: float,
           max_tokens: int, token_budget: int, batch_size: int, images: Sequence[ImageInput] = ()) -> tuple[dict[str, list[float]], int]:
    """Normalized probabilities per question key (in option order) and the input tokens used."""
    rows: list[dict[str, Any]] = []
    for key, question in questions.items():
        count = len(options(cast(Any, question))[0])
        if count > MAX_OPTIONS:
            raise CapacityError(f"at most {MAX_OPTIONS} options per choice question are supported ({count} given)")
        row: dict[str, Any] = {"state": state, "question": dict(question), "id": key, "source": {}}
        if images:
            row["images"] = list(images)
        rows.append(row)
    distributions: dict[str, list[float]] = {}
    input_tokens = 0
    for batch in microbatches(sorted(rows, key=length_estimate), batch_size, token_budget):
        try:
            prepared = model.prepare(cast(list[DecisionInput], batch), max_length=max_tokens)
        except ValueError as error:
            if "token limit" in str(error):
                raise CapacityError(f"request exceeds the maximum context length of {max_tokens} tokens") from error
            raise
        input_tokens += prepared.input_tokens
        try:
            with torch.inference_mode():
                probabilities = (model(prepared) / temperature).softmax(-1).float().cpu().tolist()
        except torch.OutOfMemoryError as error:
            del prepared
            torch.cuda.empty_cache()
            raise CapacityError("request exceeds the maximum context length that fits in GPU memory") from error
        for item, values, count in zip(batch, probabilities, prepared.counts, strict=True):
            total = sum(values[:count])
            distributions[item["id"]] = [value / total for value in values[:count]]
    return {key: distributions[key] for key in questions}, input_tokens
