"""In-process engine for the Decision Index kit (`--engine maincode_jev_serve.engine:MaincodeJevEngine`).

Optional: needs the `decision-index` package installed alongside. Same scoring as the server.
Example: python -m decision_index run --engine maincode_jev_serve.engine:MaincodeJevEngine \
             --option checkpoint=./matilda-jev-v1.5 --rows sample.jsonl.gz --out runs/x
"""

import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from decision_index.engines import Engine, Unsupported

from maincode_jev_serve.config import gpu_limits
from maincode_jev_serve.decide import CapacityError, decide
from maincode_jev_serve.model import DecisionModel


class MaincodeJevEngine(Engine):  # type: ignore[misc]
    name = "maincode-jev"
    latency = "In-process request wall time: prompt rendering, tokenization and one forward pass per question batch."

    def __init__(self, checkpoint: str, max_tokens: int | None = None, token_budget: int | None = None, batch_size: int = 64,
                 device: str | None = None, **options: Any) -> None:
        super().__init__(**options)
        self.model = DecisionModel(checkpoint=checkpoint, device=device)
        auto_context, auto_budget = gpu_limits()
        self.max_tokens, self.token_budget, self.batch_size = int(max_tokens or auto_context), int(token_budget or auto_budget), batch_size
        config = Path(checkpoint) / "decision_config.json"
        self.provenance = {"checkpoint": str(Path(checkpoint).resolve()), "revision": self.model.revision,
                           "checkpoint_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                           "temperature": self.model.temperature, "max_tokens": self.max_tokens}
        self.label = Path(checkpoint).resolve().name
        if not math.isfinite(self.model.temperature) or self.model.temperature <= 0:
            raise ValueError("Checkpoint temperature must be positive and finite")

    def synchronize(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def __call__(self, state: object, questions: dict[str, dict[str, Any]]) -> tuple[dict[str, object], None]:
        for question in questions.values():
            if question["type"] not in ("choice", "noul"):
                raise Unsupported(f"question type {question['type']!r}")
        try:
            distributions, input_tokens = decide(self.model, state, questions, temperature=self.model.temperature,
                                                 max_tokens=self.max_tokens, token_budget=self.token_budget, batch_size=self.batch_size)
        except CapacityError as error:
            raise Unsupported(str(error)) from error
        answers: dict[str, object] = {}
        for key, values in distributions.items():
            if questions[key]["type"] == "noul":
                answers[key] = {"type": "noul", "noul": values[1]}
            else:
                keys = list(questions[key]["criteria"])
                answers[key] = {"type": "choice", "choice": keys[max(range(len(keys)), key=values.__getitem__)],
                                "probabilities": dict(zip(keys, values, strict=True))}
        return {"model": self.label, "answers": answers, "usage": {"input_tokens": input_tokens}}, None
