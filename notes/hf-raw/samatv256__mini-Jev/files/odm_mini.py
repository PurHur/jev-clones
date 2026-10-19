"""Inference-only loader for ODM Mini v1.

The released repository contains only the trained DecisionHead. The frozen
Qwen/Qwen3-0.6B backbone is downloaded separately by ``from_pretrained``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache


DEFAULT_REPO_ID = "samatv256/mini-Jev"
DEFAULT_QUESTION = "What is the best next action for the agent?"


def canonical_state(state: Any) -> str:
    if isinstance(state, str):
        return state.rstrip()
    return json.dumps(state, sort_keys=True, ensure_ascii=False)


def format_state_prefix(state: Any) -> str:
    return f"STATE:\n{canonical_state(state)}\n\n"


def format_candidate_suffix_prefix(question: str, candidate_id: str) -> str:
    question = question.strip() if question and question.strip() else DEFAULT_QUESTION
    return f"QUESTION:\n{question}\n\nCANDIDATE:\n{str(candidate_id).strip()}\n\nDESCRIPTION:\n"


def mean_pool_description(
    hidden_states: torch.Tensor, description_mask: torch.Tensor
) -> torch.Tensor:
    mask = description_mask.to(
        dtype=hidden_states.dtype, device=hidden_states.device
    ).unsqueeze(-1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class DecisionHead(nn.Module):
    """Linear(1024, 256) -> GELU -> Linear(256, 1)."""

    def __init__(self, hidden_size: int = 1024, inner_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Linear(inner_size, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden.to(self.net[0].weight.dtype)).squeeze(-1)


@dataclass(frozen=True)
class DecisionCandidate:
    id: str
    description: str

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("candidate id must not be empty")
        if not self.description.strip():
            raise ValueError("candidate description must not be empty")


CandidateLike = str | Mapping[str, Any] | DecisionCandidate


def normalize_candidates(candidates: Sequence[CandidateLike]) -> list[DecisionCandidate]:
    normalized: list[DecisionCandidate] = []
    for item in candidates:
        if isinstance(item, DecisionCandidate):
            candidate = item
        elif isinstance(item, Mapping):
            candidate = DecisionCandidate(
                id=str(item["id"]), description=str(item["description"])
            )
        elif isinstance(item, str):
            candidate = DecisionCandidate(id=item, description=item)
        else:
            raise TypeError(f"unsupported candidate type: {type(item)!r}")
        normalized.append(candidate)
    if not normalized:
        raise ValueError("candidates list must not be empty")
    ids = [candidate.id for candidate in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ids must be unique within a decision")
    return normalized


@dataclass(frozen=True)
class ChoiceResponse:
    selected: str
    probabilities: dict[str, float]
    confidence: float
    decision_margin: float
    latency_ms: float


def _resolve_file(
    model_id_or_path: str | Path,
    filename: str,
    *,
    revision: str | None,
    token: str | bool | None,
) -> Path:
    local_path = Path(model_id_or_path)
    if local_path.is_dir():
        artifact = local_path / filename
        if not artifact.is_file():
            raise FileNotFoundError(f"missing release artifact: {artifact}")
        return artifact
    return Path(
        hf_hub_download(
            repo_id=str(model_id_or_path),
            filename=filename,
            revision=revision,
            token=token,
        )
    )


def replicate_cache(source_cache: DynamicCache, repeats: int) -> DynamicCache:
    new_cache = DynamicCache()
    for layer in source_cache.layers:
        keys = layer.keys.clone() if repeats == 1 else layer.keys.repeat(repeats, 1, 1, 1)
        values = (
            layer.values.clone()
            if repeats == 1
            else layer.values.repeat(repeats, 1, 1, 1)
        )
        new_cache.update(keys, values, len(new_cache.layers))
    return new_cache


class ODMMiniModel(nn.Module):
    """Frozen Qwen3-0.6B plus the trained ODM Mini DecisionHead."""

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: Any,
        head: DecisionHead,
        *,
        hidden_size: int = 1024,
        head_inner_size: int = 256,
        max_state_length: int = 4096,
        max_suffix_length: int = 256,
        candidate_batch_size: int = 32,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.head = head
        self.hidden_size = hidden_size
        self.head_inner_size = head_inner_size
        self.max_state_length = max_state_length
        self.max_suffix_length = max_suffix_length
        self.candidate_batch_size = candidate_batch_size
        self.temperature = float(temperature)
        self.escalation_threshold = 0.85
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path = DEFAULT_REPO_ID,
        *,
        revision: str | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        token: str | bool | None = None,
        candidate_batch_size: int = 32,
    ) -> "ODMMiniModel":
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        config_path = _resolve_file(
            model_id_or_path, "config.json", revision=revision, token=token
        )
        weights_path = _resolve_file(
            model_id_or_path, "model.safetensors", revision=revision, token=token
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        base_model = str(config["base_model"])
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        backbone = AutoModel.from_pretrained(base_model, dtype=dtype)
        backbone.to(device)
        backbone.eval()

        hidden_size = int(config["hidden_size"])
        actual_hidden_size = int(getattr(backbone.config, "hidden_size", hidden_size))
        if actual_hidden_size != hidden_size:
            raise ValueError(
                f"backbone hidden size {actual_hidden_size} does not match release config {hidden_size}"
            )

        head = DecisionHead(
            hidden_size=hidden_size,
            inner_size=int(config["head_inner_size"]),
        )
        head.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
        head.to(device=device, dtype=torch.float32)
        head.eval()

        return cls(
            backbone=backbone,
            tokenizer=tokenizer,
            head=head,
            hidden_size=hidden_size,
            head_inner_size=int(config["head_inner_size"]),
            max_state_length=int(config["max_state_length"]),
            max_suffix_length=int(config["max_suffix_length"]),
            candidate_batch_size=candidate_batch_size,
            temperature=float(config["temperature"]),
        )

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @torch.inference_mode()
    def encode_state(self, state: Any) -> tuple[DynamicCache, int]:
        prefix_ids = self.tokenizer.encode(
            format_state_prefix(state),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_state_length,
        )
        if not prefix_ids:
            prefix_ids = [self.tokenizer.pad_token_id or 0]
        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        output = self.backbone(input_ids=input_ids, use_cache=True)
        return output.past_key_values, len(prefix_ids)

    @torch.inference_mode()
    def score_candidates_cached(
        self,
        past_key_values: DynamicCache,
        prefix_length: int,
        candidates: Sequence[DecisionCandidate],
        question: str = DEFAULT_QUESTION,
        candidate_batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size_limit = candidate_batch_size or self.candidate_batch_size
        all_representations: list[torch.Tensor] = []

        for start in range(0, len(candidates), batch_size_limit):
            batch = candidates[start : start + batch_size_limit]
            suffixes: list[list[int]] = []
            description_spans: list[tuple[int, int]] = []
            for candidate in batch:
                suffix_prefix = format_candidate_suffix_prefix(question, candidate.id)
                prefix_ids = self.tokenizer.encode(
                    suffix_prefix,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_suffix_length // 2,
                )
                description_ids = self.tokenizer.encode(
                    candidate.description.strip(),
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(1, self.max_suffix_length - len(prefix_ids)),
                )
                suffix = prefix_ids + description_ids
                suffixes.append(suffix)
                description_spans.append((len(prefix_ids), len(suffix)))

            max_length = max(len(suffix) for suffix in suffixes)
            pad_id = int(self.tokenizer.pad_token_id or 0)
            input_ids = torch.full(
                (len(batch), max_length), pad_id, dtype=torch.long, device=self.device
            )
            attention_mask = torch.ones(
                (len(batch), prefix_length + max_length),
                dtype=torch.long,
                device=self.device,
            )
            for index, suffix in enumerate(suffixes):
                input_ids[index, : len(suffix)] = torch.tensor(
                    suffix, dtype=torch.long, device=self.device
                )
                if len(suffix) < max_length:
                    attention_mask[index, prefix_length + len(suffix) :] = 0

            output = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=replicate_cache(past_key_values, len(batch)),
                use_cache=False,
            )
            description_mask = torch.zeros(
                (len(batch), max_length), dtype=torch.bool, device=self.device
            )
            for index, (description_start, description_end) in enumerate(description_spans):
                description_mask[index, description_start:description_end] = True
            all_representations.append(
                mean_pool_description(output.last_hidden_state, description_mask)
            )

        representations = torch.cat(all_representations, dim=0)
        return self.head(representations), representations

    @torch.inference_mode()
    def predict_choice(
        self,
        state: Any,
        candidates: Sequence[CandidateLike],
        question: str = DEFAULT_QUESTION,
        temperature: float | None = None,
    ) -> ChoiceResponse:
        started = time.perf_counter()
        normalized = normalize_candidates(candidates)
        cache, prefix_length = self.encode_state(state)
        logits, _ = self.score_candidates_cached(
            cache, prefix_length, normalized, question=question
        )
        float_logits = logits.float()
        sorted_logits, _ = torch.sort(float_logits, descending=True)
        margin = float(
            (sorted_logits[0] - sorted_logits[1]).item()
            if len(sorted_logits) > 1
            else sorted_logits[0].item()
        )
        effective_temperature = self.temperature if temperature is None else float(temperature)
        probabilities = torch.softmax(float_logits / max(effective_temperature, 1e-4), dim=-1)
        selected_index = int(probabilities.argmax().item())
        probability_map = {
            candidate.id: float(probability.item())
            for candidate, probability in zip(normalized, probabilities, strict=True)
        }
        return ChoiceResponse(
            selected=normalized[selected_index].id,
            probabilities=probability_map,
            confidence=float(probabilities[selected_index].item()),
            decision_margin=margin,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
