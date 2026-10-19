"""Public, inference-only loader for the mini-Jev step 10,626 adapter and decision head."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError, validate_repo_id
from peft import PeftModel, prepare_model_for_kbit_training
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _section(name: str, content: str) -> str:
    return f"<{name}>\n{content}\n</{name}>\n"


def _option_text(option: Mapping[str, Any]) -> str:
    label = option.get("label") or option.get("name")
    if not label:
        raise ValueError("each answer option needs a nonempty label")
    fields = (
        ("type", option.get("type") or option.get("option_type") or "action"),
        ("label", label),
        ("description", option.get("description")),
        ("schema", option.get("schema") or option.get("parameters_json") or option.get("parameters")),
    )
    lines = [f"{name}: {_text(value)}" for name, value in fields if value is not None and _text(value)]
    return _section("ANSWER_OPTION", "\n".join(lines))


def _encode(tokenizer, value: str) -> list[int]:
    return list(tokenizer.encode(value, add_special_tokens=False))


def encode_branch(
    tokenizer,
    state: Mapping[str, Any] | str,
    question: str,
    question_type: str,
    option: Mapping[str, Any],
    max_tokens: int = 8192,
) -> tuple[list[int], list[bool]]:
    """Serialize one option branch; preserve the question and option within the 8K budget."""
    if not 1 <= max_tokens <= 8192:
        raise ValueError("max_tokens must be 1..8192")
    if not question or not question_type:
        raise ValueError("question and question_type must be nonempty")
    if isinstance(state, str):
        state = {"summary": state}
    if not isinstance(state, Mapping):
        raise TypeError("state must be a string or a mapping")
    history = state.get("history") or []
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise TypeError("state.history must be a sequence")
    state_open = _encode(tokenizer, "<STATE>\n")
    system = _encode(tokenizer, _section("SYSTEM", _text(state.get("system"))))
    goal = _encode(tokenizer, _section("USER_GOAL", _text(state.get("user_goal"))))
    summary_open = _encode(tokenizer, "<SUMMARY>\n")
    summary_content = _encode(tokenizer, _text(state.get("summary")))
    summary_close = _encode(tokenizer, "\n</SUMMARY>\n")
    environment_open = _encode(tokenizer, "<ENVIRONMENT>\n")
    environment_content = _encode(tokenizer, _text(state.get("environment")))
    environment_close = _encode(tokenizer, "\n</ENVIRONMENT>\n")
    open_history = _encode(tokenizer, "<HISTORY>\n")
    close_history = _encode(tokenizer, "</HISTORY>\n")
    state_close = _encode(tokenizer, "</STATE>\n")
    question_ids = _encode(tokenizer, _section("QUESTION", question))
    type_ids = _encode(tokenizer, _section("QUESTION_TYPE", question_type))
    history_ids = []
    for event in history:
        if not isinstance(event, Mapping):
            raise TypeError("history events must be mappings")
        role = str(event.get("role") or "unknown")
        label = "OBSERVATION" if role.lower() in {"tool", "observation", "environment"} else "ACTION"
        content = _text(event.get("content"))
        history_ids.append(_encode(tokenizer, _section(label, f"role: {role}\n{content}")))
    option_ids = _encode(tokenizer, _option_text(option))
    fixed = sum(map(len, (
        state_open, system, goal, summary_open, summary_close, environment_open,
        environment_close, open_history, close_history, state_close, question_ids,
        type_ids, option_ids,
    )))
    if fixed > max_tokens:
        raise ValueError("protected state, question, and option exceed 8192 tokens")
    budget = max_tokens - fixed
    summary_reserve = min(len(summary_content), max(0, budget // 3))
    budget -= summary_reserve
    environment_reserve = min(len(environment_content), max(0, budget // 3))
    budget -= environment_reserve
    selected = []
    for event, item in reversed(list(zip(history, history_ids, strict=True))):
        if len(item) <= budget:
            selected.append(item)
            budget -= len(item)
        elif not selected and budget > 0:
            role = str(event.get("role") or "unknown")
            label = "OBSERVATION" if role.lower() in {"tool", "observation", "environment"} else "ACTION"
            start = _encode(tokenizer, f"<{label}>\nrole: {role}\n")
            end = _encode(tokenizer, f"\n</{label}>\n")
            available = budget - len(start) - len(end)
            if available > 0:
                payload = _encode(tokenizer, _text(event.get("content")))
                selected.append(start + payload[-available:] + end)
                budget -= len(selected[-1])
    selected.reverse()
    extra_summary = min(budget, len(summary_content) - summary_reserve)
    kept_summary = summary_content[:summary_reserve + extra_summary]
    budget -= extra_summary
    kept_environment = environment_content[:environment_reserve + budget]
    prefix = (
        state_open + system + goal + summary_open + kept_summary + summary_close
        + environment_open + kept_environment + environment_close + open_history
        + [token for item in selected for token in item] + close_history + state_close
        + question_ids + type_ids
    )
    ids = prefix + option_ids
    if len(ids) > max_tokens:
        raise AssertionError("serialization exceeded max_tokens")
    return ids, [False] * len(prefix) + [True] * len(option_ids)


class _SetBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.RMSNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(),
                                nn.Dropout(0.0), nn.Linear(4 * width, width))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attended, _ = self.attn(normed, normed, normed,
                                key_padding_mask=~valid, need_weights=False)
        x = x + attended
        x = x + self.ff(self.norm2(x))
        return x.masked_fill(~valid.unsqueeze(-1), 0)


class _SetHead(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.input = nn.Linear(width, width)
        self.blocks = nn.ModuleList([_SetBlock(width, heads) for _ in range(layers)])
        self.output = nn.Sequential(nn.RMSNorm(width), nn.Linear(width, 1))

    def forward(self, vectors: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = self.input(vectors.float()).masked_fill(~valid.unsqueeze(-1), 0)
        for block in self.blocks:
            x = block(x, valid)
        return self.output(x).squeeze(-1).masked_fill(~valid, -torch.inf)


class MiniJev:
    """Score state + question + finite options, returning a probability per option."""

    def __init__(self, release_dir: str | Path, device: str = "cuda"):
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("this 4-bit BF16 release requires a CUDA GPU")
        root = Path(release_dir)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        quant = config["quantization"]
        if (quant["format"] != "nf4" or not quant["double_quantization"]
                or quant["compute_dtype"] != "bfloat16" or config["pooler"] != "mean"):
            raise ValueError("unsupported release configuration")
        model_id = config["base_model"]
        revision = config["base_revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        base = AutoModel.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map={"": device}, attn_implementation="sdpa",
        )
        if type(base).__name__ != "Qwen3Model":
            raise TypeError("expected a Qwen3Model hidden-state backbone")
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        self.encoder = PeftModel.from_pretrained(
            base, root / config["adapter_dir"], is_trainable=False,
        ).eval()
        self.backbone = self.encoder.base_model.model
        width = int(config["projection_dim"])
        self.projection = nn.Linear(self.backbone.config.hidden_size, width).to(device)
        self.scorer = _SetHead(width, int(config["set_layers"]), int(config["set_heads"])).to(device)
        weights = load_file(root / config["head_file"], device="cpu")
        self.projection.load_state_dict({
            key.removeprefix("projection."): value for key, value in weights.items()
            if key.startswith("projection.")
        }, strict=True)
        self.scorer.load_state_dict({
            key.removeprefix("scorer."): value for key, value in weights.items()
            if key.startswith("scorer.")
        }, strict=True)
        self.projection.eval()
        self.scorer.eval()
        self.device = device
        self.max_tokens = int(config["max_tokens"])

    @classmethod
    def load(
        cls,
        release_dir: str | Path = "samatv256/mini-Jev",
        device: str = "cuda",
        *,
        revision: str | None = None,
    ) -> MiniJev:
        """Load a local release or a cached Hub namespace/repo at a revision.

        Existing local paths and Path arguments always remain local. Revision
        applies only to Hub releases. HF_HUB_OFFLINE uses the normal Hub cache.
        """
        if isinstance(release_dir, str) and "/" in release_dir and not Path(release_dir).exists():
            try:
                validate_repo_id(release_dir)
            except HFValidationError:
                pass  # Explicit or invalid local paths retain constructor behavior.
            else:
                files = [
                    "config.json",
                    "decision_head.safetensors",
                    "adapter/adapter_config.json",
                    "adapter/adapter_model.safetensors",
                ]
                root = Path(snapshot_download(
                    repo_id=release_dir, revision=revision, allow_patterns=files,
                ))
                missing = [name for name in files if not (root / name).is_file()]
                if missing:
                    raise FileNotFoundError("missing required release files: " + ", ".join(missing))
                return cls(root, device)
        return cls(release_dir, device)

    def predict(
        self,
        state: Mapping[str, Any] | str,
        question: str,
        question_type: str,
        answer_options: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if len(answer_options) < 2:
            raise ValueError("provide at least two answer options")
        labels = [str(option.get("label") or option.get("name") or "") for option in answer_options]
        if any(not label for label in labels):
            raise ValueError("every option needs a label")
        option_ids = [str(option.get("id", i)) for i, option in enumerate(answer_options)]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("option IDs must be unique")
        vectors = []
        lengths = []
        with torch.inference_mode():
            for option in answer_options:
                ids, suffix_mask = encode_branch(
                    self.tokenizer, state, question, question_type, option, self.max_tokens,
                )
                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                mask = torch.tensor([suffix_mask], dtype=torch.bool, device=self.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden = self.backbone(input_ids=input_ids, use_cache=False).last_hidden_state
                pooled = (hidden.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                vectors.append(self.projection(pooled).squeeze(0))
                lengths.append(len(ids))
            valid = torch.ones((1, len(vectors)), dtype=torch.bool, device=self.device)
            logits = self.scorer(torch.stack(vectors).unsqueeze(0), valid)
            probs = torch.softmax(logits.float(), dim=1)[0].cpu().tolist()
        winner = max(range(len(probs)), key=lambda i: probs[i])
        return {
            "selected_id": option_ids[winner],
            "selected_label": labels[winner],
            "options": [
                {"id": option_ids[i], "label": labels[i], "probability": probs[i]}
                for i in range(len(probs))
            ],
            "branch_tokens": lengths,
        }
