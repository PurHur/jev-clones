"""
GemmaDecision: typed decisions on top of EmbeddingGemma, as a Hugging Face `transformers` model.

    state     ──► encoder ─┐
    question  ──► encoder ─┼─► field(state, question, type) ─┐
    option_i  ──► encoder ─┴──────────────────────────────── ┴─► logit_i

The encoder path reproduces the sentence-transformers pipeline used in training (task prompt → backbone → mean
pooling → dense projections → Matryoshka truncation → L2 normalization). The head scores the *text* of every option,
so the same model answers any schema of `choice`, `score` and `noul` questions.

Usage:

    from transformers import AutoModel

    model = AutoModel.from_pretrained("mrm8488/embeddinggemma-2-ft-typed-decision", trust_remote_code=True)
    answers = model.decide(state, questions, workflow="agent_trace_observability")
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor, nn
from transformers import AutoModel, PreTrainedModel
from transformers.utils import ModelOutput, logging

try:  # transformers >= 5: init helpers that respect the "already loaded" flags
    from transformers import initialization as init
except ImportError:  # transformers 4.x
    init = nn.init

from .configuration_gemmadecision import QUESTION_TYPES, GemmaDecisionConfig

logger = logging.get_logger(__name__)

Kind = Literal["choice", "score", "noul"]
SCORE = QUESTION_TYPES.index("score")
MAX_CONTEXT = 8192  # EmbeddingGemma 2 context window (model card), used when no max_seq_length is configured
ACTIVATIONS = {"identity": nn.Identity, "tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


# ---------------------------------------------------------------------------
# Schemas and rendering. These MUST match the training notebook byte for byte:
# any change produces embeddings the head has never seen, with no error.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuestionSpec:
    qid: str
    kind: Kind
    instructions: str
    options: tuple[str, ...]
    descriptions: tuple[str, ...]

    @property
    def kind_id(self) -> int:
        return QUESTION_TYPES.index(self.kind)


@dataclass(frozen=True)
class TypedAnswer:
    kind: Kind
    label: str
    value: str | int | bool
    confidence: float
    probabilities: dict[str, float]
    expected_level: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json(x: Any) -> Any:
    return json.loads(x) if isinstance(x, str) else x


def _is_int(x: Any) -> bool:
    return str(x).lstrip("-").isdigit()


def parse_question(qid: str, question: dict[str, Any], gold: dict[str, Any] | None = None) -> QuestionSpec:
    raw_kind = question.get("type") or (gold or {}).get("type") or "choice"
    kind: Kind = raw_kind if raw_kind in QUESTION_TYPES else "choice"
    crit = question.get("criteria")
    probs = (gold or {}).get("probabilities") or {}

    if kind == "noul":
        opts, desc = ("false", "true"), ("no", "yes")
    elif kind == "score":
        if isinstance(crit, list):
            opts = tuple(str(i) for i in range(len(crit)))
            desc = tuple(str(c) for c in crit)
        elif isinstance(crit, dict) and all(_is_int(k) for k in crit):
            keys = sorted(crit, key=int)
            opts, desc = tuple(str(k) for k in keys), tuple(str(crit[k]) for k in keys)
        elif isinstance(crit, dict):
            opts = tuple(str(i) for i in range(len(crit)))
            desc = tuple(f"{k}: {v}" for k, v in crit.items())
        else:
            opts = tuple(str(k) for k in sorted(probs, key=int))
            desc = opts
    else:
        if isinstance(crit, dict):
            opts, desc = tuple(map(str, crit)), tuple(str(v) for v in crit.values())
        elif isinstance(crit, list):
            opts = desc = tuple(map(str, crit))
        else:
            opts = desc = tuple(map(str, probs))

    if not opts:
        raise ValueError(f"question {qid!r} has no options")
    return QuestionSpec(qid, kind, str(question.get("instructions") or qid), opts, desc)


def gold_target(spec: QuestionSpec, gold: Any) -> tuple[list[float], int]:
    gold = gold if isinstance(gold, dict) else {"label": gold}
    norm = (lambda k: str(k).lower()) if spec.kind == "noul" else str
    label = norm(gold.get("label"))
    if label not in spec.options:
        raise ValueError(f"gold label {label!r} not among {spec.options}")
    probs = {norm(k): float(v) for k, v in (gold.get("probabilities") or {}).items()}
    t = [probs.get(o, 0.0) for o in spec.options]
    s = sum(t)
    t = [x / s for x in t] if s > 0 else [float(o == label) for o in spec.options]
    return t, spec.options.index(label)


def render_state(workflow: str, state: Any) -> str:
    # The state is rendered as readable "key: value" lines instead of raw JSON:
    # embedding models read "constraint violations: 1" much better than {"constraint_violations": 1}
    state = _json(state) if isinstance(state, str) and state.strip()[:1] in "{[" else state
    if not isinstance(state, dict):
        return f"workflow: {workflow} | state: {state}"
    lines = []

    def walk(prefix, value):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix} · {k}" if prefix else k, v)
        else:
            lines.append(f"{prefix.replace('_', ' ')}: {value}")

    walk("", state)
    return f"workflow: {workflow}\n" + "\n".join(lines)

def render_question(spec: QuestionSpec) -> str:
    return f"{spec.kind} question: {spec.instructions}"


def render_option(spec: QuestionSpec, i: int) -> str:
    key, desc = spec.options[i], spec.descriptions[i]
    if spec.kind == "noul":
        answer = desc
    elif spec.kind == "score":
        answer = f"level {key} of {len(spec.options) - 1}: {desc}"
    else:
        answer = key if desc == key else f"{key}: {desc}"
    return f"question: {spec.instructions} | answer: {answer}"


# ---------------------------------------------------------------------------
# Outputs and head
# ---------------------------------------------------------------------------
@dataclass
class GemmaDecisionOutput(ModelOutput):
    """
    Args:
        loss: Decision loss (soft CE + Brier + ordinal), when `labels` are given.
        logits: `(num_questions, max_options)` raw option logits; padded options are `-1e4`.
        probabilities: `(num_questions, max_options)` calibrated probabilities (per-type temperatures applied).
        text_embeds: `(num_texts, embed_dim)` normalized embeddings of the unique texts in the batch.
    """

    loss: Tensor | None = None
    logits: Tensor | None = None
    probabilities: Tensor | None = None
    text_embeds: Tensor | None = None


class GemmaDecisionHead(nn.Module):
    """Schema-conditioned scorer: logit_i = prior_i + sigmoid(gate) * (joint_scale * cos(field, option_i) + residual_i)."""

    def __init__(self, d: int, width: int = 256, dropout: float = 0.1, n_types: int = len(QUESTION_TYPES)) -> None:
        super().__init__()
        self.state_proj = nn.Linear(d, width)
        self.question_proj = nn.Linear(d, width)
        self.option_proj = nn.Linear(d, width)
        self.type_emb = nn.Embedding(n_types, width)
        self.field = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width)
        )
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual = nn.Sequential(
            nn.Linear(4 * width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1)
        )
        self.log_prior_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.log_joint_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, s: Tensor, q: Tensor, o: Tensor, kind: Tensor, mask: Tensor) -> Tensor:
        # s, q: (B, d); o: (B, K, d); kind: (B,); mask: (B, K) -> logits (B, K)
        S, Q, O = self.state_proj(s), self.question_proj(q), self.option_proj(o)
        f = self.field_norm(self.field(torch.cat([S, Q, S * Q], -1)) + self.type_emb(kind))
        On = self.option_norm(O)
        f = f.unsqueeze(1).expand_as(On)

        cap = math.log(100.0)
        joint = self.log_joint_scale.clamp(max=cap).exp() * F.cosine_similarity(f, On, dim=-1)
        residual = self.residual(torch.cat([f, On, f * On, (f - On).abs()], -1)).squeeze(-1)

        anchor = F.normalize(s + q, dim=-1).unsqueeze(-1)
        prior = self.log_prior_scale.clamp(max=cap).exp() * (F.normalize(o, dim=-1) @ anchor).squeeze(-1)

        logits = prior + torch.sigmoid(self.gate) * (joint + residual)
        return logits.masked_fill(~mask, -1e4)


def decision_loss(logits: Tensor, target: Tensor, mask: Tensor, ordinal: Tensor,
                  label_smoothing: float = 0.0, brier_weight: float = 0.25, ordinal_weight: float = 0.25) -> Tensor:
    log_p = logits.float().masked_fill(~mask, -1e4).log_softmax(-1)
    t = target
    if label_smoothing:
        uniform = mask.float() / mask.sum(-1, keepdim=True).clamp(min=1)
        t = (1 - label_smoothing) * target + label_smoothing * uniform
    loss = -(t * log_p).sum(-1).mean()
    p = log_p.exp()
    if brier_weight:
        loss = loss + brier_weight * ((p - target) ** 2 * mask).sum(-1).mean()
    if ordinal_weight and ordinal.any():
        levels = torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)
        dist = (levels[:, None] - levels[None, :]).abs()
        span = (mask.sum(-1) - 1).clamp(min=1).to(p.dtype)
        expected = torch.einsum("bi,ij,bj->b", p, dist, target) / span
        loss = loss + ordinal_weight * expected[ordinal].mean()
    return loss


def _exclude_prompt_from_mask(attention_mask: Tensor, prompt_length: int) -> Tensor:
    # Same rule as sentence-transformers' Pooling(include_prompt=False), for right or left padding
    attention_mask = attention_mask.clone()
    pad_lengths = attention_mask.to(torch.int32).argmax(dim=1)
    if pad_lengths.sum() == 0:
        attention_mask[:, :prompt_length] = 0
    else:
        positions = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0)
        attention_mask[positions < (pad_lengths + prompt_length).unsqueeze(1)] = 0
    return attention_mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GemmaDecisionPreTrainedModel(PreTrainedModel):
    config_class = GemmaDecisionConfig
    base_model_prefix = "gemma_decision"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, GemmaDecisionHead):
            init.constant_(module.log_prior_scale, math.log(20.0))
            init.constant_(module.log_joint_scale, math.log(10.0))
            init.constant_(module.gate, -2.0)
        elif isinstance(module, GemmaDecisionModel):
            init.ones_(module.temperatures)
        else:
            super()._init_weights(module)


class GemmaDecisionModel(GemmaDecisionPreTrainedModel):
    """EmbeddingGemma encoder + schema-conditioned head for typed decisions (`choice`, `score`, `noul`)."""

    def __init__(self, config: GemmaDecisionConfig) -> None:
        super().__init__(config)
        if config.encoder_config is None:
            raise ValueError("GemmaDecisionConfig.encoder_config is required to build the encoder")

        encoder_cls = getattr(transformers, config.encoder_class, None)
        if encoder_cls is None or encoder_cls is AutoModel:
            self.encoder = AutoModel.from_config(config.encoder_config)
        else:
            self.encoder = encoder_cls._from_config(config.encoder_config)

        self.projections = nn.ModuleList()
        self.activations = nn.ModuleList()
        for layer in config.projections:
            self.projections.append(nn.Linear(layer["in_features"], layer["out_features"], bias=layer.get("bias", True)))
            self.activations.append(ACTIVATIONS[layer.get("activation", "identity")]())
        # No size check against config.hidden_size here: some backbones (EmbeddingGemma 2 among them) project
        # their output internally (512 → 768), so hidden_size is not the output size. The exporter verifies
        # the real output against sentence-transformers instead.

        self.head = GemmaDecisionHead(config.embed_dim, config.head_width, config.head_dropout,
                                      len(config.question_types))
        # One temperature per question type, fitted after training (post-hoc calibration)
        self.register_buffer("temperatures", torch.ones(len(config.question_types)))

        self._encoder_kwargs = self._accepted_kwargs(getattr(self.encoder, config.encoder_method))
        self.post_init()

    @staticmethod
    def _accepted_kwargs(fn) -> set[str] | None:
        params = inspect.signature(fn).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return None  # accepts anything
        return set(params)

    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.encoder.set_input_embeddings(value)

    def get_tokenizer(self, tokenizer=None):
        """The tokenizer saved next to the weights, loaded once on first use (or the one you pass)."""
        if tokenizer is not None:
            return tokenizer
        if getattr(self, "_tokenizer", None) is None:
            from transformers import AutoTokenizer

            source = getattr(self.config, "_name_or_path", None) or getattr(self, "name_or_path", None)
            if not source:
                raise ValueError("No tokenizer found: load the model with from_pretrained or pass tokenizer=...")
            self._tokenizer = AutoTokenizer.from_pretrained(source)
        return self._tokenizer

    # ---- text → embedding ---------------------------------------------------
    def embed(self, input_ids: Tensor, attention_mask: Tensor, prompt_length: int | None = None) -> Tensor:
        """Token ids → normalized, truncated sentence embeddings `(N, embed_dim)` (float32)."""
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "return_dict": True}
        if self._encoder_kwargs is not None:
            kwargs = {k: v for k, v in kwargs.items() if k in self._encoder_kwargs}
        out = getattr(self.encoder, self.config.encoder_method)(**kwargs)
        for key in self.config.encoder_output:
            try:
                out = out[key]
            except (KeyError, TypeError):
                out = getattr(out, key)
        tokens = out

        if tokens.dim() == 2:
            pooled = tokens                      # the backbone already returns one vector per text
        else:
            mask = attention_mask
            if prompt_length and not self.config.pooling_include_prompt:
                mask = _exclude_prompt_from_mask(mask, int(prompt_length))
            m = mask.unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)

        for linear, activation in zip(self.projections, self.activations):
            pooled = activation(linear(pooled))
        return F.normalize(pooled[:, : self.config.embed_dim].float(), dim=-1)

    def tokenize(self, texts: list[str], tokenizer=None) -> dict[str, Any]:
        """Tokenize exactly as sentence-transformers did in training (task prompt included)."""
        tokenizer = self.get_tokenizer(tokenizer)
        prompt = self.config.prompt or ""
        # Truncate only to a real length: with max_length=None, transformers falls back to the tokenizer's
        # model_max_length, which is a huge sentinel (1e30) when unset and overflows the Rust tokenizer.
        max_length = self.config.max_seq_length
        if not max_length or max_length > 1_000_000:
            max_length = MAX_CONTEXT
        enc = tokenizer([prompt + t for t in texts], padding=True, truncation=True,
                        max_length=int(max_length), return_tensors="pt")
        features: dict[str, Any] = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if prompt and not self.config.pooling_include_prompt:
            ids = tokenizer([prompt], return_tensors="pt")["input_ids"]
            length = ids.shape[-1]
            if ids[0, -1].item() in set(getattr(tokenizer, "all_special_ids", [])):
                length -= 1
            features["prompt_length"] = length
        return features

    @torch.no_grad()
    def encode(self, texts: list[str], tokenizer=None, batch_size: int = 32) -> Tensor:
        """Embed texts, batching by length to avoid padding short texts to long ones."""
        tokenizer = self.get_tokenizer(tokenizer)
        was_training = self.training
        self.eval()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = torch.empty(len(texts), self.config.embed_dim, device=self.device)
        for j in range(0, len(order), batch_size):
            idx = order[j: j + batch_size]
            feats = self.tokenize([texts[i] for i in idx], tokenizer)
            feats = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in feats.items()}
            out[torch.tensor(idx, device=self.device)] = self.embed(**feats)
        self.train(was_training)
        return out

    # ---- tensors → decisions --------------------------------------------------
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        state_index: Tensor,
        question_index: Tensor,
        option_index: Tensor,
        option_mask: Tensor,
        question_type: Tensor,
        labels: Tensor | None = None,
        prompt_length: int | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> GemmaDecisionOutput | tuple:
        r"""
        Args:
            input_ids, attention_mask (`(num_texts, seq_len)`):
                Tokenized unique texts of the batch (states, questions, options); see [`~GemmaDecisionModel.tokenize`].
            state_index, question_index (`(num_questions,)`):
                Row in the text batch of each question's state and question text.
            option_index, option_mask (`(num_questions, max_options)`):
                Rows of each question's option texts, and which of them are real (not padding).
            question_type (`(num_questions,)`):
                Index into `config.question_types` (`0=choice, 1=score, 2=noul`).
            labels (`(num_questions, max_options)` float, or `(num_questions,)` long, *optional*):
                Soft target distributions (or gold option indices) for the decision loss.
            prompt_length (`int`, *optional*):
                Prompt length in tokens, only used when `config.pooling_include_prompt=False`.
        """
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)
        emb = self.embed(input_ids, attention_mask, prompt_length)
        emb = emb.to(self.head.state_proj.weight.dtype)
        option_mask = option_mask.bool()

        logits = self.head(emb[state_index], emb[question_index], emb[option_index], question_type, option_mask)
        logits = logits.float()
        temps = self.temperatures.float()[question_type].unsqueeze(-1)
        probabilities = (logits / temps).masked_fill(~option_mask, -1e4).softmax(-1)

        loss = None
        if labels is not None:
            if labels.dtype in (torch.long, torch.int32, torch.int64) and labels.dim() == 1:
                labels = F.one_hot(labels, logits.shape[-1]).float()
            loss = decision_loss(logits, labels.float(), option_mask, question_type == SCORE,
                                 self.config.label_smoothing, self.config.brier_weight, self.config.ordinal_weight)

        if not return_dict:
            return tuple(x for x in (loss, logits, probabilities, emb) if x is not None)
        return GemmaDecisionOutput(loss=loss, logits=logits, probabilities=probabilities, text_embeds=emb)

    # ---- high-level helpers ---------------------------------------------------
    def _build_inputs(self, examples: list[tuple[str, Any, dict[str, Any], dict[str, Any] | None]],
                      tokenizer=None) -> tuple[dict[str, Any], list[QuestionSpec], list[str]]:
        """examples: (workflow, state, questions, gold or None) → model inputs, parsed specs, unique texts."""
        bank: dict[str, int] = {}
        text_id = lambda t: bank.setdefault(t, len(bank))  # noqa: E731
        specs: list[QuestionSpec] = []
        state_idx, question_idx, options, targets = [], [], [], []

        for workflow, state, questions, gold in examples:
            s = text_id(render_state(workflow, state))
            for qid, q in _json(questions).items():
                if gold is not None and qid not in gold:
                    continue
                spec = parse_question(qid, q, gold.get(qid) if gold else None)
                specs.append(spec)
                state_idx.append(s)
                question_idx.append(text_id(render_question(spec)))
                options.append([text_id(render_option(spec, i)) for i in range(len(spec.options))])
                if gold is not None:
                    targets.append(gold_target(spec, gold[qid])[0])

        k = max(len(o) for o in options)
        option_index = torch.zeros(len(specs), k, dtype=torch.long)
        option_mask = torch.zeros(len(specs), k, dtype=torch.bool)
        for i, o in enumerate(options):
            option_index[i, : len(o)] = torch.tensor(o)
            option_mask[i, : len(o)] = True

        inputs = {
            **self.tokenize(list(bank), tokenizer),
            "state_index": torch.tensor(state_idx),
            "question_index": torch.tensor(question_idx),
            "option_index": option_index,
            "option_mask": option_mask,
            "question_type": torch.tensor([sp.kind_id for sp in specs]),
        }
        if targets:
            labels = torch.zeros(len(specs), k)
            for i, t in enumerate(targets):
                labels[i, : len(t)] = torch.tensor(t)
            inputs["labels"] = labels
        return inputs, specs, list(bank)

    def collate(self, rows: list[dict[str, Any]], tokenizer=None) -> dict[str, Any]:
        """Data collator for rows shaped like LocalLLaMA/typed-decisions ({workflow, state, questions, gold})."""
        examples = [(r.get("workflow", ""), r["state"], _json(r["questions"]), _json(r["gold"])) for r in rows]
        return self._build_inputs(examples, tokenizer)[0]

    @torch.no_grad()
    def decide(self, state: Any, questions: dict[str, dict[str, Any]], workflow: str = "",
               tokenizer=None) -> dict[str, TypedAnswer]:
        """Answer every question of a schema about one state. Returns `{question_id: TypedAnswer}`."""
        was_training = self.training
        self.eval()
        inputs, specs, _ = self._build_inputs([(workflow, state, _json(questions), None)], tokenizer)
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        probs = self(**inputs).probabilities.cpu()
        self.train(was_training)

        answers: dict[str, TypedAnswer] = {}
        for i, spec in enumerate(specs):
            p = probs[i, : len(spec.options)]
            j = int(p.argmax())
            label = spec.options[j]
            value: str | int | bool = (
                (label == "true") if spec.kind == "noul" else int(label) if spec.kind == "score" else label
            )
            expected = (float(sum(int(o) * float(pi) for o, pi in zip(spec.options, p)))
                        if spec.kind == "score" else None)
            answers[spec.qid] = TypedAnswer(spec.kind, label, value, float(p[j]),
                                            dict(zip(spec.options, p.tolist())), expected)
        return answers


__all__ = [
    "GemmaDecisionPreTrainedModel",
    "GemmaDecisionModel",
    "GemmaDecisionHead",
    "GemmaDecisionOutput",
    "QuestionSpec",
    "TypedAnswer",
    "parse_question",
    "gold_target",
    "render_state",
    "render_question",
    "render_option",
    "decision_loss",
]