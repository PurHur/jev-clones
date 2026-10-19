"""Run Nimble-v4-max through the Decision Index reproduction kit (github.com/apolinario/decision-index).

    hf download bespokelabs/Nimble-v4-max --local-dir Nimble-v4-max
    PYTHONPATH=code python -m decision_index run --engine nimble.evaluation.decision_index_engine:NimbleEngine \
        --option adapter=Nimble-v4-max --rows <suite rows> --out <run dir>

`adapter` is the downloaded model directory: the full BF16 weights (config.json and safetensors shards) and the
schema_config.json contract that pins the prompt code and the option-code tokens. The tokenizer is the base
model's, at the revision the contract names.

The kit calls the engine once per request. Every question of a request is answered from one prompt layout: the
whole request (state and every question) in compact plain text, then "Requested field: <name>". The request part
is read both ways in the full-attention layers (nimble/training/prefix_attention.py); the field name and the
assistant tail are read one way. The request part runs once, and each question's short tail runs on a copy of
its cache (shared_prefix, the fastest path). Each option is one token; its probability is the softmax over the
options' logits at the answer position. Nothing is truncated: a request longer than the context is refused as
Unsupported, which the index counts as wrong. An empty string state is passed as its JSON form.
"""
import copy
import hashlib
import json
from pathlib import Path

import torch
from decision_index.engines import Engine, Unsupported

COMPACT_TASK = "schema_candidate_classification_compact_v1"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def weights_sha256(path):
    """One SHA-256 for the weights: that of weights_sha256.json, the per-shard hashes saved with the model."""
    for name in ("weights_sha256.json", "model.safetensors.index.json"):
        if (Path(path) / name).exists():
            return sha(Path(path) / name)
    raise FileNotFoundError(f"No weights in {path}")


def native_question(question):
    """Nimble reads a yes/no question from true/false criteria; map yes/no keys onto them."""
    if question["type"] != "noul":
        return question
    criteria = question.get("criteria") or {}
    if set(criteria) <= {"true", "false"}:
        return question
    if set(criteria) == {"yes", "no"}:
        return dict(question, criteria={"true": criteria["yes"], "false": criteria["no"]})
    raise Unsupported(f"yes/no question with criteria keys {sorted(criteria)} has no true/false reading")


def native_schema(body):
    """The prompt's context text and field schema for a System One request."""
    def text(v):
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    schema = {}
    for name, q in body["questions"].items():
        field = {"description": text(q["instructions"])}
        criteria = q.get("criteria") or {}
        if q["type"] == "noul":
            field.update(type="boolean", choices=[False, True],
                         choice_descriptions={k: text(v) for k, v in criteria.items() if v is not None})
        elif q["type"] == "choice":
            field.update(type="enum", choices=list(criteria),
                         choice_descriptions={k: text(v) for k, v in criteria.items() if v is not None})
        elif q["type"] == "score":
            field.update(type="enum", choices=[str(i) for i in range(len(criteria))],
                         choice_descriptions={str(i): text(v) for i, v in enumerate(criteria)})
        else:
            raise ValueError("Unsupported primitive")
        schema[name] = field
    return text(body["state"]), schema


def chunks(rows, budget):
    """Consecutive groups whose padded size, longest prompt times rows, stays within budget."""
    group, longest = [], 0
    for row in rows:
        size = len(row["input_ids"])
        if group and max(longest, size) * (len(group) + 1) > budget:
            yield group
            group, longest = [], 0
        group.append(row)
        longest = max(longest, size)
    if group:
        yield group


def collate(rows, pad_id):
    """Left-padded prompts and right-padded option tokens for one batch."""
    length = max(len(r["input_ids"]) for r in rows)
    width = max(len(r["candidate_ids"]) for r in rows)
    return {
        "input_ids": torch.tensor([[pad_id] * (length - len(r["input_ids"])) + r["input_ids"] for r in rows]),
        "attention_mask": torch.tensor([[0] * (length - len(r["input_ids"])) + [1] * len(r["input_ids"]) for r in rows]),
        "candidate_ids": torch.tensor([r["candidate_ids"] + [0] * (width - len(r["candidate_ids"])) for r in rows]),
        "candidate_mask": torch.tensor([[True] * len(r["candidate_ids"]) + [False] * (width - len(r["candidate_ids"]))
                                        for r in rows]),
    }


def candidate_logits(model, inputs, prefix_attention):
    """Logits of each row's option tokens at its last position."""
    with prefix_attention.request_blocks(inputs["input_ids"], inputs["attention_mask"]):
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                       use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
    selected = logits.gather(1, inputs["candidate_ids"])
    return selected.masked_fill(~inputs["candidate_mask"], -torch.inf)


def decision_result(row, logits, choice_key):
    """The most probable option and the softmax over the options."""
    logits = logits[:len(row["choices"])].double()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite candidate logits")
    probs = logits.softmax(-1)
    choices = row["choices"]
    return {"prediction": choices[logits.argmax().item()],
            "probabilities": {choice_key(k): v for k, v in zip(choices, probs.tolist())}}


def load_model(path):
    """The full BF16 model on this process's GPU."""
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
    config = AutoConfig.from_pretrained(path, trust_remote_code=False)
    if config.model_type != "qwen3_5":
        raise ValueError(f"Unsupported architecture: {config.model_type}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, config=config, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False,
        device_map={"": torch.cuda.current_device()})
    model.config.use_cache = False
    return model.eval()


class NimbleEngine(Engine):
    name = "nimble"
    latency = "In-process request wall time on one GPU: prompt construction and batched forward passes."

    def __init__(self, adapter, max_tokens=32768, batch_tokens=65536, model=None, revision=None, shared_prefix=True,
                 cache_tokens=131072, **options):
        super().__init__(adapter=adapter, max_tokens=max_tokens, batch_tokens=batch_tokens,
                         shared_prefix=shared_prefix, cache_tokens=cache_tokens, **options)
        from transformers import AutoTokenizer
        from nimble.scoring import extended_schema
        from nimble.training import compact_fast, compact_schema, prefix_attention
        adapter = Path(adapter)
        self.contract = json.loads((adapter / "schema_config.json").read_text())
        if self.contract.get("task") != COMPACT_TASK:
            raise ValueError(f"{adapter} is not a compact_v1 Nimble model")
        model = model or self.contract["model"]
        revision = revision or self.contract["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Refuses a model whose prompt code, option codes or their token ids differ from this code's.
        compact_schema.validate_contract(self.contract, self.tokenizer)
        prefix_attention.configure(self.contract.get("prefix_attention"), self.tokenizer)
        self.prefix_attention = prefix_attention
        self.prepare = compact_fast.prepare_prompts  # compact_schema's prompts token for token, faster
        self.choice_key = extended_schema.choice_key
        self.shared_prefix, self.cache_tokens = bool(shared_prefix), int(cache_tokens)
        # How many question tails share one copied cache at most. Each tail gets its own copy of every
        # linear-attention layer's state, about 150 MB for a 27B, so the token budget alone can ask for more
        # copies than fit. An out-of-memory error halves the batch, and the smaller size is kept from then on.
        self.tail_cap = int(options.get("max_tails") or 0) or None
        self.model = load_model(adapter)
        self.max_tokens, self.batch_tokens = int(max_tokens), int(batch_tokens)
        self.provenance = {"adapter": str(adapter), "weights": "full_model", "weights_sha256": weights_sha256(adapter),
                           "base_model": model, "revision": revision, "contract_task": self.contract.get("task"),
                           "max_tokens": self.max_tokens, "prefix_attention": self.contract.get("prefix_attention")}

    def __call__(self, state, questions):
        # The contract needs nonempty context text. An empty string state is passed in its JSON form, "",
        # the same way an empty object state reaches the prompt as {}. This applies to every benchmark.
        if isinstance(state, str) and not state.strip():
            state = json.dumps(state)
        body = {"state": state, "questions": {k: native_question(q) for k, q in questions.items()}}
        context, schema = native_schema(body)
        try:
            prompt = self.prepare(self.tokenizer, context, schema, self.max_tokens)
        except ValueError as exc:
            if "limit is" in str(exc):
                raise Unsupported(f"maximum context length: {exc}") from None
            raise
        rows = [dict(input_ids=prompt.full_ids[i], candidate_ids=prompt.candidate_ids[i], choices=prompt.choices[i])
                for i in range(len(prompt.names))]
        results = self.score_prompt(prompt, rows)
        answers = {}
        for name, out in zip(prompt.names, results):
            probabilities = out["probabilities"]
            if questions[name]["type"] == "noul":
                answers[name] = {"type": "noul", "noul": probabilities["true"]}
            else:
                answers[name] = {"type": "choice", "choice": out["prediction"], "probabilities": probabilities}
        return {"model": self.name, "answers": answers, "usage": {"input_tokens": sum(map(len, prompt.full_ids))}}, None

    def score_prompt(self, prompt, rows):
        """One scored row per prompt."""
        # A single long question also goes the shared way: the request then runs with SDPA's fused kernels instead
        # of against a materialized square mask. Short ones stay in one forward pass, which costs less than the
        # second pass the shared way needs.
        if self.shared_prefix and (len(rows) > 1 or self.prefix_attention.enabled() and len(rows[0]["input_ids"]) > 4096):
            prefix, tails = self.split(prompt)
            if prefix is not None and len(prefix) >= 256:
                return self.score_shared(prefix, tails, rows)
        return self.score(rows)

    def split(self, prompt):
        """The tokens all of a request's prompts share, and each prompt's tail after them.

        Request tokens are read both ways and the rest one way, so the shared part must end exactly where the
        request does (after the last "Requested field:"). When the prompts do not agree there, (None, None) sends
        the request through ordinary batches.
        """
        if not self.prefix_attention.enabled():
            return prompt.prefix_ids, prompt.suffix_ids
        ends = {int(self.prefix_attention.request_ends(torch.tensor([ids]))[0]) for ids in prompt.full_ids}
        if len(ends) != 1:
            return None, None
        end = ends.pop()
        prefix = prompt.full_ids[0][:end]
        if any(ids[:end] != prefix for ids in prompt.full_ids):
            return None, None
        return prefix, [ids[end:] for ids in prompt.full_ids]

    def score(self, rows):
        results = []
        for group in chunks(rows, self.batch_tokens):
            results.extend(self.score_group(group))
        return results

    def score_group(self, group):
        """Score one batch; on out-of-memory, split it, and refuse only a single prompt that does not fit."""
        try:
            batch = {k: v.to("cuda") for k, v in collate(group, self.tokenizer.pad_token_id).items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = candidate_logits(self.model, batch, self.prefix_attention).cpu()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(group) == 1:
                raise Unsupported(f"out of GPU memory for one {len(group[0]['input_ids'])}-token prompt") from None
            half = len(group) // 2
            return self.score_group(group[:half]) + self.score_group(group[half:])
        return [decision_result(row, logits[i], self.choice_key) for i, row in enumerate(group)]

    def score_shared(self, prefix, tails, rows):
        """Run the questions' shared request part once, then only each question's short tail on a copy of its cache.

        The prefix runs once with the cache on; the cache (attention keys and values, and the linear-attention
        layers' convolution and recurrent state) is copied across a batch of tails, which are right-padded so
        padding never comes between the prefix and a tail. Each tail's answer is read at its own last token.
        """
        per_batch = max(1, self.cache_tokens // len(prefix))
        if self.tail_cap:
            per_batch = min(per_batch, self.tail_cap)
        results = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            try:
                with self.prefix_attention.whole_request(self.model):
                    out = self.model(input_ids=torch.tensor([prefix], device="cuda"), use_cache=True,
                                     logits_to_keep=1)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return self.score(rows)
            cache = out.past_key_values
            start = 0
            while start < len(rows):
                group = list(range(start, min(start + per_batch, len(rows))))
                # The last question alone can run on the prefix cache itself: nothing reads the cache after it,
                # and a one-question batch that runs out of memory is refused rather than retried.
                last_alone = len(group) == 1 and group[-1] == len(rows) - 1
                try:
                    results.extend(self.score_tails(cache, prefix, tails, rows, group, reuse=last_alone))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if per_batch == 1 or last_alone:  # a reused cache may be half extended, so no retry
                        raise Unsupported(f"out of GPU memory for one question after a {len(prefix)}-token prefix") from None
                    per_batch = max(1, per_batch // 2)
                    self.tail_cap = per_batch
                    continue
                start += len(group)
        return results

    def score_tails(self, cache, prefix, tails, rows, group, reuse=False):
        """The tails of `group` on copies of the prefix cache; right padding keeps each tail next to the prefix.
        With reuse, a single tail runs on the cache itself, which it extends."""
        lengths = [len(tails[i]) for i in group]
        width = max(lengths)
        ids = torch.tensor([tails[i] + [self.tokenizer.pad_token_id] * (width - len(tails[i])) for i in group],
                           device="cuda")
        mask = torch.tensor([[1] * len(prefix) + [1] * n + [0] * (width - n) for n in lengths], device="cuda")
        if reuse:
            batch_cache = cache
        else:
            batch_cache = copy.deepcopy(cache)
            batch_cache.reorder_cache(torch.zeros(len(group), dtype=torch.long, device="cuda"))
        logits = self.model(input_ids=ids, attention_mask=mask, past_key_values=batch_cache, use_cache=True,
                            logits_to_keep=width).logits
        last = logits[torch.arange(len(group), device="cuda"), torch.tensor(lengths, device="cuda") - 1].float()
        results = []
        for j, i in enumerate(group):
            candidates = torch.tensor(rows[i]["candidate_ids"], device="cuda")
            results.append(decision_result(rows[i], last[j, candidates].cpu(), self.choice_key))
        del batch_cache, logits
        return results

    def runtime(self):
        return {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()}

    def synchronize(self):
        torch.cuda.synchronize()
