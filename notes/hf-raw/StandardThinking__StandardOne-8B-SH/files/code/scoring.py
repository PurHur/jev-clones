"""In-process /v1/systemone scoring with the joint schema head (shared by serve_head.py and eval_head.py).

Response JSON follows the jev-adapter contract (protocol.reduce_probabilities / service.score), so the unchanged
runners (for example `jev_adapter.benchmarks.run`) work as is:
  answers[q] = {type, probabilities | noul, choice, confidence, score, legend, logprobs?}
  usage = {input_tokens, output_tokens: 0}
  metadata.evaluations = questions x permutations (the runners' per-question accounting; the head needs only
  `prefills` = permutations forward passes for the whole request), plus prefill_ms / head_ms.
Option permutations (options.permutations = K) rotate choice and noul options (display position p shows original
(p + r) mod n, as the adapter's rotation_order); score levels always stay in ascending order.
"""
import json
import math
import time
from pathlib import Path

import torch

from render import EncodeError, entropy_confidence, request_from_systemone, token_roles

CONFIDENCE_METHOD = "1 - normalized_entropy"


class RequestError(ValueError):
    def __init__(self, code, message, field=None, status=422):
        super().__init__(message)
        self.code, self.field, self.status = code, field, status


def validate_body(body):
    """Light structural validation (the adapter's pydantic schema is used when importable)."""
    try:
        from jev_adapter.protocol import SystemOneRequest  # noqa: F401
        SystemOneRequest.model_validate(body)
    except ImportError:
        pass
    except Exception as e:  # pydantic ValidationError
        errors = getattr(e, "errors", lambda: [{"msg": str(e), "loc": ()}])()
        first = errors[0] if errors else {"msg": str(e), "loc": ()}
        raise RequestError("invalid_request", first.get("msg", str(e)),
                           ".".join(str(p) for p in first.get("loc", ()))) from None
    if not isinstance(body, dict) or not isinstance(body.get("questions"), dict) or not body["questions"]:
        raise RequestError("invalid_request", "questions are required", "questions")


class HeadScorer:
    def __init__(self, model, encoder, served_model, pad_id, temperature_by_type=None, max_concurrency=1,
                 max_batch_tokens=32768, preview=True):
        self.model, self.encoder, self.served_model, self.pad_id = model, encoder, served_model, pad_id
        self.preview = bool(preview)   # must match training (head config "preview"; --no-preview ablation)
        self.temperature_by_type = dict(temperature_by_type or {})
        self.max_concurrency = max_concurrency
        self.max_batch_tokens = max_batch_tokens
        self.device = model.device()

    def plan(self, body):
        validate_body(body)
        req = request_from_systemone(body)
        opts = body.get("options") or {}
        k = int(opts.get("permutations", 1))
        encs = []
        for r in range(k):
            orders = {}
            for key, q in req["questions"].items():
                n = len(q["options"])
                if q["type"] != "score" and r:
                    orders[key] = [(p + r) % n for p in range(n)]
            try:
                encs.append(self.encoder.encode(req, "W0", list(req["questions"]), orders, preview=self.preview))
            except EncodeError as e:
                raise RequestError(e.reason, str(e), "questions" if e.reason != "too_long" else "state") from None
        return req, opts, encs

    @torch.no_grad()
    def run(self, encs):
        """Forward the encodings (one padded micro-batch); returns (per-encoding logits lists, prefill_s, head_s)."""
        from model import make_batch
        dev = self.device
        sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
        batch = make_batch(encs, self.pad_id, dev)
        sync()
        t0 = time.perf_counter()
        states = self.model.backbone_states(batch)
        sync()
        t1 = time.perf_counter()
        outs = []
        with torch.autocast(device_type=dev.type, enabled=False):
            for i, enc in enumerate(encs):
                n = enc["n_tokens"]
                taps = [s[i, :n].float() for s in states]
                roles = torch.tensor(token_roles(enc), device=dev)
                outs.append([z.float().cpu() for z in self.model.head.forward_one(taps, enc, roles)])
        sync()
        t2 = time.perf_counter()
        return outs, t1 - t0, t2 - t1

    def respond(self, req, opts, encs, outs, prefill_s, head_s, started):
        temperature = float(opts.get("temperature", 1.0)) if opts.get("temperature_scaling", True) else 1.0
        return_logprobs = bool(opts.get("return_logprobs", False))
        answers = {}
        keys = list(req["questions"])
        for key in keys:
            q = req["questions"][key]
            names = [o["name"] for o in q["options"]]
            t_type = self.temperature_by_type.get(q["type"], 1.0)
            probs_sum = [0.0] * len(names)
            raw = []
            for enc, out in zip(encs, outs):
                z = out[enc["keys"].index(key)].double()
                logp = torch.log_softmax(z, -1)
                raw.append({n: float(v) for n, v in zip(names, logp.tolist())})
                p = torch.softmax(z / (t_type * temperature), -1).tolist()
                for j, v in enumerate(p):
                    probs_sum[j] += v
            total = math.fsum(probs_sum)
            probs = [v / total for v in probs_sum]
            ans = {"type": q["type"]}
            if q["type"] == "noul":
                ans["noul"] = probs[0]
            else:
                ans["probabilities"] = dict(zip(names, probs))
                ans["confidence"] = entropy_confidence(probs)
                if q["type"] == "choice":
                    ans["choice"] = names[max(range(len(names)), key=probs.__getitem__)]
                else:
                    ans["score"] = math.fsum(i * p for i, p in enumerate(probs))
                    ans["legend"] = {o["name"]: o.get("description") for o in q["options"]}
            if return_logprobs:
                ans["logprobs"] = raw
            answers[key] = ans
        input_tokens = sum(e["n_tokens"] for e in encs)
        usage = {"input_tokens": input_tokens, "output_tokens": 0}
        if req["images"]:
            usage["input_tokens_details"] = {"image_tokens": sum(e["image_tokens"] for e in encs)}
        return {"model": self.served_model, "answers": answers, "usage": usage,
                "metadata": {"confidence_method": CONFIDENCE_METHOD, "temperature": temperature,
                             **({"temperature_by_type": self.temperature_by_type} if self.temperature_by_type else {}),
                             "evaluations": len(keys) * len(encs), "prefills": len(encs),
                             "service_max_concurrency": self.max_concurrency, "label_scheme": "schema-v1",
                             "cached_tokens": 0, "prefill_ms": prefill_s * 1000, "head_ms": head_s * 1000,
                             "adapter_elapsed_ms": (time.perf_counter() - started) * 1000}}

    def score(self, body):
        started = time.perf_counter()
        if body.get("model") not in (None, self.served_model, "jev-latest"):
            raise RequestError("model_not_found", "Requested model is not configured.", "model", 404)
        req, opts, encs = self.plan(body)
        outs, ps, hs = self.run(encs)
        return self.respond(req, opts, encs, outs, ps, hs, started)


def load_scorer(snapshot=None, head_dir=None, adapter=None, tiny=None, tokenizer=None, served_model=None,
                temperature_file=None, device="auto", merge=True, max_concurrency=1, memory_cap_bytes=0):
    """Backbone (+ LoRA adapter merged) + trained head -> HeadScorer.
    memory_cap_bytes > 0 caps the torch allocator (set_per_process_memory_fraction) before the model is loaded."""
    from model import build_model, load_head_weights
    from render import Encoder
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    if dev.type == "cuda" and memory_cap_bytes:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(memory_cap_bytes) / total), 0)
    cfg = json.loads((Path(head_dir) / "schema_head_config.json").read_text()) if head_dir else {}
    hc = cfg.get("head", {})
    taps = cfg.get("taps") or hc.get("taps") or ["final", 25]
    overrides = {k: hc[k] for k in ("d", "heads", "ffn", "route_blocks", "interact_blocks", "ord_hidden", "max_nodes")
                 if k in hc}
    overrides["dropout"] = 0.0
    overrides["grad_checkpoint"] = False
    model, info = build_model(snapshot, tiny=tiny, lora_rank=0, taps=taps, device=dev, grad_checkpointing=False,
                              init_adapter=adapter, head_overrides=overrides)
    if adapter and merge and model.peft_model is not None:
        model.backbone = model.peft_model.merge_and_unload()
        model.peft_model = None
    if head_dir:
        load_head_weights(model.head, head_dir)
    model.eval()
    if dev.type == "cuda":
        model.backbone.to(torch.bfloat16)
    encoder = Encoder.from_snapshot(tokenizer or snapshot, with_processor=True)
    pad = encoder.tokenizer.pad_token_id if encoder.tokenizer.pad_token_id is not None else encoder.tokenizer.eos_token_id
    temps = json.loads(Path(temperature_file).read_text()).get("temperature_by_type") if temperature_file else None
    return HeadScorer(model, encoder, served_model or "standardthinking/standard-schema-8b", pad, temps,
                      max_concurrency=max_concurrency, preview=cfg.get("preview", True)), info
