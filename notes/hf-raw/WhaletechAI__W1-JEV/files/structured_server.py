"""Serve w1-jev decisions with the djev schema and template."""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import gc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from queue import Empty, Queue
import random
import threading
import time
import traceback

import torch
from transformers import PreTrainedTokenizerFast

from batch import batch_logits
from checkpoint import load_model_state
import djev_template as djev
from model import create_model

BASE = Path(__file__).resolve().parent
MODEL_NAME = "w1-jev"
DEFAULTS = {"steps": 1, "samples": 1, "think": 0, "timestep": 0.5, "mode": "single"}


def validate_options(value):
    for key, expected in DEFAULTS.items():
        actual = value.get(key, expected)
        if actual != expected or isinstance(actual, bool):
            raise ValueError(f"w1-jev requires {key}={expected!r}")
    if value.get("images") or value.get("stream"):
        raise ValueError("Only non-streaming text decisions are supported")
    chunk_rows = value.get("chunk_rows")
    if chunk_rows is not None and (type(chunk_rows) is not int or not 8 <= chunk_rows <= djev.CANVAS_LEN):
        raise ValueError(f"chunk_rows must be an integer from 8 to {djev.CANVAS_LEN}")
    seed = value.get("seed", 42)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return {**value, **DEFAULTS, "seed": seed}


def chunk_groups(schema, questions, conditioned=False):
    """Split templates using the delimiter each compiled canvas will contain."""
    limit = min(schema.get("chunk_rows") or djev.CANVAS_LEN, djev.CANVAS_LEN)
    join = djev.FORMATS[schema["format"]][0]
    groups, group = [], []

    def rows(items):
        lead = join if conditioned or (schema["sequential"] and groups) else ""
        return len(djev.enc(lead + djev.answer_text(items, [0] * len(items), schema["format"]))) + 1

    def check_single(question):
        size = rows([question])
        if size > limit:
            raise djev.SchemaError(
                f"question {question['id']!r} alone needs {size} canvas rows; maximum is {limit}"
            )

    for question in questions:
        if question["alone"]:
            if group:
                groups.append(group)
                group = []
            check_single(question)
            groups.append([question])
            continue
        if group and rows(group + [question]) > limit:
            groups.append(group)
            group = []
        if not group:
            check_single(question)
        group.append(question)
    if group:
        groups.append(group)
    return groups


class TemplateCompiler:
    def __init__(self):
        self.config = json.loads((BASE / "config.json").read_text())
        tokenizer_config = json.loads((BASE / "tokenizer_config.json").read_text())
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(BASE / "tokenizer.json"),
            **{key: tokenizer_config[key] for key in ("bos_token", "eos_token", "unk_token", "pad_token", "mask_token")})
        djev.TOK = self.tokenizer
        self.max_seq_len = self.config["model"]["max_seq_len"]
        self.vocab_size = self.config["model"]["vocab_size"]

    def prompt_ids(self, system, state):
        # W1 role framing around the djev question template and user state.
        state = state.replace("\x00", " ").strip()
        prefix = (f"<|system|>\n{system.strip()}\n" if system.strip() else "")
        prefix += f"<|user|>\n{state}\n<|assistant|>\n"
        return self.tokenizer.encode(prefix, add_special_tokens=False)

    def compile(self, schema, system, state, seed=42, prefix=None, lead=""):
        prefix_ids = self.prompt_ids(system, state) if prefix is None else list(prefix)
        canvas, slots = djev.resolve_template(schema["questions"], [], lead, schema["format"])
        canvas.append(self.tokenizer.eos_token_id)
        # djev's build_canvas rule, using this model's vocabulary size.
        rng = random.Random(seed)
        for slot in slots:
            canvas[slot["pos"]] = rng.randrange(self.vocab_size)
        ids = prefix_ids + canvas
        if len(ids) > self.max_seq_len:
            raise ValueError(f"Input has {len(ids)} tokens; maximum is {self.max_seq_len}")
        if any(not 0 <= i < self.vocab_size for i in ids):
            raise ValueError("Input token outside model vocabulary")
        return {"schema": schema, "prefix_ids": prefix_ids, "input_ids": ids,
                "canvas": canvas, "slots": slots}


class Engine:
    def __init__(self, compiler, checkpoint):
        if not torch.cuda.is_available():
            raise ValueError("A CUDA GPU is required")
        if not checkpoint.is_file():
            raise ValueError(f"Place w1-jev.pt beside this script, or use --checkpoint: {checkpoint}")
        self.compiler = compiler
        torch.set_num_threads(4)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        state = saved.get("model", saved)  # Training checkpoint or model-only state_dict.
        with torch.device("meta"):
            self.model = create_model(compiler.config)
        self.model.to(dtype=torch.bfloat16).to_empty(device="cuda")
        load_model_state(self.model, state, compiler.config["model"])
        del saved, state
        gc.collect()
        self.model.eval()
        torch.cuda.synchronize()

    @torch.inference_mode()
    def read_many(self, compiled):
        positions = [[len(c["prefix_ids"]) + s["pos"] for s in c["slots"]] for c in compiled]
        device = next(self.model.parameters()).device
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = batch_logits(self.model, [c["input_ids"] for c in compiled], positions)
        results, offset = [], 0
        for c in compiled:
            answers, diagnostics = {}, {}
            for q, slot in zip(c["schema"]["questions"], c["slots"]):
                row = logits[offset]
                offset += 1
                candidate = row[slot["label_ids"]]
                probs = torch.softmax(candidate, -1).tolist()
                best = max(range(len(probs)), key=probs.__getitem__)
                names = [choice[0] for choice in q["choices"]]
                answer = {"type": q["type"], "label": q["labels"][best],
                          "confidence": probs[best], "probabilities": dict(zip(names, probs))}
                if q["type"] == "noul":
                    answer["noul"] = probs[0]
                elif q["type"] == "choice":
                    answer["choice"] = names[best]
                else:
                    answer.update(score=sum((i + 1) * p for i, p in enumerate(probs)), level=names[best])
                answers[q["id"]] = answer
                diagnostics[q["id"]] = {
                    "pos": slot["pos"], "entropy": [-sum(p * math.log(p) for p in probs if p)],
                    "label_mass": float(torch.exp(torch.logsumexp(candidate, 0) - torch.logsumexp(row, 0))),
                    "argmax_is_label": int(row.argmax()) in slot["label_ids"]}
            results.append({"answers": answers, "diagnostics": diagnostics})
        return results


class Batcher:
    """One GPU owner; concurrent HTTP requests share a short batching window."""
    def __init__(self, engine, size=2, wait_ms=5):
        self.engine, self.size, self.wait = engine, size, wait_ms / 1000
        self.queue = Queue()
        threading.Thread(target=self.work, daemon=True).start()

    def submit(self, compiled):
        future = Future()
        self.queue.put((compiled, future))
        return future.result(timeout=600)

    def work(self):
        while True:
            batch = [self.queue.get()]
            deadline = time.perf_counter() + self.wait
            while len(batch) < self.size:
                try:
                    batch.append(self.queue.get(timeout=max(0, deadline - time.perf_counter())))
                except Empty:
                    break
            try:
                results = self.engine.read_many([c for c, _ in batch])
                for (_, future), result in zip(batch, results):
                    future.set_result(result)
            except Exception as exc:
                traceback.print_exc()
                for _, future in batch:
                    future.set_exception(exc)


class Decisions:
    def __init__(self, compiler, batcher):
        self.compiler, self.batcher = compiler, batcher

    def decide(self, schema, state, seed):
        started = time.perf_counter()
        qs = [q for q in schema["questions"] if not schema["ask"] or q["id"] in schema["ask"]]
        levels = djev.schedule(qs)
        chained = len(levels) > 1 or schema["sequential"]
        join = djev.FORMATS[schema["format"]][0]
        system = djev.system_text(schema)
        base_ids = self.compiler.prompt_ids(system, state) if chained else None
        answers, lines, diagnostics, stages, skipped = {}, [], {}, [], {}
        by_id = {q["id"]: q for q in qs}
        reads, input_tokens, output_tokens = 0, 0, 0

        def run(group, index, conditioned):
            sub = dict(schema, questions=group)
            prefix = base_ids + djev.enc(join.join(lines)) if conditioned else None
            lead = join if conditioned else ""
            sys_text = system if chained else djev.system_text(
                schema if schema["chunk_prompt"] == "shared" else sub,
                chunked=schema["chunk_prompt"] == "shared")
            c = self.compiler.compile(sub, sys_text, state, seed + 104729 * index, prefix, lead)
            return self.batcher.submit(c), c

        def collect_result(group, result):
            nonlocal reads, input_tokens, output_tokens
            body, compiled = result
            answers.update(body["answers"])
            diagnostics.update(body["diagnostics"])
            lines.append(djev.answer_text(group, [q["labels"].index(answers[q["id"]]["label"]) for q in group], schema["format"]))
            reads += 1
            input_tokens = max(input_tokens, len(compiled["prefix_ids"]))
            output_tokens += len(compiled["canvas"])

        for level in levels:
            asked = []
            for q in level:
                if any(djev.answer_name(by_id[dep], answers.get(dep)) not in vals for dep, vals in q["ask_if"].items()):
                    answers[q["id"]] = None
                    skipped[q["id"]] = True
                else:
                    asked.append(q)
            if not asked:
                continue
            stages.append([q["id"] for q in asked])
            conditioned = bool(lines) and chained
            groups = chunk_groups(schema, asked, conditioned)
            if schema["sequential"] or len(groups) == 1:
                for group in groups:
                    collect_result(group, run(group, reads, conditioned or (schema["sequential"] and bool(lines))))
            else:
                # Read independent chunks before adding their answers to the prefix.
                with ThreadPoolExecutor(max_workers=min(16, len(groups))) as pool:
                    futures = [pool.submit(run, group, reads + i, conditioned) for i, group in enumerate(groups)]
                    results = [future.result() for future in futures]
                for group, result in zip(groups, results):
                    collect_result(group, result)
        return {"model": MODEL_NAME, "answers": {q["id"]: answers[q["id"]] for q in qs},
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                "diagnostics": {"questions": diagnostics, "stages": stages, "skipped": skipped,
                                "timing": {"total_ms": (time.perf_counter() - started) * 1000, "reads": reads}},
                "runtime": {**DEFAULTS, "seed": seed, "dtype": "bfloat16", "noise_profile": "djev_random"}}

    def handle(self, body):
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")
        if body.get("model", MODEL_NAME) != MODEL_NAME:
            raise ValueError(f"model must be {MODEL_NAME!r}")
        body = validate_options(body)
        messages = body.get("messages", [])
        if (not isinstance(messages, list) or len(messages) != 2
            or any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in messages)
            or messages[0].get("role") not in ("system", "developer") or messages[1].get("role") != "user"):
            raise ValueError("Use two text messages: system schema JSON, then user state JSON")
        value = json.loads(messages[0]["content"])
        if not isinstance(value, dict):
            raise ValueError("System schema must be a JSON object")
        schema = djev.parse_schema(validate_options(value))
        state = messages[1]["content"].strip()
        json.loads(state)
        if any(len({name for name, _ in q["choices"]}) != len(q["choices"]) for q in schema["questions"]):
            raise ValueError("Alternative names must be unique")
        result = self.decide(schema, state, body["seed"])
        usage = result["usage"]
        return {"id": f"chatcmpl-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
                "model": MODEL_NAME, "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(result)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"],
                          "total_tokens": usage["input_tokens"] + usage["output_tokens"]}}


def make_server(decisions, host, port):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, body):
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self.send_json(200, {"status": "ready", "model": MODEL_NAME})
            if self.path == "/v1/models":
                return self.send_json(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "local"}]})
            self.send_json(404, {"error": {"message": "Unknown endpoint"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.send_json(404, {"error": {"message": "Unknown endpoint"}})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ValueError("Expected a JSON body of at most 4 MiB")
                result = decisions.handle(json.loads(self.rfile.read(length)))
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                return self.send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            except Exception:
                traceback.print_exc()
                return self.send_json(500, {"error": {"message": "Inference failed"}})
            self.send_json(200, result)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=BASE / "w1-jev.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    compiler = TemplateCompiler()
    engine = Engine(compiler, args.checkpoint)
    decisions = Decisions(compiler, Batcher(engine, args.batch_size))
    server = make_server(decisions, args.host, args.port)
    print(f"{MODEL_NAME} ready at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
