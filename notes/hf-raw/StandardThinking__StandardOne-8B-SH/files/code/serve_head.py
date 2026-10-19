#!/usr/bin/env python3
"""HTTP server implementing the jev-adapter /v1/systemone contract with the joint schema head (DESIGN.md 5.1).

  python3 serve_head.py --snapshot <base or merged dir> --adapter <run>/final/adapter --head <run>/final/head \
      --port 30171 [--temperature-file calib.json]

Endpoints: POST /v1/systemone, GET /v1/models, /model_info, /server_info (also /get_model_info, /get_server_info),
/health. The engine snapshot fields the benchmark runner checks are reported (disable_radix_cache=true,
mm_preprocess_cache_size_mb=0, speculative_algorithm=null): every request is a full prefill, nothing is cached.
Dynamic batching: requests waiting in the queue are forwarded together while the padded token count stays within
--max-batch-tokens (image requests run alone). Stdlib HTTP server; binds 127.0.0.1 by default.
"""
import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scoring import RequestError, load_scorer  # noqa: E402


class Batcher:
    def __init__(self, scorer, max_batch_tokens, max_batch_requests, wait_ms):
        self.scorer, self.max_tokens, self.max_requests, self.wait = scorer, max_batch_tokens, max_batch_requests, wait_ms / 1000
        self.q = queue.Queue()
        threading.Thread(target=self.loop, daemon=True).start()

    def submit(self, body):
        started = time.perf_counter()
        req, opts, encs = self.scorer.plan(body)        # validation + encoding in the caller's thread
        job = {"req": req, "opts": opts, "encs": encs, "started": started, "done": threading.Event()}
        self.q.put(job)
        job["done"].wait()
        if "error" in job:
            raise job["error"]
        return job["response"]

    def loop(self):
        while True:
            jobs = [self.q.get()]
            image = bool(jobs[0]["req"]["images"])
            deadline = time.perf_counter() + self.wait
            while not image and len(jobs) < self.max_requests:
                try:
                    nxt = self.q.get(timeout=max(0.0, deadline - time.perf_counter()))
                except queue.Empty:
                    break
                encs = [e for j in jobs + [nxt] for e in j["encs"]]
                cost = len(encs) * max(e["n_tokens"] for e in encs)
                if nxt["req"]["images"] or cost > self.max_tokens:
                    self.q.put(nxt)   # goes to the next batch
                    break
                jobs.append(nxt)
            try:
                encs = [e for j in jobs for e in j["encs"]]
                outs, ps, hs = self.scorer.run(encs)
                k = 0
                for j in jobs:
                    n = len(j["encs"])
                    j["response"] = self.scorer.respond(j["req"], j["opts"], j["encs"], outs[k:k + n], ps, hs, j["started"])
                    j["response"]["metadata"]["batch_requests"] = len(jobs)
                    k += n
            except Exception as e:  # noqa: BLE001
                for j in jobs:
                    j["error"] = e
            for j in jobs:
                j["done"].set()


def make_handler(scorer, batcher, info):
    model_info = {"model_path": info["model_path"], "served_model_name": scorer.served_model, "is_generation": False,
                  "has_image_understanding": True, "model_type": "schema-head", "architectures": ["SchemaHeadModel"]}
    server_info = {"model_path": info["model_path"], "served_model_name": scorer.served_model,
                   "disable_radix_cache": True, "mm_preprocess_cache_size_mb": 0, "enable_prefix_mm_cache": False,
                   "enable_mm_global_cache": False, "speculative_algorithm": None, "context_length": 32768,
                   "dtype": info["dtype"], "attention_backend": "torch-sdpa", "version": "schema-head-serve-v1"}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path == "/health":
                return self._send(200, {"status": "ok"})
            if path == "/v1/models":
                return self._send(200, {"object": "list", "data": [{"id": scorer.served_model, "object": "model",
                                                                     "owned_by": "schema-head", "label_scheme": "schema-v1"}]})
            if path in ("/model_info", "/get_model_info"):
                return self._send(200, model_info)
            if path in ("/server_info", "/get_server_info"):
                return self._send(200, server_info)
            return self._send(404, {"error": {"code": "not_found", "message": path}})

        def do_POST(self):
            if self.path.split("?")[0].rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": {"code": "not_found", "message": self.path}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                return self._send(200, batcher.submit(body))
            except RequestError as e:
                return self._send(e.status, {"error": {"code": e.code, "message": str(e), "field": e.field}})
            except json.JSONDecodeError as e:
                return self._send(400, {"error": {"code": "invalid_json", "message": str(e)}})
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"error": {"code": "scoring_failed", "message": f"{type(e).__name__}: {e}"}})
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--tiny", help="JSON tiny config (tests only)")
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--served-model-name", default="standardthinking/standard-schema-8b")
    ap.add_argument("--temperature-file", type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30171)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-batch-tokens", type=int, default=32768)
    ap.add_argument("--max-batch-requests", type=int, default=8)
    ap.add_argument("--batch-wait-ms", type=float, default=5.0)
    ap.add_argument("--memory-cap-bytes", type=float, default=0, help="torch allocator cap on GPU (0 = none)")
    args = ap.parse_args()
    tiny = json.loads(args.tiny) if args.tiny else None
    scorer, info = load_scorer(args.snapshot, args.head, args.adapter, tiny, args.tokenizer, args.served_model_name,
                               args.temperature_file, args.device, max_concurrency=args.max_batch_requests,
                               memory_cap_bytes=args.memory_cap_bytes)
    batcher = Batcher(scorer, args.max_batch_tokens, args.max_batch_requests, args.batch_wait_ms)
    meta = {"model_path": str(args.snapshot or "tiny"), "dtype": "bfloat16" if scorer.device.type == "cuda" else "float32"}
    server = ThreadingHTTPServer((args.host, args.port), make_handler(scorer, batcher, meta))
    print(json.dumps({"serving": f"http://{args.host}:{args.port}", "model": args.served_model_name,
                      "head_params": info["head_params"], "device": str(scorer.device)}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
