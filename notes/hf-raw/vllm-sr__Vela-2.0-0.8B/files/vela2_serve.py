"""Minimal SystemOne-compatible HTTP server for a Vela 2.0 member (optional; needs fastapi and uvicorn).

    pip install fastapi uvicorn
    python vela2_serve.py --model . --device cuda --port 8001

Routes
    POST /v1/systemone   TypeSafe SystemOne request / response ({"state", "model", "questions"} -> {"model",
                         "answers", "usage"}), plus the span / set extensions of vela2_inference.system_one
    GET  /v1/models      {"models": [{"name", "description", "release_date"}]}
    GET  /health, /ready

The typesafe-sdk client works against it: TypeSafeClient(base_url="http://127.0.0.1:8001", api_key="local",
model=<the served name, config.json "model_name">). An Authorization header is accepted and ignored unless
VELA2_API_KEY is set, in which case "Authorization: Bearer $VELA2_API_KEY" is required (401 otherwise).
Errors: 422 {"detail": [{"loc", "msg", "type"}]} for an invalid request or another model name; 413 when the
questions alone do not fit the input window (config.json "max_length"); 401 as above.
"""
import argparse
import hmac
import os
import sys
import threading
from pathlib import Path

try:  # optional dependencies, only needed to serve
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except ImportError as e:  # pragma: no cover
    raise ImportError("vela2_serve needs fastapi and uvicorn: pip install fastapi uvicorn") from e

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vela2_inference import DEFAULT_MODEL_NAME, SystemOneError, Vela2, system_one_request  # noqa: E402


def create_app(engine, model_name=None, api_key=None):
    model_name = model_name or getattr(engine, "model_name", DEFAULT_MODEL_NAME)
    app = FastAPI(title="Vela 2.0 SystemOne", version="1")
    lock = threading.Lock()  # one forward pass at a time (the engine keeps a tokenisation cache)
    served = {model_name}

    def unauthorized(request):
        if not api_key:
            return None
        got = request.headers.get("authorization", "")
        if hmac.compare_digest(got, f"Bearer {api_key}"):
            return None
        return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key"})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": True}

    @app.get("/v1/models")
    def models(request: Request):
        cfg = getattr(engine, "config", {}) or {}
        return unauthorized(request) or {"models": [{"name": model_name, "release_date": cfg.get("release_date", ""),
                                                     "description": cfg.get("description", "Vela 2.0: SystemOne Choice / Noul / "
                                                                    "Score with span and set extensions")}]}

    @app.post("/v1/systemone")
    def systemone(request: Request, body: dict):  # sync handler: FastAPI runs it in its thread pool
        err = unauthorized(request)
        if err is not None:
            return err
        try:
            with lock:
                return system_one_request(engine, body, served=served)
        except SystemOneError as e:
            return JSONResponse(status_code=413 if e.type == "too_long" else 422, content={"detail": [e.detail()]})

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent), help="release dir or Hub repo id")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"], help="onnx: encoder release only")
    ap.add_argument("--device", default="cpu", help="torch device, e.g. cuda")
    ap.add_argument("--onnx-file", default="onnx/model.onnx")
    ap.add_argument("--name", default=os.environ.get("VELA2_MODEL_NAME"),
                    help="model name accepted in the request's model field (default: config.json model_name)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args(argv)
    kw = dict(backend=a.backend)
    if a.backend == "torch":
        kw["device"] = a.device
    else:
        kw["onnx_file"] = a.onnx_file
    engine = Vela2.from_pretrained(a.model, **kw)
    uvicorn.run(create_app(engine, a.name, os.environ.get("VELA2_API_KEY") or None), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
