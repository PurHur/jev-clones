"""HTTP API and browser demo for calibrated decisions."""

import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from config import SpaceConfig, resolve_checkpoint, resolve_device
from inference import DecisionEngine, MODEL_NAME, DEFAULT_TEMPERATURE, IMAGE_MAX_PIXELS
from pipeline import DecisionPipeline
from examples import examples
from uploads import MAX_UPLOAD_BYTES, MAX_UPLOAD_TOTAL, materialize_images

STATIC = Path(__file__).with_name("static")


def create_app(pipeline=None):
    @asynccontextmanager
    async def lifespan(app):
        if pipeline is None:
            config = SpaceConfig.from_env()
            device = resolve_device(config.device)
            print(f"Loading Intern-Decision-0.8B on {device} ({config.dtype}).", flush=True)
            if device == "cpu":
                print("CPU inference can be slow. Select GPU hardware in Space Settings for interactive use.", flush=True)
            try:
                checkpoint = resolve_checkpoint(config)
                engine = DecisionEngine(
                    checkpoint, temperature=DEFAULT_TEMPERATURE, backend="hf",
                    max_length=config.max_length, device=device,
                    dtype=config.dtype, attn_implementation=config.attn_implementation,
                )
                app.state.pipeline = DecisionPipeline(engine)
                yield
            finally:
                app.state.pipeline = None
        else:
            app.state.pipeline = pipeline
            yield

    app = FastAPI(title="Intern-Decision", lifespan=lifespan)
    app.state.pipeline = pipeline

    def current():
        if app.state.pipeline is None:
            raise HTTPException(503, "Inference is not ready.")
        return app.state.pipeline

    @app.get("/health")
    def health():
        return {"status": "ok" if app.state.pipeline else "unavailable", "model_loaded": app.state.pipeline is not None}

    @app.get("/demo-meta")
    def metadata():
        current()
        return {
            "model": MODEL_NAME,
            "model_repo": "InternLM/Intern-Decision-0.8B",
            "calibration_temperature": DEFAULT_TEMPERATURE,
            "image_preprocessing": {
                "min_pixels": 1,
                "max_pixels": IMAGE_MAX_PIXELS,
                "upscale_small_images": False,
            },
            "uploads": {
                "enabled": True,
                "max_images": 8,
                "max_bytes": MAX_UPLOAD_BYTES,
                "max_total_bytes": MAX_UPLOAD_TOTAL,
            },
        }

    @app.post("/v1/jev")
    @app.post("/v1/decisions")
    def predict(request: dict):
        # Validate the public upload path before any local filesystem access.
        upload_dir = None
        try:
            request, upload_dir = materialize_images(request)
            result = current().predict(request, allow_images=bool(upload_dir))
            return result
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/demo.js")
    def javascript():
        return FileResponse(STATIC / "demo.js", media_type="application/javascript")

    @app.get("/examples")
    def example_requests():
        return examples()

    from limits import RequestSizeLimit
    app.add_middleware(RequestSizeLimit, max_bytes=48 * 1024 * 1024)
    return app


app = create_app()
