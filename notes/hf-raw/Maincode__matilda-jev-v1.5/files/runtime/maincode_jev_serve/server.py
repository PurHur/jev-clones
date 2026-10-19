"""Serve a Maincode decision checkpoint over `POST /v1/systemone`.

Endpoints: POST /v1/systemone (decisions), GET /health, GET /v1/models, GET / (browser playground).
Scoring is maincode_jev_serve.decide (same prompt, batching and stored temperature as the benchmarked
engine). Requests that do not fit are refused with HTTP 413 "maximum context length", never truncated.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import RequestResponseEndpoint

from maincode_jev_serve import config
from maincode_jev_serve.types import Answer, DecisionResponse, JSONValue, Question as DecisionQuestion

if TYPE_CHECKING:
    from maincode_jev_serve.model import DecisionModel

type Content = str | dict[str, JsonValue] | list[JsonValue]
ALIAS = "maincode-jev-latest"


@dataclass
class Service:
    settings: config.Settings | None = None
    model: DecisionModel | None = None
    release_date: str = ""
    warmup_ms: list[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def names(self) -> set[str]:
        return {self.settings.model_name, ALIAS} if self.settings else {ALIAS}


service = Service()


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: Content | None = None


class Choice(Question):
    type: Literal["choice"]
    criteria: dict[str, Content | None] = Field(min_length=1, max_length=255)


class Score(Question):
    type: Literal["score"]
    criteria: list[Content] = Field(min_length=2, max_length=10)


class Noul(Question):
    type: Literal["noul"]
    criteria: dict[Literal["true", "false"], Content | None] | None = None


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    state: Content
    questions: dict[str, Annotated[Choice | Score | Noul, Field(discriminator="type")]] = Field(min_length=1)
    images: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("model")
    @classmethod
    def known_model(cls, value: str) -> str:
        if value not in service.names:
            raise ValueError(f"Unknown model. Use {' or '.join(sorted(service.names))}.")
        return value

    @field_validator("images")
    @classmethod
    def valid_images(cls, values: list[str]) -> list[str]:
        for value in values:
            if len(value) > 12_000_000:
                raise ValueError("Each image must be at most 8 MB before base64 encoding.")
            header, separator, encoded = value.partition(",")
            if not separator or header not in {"data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"}:
                raise ValueError("Images must be base64 PNG, JPEG, or WebP data URLs.")
            try:
                content = base64.b64decode(encoded, validate=True)
                if len(content) > 8_000_000:
                    raise ValueError("Each image must be at most 8 MB.")
                with Image.open(BytesIO(content)) as image:
                    if image.width * image.height > 16_000_000:
                        raise ValueError("Each image must have at most 16 million pixels.")
                    if image.format not in {"PNG", "JPEG", "WEBP"}:
                        raise ValueError("Unsupported image format.")
                    image.verify()
            except (binascii.Error, OSError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                raise ValueError("Invalid image data.") from error
        return values


def authenticate(authorization: str | None = Header(default=None)) -> None:
    key = service.settings.api_key if service.settings else None
    if key and not hmac.compare_digest((authorization or "").encode(), f"Bearer {key}".encode()):
        raise HTTPException(401, "Missing or invalid API key.", headers={"WWW-Authenticate": "Bearer"})


def predict(model: DecisionModel, state: Content, questions: dict[str, DecisionQuestion], images: list[str]) -> DecisionResponse:
    from maincode_jev_serve.decide import decide
    from maincode_jev_serve.model import answer

    settings = cast(config.Settings, service.settings)
    distributions, input_tokens = decide(model, state, questions, temperature=model.temperature, max_tokens=settings.max_context,
                                         token_budget=settings.token_budget, batch_size=settings.batch_size, images=images)
    answers: dict[str, Answer] = {key: answer(questions[key], values) for key, values in distributions.items()}
    return {"model": settings.model_name, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}


def warm_up(model: DecisionModel) -> list[float]:
    """Compile GPU kernels for short, medium and long prompts (sized to the context limit) before serving."""
    from maincode_jev_serve.decide import CapacityError

    settings = cast(config.Settings, service.settings)
    timings = []
    for words in (20, 600, 2500):
        words = min(words, max(1, settings.max_context // 4))  # ~2 tokens per "warm-up" word, plus the prompt template
        started = time.perf_counter()
        try:
            predict(model, "warm-up " * words, {"q": {"type": "choice", "instructions": "Pick one.", "criteria": {"a": None, "b": None}},
                                                "n": {"type": "noul", "instructions": "Is it true?"}}, [])
        except CapacityError as error:  # warm-up must never stop the server
            print(f"warm-up request of {words} words skipped: {error}", flush=True)
            continue
        timings.append(round((time.perf_counter() - started) * 1000, 1))
    return timings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from maincode_jev_serve.model import DecisionModel

    settings = service.settings or config.load()
    service.settings = settings
    if not (Path(settings.checkpoint) / "decision_config.json").is_file():
        raise RuntimeError(f"{settings.checkpoint} is not a decision checkpoint (no decision_config.json)")
    print(f"loading {settings.checkpoint} as {settings.model_name} (context {settings.max_context}, batch budget "
          f"{settings.token_budget}, auth {'on' if settings.api_key else 'OFF'})", flush=True)
    service.model = await run_in_threadpool(DecisionModel, checkpoint=settings.checkpoint, device=settings.device)
    if settings.warmup:
        service.warmup_ms = await run_in_threadpool(warm_up, service.model)
        print(f"warm-up done: {service.warmup_ms} ms", flush=True)
    modified = (Path(settings.checkpoint) / "decision_config.json").stat().st_mtime
    service.release_date = datetime.fromtimestamp(modified, timezone.utc).date().isoformat()
    try:
        yield
    finally:
        service.model = None


app = FastAPI(title="maincode-jev", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_metadata(request: Request, call_next: RequestResponseEndpoint) -> Response:
    started, identifier = time.perf_counter(), uuid.uuid4().hex
    response = await call_next(request)
    response.headers["x-request-id"] = identifier
    response.headers["server-timing"] = f"total;dur={(time.perf_counter() - started) * 1000:.1f}"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": [
        {"loc": item["loc"], "msg": item["msg"], "type": item["type"]} for item in error.errors()]})


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def playground() -> str:
    return Path(__file__).with_name("playground.html").read_text()


@app.get("/health", response_model=None)
def health() -> dict[str, JSONValue]:
    model, settings = service.model, service.settings
    if settings is None:
        return {"status": "loading"}
    device = None
    if model is not None:
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"status": "ready" if model is not None else "loading", "model": settings.model_name, "alias": ALIAS,
            "checkpoint": str(Path(settings.checkpoint).resolve()), "temperature": model.temperature if model else None,
            "max_context_tokens": settings.max_context, "device": device, "warmup_ms": cast(JSONValue, service.warmup_ms),
            "authentication": bool(settings.api_key), "modalities": ["text", "image"]}


@app.get("/v1/models", dependencies=[Depends(authenticate)], response_model=None)
def models() -> dict[str, JSONValue]:
    return {"models": [{"name": name, "description": "Maincode one-pass typed decisions (text and image).",
                        "release_date": service.release_date} for name in sorted(service.names)]}


@app.post("/v1/systemone", dependencies=[Depends(authenticate)], response_model=None)
async def system_one(body: EvaluationRequest) -> DecisionResponse:
    from maincode_jev_serve.decide import CapacityError

    model, settings = service.model, service.settings
    if model is None or settings is None:
        raise HTTPException(503, "The model is not ready.")
    if not await run_in_threadpool(service.lock.acquire, True, settings.queue_seconds):
        raise HTTPException(529, "The model is busy. Retry shortly.", headers={"Retry-After": "1"})
    try:
        questions = {key: cast(DecisionQuestion, question.model_dump(exclude_none=True)) for key, question in body.questions.items()}
        return await run_in_threadpool(predict, model, body.state, questions, list(body.images))
    except CapacityError as error:
        raise HTTPException(413, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    finally:
        service.lock.release()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve a Maincode decision checkpoint. Flags override MJ_* env vars (see serve.env).")
    parser.add_argument("--checkpoint", help="Decision checkpoint directory (MJ_CHECKPOINT)")
    parser.add_argument("--host", help="Bind address (MJ_HOST, default 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Port (MJ_PORT, default 8000)")
    parser.add_argument("--model-name", help="Served model name (MJ_MODEL_NAME, default: checkpoint directory name)")
    parser.add_argument("--max-context", type=int, help="Max prompt tokens per question (MJ_MAX_CONTEXT, default: sized to GPU)")
    parser.add_argument("--token-budget", type=int, help="Max padded tokens per forward batch (MJ_TOKEN_BUDGET)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip kernel warm-up at startup (MJ_WARMUP=0)")
    args = parser.parse_args()
    for flag, variable in (("checkpoint", "MJ_CHECKPOINT"), ("host", "MJ_HOST"), ("port", "MJ_PORT"), ("model_name", "MJ_MODEL_NAME"),
                           ("max_context", "MJ_MAX_CONTEXT"), ("token_budget", "MJ_TOKEN_BUDGET")):
        if getattr(args, flag) is not None:
            os.environ[variable] = str(getattr(args, flag))
    if args.no_warmup:
        os.environ["MJ_WARMUP"] = "0"
    service.settings = config.load()
    uvicorn.run(app, host=service.settings.host, port=service.settings.port)


if __name__ == "__main__":
    main()
