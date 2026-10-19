"""Validate requests and serialize model inference."""
import copy
import math
import secrets
import threading
import time
from inference import ANSWER_SYMBOLS, compile_row, MODEL_NAME

def validate_request(request, allow_images=False):
    if not isinstance(request, dict) or "state" not in request:
        raise ValueError("Supply a JSON object containing state and questions.")
    images = request.get("images") or []
    if images and not allow_images:
        raise ValueError("Pass image bytes through the HTTP upload interface.")
    if images:
        if not isinstance(images, list) or not 1 <= len(images) <= 8:
            raise ValueError("Upload 1–8 images.")
        if not all(isinstance(image, str) and image.startswith("/") for image in images):
            raise ValueError("Uploaded images must be server-managed files.")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 16:
        raise ValueError("Supply 1–16 questions.")
    clean = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Question names must be nonempty strings.")
        if not isinstance(question, dict) or question.get("type") not in {"choice", "score", "noul"}:
            raise ValueError("Question type must be choice, score, or noul.")
        kind = question["type"]
        criteria = question.get("criteria")
        if kind == "choice" and not isinstance(criteria, dict):
            raise ValueError("Choice criteria must be an object.")
        if kind == "score":
            if not isinstance(criteria, (list, dict)):
                raise ValueError("Score criteria must be a list or object.")
            if isinstance(criteria, dict):
                try:
                    if not all(math.isfinite(float(key)) for key in criteria):
                        raise ValueError
                except (ValueError, TypeError) as exc:
                    raise ValueError("Score option keys must be finite numbers.") from exc
        if kind != "noul" and not 1 <= len(criteria) <= len(ANSWER_SYMBOLS):
            raise ValueError(f"Supply 1–{len(ANSWER_SYMBOLS)} options.")
        clean[name] = {
            key: copy.deepcopy(question[key]) for key in ("type", "instructions", "criteria") if key in question
        }
    row = {"state": copy.deepcopy(request["state"]), "questions": clean}
    if images:
        row["images"] = list(images)
    compile_row(row)
    return row


class DecisionPipeline:
    def __init__(self, engine):
        self.engine = engine
        self._model_lock = threading.Lock()

    def predict(self, request, allow_images=False):
        started = time.perf_counter()
        row = validate_request(request, allow_images=allow_images)
        with self._model_lock:
            queued_ms = (time.perf_counter() - started) * 1000
            result = self.engine.predict(row)
        result.update(model=MODEL_NAME, request_id=secrets.token_hex(16))
        result["usage"]["decision_count"] = len(result["answers"])
        result.setdefault("timing", {}).update(
            queue_ms=round(queued_ms, 2),
            server_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return result
