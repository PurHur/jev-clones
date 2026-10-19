"""The Box question: a bounding box for a referring phrase in an image.

Request: {"type": "box", "instructions": "the coffee mug in `photo`"} (no criteria; backticks name a state key that
holds an image; a request with no image is a 400). Answer: {"type": "box", "box": {"top", "bottom", "left", "right"}
as fractions 0-1 of the image measured from the top-left corner, "confidence": 0-1, "present": 0-1}.

Three stages behind one answer, all on the same engine:
  1. grounding: one base-mode chat generation (temperature 0, thinking off) asking for the target's box in the
     configured convention (for example "qwen1000": Qwen's own bbox_2d integers 0..1000; "gemma_y1x1": Gemma's
     box_2d, y first). The JSON in the reply is parsed, clamped and ordered. A reply without a box is "no box" (the
     full image with confidence 0).
  2. calibrated confidence: NOT the model's self-stated number. A decision readout on the image with the box drawn:
     a yes/no question "does the red box tightly contain <target>?", read from the one-token readout like every
     other question, with its own temperature (serve_config box.temperature; 1.0 until fitted). It estimates
     P(IoU >= 0.5). With box.confidence "generation" (serve_config's choice), the confidence comes instead from the
     one grounding generation's own token probabilities, calibrated by box.generation_calibration, and no drawn-box
     readout is made.
  3. presence: a yes/no readout on the untouched image, "is <target> in the image?", returned as `present`, so a
     caller can gate the box.

The referring phrase is passed through verbatim (relational targets among identical objects, region targets), never
reduced to a class name; only the readout questions strip the backticked image name.

The runner is pure: `run()` takes callables for the generation and the readout, so the shim wires its own vLLM client
and the fake engines in tests wire theirs.
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
from collections.abc import Callable, Mapping
from typing import Final

GENERIC_PROMPT: Final = (
    "{instructions}\n\nFind it in the image and return exactly one bounding box. Reply with JSON only, no explanation.\n"
    'Format: {{"left": L, "top": T, "right": R, "bottom": B}} where L, R are fractions (0 to 1) of the image width and '
    "T, B fractions of the image height, measured from the top-left corner."
)
QWEN_PROMPT: Final = (
    "{instructions}\n\nFind it in the image and return exactly one bounding box. Reply with JSON only, no explanation.\n"
    'Format: {{"bbox_2d": [x1, y1, x2, y2]}} where the coordinates are integers from 0 to 1000, relative to the image '
    "width and height."
)
GEMMA_Y_PROMPT: Final = (
    "{instructions}\n\nFind it in the image and return exactly one bounding box. Reply with JSON only, no explanation.\n"
    'Format: {{"box_2d": [y1, x1, y2, x2]}} where the coordinates are integers from 0 to 1000, relative to the image '
    "height and width."
)
# the grounding reply's JSON schema per convention (vLLM structured outputs through the OpenAI response_format
# json_schema: a short constrained decode, valid by construction, ints 0..1000)
SCHEMAS: Final = {
    key: {
        "type": "object",
        "properties": {name: {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}, "minItems": 4, "maxItems": 4}},
        "required": [name],
        "additionalProperties": False,
    }
    for key, name in (("qwen1000", "bbox_2d"), ("gemma_y1x1", "box_2d"))
}
GUIDED_MAX_TOKENS: Final = 32  # '{"bbox_2d": [1000, 1000, 1000, 1000]}' is ~20 tokens
# conventions: "qwen1000" (Qwen's bbox_2d x-first), "gemma_y1x1" (box_2d y-first),
# "fractions" (named 0..1 edges)
PROMPTS: Final = {"qwen1000": QWEN_PROMPT, "gemma_y1x1": GEMMA_Y_PROMPT, "fractions": GENERIC_PROMPT}
# the grounding call's system line (the decision engine's own system prompt asks for a single letter, which is wrong
# for a free JSON reply)
SYSTEM: Final = "You are a vision grounding engine. You locate the target named in the request in the image and reply with its bounding box as JSON."
SYSTEM_BY_CONVENTION: Final = {k: SYSTEM for k in PROMPTS}  # one system line for every family today; per-convention if a profile needs its own
DEFAULT_CONVENTION: Final = "qwen1000"
TIGHT_QUESTION: Final = "Does the red box drawn on the image tightly contain {target}, with the whole of it inside and little else?"
PRESENT_QUESTION: Final = "Is {target} visible in the image?"
_JSON: Final = re.compile(r"\{.*\}", re.S)
_BACKTICK: Final = re.compile(r"`([^`]+)`")
FULL: Final = {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}


# box.mode (each coordinate is the expected value of its edge's predicted position):
# "generate" = the constrained generation above (works out of the box on an untrained model);
# "edges" = four readouts over EDGE_BINS position bins per edge, coordinate = the expected value (a trained model's
# path); "hybrid" = both boxes formed, each read for tightness, the tighter one answered with its probability.
MODES: Final = ("generate", "edges", "hybrid")
HYBRID_THRESHOLD: Final = 0.7  # hybrid's early exit: a generated box this tight is answered without the edge path
EDGE_BINS: Final = 100
EDGE_QUESTIONS: Final = {
    "left": "Where is the left edge of {target}? Answer with its position across the image width as a percentage from the left border (0 = at the left border, 99 = at the right border).",
    "right": "Where is the right edge of {target}? Answer with its position across the image width as a percentage from the left border (0 = at the left border, 99 = at the right border).",
    "top": "Where is the top edge of {target}? Answer with its position down the image height as a percentage from the top border (0 = at the top border, 99 = at the bottom border).",
    "bottom": "Where is the bottom edge of {target}? Answer with its position down the image height as a percentage from the top border (0 = at the top border, 99 = at the bottom border).",
}
EDGE_ORDER: Final = ("left", "right", "top", "bottom")
# the two-level scheme for shims with fewer single-token labels than bins (the gemma profiles' 26 letters): a coarse
# readout over K equal bands, then a fine readout over K sub-bands inside the most probable band; resolution 1/K^2
_AXIS: Final = {"left": ("across the image width", "left"), "right": ("across the image width", "left"),
                "top": ("down the image height", "top"), "bottom": ("down the image height", "top")}
COARSE_QUESTION: Final = ("Where is the {edge} edge of {target}? Answer with the band it lies in, counting {k} equal bands {axis} "
                          "from the {origin} border (0 = the first band at the {origin} border, {last} = the last band).")
FINE_QUESTION: Final = ("The {edge} edge of {target} lies within band {band} of {k} {axis}, from {lo}% to {hi}% of the image measured from "
                        "the {origin} border. Where is it inside that band? Answer with the sub-band it lies in, counting {k} equal "
                        "sub-bands from the band's {origin} side (0 = the first sub-band, {last} = the last sub-band).")
MIN_EXTENT: Final = 0.005  # an edges box thinner than this is "not found"


def edge_options(bins: int = EDGE_BINS) -> dict[str, str]:
    """The choice criteria of an edge readout: option names "0".."bins-1", each described as its percentage."""
    return {str(i): f"{i}%" for i in range(bins)}


def expected_position(probabilities: list[float], bins: int = EDGE_BINS) -> float:
    """The expected value of the edge over the bins (bin i covers [i/bins, (i+1)/bins), its centre (i+0.5)/bins)."""
    if len(probabilities) != bins:
        raise ValueError(f"edge readout returned {len(probabilities)} probabilities for {bins} bins")
    return round(float(sum(p * (i + 0.5) / bins for i, p in enumerate(probabilities))), 6)


def band_options(k: int, fine: bool = False) -> dict[str, str]:
    """The choice criteria of a two-level readout: names "0".."k-1" described as band / sub-band i."""
    return {str(i): f"{'sub-band' if fine else 'band'} {i}" for i in range(k)}


def two_level_position(read_edge: Callable[[str, str, int], tuple[list[float], int]], image: str, edge: str, target: str, k: int) -> tuple[float, int]:
    """(the edge's position, prompt tokens): a coarse readout over k bands, the fine readout inside the most probable
    band, position = band/k + E[fine]/k (the fine expected value scaled into the band)."""
    axis, origin = _AXIS[edge]
    coarse, used = read_edge(image, COARSE_QUESTION.format(edge=edge, target=target, k=k, axis=axis, origin=origin, last=k - 1), k)
    if len(coarse) != k:
        raise ValueError(f"coarse edge readout returned {len(coarse)} probabilities for {k} bands")
    band = max(range(k), key=lambda i: coarse[i])
    lo, hi = round(100 * band / k, 1), round(100 * (band + 1) / k, 1)
    fine, n = read_edge(image, FINE_QUESTION.format(edge=edge, target=target, band=band, k=k, axis=axis, lo=lo, hi=hi, origin=origin, last=k - 1), k)
    used += n
    if len(fine) != k:
        raise ValueError(f"fine edge readout returned {len(fine)} probabilities for {k} sub-bands")
    return round(band / k + expected_position(fine, k) / k, 6), used


def edges_box(positions: Mapping[str, float]) -> dict[str, float] | None:
    """The box from four expected edge positions: clamped, ordered; None when it has no extent."""
    box = {k: min(1.0, max(0.0, float(positions[k]))) for k in EDGE_ORDER}
    if box["left"] > box["right"]:
        box["left"], box["right"] = box["right"], box["left"]
    if box["top"] > box["bottom"]:
        box["top"], box["bottom"] = box["bottom"], box["top"]
    if box["right"] - box["left"] < MIN_EXTENT or box["bottom"] - box["top"] < MIN_EXTENT:
        return None
    return box


# box.confidence
# "generation": the ONE guided generation is sent with per-position logprobs; each coordinate is the expected value
# over its digit tokens' distributions (place value combined), its confidence the chosen leading digits' probability
# (renormalised over digit-only alternatives, sharpened by `temperature`); no drawn-image tightness prefill, and the
# presence readout only when box.presence is on. Cost: one image prefill + ~20 decode steps.
CONFIDENCE_SOURCES: Final = ("tightness", "generation")


def calibrate(p: float, a: float = 1.0, b: float = 0.0) -> float:
    """Platt calibration of a confidence: sigmoid(a * logit(p) + b); (1, 0) is the identity. The generation confidence
    (a product of four leading-digit probabilities) runs low, and a temperature alone cannot shift it; the fitted (a, b)
    come from serve_config box.generation_calibration."""
    p = min(1.0 - 1e-6, max(1e-6, float(p)))
    z = math.log(p / (1.0 - p))
    return round(1.0 / (1.0 + math.exp(-(a * z + b))), 6)
_DIGITS: Final = re.compile(r"\d+")


def _ascii_digits(s: str) -> bool:
    """ASCII 0-9 only: str.isdigit() is true for superscripts and other Unicode digits a tokenizer may offer as
    alternatives (vLLM top_logprobs returned "\u00b9"), which int() refuses."""
    return bool(s) and all("0" <= ch <= "9" for ch in s)


def _number_spans(text: str, convention: str) -> list[tuple[int, int]]:
    """The character spans of the four coordinates in the reply, in the reply's order (x1,y1,x2,y2 or y1,x1,y2,x2)."""
    m = _JSON.search(text or "")
    if not m:
        return []
    key = "box_2d" if "box_2d" in m.group(0) and "bbox_2d" not in m.group(0) else "bbox_2d"
    k = m.group(0).find(key)
    if k < 0:
        return []
    start = m.start() + k + len(key)  # after the key (whose own digit, "2d", is not a coordinate)
    spans = [(start + n.start(), start + n.end()) for n in _DIGITS.finditer(m.group(0)[k + len(key):])]
    return spans[:4] if len(spans) >= 4 else []


def parse_box_logprobs(tokens: list[tuple[str, float, list[tuple[str, float]]]], convention: str, temperature: float = 1.0,
                       coordinates: str = "mode") -> tuple[dict[str, float] | None, float]:
    """(the box, confidence) from the generation's tokens. coordinates="mode" (default): the box is the generated
    text's own numbers (parse_box: the argmax decode); the per-digit
    expected value ("expected") would move the box: digit distributions are
    not position distributions, their mass on neighbouring digits shifts a box by whole digit places. The confidence
    comes from the distributions either way. Tokens:
    [(token_text, logprob, [(alternative_text, logprob), ...]), ...] as vLLM returns them with logprobs/top_logprobs.
    For each coordinate, every token covering its digits contributes E[chunk] over its digit-only alternatives of the
    same length (the chosen token included), combined by place value; a token that is not a whole digit chunk counts as
    certain. Confidence = the product over the four coordinates of the chosen LEADING chunk's probability (renormalised
    over digit-only alternatives, p^(1/T) with T = `temperature`): the model's own certainty about where each edge starts."""
    text = "".join(t for t, _, _ in tokens)
    spans = _number_spans(text, convention)
    if len(spans) != 4:
        return None, 0.0
    offsets, pos = [], 0
    for t, _, _ in tokens:
        offsets.append((pos, pos + len(t)))
        pos += len(t)
    values: list[float] = []
    leading: list[float] = []
    for a, b in spans:
        chunks = []  # (expected value of the chunk, digit length, chosen prob)
        for (ta, tb), (t, lp, alts) in zip(offsets, tokens):
            if tb <= a or ta >= b:
                continue
            piece = text[max(ta, a):min(tb, b)]
            n = len(piece)
            if _ascii_digits(t) and len(t) == n:  # a whole digit chunk: its alternatives are the same-length digit strings
                cands = {t: math.exp(lp / temperature)}
                for alt, alp in alts:
                    if _ascii_digits(alt) and len(alt) == n and alt not in cands:
                        cands[alt] = math.exp(alp / temperature)
                z = sum(cands.values())
                exp_v = sum(float(int(d)) * w for d, w in cands.items()) / z
                chunks.append((exp_v, n, cands[t] / z))
            else:
                chunks.append((float(int(piece)), n, 1.0))
        value = 0.0
        for i, (exp_v, n, _) in enumerate(chunks):
            value += exp_v * (10 ** sum(m for _, m, _ in chunks[i + 1:]))
        values.append(value)
        leading.append(chunks[0][2] if chunks else 1.0)
    conf = 1.0
    for p_ in leading:
        conf *= p_
    if coordinates == "mode":
        return parse_box(text, convention), round(conf, 6)
    scale = 1000.0 if max(values) > 1.0 else 1.0
    if convention == "gemma_y1x1":
        y1, x1, y2, x2 = values
    else:
        x1, y1, x2, y2 = values
    box = {"left": x1 / scale, "top": y1 / scale, "right": x2 / scale, "bottom": y2 / scale}
    box = {k: min(1.0, max(0.0, v)) for k, v in box.items()}
    if box["left"] > box["right"]:
        box["left"], box["right"] = box["right"], box["left"]
    if box["top"] > box["bottom"]:
        box["top"], box["bottom"] = box["bottom"], box["top"]
    if box["right"] - box["left"] <= 0 or box["bottom"] - box["top"] <= 0:
        return None, 0.0
    return {k: round(v, 6) for k, v in box.items()}, round(conf, 6)


class BoxError(ValueError):
    """A 400: the box question cannot be served as asked (no image in the request, bad instructions)."""


def target_text(instructions: object) -> str:
    """The target as plain text for the readout questions: the instructions with the image's backtick name removed
    ("the coffee mug in `photo`" -> "the coffee mug"); structured instructions are rendered as JSON."""
    text = instructions if isinstance(instructions, str) else json.dumps(instructions, ensure_ascii=False)
    text = re.sub(r"\s+in\s+`[^`]+`\s*$", "", text.strip())
    return _BACKTICK.sub(lambda m: m.group(1), text).strip() or "the target"


def parse_box(text: str, convention: str) -> dict[str, float] | None:
    """The box in a generation reply, as ordered, clamped fractions; None when there is none."""
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    try:
        if "bbox_2d" in obj or "box_2d" in obj:
            vals = [float(v) for v in (obj["bbox_2d"] if "bbox_2d" in obj else obj["box_2d"])]
            # box_2d is the y-first order [y1, x1, y2, x2]; bbox_2d is Qwen's x-first [x1, y1, x2, y2]
            x1, y1, x2, y2 = (vals[1], vals[0], vals[3], vals[2]) if "bbox_2d" not in obj else vals
            scale = 1000.0 if max(x1, y1, x2, y2) > 1.0 else 1.0  # integers 0..1000, or fractions whatever was asked
            box = {"left": x1 / scale, "top": y1 / scale, "right": x2 / scale, "bottom": y2 / scale}
        else:
            box = {k: float(obj[k]) for k in ("left", "top", "right", "bottom")}
    except (KeyError, TypeError, ValueError):
        return None
    box = {k: min(1.0, max(0.0, v)) for k, v in box.items()}
    if box["left"] > box["right"]:
        box["left"], box["right"] = box["right"], box["left"]
    if box["top"] > box["bottom"]:
        box["top"], box["bottom"] = box["bottom"], box["top"]
    if box["right"] - box["left"] <= 0 or box["bottom"] - box["top"] <= 0:
        return None
    return box


def draw_box(data_uri: str, box: Mapping[str, float]) -> str | None:
    """The image with the box drawn in red (a 3 px line scaled to the image), as a PNG data URI; None when PIL cannot
    read it (the tightness readout is then skipped: a confidence about a box the model never saw is no confidence)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover - Pillow is a declared dependency
        return None
    header, _, payload = data_uri.partition(",")
    try:
        image = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    except Exception:  # noqa: BLE001 - not an image PIL reads
        return None
    w, h = image.size
    draw = ImageDraw.Draw(image)
    width = max(2, round(min(w, h) / 200))
    draw.rectangle([box["left"] * w, box["top"] * h, box["right"] * w, box["bottom"] * h], outline=(255, 0, 0), width=width)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def image_for(question: Mapping[str, object], state: object, images: list[str]) -> int:
    """Which of the request's images the question targets: the one a backtick name in the instructions points at
    (a state key holding an `[image N]` placeholder), else the first image."""
    ins = question.get("instructions")
    if isinstance(ins, str) and isinstance(state, Mapping):
        for name in _BACKTICK.findall(ins):
            value = state.get(name)
            m = re.fullmatch(r"\[image (\d+)\]", str(value).strip()) if value is not None else None
            if m and 1 <= int(m.group(1)) <= len(images):
                return int(m.group(1)) - 1
    return 0


def run(
    question: Mapping[str, object],
    state: object,
    images: list[str],
    *,
    generate: Callable[[str, str], tuple[str, int]],
    read_noul: Callable[[str, str], tuple[float, int]],
    convention: str = DEFAULT_CONVENTION,
    mode: str = "generate",
    read_edge: Callable[[str, str, int], tuple[list[float], int]] | None = None,
    bins: int = EDGE_BINS,
    two_level: int | None = None,
    submit: Callable[..., object] | None = None,
    hybrid_threshold: float = HYBRID_THRESHOLD,
    confidence: str = "tightness",
    presence: bool = True,
    generation_temperature: float = 1.0,
    generation_calibration: tuple[float, float] = (1.0, 0.0),
) -> tuple[dict[str, object], int]:
    """(the answer, input tokens used). `generate(image_uri, prompt) -> (text, prompt_tokens)`;
    `read_noul(image_uri, question_text) -> (p_yes, prompt_tokens)`;
    `read_edge(image_uri, question_text, n) -> (probabilities over the n options, prompt_tokens)` (modes edges/hybrid):
    flat over `bins` options, or, with `two_level=K`, a coarse + fine readout over K bands each (the shim's option
    count, e.g. 26) — the caller's shim renders the options in the order given, never permuted (ordinal labels).
    `submit(fn, *args) -> future`: with it, the independent calls of a box (the grounding generation, the four edge
    readouts, the presence readout) run CONCURRENTLY on the same engine, sharing the image's prefix cache (an edges box
    should cost about one or two yes/no latencies, not six); only the tightness readouts
    wait for their boxes. Without it, everything runs in order (tests, package shims without a pool)."""
    if not images:
        raise BoxError("a box question needs an image in the request (an image in the state, an asset, or images)")
    if convention not in PROMPTS:
        raise BoxError(f"unknown box convention {convention!r}")
    if mode not in MODES:
        raise BoxError(f"unknown box mode {mode!r}")
    if confidence not in CONFIDENCE_SOURCES:
        raise BoxError(f"unknown box confidence source {confidence!r}")
    if mode != "generate" and read_edge is None:
        raise BoxError(f"box mode {mode!r} needs an edge readout")
    target: Final = target_text(question.get("instructions"))
    image: Final = images[image_for(question, state, images)]

    class _Now:  # the sequential stand-in for a future
        def __init__(self, fn, *args):
            self.value = fn(*args)

        def result(self):
            return self.value

    start = submit if submit is not None else _Now
    used = 0
    jobs: dict[str, object] = {}
    tight_q: Final = TIGHT_QUESTION.format(target=target)

    def start_edges() -> None:
        for edge in EDGE_ORDER:
            if two_level:
                jobs[edge] = start(two_level_position, read_edge, image, edge, target, two_level)
            else:
                jobs[edge] = start(read_edge, image, EDGE_QUESTIONS[edge].format(target=target), bins)

    if mode in ("generate", "hybrid"):
        jobs["generate"] = start(generate, image, PROMPTS[convention].format(instructions=str(question.get("instructions") or target)))
    if mode == "edges":
        start_edges()
    if presence:
        jobs["present"] = start(read_noul, image, PRESENT_QUESTION.format(target=target))
    candidates: list[dict[str, float]] = []
    if "generate" in jobs:
        got = jobs["generate"].result()  # type: ignore[attr-defined]
        text, n = got[0], got[1]
        used += n
        if confidence == "generation" and mode == "generate":
            # the generation's own distributions: expected-value box and its confidence, no tightness prefill
            lp_tokens = got[2] if len(got) > 2 else None
            box, conf = parse_box_logprobs(lp_tokens, convention, generation_temperature) if lp_tokens else (parse_box(text, convention), 0.0)
            if lp_tokens and box is not None:
                conf = calibrate(conf, *generation_calibration)  # the fitted Platt map (serve_config box.generation_calibration)
            present_v = jobs["present"].result() if "present" in jobs else None  # type: ignore[attr-defined]
            if present_v is not None:
                used += present_v[1]
            answer: dict[str, object] = {"type": "box", "box": box if box is not None else dict(FULL), "confidence": conf if box is not None else 0.0,
                                         "found": box is not None}
            # `present` keeps its place and meaning in the answer (never dropped): the presence
            # readout when box.presence is on; with it off, the generation's own evidence that the target was located:
            # the calibrated box confidence (a located box with a confident grounding), 0.0 when no box came back
            answer["present"] = present_v[0] if present_v is not None else (answer["confidence"] if box is not None else 0.0)
            return answer, used
        box = parse_box(text, convention)
        if box is not None:
            candidates.append(box)
    if mode == "hybrid":
        # the early exit: the generated box is read for tightness first; when it is tight
        # enough the edge path never runs, so the common case costs what generate costs
        drawn = draw_box(image, candidates[0]) if candidates else None
        if drawn is not None:
            tight, n = read_noul(drawn, tight_q)
            used += n
            if tight >= hybrid_threshold:
                out_h: dict[str, object] = {"type": "box", "box": candidates[0], "confidence": tight, "found": True}
                if "present" in jobs:
                    present, n = jobs["present"].result()  # type: ignore[attr-defined]
                    used += n
                    out_h["present"] = present
                return out_h, used
            jobs["tight:0"] = _Now(lambda: (tight, 0))  # already read: reused below, not read again
        start_edges()
    if mode in ("edges", "hybrid"):
        positions = {}
        for edge in EDGE_ORDER:
            value, n = jobs[edge].result()  # type: ignore[attr-defined]
            positions[edge] = value if two_level else expected_position(value, bins)
            used += n
        box = edges_box(positions)
        if box is not None:
            candidates.append(box)
    present = None
    if "present" in jobs:
        present, n = jobs["present"].result()  # type: ignore[attr-defined]
        used += n
    if not candidates:
        out: dict[str, object] = {"type": "box", "box": dict(FULL), "confidence": 0.0, "found": False}
        if present is not None:
            out["present"] = present
        return out, used
    tight_jobs = []  # hybrid: both candidates read at once; the tighter by its own readout wins
    for i, box in enumerate(candidates):
        if f"tight:{i}" in jobs:  # hybrid's early-exit readout, already taken
            tight_jobs.append((box, jobs[f"tight:{i}"]))
            continue
        drawn = draw_box(image, box)
        tight_jobs.append((box, None if drawn is None else start(read_noul, drawn, tight_q)))
    best: dict[str, float] | None = None
    best_tight = -1.0
    for box, job in tight_jobs:
        if job is None:  # the box could not be drawn: found, but no readout to back a confidence
            tight = 0.0
        else:
            tight, n = job.result()  # type: ignore[attr-defined]
            used += n
        if tight > best_tight:
            best, best_tight = box, tight
    final: dict[str, object] = {"type": "box", "box": best, "confidence": max(0.0, best_tight), "found": True}
    if present is not None:
        final["present"] = present
    return final, used
