# SPDX-License-Identifier: Apache-2.0
"""Speech to Decisions with Granite: one model sizes up what you're saying, while you say it.

Speak into the microphone. One model, Granite Switch 4.1 8B on one Inference
Endpoint, does all of it:

- its built-in **speech recognition** turns the audio into text;
- the **base model** says what the speaker is talking about, in 1 to 5 words;
- the **guardian-core** adapter screens the words for harm, profanity and
  violence, one call per category;
- the **decision** adapter answers each yes/no question on the page.

Every safety category and every question comes back as a *noul*: the
probability that the answer is yes.

The flow
--------
1. The page (static/, a hand-built page with no framework) captures the
   microphone at 16 kHz and posts it to /api/chunk every half second. Each
   chunk is appended to the session's open *segment* of audio.
2. Speech recognition is a cascade inside the endpoint: an `input_audio` part
   in a chat message is transcribed by a small Granite Speech CTC model, and
   the transcript is spliced into the prompt as ordinary text. It takes a
   whole clip, with no streaming mode, so about twice a second the open segment
   is sent again in full, and the transcript read back from the prompt
   replaces the segment's *live* text (shown in grey).
3. When the speaker pauses, the segment's text is *committed* to the
   transcript and a new segment starts, so no request ever carries more than
   MAX_SEGMENT_S of audio. A speaker who doesn't pause is cut at the quietest
   moment of the segment's last few seconds.
4. Each time the text changes, its last RECENT_WORDS words go back to Granite
   (after QUIET_AFTER_S of silence the judgments are wiped, and the next ones
   are about the words said after the silence):
   - The decisions: 3 guardian-core calls and one decision call
     per question, all sent at once. Both adapters are aLoRAs, and every
     call (the label's too) opens with the same user turn holding the words,
     "State:" and then the words, as the decision adapter's model card writes
     it, so that turn is base-model KV the endpoint computes once and reuses.
   - The label: one base-model chat completion, on its own thread so it never
     holds up the transcript or the decisions.

The transcription and the decisions run on a background thread per session,
so the microphone stream is never held up. Every /api/chunk reply carries the
session's latest results, and the page redraws from it.

The API
-------
GET  /api/config    what the page needs to draw itself
POST /api/session   set the yes/no questions (and make a session if needed)
POST /api/chunk     a chunk of 16-bit mono PCM; ?sid= and ?rate=
POST /api/stop      recording stopped: finish with the audio that's left
POST /api/clear     drop the session and start a new one
Every POST returns the session's state as JSON. Everything else under / is
served from static/.
"""

import base64
import io
import math
import os
import threading
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Everything is Granite Switch 4.1 8B on a Hugging Face Inference Endpoint
# (vLLM, OpenAI-compatible), reached with an HF token. The URL ends in /v1.
GRANITE_URL = (os.environ.get("GRANITE_ENDPOINT_URL") or os.environ.get("HF_ENDPOINT_URL_SWITCH_8B") or "").rstrip("/")
GRANITE_NAME = "Granite Switch 4.1 8B"
# The endpoint's served model name, read from /models when this isn't set.
GRANITE_MODEL_ID = os.environ.get("GRANITE_MODEL_ID", "")
# Speech recognition is the checkpoint's own: its config leaves asr_model_id
# unset, which means the default, granite-speech-5.0-470m-turboctc, a 470M
# English CTC encoder. It writes lowercase words with no punctuation.
# What the page calls each part of the model.
LABELS = {
    "stt": "Granite Switch: ASR Encoder",
    "describe": "Granite Switch: Base Model",
    "safety": "Granite Switch: Guardian Adapter",
    "decide": "Granite Switch: Decisions Adapter",
}
# The endpoint scales to zero, and the first request after that waits minutes.
# Opening the page wakes it, at most this often.
WAKE_EVERY_S = 60.0

# The label: what the speaker is talking about, from the base model (no adapter).
DESCRIBE_MAX_TOKENS = 16
DESCRIBE_MAX_WORDS = 5
# Up to this many label requests run at once, the newest answer wins, and a
# request that takes this long is given up on.
DESCRIBE_IN_FLIGHT = 3
DESCRIBE_TIMEOUT_S = 5.0
# It follows the words, in a second user turn, so the words' turn is the same
# prefix every decision opens with and its KV is shared with them.
DESCRIBE_PROMPT = (
    "The message above is live speech, transcribed, and may be cut off mid-sentence. "
    "In 1 to 5 words, what is the speaker talking about? Reply with the topic as a short noun "
    "phrase taken from what they said. No punctuation, no quotes, nothing else."
)

# The yes/no questions: the decision adapter, an aLoRA, called as its model
# card shows. A user turn holding the state, then a user turn opening with the
# `### DECISION` marker (where the chat template, given adapter_name, puts the
# adapter's control token) with the question, the options as "A: ...",
# "B: ..." and DECISION_REPLY. The answer is the first assistant token, read
# the way the adapter is trained to answer: a softmax over the label tokens
# alone. So a request names those tokens in logprob_token_ids (a vLLM
# extension), which returns their logprobs however unlikely they are.
DECISION_ADAPTER = "decision"
DECISION_INVOCATION = "### DECISION"
DECISION_REPLY = "Reply with exactly one option label."
DECISION_OPTIONS = ("Yes", "No")
DECISION_LABELS = tuple(chr(ord("A") + i) for i in range(len(DECISION_OPTIONS)))
# guardian-core's score word can be spelled several ways ("yes" and " yes"),
# so ask for a window wide enough to see every spelling.
TOP_LOGPROBS = 20

# The safety check: guardian-core, an aLoRA that activates at `<guardian>`.
# Its io.yaml gives the envelope (the instruction below) and the output, a JSON
# object {"score": "yes"|"no"}; the noul is P("yes") at the score token.
GUARDIAN_ADAPTER = "guardian-core"
GUARDIAN_INSTRUCTION = (
    "<guardian>As a judge agent, your role is to help assess whether the provided text meets the "
    "given judging criteria, utilizing all available information, including conversations, "
    "documents, and tools.\n\n### Criteria: {criteria}\n\n### Scoring Schema: {scoring_schema}"
)
# The words are in a user turn, so the judge is asked about the user's text.
GUARDIAN_SCORING = "If the last user's text meets the criteria, return 'yes'; otherwise, return 'no'."
GUARDIAN_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "guardian",
        "schema": {
            "type": "object",
            "properties": {"score": {"type": "string", "enum": ["yes", "no"]}},
            "required": ["score"],
            "additionalProperties": False,
        },
    },
}
GUARDIAN_MAX_TOKENS = 15

# The chat template's markers around a user turn, which the transcript sits between.
USER_TURN = "user<|end_of_role|>"
END_OF_TURN = "<|end_of_text|>"

SAMPLE_RATE = 16000  # what the audio is resampled to before it's sent
STREAM_EVERY_S = 0.5  # how often the page sends a chunk
MAX_CHUNK_BYTES = 2_000_000  # a chunk is about 16 kB; this is far above any real one
STATIC_DIR = Path(__file__).parent / "static"
# Transcribe again once this much new audio has arrived: a pass every half
# second, as often as the page sends a chunk, so a word that tips a question
# over 0.5 is judged within a chunk of being said. A pass is about 0.35 s of
# endpoint time (transcription, then every judgment at once), so this keeps
# one in flight nearly all the time while someone talks.
MIN_NEW_AUDIO_S = float(os.environ.get("MIN_NEW_AUDIO_S", "0.5"))
# A segment is committed at a pause once it's at least MIN_COMMIT_S long, and
# cut at its quietest recent moment once it reaches MAX_SEGMENT_S.
MIN_COMMIT_S = 4.0
MAX_SEGMENT_S = 15.0
PAUSE_S = 0.7
# After this many seconds with no speech the board is wiped, and it shows no
# judgments until the speaker starts again. The next judgments are about what's
# said after the silence only, not the words before it. Silence is counted in
# 100 ms frames, so it's measured from the last loud frame, not the last chunk.
QUIET_AFTER_S = float(os.environ.get("QUIET_AFTER_S", "0.5"))
# A 100 ms frame louder than this (RMS, as a fraction of full scale) counts as
# speech. Audio with no such frame isn't transcribed at all.
VOICE_RMS = float(os.environ.get("VOICE_RMS", "0.01"))
FRAME = SAMPLE_RATE // 10

# The model judges what's being said right now: the last RECENT_WORDS words of
# the transcript, so the board follows the speaker from one subject to the next.
RECENT_WORDS = 25
# A pass judges the words only once there are this many: the first pass of a
# phrase often hears just its first word, or mishears a fragment ("as"), and
# judging that flashes a wrong label up. A shorter phrase is judged when the
# silence after it falls (judge_owed).
MIN_WORDS = 2
# How long the late answers about a phrase the silence overtook stay up.
LATE_SHOW_S = 3.0
MAX_NOULS = 50
SESSION_TTL_S = 30 * 60

# The safety check: guardian-core answers one noul per category about the same
# words. key -> (name on the page, the criteria). The criteria are the Granite
# Guardian definitions, as Mellea's CRITERIA_BANK has them. A category above
# SAFETY_THRESHOLD flags the text.
SAFETY = {
    "harm": (
        "Harm",
        "Texts deemed universally harmful are those that pose harm across all contexts and audiences.",
    ),
    "profanity": (
        "Profanity",
        (
            "Profanity refers to the use of language deemed offensive or socially unacceptable in a "
            "given context. This primarily includes curse words, swear words, and crude expressions, "
            "though it typically excludes slurs or derogatory terms specifically targeting certain groups."
        ),
    ),
    "violence": (
        "Violence",
        (
            "Violence refers to content that promotes or depicts physical harm to individuals or "
            "groups. This includes depictions of assault, self-harm, or the creation of a hostile "
            "environment through threats or intimidation."
        ),
    ),
}
SAFETY_THRESHOLD = 0.5

DEFAULT_NOULS = [
    "Is the speaker asking a question?",
    "Is the speaker making a decision?",
    "Is the speaker mad?",
    "Does the speaker name a company, product or brand?",
    "Does the speaker mention money or a price?",
    "Does the speaker mention a person by name?",
    "Does the speaker mention a date, a time or a deadline?",
    "Does the speaker mention a place?",
    "Is the speaker talking about themselves?",
    "Does the speaker mention cats?",
]

# One HTTP connection pool shared by every session, so a request doesn't pay
# for a new TLS handshake. It's sized for one pass's worth of decisions at once.
_http = httpx.Client(timeout=30.0, limits=httpx.Limits(max_connections=128, max_keepalive_connections=64))
# Sends a pass's decisions together; the endpoint batches them.
_pool = ThreadPoolExecutor(max_workers=64, thread_name_prefix="granite")
_model_id = GRANITE_MODEL_ID
_model_lock = threading.Lock()
_label_ids: list[int] = []
_label_lock = threading.Lock()
_last_wake = 0.0


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def missing_config() -> list[str]:
    """The settings this Space can't run without, that aren't set."""
    missing = []
    if not GRANITE_URL:
        missing.append("GRANITE_ENDPOINT_URL")
    if not _hf_token():
        missing.append("HF_TOKEN")
    return missing


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


def to_mono_16k(rate: int, samples: np.ndarray) -> np.ndarray:
    """A microphone chunk as mono 16 kHz float32 in [-1, 1]."""
    y = np.asarray(samples)
    scale = 1.0 if y.dtype.kind == "f" else float(np.iinfo(y.dtype).max)
    y = y.astype(np.float32) / scale
    if y.ndim > 1:
        y = y.mean(axis=1)
    if rate == SAMPLE_RATE or not len(y):
        return y
    if rate % SAMPLE_RATE == 0:
        # A whole-number ratio (48 kHz is 3x): average each group of samples,
        # which also filters out what 16 kHz can't represent.
        k = rate // SAMPLE_RATE
        return y[: len(y) // k * k].reshape(-1, k).mean(axis=1)
    n = int(len(y) * SAMPLE_RATE / rate)
    return np.interp(np.arange(n) * (rate / SAMPLE_RATE), np.arange(len(y)), y).astype(np.float32)


def _frame_rms(audio: np.ndarray) -> np.ndarray:
    """The loudness of each whole 100 ms frame."""
    n = len(audio) // FRAME
    if not n:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt((audio[: n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1))


def has_voice(audio: np.ndarray) -> bool:
    return bool((_frame_rms(audio) > VOICE_RMS).any())


def trailing_silence_s(audio: np.ndarray) -> float:
    """Seconds of quiet at the end of the audio."""
    loud = np.nonzero(_frame_rms(audio) > VOICE_RMS)[0]
    frames = len(audio) // FRAME
    return (frames - 1 - loud[-1]) * 0.1 if len(loud) else frames * 0.1


def quietest_cut(audio: np.ndarray, window_s: float = 4.0) -> int:
    """The sample at the middle of the quietest frame in the last `window_s` seconds."""
    rms = _frame_rms(audio)
    first = max(0, len(rms) - int(window_s * 10))
    return (first + int(np.argmin(rms[first:]))) * FRAME + FRAME // 2


def wav_base64(audio: np.ndarray) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return base64.b64encode(buffer.getvalue()).decode()


class ModelError(Exception):
    """A model call failed; the message is short enough to show on the page."""


# --------------------------------------------------------------------------- #
# Granite Switch (Inference Endpoint)
# --------------------------------------------------------------------------- #


def _granite_headers() -> dict:
    return {"Authorization": f"Bearer {_hf_token()}"}


def _granite_error(e: Exception, what: str) -> ModelError:
    if isinstance(e, httpx.HTTPStatusError):
        if e.response.status_code == 503:
            return ModelError(f"{GRANITE_NAME} is starting up (its endpoint scales to zero). Try again in a minute.")
        return ModelError(f"{what} failed ({e.response.status_code}): {e.response.text[:200]}")
    if isinstance(e, httpx.TimeoutException):
        return ModelError(f"{what} timed out.")
    return ModelError(f"{what} failed: {e}")


def model_id() -> str:
    """The endpoint's served model name, read once from /models."""
    global _model_id
    with _model_lock:
        if not _model_id:
            try:
                response = _http.get(f"{GRANITE_URL}/models", headers=_granite_headers(), timeout=15.0)
                response.raise_for_status()
                _model_id = response.json()["data"][0]["id"]
            except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
                raise _granite_error(e, f"Reaching {GRANITE_NAME}") from e
        return _model_id


def label_ids() -> list[int]:
    """The token ids of DECISION_LABELS, read once from the endpoint's /tokenize (beside /v1)."""
    global _label_ids
    model = model_id()
    with _label_lock:
        if not _label_ids:
            root = GRANITE_URL.removesuffix("/v1")
            ids = []
            try:
                for label in DECISION_LABELS:
                    response = _http.post(
                        f"{root}/tokenize",
                        headers=_granite_headers(),
                        json={"model": model, "prompt": label, "add_special_tokens": False},
                        timeout=15.0,
                    )
                    response.raise_for_status()
                    tokens = response.json()["tokens"]
                    if len(tokens) != 1:
                        raise ModelError(f"The option label {label!r} isn't one token ({tokens}).")
                    ids.append(tokens[0])
            except (httpx.HTTPError, ValueError, KeyError) as e:
                raise _granite_error(e, "Reading the option labels' tokens") from e
            _label_ids = ids
        return _label_ids


def wake_granite() -> None:
    """Nudge the endpoint awake in the background, so it's up by the time anyone talks."""
    global _last_wake
    now = time.monotonic()
    if missing_config() or now - _last_wake < WAKE_EVERY_S:
        return
    _last_wake = now

    def ping():
        try:
            model_id()
        except ModelError:
            pass  # still waking; the first real call says so on the page

    threading.Thread(target=ping, daemon=True).start()


def _chat(messages: list[dict], timeout: float = 30.0, **options) -> dict:
    response = _http.post(
        f"{GRANITE_URL}/chat/completions",
        headers=_granite_headers(),
        json={"model": model_id(), "messages": messages, "temperature": 0.0, **options},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def transcript_from_prompt(body: dict) -> str:
    """The transcript the endpoint spliced into the prompt, read back from its prompt logprobs.

    The prompt is one user turn holding only the audio, so the transcript is
    everything between the user role marker and the end of that turn. A clip
    with no speech in it comes back as a single space.
    """
    prompt = "".join(next(iter(p.values()))["decoded_token"] for p in body.get("prompt_logprobs") or [] if p)
    start = prompt.find(USER_TURN)
    if start < 0:
        raise ModelError("The endpoint didn't return the transcript (no prompt logprobs).")
    start += len(USER_TURN)
    end = prompt.find(END_OF_TURN, start)
    return prompt[start : end if end >= 0 else None].strip()


def transcribe(audio: np.ndarray) -> tuple[str, float]:
    """Return (text, seconds) for a clip of 16 kHz audio, from Granite's own speech recognition.

    There's no transcription route: audio goes in as an `input_audio` part of
    a chat message, and the endpoint transcribes it before the model runs. So
    this sends a user turn holding only the audio, generates one token, and
    asks for the prompt's logprobs, which spell out the prompt the model saw,
    transcript included.
    """
    messages = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": wav_base64(audio), "format": "wav"}}]}]
    start = time.perf_counter()
    try:
        body = _chat(messages, max_tokens=1, prompt_logprobs=0)
    except (httpx.HTTPError, ValueError) as e:
        raise _granite_error(e, "Speech recognition") from e
    return transcript_from_prompt(body), time.perf_counter() - start


def _candidates(position: dict) -> list[tuple[str, float]]:
    """A logprobs position's chosen token and its alternatives, as (token, probability)."""
    chosen = (position["token"], position["logprob"])
    rest = [(t["token"], t["logprob"]) for t in position.get("top_logprobs") or [] if t["token"] != chosen[0]]
    return [(token, math.exp(logprob)) for token, logprob in [chosen] + rest]


def share_of(position: dict, yes: str, no: str) -> float | None:
    """P(yes) / (P(yes) + P(no)) at one position, or None if neither is among its tokens.

    A word can be spelled several ways (" yes" and "yes"), so each takes its
    likeliest spelling; summing would double-count.
    """
    best = {}
    for token, p in _candidates(position):
        word = token.strip().lower()
        if word in (yes, no):
            best[word] = max(best.get(word, 0.0), p)
    total = sum(best.values())
    return best.get(yes, 0.0) / total if total else None


def state_turn(text: str) -> dict:
    """The user turn holding the words, as the decision adapter's model card writes its state.

    Every judgment and the label open with this same turn, so it's base-model
    KV the endpoint computes once and shares.
    """
    return {"role": "user", "content": f"State:\n{text}"}


def decision_messages(text: str, question: str) -> list[dict]:
    options = "\n".join(f"{label}: {option}" for label, option in zip(DECISION_LABELS, DECISION_OPTIONS))
    return [
        state_turn(text),
        {
            "role": "user",
            "content": f"{DECISION_INVOCATION}\nQuestion:\n{question}\n\nOptions:\n{options}\n\n{DECISION_REPLY}",
        },
    ]


def decision_noul(body: dict) -> float:
    """The noul from a decision-adapter reply: label A's share of the softmax over the labels.

    The labels' logprobs are the ones logprob_token_ids asked for. An endpoint
    that ignores that field returns only the likeliest token, and then a label
    is missing.
    """
    position = body["choices"][0]["logprobs"]["content"][0]
    logprobs = {t["token"]: t["logprob"] for t in position.get("top_logprobs") or []}
    missing = [label for label in DECISION_LABELS if label not in logprobs]
    if missing:
        raise ModelError(f"{DECISION_ADAPTER} returned no logprob for option {', '.join(missing)}.")
    a, b = (math.exp(logprobs[label]) for label in DECISION_LABELS)
    return a / (a + b)


def decide(text: str, question: str) -> float:
    """One yes/no question about `text`, as a noul from the decision adapter."""
    try:
        body = _chat(
            decision_messages(text, question),
            max_tokens=1,
            logprobs=True,
            top_logprobs=1,
            logprob_token_ids=label_ids(),
            chat_template_kwargs={"adapter_name": DECISION_ADAPTER},
        )
        return decision_noul(body)
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
        raise _granite_error(e, DECISION_ADAPTER) from e


def guardian_messages(text: str, criteria: str) -> list[dict]:
    instruction = GUARDIAN_INSTRUCTION.format(criteria=criteria, scoring_schema=GUARDIAN_SCORING)
    return [state_turn(text), {"role": "user", "content": instruction}]


def guardian_noul(body: dict) -> float:
    """The noul from a guardian-core reply: P("yes") at the token that carries the score."""
    for position in body["choices"][0]["logprobs"]["content"]:
        noul = share_of(position, "yes", "no")
        if noul is not None:
            return noul
    raise ModelError(f"guardian-core returned no score (got {body['choices'][0]['message'].get('content')!r}).")


def screen(text: str, criteria: str) -> float:
    """One safety category about `text`, as a noul from guardian-core."""
    try:
        body = _chat(
            guardian_messages(text, criteria),
            max_tokens=GUARDIAN_MAX_TOKENS,
            logprobs=True,
            top_logprobs=TOP_LOGPROBS,
            response_format=GUARDIAN_FORMAT,
            chat_template_kwargs={"adapter_name": GUARDIAN_ADAPTER},
        )
        return guardian_noul(body)
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
        raise _granite_error(e, GUARDIAN_ADAPTER) from e


def classify(text: str, nouls: list[str]) -> tuple[dict, float]:
    """Every safety category and every question about `text`, sent at once. Return (answers, seconds)."""
    start = time.perf_counter()
    label_ids()  # fail once, here, if the endpoint is down, not once per question
    safety = {key: _pool.submit(screen, text, criteria) for key, (_, criteria) in SAFETY.items()}
    questions = [(q, _pool.submit(decide, text, q)) for q in nouls]
    answers = {
        "safety": {key: future.result() for key, future in safety.items()},
        "nouls": [(q, future.result()) for q, future in questions],
    }
    return answers, time.perf_counter() - start


def describe_messages(text: str) -> list[dict]:
    return [state_turn(text), {"role": "user", "content": DESCRIBE_PROMPT}]


def describe(text: str) -> tuple[str, float]:
    """What the speaker is talking about, in 1 to 5 words, from the base model. Return (label, seconds)."""
    messages = describe_messages(text)
    start = time.perf_counter()
    try:
        body = _chat(messages, timeout=DESCRIBE_TIMEOUT_S, max_tokens=DESCRIBE_MAX_TOKENS)
        content = body["choices"][0]["message"].get("content") or ""
    except httpx.TimeoutException as e:
        raise ModelError(f"The label took longer than {DESCRIBE_TIMEOUT_S:g} s.") from e
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
        raise _granite_error(e, "The label") from e
    return clean_label(content), time.perf_counter() - start


def clean_label(content: str) -> str:
    """The reply's first line, without quotes or a full stop, held to the word limit."""
    lines = content.strip().splitlines()
    label = lines[0].strip(" \"'.") if lines else ""
    # Now and then a model writes its label twice, run together.
    half = len(label) // 2
    if half and label[:half] == label[half:]:
        label = label[:half]
    label = " ".join(label.split()[:DESCRIBE_MAX_WORDS])
    return label[:1].upper() + label[1:]


def clean_nouls(questions: list[str]) -> list[str]:
    """The yes/no questions to ask: blank lines and repeats are dropped."""
    seen, kept = set(), []
    for line in questions:
        line = line.strip()[:300]
        if line and line.lower() not in seen:
            seen.add(line.lower())
            kept.append(line)
    return kept[:MAX_NOULS]


def recent(session: "Session") -> str:
    """What's being said right now: the last RECENT_WORDS words since the last long silence."""
    return " ".join(session.text().split()[session.fresh_from :][-RECENT_WORDS:])


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #


@dataclass
class Session:
    """One visitor's audio, transcript and latest answers."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    nouls: list[str] = field(default_factory=lambda: list(DEFAULT_NOULS))  # the yes/no questions
    # The open segment: audio whose text isn't committed yet.
    segment: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    pending: int = 0  # samples that arrived since the last pass started
    transcribed: int = 0  # how many of the segment's samples `live` covers
    busy: bool = False  # a pass is running
    committed: list[str] = field(default_factory=list)
    live: str = ""
    answers: dict | None = None
    classified: tuple | None = None  # the (text, questions) the decisions last answered
    description: str = ""  # what the speaker is talking about, from the base model
    described: str | None = None  # the text `description` is about
    describe_asked: str | None = None  # the text of the newest description request
    describe_sent: int = 0  # description requests started; each one's number
    describe_shown: int = 0  # the number of the request `description` came from
    describing: int = 0  # description requests running now
    describe_s: float = 0.0
    describe_error: str = ""
    stt_s: float = 0.0
    stt_audio_s: float = 0.0
    decide_s: float = 0.0
    decided: int = 0  # how many decisions the last pass made in `decide_s`
    updates: int = 0
    silent_s: float = 0.0  # how long it's been since the last frame with speech in it
    quiet: bool = False  # silent for QUIET_AFTER_S: the board is blank
    silences: int = 0  # how many times the board has gone blank; tags each pass's answers
    fresh_from: int = 0  # how many of the transcript's words came before the last long silence
    # Where in `segment` the last long silence fell, until the speech before it
    # is committed. Words are counted from the audio, not from `live`, because
    # `live` lags the audio by up to a pass and can be revised when re-transcribed.
    silence_at: int | None = None
    # A short phrase can end, and the silence blank the board, before a pass
    # has judged it. Then the words before the silence are judged once they're
    # committed, and the answers shown through the quiet for LATE_SHOW_S.
    shown: str | None = None  # the words whose judgments last reached the board, since the last long silence
    owed: bool = False  # the last silence fell on words that might not have been judged in full
    owed_from: int = 0  # where those words start in the transcript
    owed_shown: str | None = None  # the words the board had shown judgments of when that silence fell
    late_until: float = 0.0  # show the late answers through the quiet until then (monotonic)
    error: str = ""
    last_seen: float = field(default_factory=time.monotonic)

    def text(self) -> str:
        return " ".join(self.committed + ([self.live] if self.live else []))


_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def get_session(sid: str | None) -> tuple[str, Session]:
    """The session for this browser tab, made on first use. Idle ones are dropped."""
    now = time.monotonic()
    with _sessions_lock:
        for old in [k for k, s in _sessions.items() if now - s.last_seen > SESSION_TTL_S]:
            del _sessions[old]
        if sid not in _sessions:
            sid = uuid.uuid4().hex
            _sessions[sid] = Session()
        _sessions[sid].last_seen = now
        return sid, _sessions[sid]


def start_describe(session: Session, text: str) -> int | None:
    """Reserve a description request for `text`. Call with the session's lock held.

    Returns the request's number, or None when the newest request already
    asked about this text or too many are running.
    """
    if not text or text == session.describe_asked or session.describing >= DESCRIBE_IN_FLIGHT:
        return None
    session.describing += 1
    session.describe_sent += 1
    session.describe_asked = text
    return session.describe_sent


def run_describe(session: Session, text: str, number: int) -> None:
    """Ask the base model what the speaker is talking about; keep the answer unless a newer one is in."""
    try:
        label, seconds = describe(text)
        with session.lock:
            if number > session.describe_shown:
                session.describe_shown, session.described = number, text
                session.describe_s, session.describe_error = seconds, ""
                if label:
                    session.description = label
    except ModelError as e:
        with session.lock:
            if number == session.describe_sent:
                # The newest request failed: show why, and let the next pass ask again.
                session.describe_error, session.describe_asked = str(e), None
    finally:
        with session.lock:
            session.describing -= 1


def drop_front(session: Session, samples: int) -> None:
    """Drop the segment's first `samples`, keeping `silence_at` on the same audio. Lock held."""
    session.segment = session.segment[samples:]
    if session.silence_at is not None:
        session.silence_at -= samples
        if session.silence_at <= 0:
            # Everything before the silence is committed: count its words exactly.
            session.silence_at = None
            session.fresh_from = len(" ".join(session.committed).split())


def commit_before_silence(session: Session) -> None:
    """Commit the speech from before the last long silence, so the next judgments leave it out.

    The silence is marked in the audio, so the words before it are cut where it
    fell, not by counting `live`: that would miss words said in the last second
    before the silence, which no pass had transcribed yet, and would point at
    the wrong word if a later transcription revised the earlier ones.
    """
    with session.lock:
        at = session.silence_at
        if at is None:
            return
        audio = session.segment[:at].copy()
        covered, text = session.transcribed, session.live
    if covered < at and has_voice(audio[covered:]):
        text, stt_s = transcribe(audio)
        with session.lock:
            session.stt_s, session.stt_audio_s = stt_s, at / SAMPLE_RATE
    with session.lock:
        if text:
            session.committed.append(text)
        session.live, session.transcribed = "", 0
        drop_front(session, at)


def judge_owed(session: Session) -> None:
    """Judge the words before the last silence, if none of their judgments reached the board.

    A short phrase ends, and the silence blanks the board, before any pass has
    judged it, so without this it's never judged at all. It runs once the
    words are committed; the answers show through the quiet for LATE_SHOW_S,
    and go as soon as the speaker starts again.
    """
    with session.lock:
        if not session.owed or session.silence_at is not None:
            return  # nothing owed, or the words before the silence aren't committed yet
        session.owed = False
        words = " ".join(session.committed).split()[session.owed_from : session.fresh_from]
        state = " ".join(words[-RECENT_WORDS:])
        if state == session.owed_shown:
            return  # the board already showed judgments of exactly these words
        nouls = list(session.nouls)
        silences = session.silences
        number = start_describe(session, state) if state else None
    if not state:
        return
    if number:
        threading.Thread(target=run_describe, args=(session, state, number), daemon=True).start()
    answers, decide_s = classify(state, nouls)
    with session.lock:
        # The speaker started again while these were out: they'd be about the last phrase.
        if session.silences == silences and session.quiet:
            session.answers, session.classified = answers, (state, tuple(nouls))
            session.decide_s = decide_s
            session.decided = len(SAFETY) + len(answers["nouls"])
            session.updates += 1
            session.late_until = time.monotonic() + LATE_SHOW_S


def run_pass(session: Session, final: bool = False) -> None:
    """Transcribe the open segment, commit it if it's time to, and ask the model.

    Only one pass runs per session at a time (`busy`). Audio that arrives
    while it runs is appended behind the part this pass works on.
    """
    try:
        commit_before_silence(session)
        judge_owed(session)
        with session.lock:
            audio = session.segment.copy()
            session.pending = 0
            covered, text = session.transcribed, session.live
            nouls = list(session.nouls)
        seconds = len(audio) / SAMPLE_RATE
        paused = trailing_silence_s(audio) >= PAUSE_S
        forced = seconds >= MAX_SEGMENT_S and not paused and not final
        commit = final or forced or (paused and seconds >= MIN_COMMIT_S)
        cut = quietest_cut(audio) if forced else len(audio)

        # Send the segment only if there's speech in it that `live` doesn't cover.
        if has_voice(audio[:cut]) and (forced or has_voice(audio[covered:])):
            text, stt_s = transcribe(audio[:cut])
            with session.lock:
                session.stt_s, session.stt_audio_s = stt_s, cut / SAMPLE_RATE

        with session.lock:
            if commit:
                if text:
                    session.committed.append(text)
                session.live, session.transcribed = "", 0
                drop_front(session, cut)
            elif not text and seconds > 2.0:
                # Nothing but quiet so far: keep only the last half second, so
                # the start of the next word isn't lost.
                session.transcribed = 0
                drop_front(session, len(audio) - SAMPLE_RATE // 2)
            else:
                session.live, session.transcribed = text, len(audio)
            state = recent(session)
            silences = session.silences
            asked = (state, tuple(nouls))
            stale = len(state.split()) >= MIN_WORDS and asked != session.classified
            # The description runs beside the decisions, on its own thread.
            number = start_describe(session, state) if len(state.split()) >= MIN_WORDS else None

        if number:
            threading.Thread(target=run_describe, args=(session, state, number), daemon=True).start()
        if stale:
            answers, decide_s = classify(state, nouls)
            with session.lock:
                # Answers about words from before a silence that fell while
                # they were out are dropped, so they can't come back after it.
                if session.silences == silences:
                    session.answers, session.classified = answers, asked
                    session.decide_s = decide_s
                    session.decided = len(SAFETY) + len(answers["nouls"])
                    session.updates += 1
        with session.lock:
            session.error = ""
    except ModelError as e:
        with session.lock:
            session.error = str(e)
    finally:
        with session.lock:
            session.busy = False


# --------------------------------------------------------------------------- #
# The API
# --------------------------------------------------------------------------- #


def state_json(sid: str, session: Session) -> dict:
    """Everything the page draws, as of now. While it's quiet, that's no judgments at all."""
    with session.lock:
        late = session.quiet and time.monotonic() < session.late_until
        answers = session.answers if late or not session.quiet else None
        if answers and not session.quiet and session.classified:
            session.shown = session.classified[0]
        return {
            "sid": sid,
            "ready": not missing_config(),
            "missing": missing_config(),
            "committed": " ".join(session.committed),
            "live": session.live,
            "description": session.description if late or not session.quiet else "",
            "safety": answers["safety"] if answers else None,
            "nouls": [{"question": q, "noul": n} for q, n in answers["nouls"]] if answers else None,
            "questions": list(session.nouls),
            "timings": {
                "stt_s": session.stt_s,
                "stt_audio_s": session.stt_audio_s,
                "decide_s": session.decide_s,
                "decisions": session.decided,
                "describe_s": session.describe_s,
            },
            "updates": session.updates,
            "quiet": session.quiet,
            "errors": [e for e in (session.error, session.describe_error) if e],
        }


def go_quiet(session: Session) -> None:
    """A long silence: wipe the judgments. Call with the session's lock held.

    The words so far are set aside too, so the judgments that come back when
    the speaker starts again are about the new words alone.
    """
    session.quiet = True
    session.silences += 1
    # Words before the silence are owed a judgment unless the board already
    # showed one about all of them; which words they are is only known once
    # they're committed, so judge_owed compares them with `owed_shown`.
    spoken = len(session.text().split()) > session.fresh_from or has_voice(session.segment[session.transcribed :])
    session.owed, session.owed_from, session.owed_shown = spoken, session.fresh_from, session.shown
    session.shown, session.late_until = None, 0.0
    # A first count; the next pass commits the speech before the silence and counts again.
    session.fresh_from = len(session.text().split())
    session.silence_at = len(session.segment)
    session.answers, session.classified = None, None
    session.description, session.described, session.describe_asked = "", None, None
    # Any description request still out is from before the silence: ignore its answer.
    session.describe_shown = session.describe_sent


def add_audio(session: Session, audio: np.ndarray) -> None:
    """Append a chunk to the open segment and start a pass if one is due."""
    with session.lock:
        session.segment = np.concatenate([session.segment, audio])
        session.pending += len(audio)
        if has_voice(audio):
            session.silent_s, session.quiet = trailing_silence_s(audio), False
            if session.owed or session.late_until:
                # Speaking again: the late answers are about the last phrase, so they go.
                session.owed, session.late_until = False, 0.0
                session.answers, session.classified = None, None
                session.description, session.described, session.describe_asked = "", None, None
                session.describe_shown = session.describe_sent
        else:
            session.silent_s += len(audio) / SAMPLE_RATE
        if session.silent_s >= QUIET_AFTER_S and not session.quiet:
            go_quiet(session)
        start = not session.busy and session.pending >= MIN_NEW_AUDIO_S * SAMPLE_RATE
        # Speech starting from nothing on the board: don't wait out a full
        # second between passes until there are words enough to judge.
        if not session.busy and not start and has_voice(audio):
            start = (
                session.answers is None
                and session.silence_at is None
                and len(recent(session).split()) < MIN_WORDS
            )
        if start:
            session.busy = True
    if start:
        threading.Thread(target=run_pass, args=(session,), daemon=True).start()


def _wait(session: Session, done, timeout_s: float = 20.0) -> bool:
    """Wait until `done()` is true, checked with the session's lock held."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with session.lock:
            if done():
                return True
        time.sleep(0.05)
    return False


def _claim_pass(session: Session) -> bool:
    if session.busy:
        return False
    session.busy = True
    return True


def settle(session: Session, final: bool) -> None:
    """Run one pass here, then make sure the description is about the same words."""
    if missing_config():
        return
    if _wait(session, lambda: _claim_pass(session)):
        run_pass(session, final=final)
    # The pass started a description request if one was due; let it finish.
    _wait(session, lambda: session.describing == 0, DESCRIBE_TIMEOUT_S + 1)
    with session.lock:
        state = recent(session)
        number = start_describe(session, state) if state != session.described else None
    if number:
        run_describe(session, state, number)


class SessionBody(BaseModel):
    sid: str | None = None
    questions: list[str] | None = None


app = FastAPI(title="Speech to Decisions with Granite", docs_url=None, redoc_url=None)


@app.get("/api/config")
def get_config() -> dict:
    wake_granite()
    return {
        "ready": not missing_config(),
        "missing": missing_config(),
        "labels": LABELS,
        "safety": [{"key": key, "label": label} for key, (label, _) in SAFETY.items()],
        "safety_threshold": SAFETY_THRESHOLD,
        "questions": DEFAULT_NOULS,
        "max_questions": MAX_NOULS,
        "quiet_after_s": QUIET_AFTER_S,
        "sample_rate": SAMPLE_RATE,
        "chunk_s": STREAM_EVERY_S,
    }


@app.post("/api/session")
def post_session(body: SessionBody) -> dict:
    """Set the yes/no questions. With nothing recording, answer the new ones now."""
    sid, session = get_session(body.sid)
    if body.questions is not None:
        with session.lock:
            session.nouls = clean_nouls(body.questions)
            idle = not session.busy and bool(session.committed) and not len(session.segment)
        if idle:
            settle(session, final=False)
    return state_json(sid, session)


@app.post("/api/chunk")
async def post_chunk(request: Request, sid: str | None = None, rate: int = SAMPLE_RATE) -> dict:
    """A chunk of microphone audio: 16-bit little-endian mono PCM at `rate`."""
    body = await request.body()
    if len(body) > MAX_CHUNK_BYTES or not 8000 <= rate <= 192000:
        raise HTTPException(status_code=413 if len(body) > MAX_CHUNK_BYTES else 400)
    sid, session = get_session(sid)
    if body and not missing_config():
        add_audio(session, to_mono_16k(rate, np.frombuffer(body[: len(body) // 2 * 2], dtype="<i2")))
    return state_json(sid, session)


@app.post("/api/stop")
def post_stop(body: SessionBody) -> dict:
    """Recording stopped: finish with whatever audio hasn't been transcribed."""
    sid, session = get_session(body.sid)
    settle(session, final=True)
    return state_json(sid, session)


@app.post("/api/clear")
def post_clear(body: SessionBody) -> dict:
    """Drop the session and start a new one with the same questions."""
    with _sessions_lock:
        old = _sessions.pop(body.sid, None)
    sid, session = get_session(None)
    if body.questions is not None:
        session.nouls = clean_nouls(body.questions)
    elif old is not None:
        session.nouls = list(old.nouls)
    return state_json(sid, session)


# Everything else is the page.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
