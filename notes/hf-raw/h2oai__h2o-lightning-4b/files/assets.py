"""`assets`: images referenced by name anywhere in a request.

    "assets": {"<ref>": {"type": "image", "format": "png"|"jpg", "data_base64": "..."}
               | {"type": "image", "part": "<multipart file part name>"}}

The ref string is found and replaced wherever it occurs in the request's STRINGS: references are matched as plain
text anywhere in the request's strings, so a reference works mid-sentence as well as as a whole value. Whole values
match first. In longer strings, references are replaced from left to right, and matches never overlap. A short
reference such as `cat` could also match inside `category`, so use references that cannot occur in your text by
accident. Object keys are not replaced. So: plain substring matching, no word boundaries (a ref inside a longer word
IS replaced; that is the caller's contract), left to right, non-overlapping, the longest ref winning where two refs
start at the same place; keys never.
Backticks are NOT part of a reference: a question's backticked name refers to a STATE field ("the mug in `photo`",
where state.photo holds the ref), and stays text. Every asset must be referenced somewhere, and a multipart file part
must belong to exactly one asset; otherwise 400. PNG and JPEG only, 16 MiB per file, no URL fetching.

How it binds to the model: the shim's image path puts every image of a request ahead of the user text as image
parts and leaves `[image N]` placeholders in the text (lift_images). bind() turns each ref into exactly that: the
asset's data URI is appended to the request's image list ONCE per distinct reference, at its first use in order of
appearance (state, then questions in order: instructions, then criteria), and every use, whole-string or inside
text, becomes that reference's `[image N]` (N = its position in the top-level images list; images the caller
embedded in the state itself are numbered after the top-level list by lift_images, as before). Usage then counts
the image once.
"""
from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping
from typing import Final

MAX_FILE_BYTES: Final = 16 * 1024 * 1024
FORMATS: Final = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
_PNG: Final = b"\x89PNG\r\n\x1a\n"
_JPEG: Final = b"\xff\xd8\xff"
PLACEHOLDER: Final = "[image {n}]"


class AssetError(ValueError):
    """A 400: the assets map, a reference or a part is not as the wire format says."""


def _sniff(raw: bytes) -> str | None:
    if raw.startswith(_PNG):
        return "image/png"
    if raw.startswith(_JPEG):
        return "image/jpeg"
    return None


def _decode(ref: str, entry: Mapping[str, object], parts: Mapping[str, bytes] | None) -> bytes:
    if not isinstance(entry, Mapping) or entry.get("type") != "image":
        raise AssetError(f"asset {ref!r} must be an object with type \"image\"")
    if "part" in entry:
        name = entry.get("part")
        if not isinstance(name, str) or not name:
            raise AssetError(f"asset {ref!r}: part must name a multipart file part")
        if parts is None or name not in parts:
            raise AssetError(f"asset {ref!r} names file part {name!r}, which the request does not carry")
        return parts[name]
    data = entry.get("data_base64")
    if not isinstance(data, str) or not data:
        raise AssetError(f"asset {ref!r} needs data_base64 or part")
    try:
        return base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as error:
        raise AssetError(f"asset {ref!r}: data_base64 is not valid base64") from error


def resolve(assets: object, parts: Mapping[str, bytes] | None = None) -> dict[str, str]:
    """{ref: data URI} for the assets map, validated (type, format, PNG/JPEG bytes, 16 MiB per file, parts)."""
    if not isinstance(assets, Mapping):
        raise AssetError("assets must be an object mapping reference strings to image assets")
    resolved: dict[str, str] = {}
    used_parts: set[str] = set()
    for ref, entry in assets.items():
        if not isinstance(ref, str) or not ref.strip():
            raise AssetError("asset references must be non-empty strings")
        raw = _decode(ref, entry, parts)
        if len(raw) > MAX_FILE_BYTES:
            raise AssetError(f"asset {ref!r} exceeds 16 MiB")
        mime = _sniff(raw)
        declared = entry.get("format") if isinstance(entry, Mapping) else None
        if declared is not None and str(declared).lower() not in FORMATS:
            raise AssetError(f"asset {ref!r}: format must be png or jpg")
        if mime is None:
            raise AssetError(f"asset {ref!r} is not a PNG or JPEG image")
        if declared is not None and FORMATS[str(declared).lower()] != mime:
            raise AssetError(f"asset {ref!r}: the bytes are {mime}, not {declared}")
        if isinstance(entry, Mapping) and "part" in entry:
            if entry["part"] in used_parts:
                raise AssetError(f"file part {entry['part']!r} belongs to more than one asset")
            used_parts.add(str(entry["part"]))
        resolved[ref] = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    if parts:
        stray = sorted(set(parts) - used_parts)
        if stray:
            raise AssetError(f"file part(s) {stray} belong to no asset; each file part must be named by exactly one asset")
    return resolved


class _Binder:
    """Replaces refs in strings; counts uses; appends one image per distinct ref to `images`."""

    def __init__(self, resolved: Mapping[str, str], images: list[str]) -> None:
        self.resolved = resolved
        self.images = images
        self.uses: dict[str, int] = dict.fromkeys(resolved, 0)
        self.placeholder: dict[str, str] = {}  # ref -> its `[image N]`, set on the first use
        # longest refs first, so a ref that contains another is matched whole
        self.pattern = re.compile("|".join(re.escape(r) for r in sorted(resolved, key=len, reverse=True))) if resolved else None

    def text(self, s: str) -> str:
        """Every use, whole-string or inside text, becomes `[image N]`; each DISTINCT reference is bound once (its
        image appended to the top-level list on first use) and every later use points at the same placeholder (a
        reference used twice reaches the model once). One numbering for all uses; a whole-string data URI would be lifted and numbered a second time."""
        if self.pattern is None:
            return s

        def one(m: re.Match[str]) -> str:
            ref = m.group(0)
            self.uses[ref] += 1
            if ref not in self.placeholder:
                self.images.append(self.resolved[ref])
                self.placeholder[ref] = PLACEHOLDER.format(n=len(self.images))
            return self.placeholder[ref]

        return self.pattern.sub(one, s)

    def walk(self, value: object) -> object:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Mapping):
            return {k: self.walk(v) for k, v in value.items()}  # keys are never replaced
        if isinstance(value, list):
            return [self.walk(v) for v in value]
        return value


def bind(body: Mapping[str, object], parts: Mapping[str, bytes] | None = None) -> dict[str, object]:
    """The request with its assets bound: a copy whose state, questions (instructions, criteria) carry the images
    in place and whose top-level `images` list holds them (one entry per use, in order), `assets` and the parts
    removed. A request without `assets` is returned as is. Raises AssetError (a 400) on a malformed map, an
    unreferenced asset or a stray file part."""
    if "assets" not in body:
        if parts:
            raise AssetError("multipart file parts need an `assets` map naming them (asset: {'type': 'image', 'part': ...})")
        return dict(body)
    resolved: Final = resolve(body["assets"], parts)
    out: dict[str, object] = {k: v for k, v in body.items() if k != "assets"}
    have = out.get("images")
    images: list[str] = list(have) if isinstance(have, list) else ([have] if isinstance(have, str) else [])
    binder: Final = _Binder(resolved, images)
    if "state" in out:
        out["state"] = binder.walk(out["state"])
    if "input" in out:  # the OpenAI wire's state (a string, or user-message parts), before its questions
        out["input"] = binder.walk(out["input"])
    questions = out.get("questions")
    if isinstance(questions, Mapping):
        bound = {}
        for qid, q in questions.items():
            if isinstance(q, Mapping):
                q2 = dict(q)
                for field in ("instructions", "criteria"):
                    if field in q2:
                        q2[field] = binder.walk(q2[field])
                bound[qid] = q2
            else:
                bound[qid] = q
        out["questions"] = bound
    elif isinstance(questions, list):  # the OpenAI wire: a list of typed questions (choices/levels hold the text)
        bound_list = []
        for q in questions:
            if isinstance(q, Mapping):
                q2 = dict(q)
                for field in ("instructions", "choices", "levels", "criteria"):
                    if field in q2:
                        q2[field] = binder.walk(q2[field])
                bound_list.append(q2)
            else:
                bound_list.append(q)
        out["questions"] = bound_list
    unused: Final = [ref for ref, n in binder.uses.items() if n == 0]
    if unused:
        raise AssetError(f"asset(s) {unused} are not referenced anywhere in the request; every asset must be referenced")
    if images:
        out["images"] = images
    return out
