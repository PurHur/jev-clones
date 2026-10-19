#!/usr/bin/env python3
"""Small labeled image/video integration benchmark for a System One endpoint."""

import argparse
import base64
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="xor-1.2")
    parser.add_argument("--fixtures", type=Path,
                        default=Path(__file__).resolve().parent / "tests/fixtures")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for medium, extension, mime in (("image", "png", "image/png"),
                                    ("video", "mp4", "video/mp4")):
        for color in ("red", "blue"):
            path = args.fixtures / f"{color}.{extension}"
            fixture = path.read_bytes()
            data_url = f"data:{mime};base64,{base64.b64encode(fixture).decode()}"
            payload = {
                "state": f"Examine the attached {medium}.",
                "model": args.model,
                "questions": {"color": {
                    "type": "choice",
                    "instructions": f"What color fills the {medium}?",
                    "criteria": {"red": "red", "blue": "blue",
                                 "green": "green", "yellow": "yellow"},
                }},
            }
            payload["images" if medium == "image" else "video"] = (
                [data_url] if medium == "image" else data_url)
            request = urllib.request.Request(
                args.endpoint.rstrip("/") + "/v1/systemone",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.load(response)
                error = None
            except Exception as exc:
                result, error = None, str(exc)
            answer = result["answers"]["color"] if result else None
            rows.append({"medium": medium, "fixture": path.name,
                         "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
                         "expected": color,
                         "predicted": answer["choice"] if answer else None,
                         "correct": answer["choice"] == color if answer else False,
                         "latency_ms": (time.perf_counter() - started) * 1000,
                         "answer": answer, "error": error})
            print(json.dumps(rows[-1]), flush=True)
    (args.out / "traces.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    report = {medium: {
        "n": sum(row["medium"] == medium for row in rows),
        "correct": sum(row["correct"] for row in rows if row["medium"] == medium),
        "errors": sum(row["error"] is not None for row in rows if row["medium"] == medium),
    } for medium in ("image", "video")}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
