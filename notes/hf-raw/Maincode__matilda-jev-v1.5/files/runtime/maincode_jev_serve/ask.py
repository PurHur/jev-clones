"""Command-line client for a running maincode-jev server (standard library only).

Server URL: --url, else $MJ_URL, else run/endpoint.json (written by scripts/serve.sh), else http://127.0.0.1:8000.
API key: $MJ_API_KEY, else the file named by $MJ_API_KEY_FILE.

  maincode-jev-ask                                         # server health
  maincode-jev-ask --state "10 units; 25+ get a discount" --instructions "Which tier?" --options list="List price" volume="Volume discount"
  maincode-jev-ask --state "Worst purchase this year." --noul "Is the review positive?"
  maincode-jev-ask examples/request.json                    # full request: {state, questions}; '-' reads stdin
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def server_url(explicit: str | None) -> str:
    if explicit or os.getenv("MJ_URL"):
        return str(explicit or os.getenv("MJ_URL")).rstrip("/")
    endpoint = REPO / "run" / "endpoint.json"
    if endpoint.is_file():
        return str(json.loads(endpoint.read_text())["url"]).rstrip("/")
    return "http://127.0.0.1:8000"


def api_key() -> str | None:
    if os.getenv("MJ_API_KEY"):
        return os.environ["MJ_API_KEY"]
    path = os.getenv("MJ_API_KEY_FILE")
    return Path(path).read_text().strip() if path and Path(path).is_file() else None


def call(url: str, path: str, body: dict[str, object] | None, key: str | None) -> tuple[int, dict[str, object], float]:
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    request = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None,
                                     method="POST" if body is not None else "GET", headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read()), (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}"), (time.perf_counter() - started) * 1000
    except urllib.error.URLError as error:
        raise SystemExit(f"Cannot reach {url}: {error.reason}. Is the server running? (scripts/serve.sh)") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("request", nargs="?", help="JSON file with {state, questions} ('-' for stdin)")
    parser.add_argument("--url")
    parser.add_argument("--state", help="State text for a single question")
    parser.add_argument("--instructions", default="Choose the best matching option.")
    parser.add_argument("--options", nargs="+", help="Choice options: key or key=description")
    parser.add_argument("--noul", metavar="QUESTION", help="Ask a yes/no question instead")
    parser.add_argument("--model", default="maincode-jev-latest")
    parser.add_argument("--raw", action="store_true", help="Print the raw JSON response")
    args = parser.parse_args()
    url, key = server_url(args.url), api_key()
    if args.request:
        body = json.loads(sys.stdin.read() if args.request == "-" else Path(args.request).read_text())
    elif args.state is not None and (args.options or args.noul):
        if args.noul:
            question: dict[str, object] = {"type": "noul", "instructions": args.noul}
        else:
            criteria = dict(item.split("=", 1) if "=" in item else (item, None) for item in args.options)
            question = {"type": "choice", "instructions": args.instructions, "criteria": criteria}
        body = {"state": args.state, "questions": {"answer": question}}
    else:
        print(f"{url}:")
        print(json.dumps(call(url, "/health", None, key)[1], indent=1))
        return
    body.setdefault("model", args.model)
    status, response, milliseconds = call(url, "/v1/systemone", body, key)
    if args.raw or status != 200:
        print(f"HTTP {status}  {milliseconds:.0f} ms")
        print(json.dumps(response, indent=1))
        raise SystemExit(0 if status == 200 else 1)
    print(f"{response['model']}  {milliseconds:.0f} ms  ({response['usage']['input_tokens']} input tokens)")  # type: ignore[index]
    for name, answer in response["answers"].items():  # type: ignore[union-attr]
        if answer["type"] == "noul":
            print(f"  {name}: P(yes) = {answer['noul']:.3f}")
        else:
            ranked = sorted(answer["probabilities"].items(), key=lambda item: -item[1])
            print(f"  {name}: {answer.get('choice', ranked[0][0])}  " + "  ".join(f"{k}={v:.3f}" for k, v in ranked[:5]))


if __name__ == "__main__":
    main()
