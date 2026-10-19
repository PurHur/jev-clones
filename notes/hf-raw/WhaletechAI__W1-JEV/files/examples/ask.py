"""Send a choice question to a running local W1-JEV server."""
import argparse
import json
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8011")
    parser.add_argument("--question", default="What is the capital of France?")
    parser.add_argument("--options", nargs="+", default=["Paris", "London", "Berlin", "Rome"])
    args = parser.parse_args()
    if not 2 <= len(args.options) <= 26:
        parser.error("Provide 2 to 26 options")
    schema = {"questions": [{
        "id": "answer", "type": "choice",
        "instructions": "Choose the correct answer to the question.",
        "options": args.options,
    }]}
    body = {"model": "w1-jev", "seed": 42, "messages": [
        {"role": "system", "content": json.dumps(schema)},
        {"role": "user", "content": json.dumps({"question": args.question})},
    ]}
    request = Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=600) as response:
        reply = json.load(response)
    result = json.loads(reply["choices"][0]["message"]["content"])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
