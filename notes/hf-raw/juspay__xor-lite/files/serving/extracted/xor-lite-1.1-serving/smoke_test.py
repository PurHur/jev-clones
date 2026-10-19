#!/usr/bin/env python3
import json
import sys
import urllib.error
import urllib.request


def post(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body[:500]


def get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def wide_choice(n):
    return {
        "state": "A support ticket was filed.",
        "model": "xor-lite",
        "questions": {"route": {
            "type": "choice",
            "instructions": "Which queue should take the ticket?",
            "criteria": {f"q{i:03d}": f"Queue number {i}" for i in range(n)},
        }},
    }


def main():
    base_url = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:30002"
    ids = [m["id"] for m in get(f"{base_url}/v1/models")["data"]]
    assert ids == ["xor-lite"], ids

    status, result = post(
        f"{base_url}/v1/systemone",
        {
            "state": "The customer was charged twice and needs a refund.",
            "model": "xor-lite",
            "questions": {
                "urgent": {
                    "type": "noul",
                    "instructions": "Does the customer need prompt help?",
                },
                "department": {
                    "type": "choice",
                    "instructions": "Which department should handle this?",
                    "criteria": {
                        "billing": "Payments, refunds, and invoices",
                        "technical": "Software bugs and outages",
                        "sales": "Pricing and purchasing",
                    },
                },
                "mood": {
                    "type": "score",
                    "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very frustrated"],
                },
            },
        },
    )
    assert status == 200, (status, result)
    assert result["model"] == "xor-lite"
    assert result["answers"]["urgent"]["type"] == "noul"
    assert 0.0 <= result["answers"]["urgent"]["noul"] <= 1.0
    assert result["answers"]["department"]["type"] == "choice"
    assert result["answers"]["mood"]["type"] == "score"
    for key in ("department", "mood"):
        total = sum(result["answers"][key]["probabilities"].values())
        assert abs(total - 1.0) <= 0.02, (key, total)

    status, wide = post(f"{base_url}/v1/systemone", wide_choice(255))
    assert status == 200, (status, wide)
    assert len(wide["answers"]["route"]["probabilities"]) == 255
    status, _ = post(f"{base_url}/v1/systemone", wide_choice(256))
    assert status == 422, status

    print(json.dumps({"status": "ok", "model": result["model"]}))


if __name__ == "__main__":
    main()
