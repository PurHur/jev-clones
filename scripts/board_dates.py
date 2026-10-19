"""Creation date of each Decision Index entrant's HF repo -> notes/board_dates.json (repo id -> YYYY-MM-DD).

Usage: python3 -I scripts/board_dates.py
"""

import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def repo_of(url: str) -> str | None:
    m = re.match(r"https://huggingface\.co/([^/]+/[^/]+)", url or "")
    return m.group(1) if m else None


def created(repo: str) -> tuple[str, str | None]:
    try:
        req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}?expand[]=createdAt",
                                     headers={"User-Agent": "jev-clones-survey"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return repo, (json.loads(r.read()).get("createdAt") or "")[:10] or None
    except Exception:
        return repo, None


def main() -> None:
    repos = set()
    for line in (ROOT / "notes/hf-raw/_spaces/dix_table.tsv").read_text().splitlines():
        cells = line.split("\t")
        if len(cells) >= 10 and (r := repo_of(cells[8])):
            repos.add(r)
    with ThreadPoolExecutor(8) as pool:
        dates = dict(pool.map(created, sorted(repos)))
    (ROOT / "notes/board_dates.json").write_text(json.dumps(dates, indent=1, sort_keys=True))
    print(f"{sum(1 for v in dates.values() if v)} of {len(dates)} repos dated")


if __name__ == "__main__":
    main()
