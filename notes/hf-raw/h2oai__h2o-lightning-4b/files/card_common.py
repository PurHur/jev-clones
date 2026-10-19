"""Helpers for the model-card conformance tests (test_card.py): the card is parsed, its commands are run and its
examples are sent as written, so the card and the tests cannot drift apart.

Inputs (environment):
  CARD_REPO   the Hub repo whose card is tested (default h2oai/h2o-lightning-4b)
  CARD_REV    the exact 40-hex commit under test (required unless CARD_DIR is given)
  CARD_DIR    a folder that already holds that commit's files (optional; else a fresh snapshot is downloaded)
  CARD_LIVE   1 to run the live tier (a GPU, vLLM 0.30.0 and the card's launch commands)
  CARD_REPORT where the JSON report goes (default card_report.json in the working directory)
Tiers are pytest markers: static (CPU, seconds, the default), live (serves the model), slow (recomputes numbers).
Nothing here writes to the Hub."""
import atexit
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request

DEFAULT_REPO = "h2oai/h2o-lightning-4b"
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


class Report:
    """Collects what the tests read and found; written as JSON when the interpreter exits."""

    def __init__(self):
        self.data = {"repo": None, "revision": None, "snapshot_commit": None, "files_read": {}, "results": {}, "notes": []}
        atexit.register(self.write)

    def read(self, path):
        p = os.path.abspath(path)
        if p not in self.data["files_read"] and os.path.isfile(p):
            self.data["files_read"][p] = sha256_file(p)
        return p

    def result(self, name, value):
        self.data["results"][name] = value

    def note(self, text):
        self.data["notes"].append(text)

    def write(self):
        with open(env("CARD_REPORT", "card_report.json"), "w") as f:
            json.dump(self.data, f, indent=1, sort_keys=True)


REPORT = Report()


def fresh_snapshot(repo, rev, ignore_patterns=None, token=None):
    """The exact commit's files: CARD_DIR when given (its README must be that commit's), else a snapshot into a fresh
    HF_HOME under the working directory; returns (folder, commit sha the Hub reported). ignore_patterns leaves out
    files the static tier does not read (the GGUF weights); token=False proves a public repo needs no login."""
    d = env("CARD_DIR")
    if d:
        REPORT.data.update(repo=repo, revision=rev, snapshot_commit=rev)
        return os.path.abspath(d), rev
    if not (rev and HEX40.match(rev)):
        raise RuntimeError("CARD_REV must be the full 40-hex commit under test")
    from huggingface_hub import HfApi, snapshot_download
    home = os.path.abspath(f"card_snapshot_{rev[:12]}")
    os.environ["HF_HOME"] = os.path.join(home, "hf_home")
    folder = snapshot_download(repo, revision=rev, local_dir=os.path.join(home, "files"), ignore_patterns=ignore_patterns,
                               token=token)
    got = HfApi().model_info(repo, revision=rev, token=token).sha
    REPORT.data.update(repo=repo, revision=rev, snapshot_commit=got)
    return folder, got


def readme_blocks(text):
    """Every fenced block of the card: {"index", "lang", "code", "heading"} with the nearest heading above it."""
    out, heading = [], ""
    for m in re.finditer(r"^(#{1,6})[ \t]+([^\n]*)$|```(\w*)\n(.*?)```", text, re.S | re.M):
        if m.group(1):
            heading = m.group(2).strip()
        else:
            out.append({"index": len(out), "lang": m.group(3) or "", "code": m.group(4), "heading": heading, "pos": m.start()})
    return out


def referenced_paths(text):
    """File names the card mentions in backticks that look like package files."""
    names = set(re.findall(r"`([\w./-]+\.(?:py|json|sh|jinja|lightning|safetensors|md|txt))`", text))
    return sorted(n.split("/")[-1] if n.startswith("h2o-lightning-4b/") else n for n in names)


def claims_check(text, claims):
    """Each registered claim quote must appear verbatim in the card; returns the missing ones."""
    return [c["quote"] for c in claims if c["quote"] not in text]


NUMBER = re.compile(r"\d")


def numeric_sentences(text, headings):
    """Sentences with a number under the given headings (prose only, fenced blocks and tables excluded)."""
    out = []
    section = None
    for line in text.splitlines():
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            continue
        if section in headings and not line.startswith(("|", "```", ">", "    ")) and NUMBER.search(line):
            out.append((section, line.strip()))
    return out


def substitute(code, mapping):
    for k, v in mapping.items():
        code = code.replace(k, v)
    return code


def run_card_command(code, cwd, log_path, extra_env=None):
    """Runs a bash block as written (after substitutions) in the background; returns the Popen."""
    e = dict(os.environ)
    e.update(extra_env or {})
    log = open(log_path, "ab")
    return subprocess.Popen(["bash", "-c", code], cwd=cwd, env=e, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def wait_http(url, seconds, proc=None):
    """Until the URL answers 200; a process that exits first is a failure."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def wait_log(path, needle, seconds, proc=None):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if os.path.isfile(path) and needle in open(path, "rb").read().decode("utf-8", "replace"):
            return True
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(3)
    return False


def stop_tree(proc):
    """Stops the process group a card command started (its vLLM and shim children included)."""
    import signal
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(40):
        if proc.poll() is not None:
            break
        time.sleep(1)
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    time.sleep(5)


def post_example(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw[:500]}


def curl_json_bodies(code):
    """The JSON bodies of the curl commands in a bash block (the -d '...' arguments), parsed; and their URLs."""
    out = []
    for m in re.finditer(r"curl\s+(.*?)-d\s+'(.*?)'", code, re.S):
        url = re.search(r"https?://[^\s'\"]+", m.group(1))
        body = m.group(2)
        try:
            out.append((url.group(0) if url else None, json.loads(body)))
        except ValueError:
            pass
    return out


def safetensors_keys(path):
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", None)
    return h, meta


def weights_differ_from_base(folder, base_repo, base_rev, max_bytes=1 << 20, tensor="q_proj.weight"):
    """One tensor of the shipped weights against the same tensor of the base model on the Hub (range read): the
    fraction of equal bf16 values must be below 0.5, so the shipped file is not the base under another name.
    The base may be one model.safetensors or sharded (its index names the shard)."""
    import struct
    hdr, _ = safetensors_keys(os.path.join(folder, "model.safetensors"))
    name = sorted(k for k in hdr if k.endswith(tensor))[0]
    ours = hdr[name]
    with open(os.path.join(folder, "model.safetensors"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + n + ours["data_offsets"][0])
        a = f.read(min(max_bytes, ours["data_offsets"][1] - ours["data_offsets"][0]))
    base_name, b = hub_safetensors_head(base_repo, base_rev, name, len(a), fallback=name.replace("model.language_model.", "model."))
    return name, bf16_equal_share(a, b[:len(a)])


# --- additions for the GGUF and Nemotron-Nano cards (range reads, claim registry, example shapes) ---

def hub_safetensors_head(repo, rev, tensor, n=1 << 20, fallback=None):
    """First n bytes of one tensor of a Hub safetensors model (single file or sharded), by range reads only.
    `tensor` may be a full name or a suffix; `fallback` is tried when the name is not in the model."""
    import struct
    from huggingface_hub import HfFileSystem, get_safetensors_metadata
    m = get_safetensors_metadata(repo, revision=rev, token=False)
    names = [tensor, fallback] if fallback else [tensor]
    name = next((x for x in names if x in m.weight_map), None) or next(k for k in m.weight_map if k.endswith(tensor))
    fname = m.weight_map[name]
    with HfFileSystem(token=False).open(f"{repo}@{rev}/{fname}", "rb", block_size=n) as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + hlen + m.files_metadata[fname].tensors[name].data_offsets[0])
        return name, f.read(n)


def hub_gguf_head(repo, rev, fname, tensor, n=1 << 20):
    """First n bytes of one tensor of a Hub GGUF file (v2/v3), by range reads; returns (dims, ggml type, bytes)."""
    import struct
    from huggingface_hub import HfFileSystem
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with HfFileSystem(token=False).open(f"{repo}@{rev}/{fname}", "rb", block_size=16 << 20) as f:
        u32 = lambda: struct.unpack("<I", f.read(4))[0]
        u64 = lambda: struct.unpack("<Q", f.read(8))[0]
        s = lambda: f.read(u64()).decode("utf-8", "replace")

        def val(t):
            if t == 8:
                return s()
            if t == 9:
                et, k = u32(), u64()
                return [val(et) for _ in range(k)]
            return f.read(sizes[t])

        if f.read(4) != b"GGUF":
            raise ValueError(f"{fname} is not GGUF")
        u32()
        nt, nkv = u64(), u64()
        align = 32
        for _ in range(nkv):
            k, t = s(), u32()
            v = val(t)
            if k == "general.alignment":
                align = struct.unpack("<I", v)[0]
        infos = {}
        for _ in range(nt):
            name, nd = s(), u32()
            dims = [u64() for _ in range(nd)]
            infos[name] = (dims, u32(), u64())
        base = (f.tell() + align - 1) // align * align
        dims, ty, off = infos[tensor]
        f.seek(base + off)
        return dims, ty, f.read(n)


def bf16_equal_share(a, b):
    """Share of equal 16-bit values in two equal-length byte strings."""
    n = min(len(a), len(b)) // 2
    return sum(a[2 * i:2 * i + 2] == b[2 * i:2 * i + 2] for i in range(n)) / max(1, n)


NUMERIC_CLAIM = re.compile(r"\d+\.\d+|\d+(?:\.\d+)?\s?%|\b\d+\s?/\s?\d+\b|"
                           r"\b\d[\d,]*\s?(?:GB|MB|MiB|GiB|ms|s|tokens|questions|items)\b")


def prose_lines(text):
    """(line number, line) of the card outside the front matter and fenced blocks."""
    out, fenced, start = [], False, 0
    lines = text.splitlines()
    if lines and lines[0] == "---":
        start = lines.index("---", 1) + 1
    for i, line in enumerate(lines[start:], start=start + 1):
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            out.append((i, line))
    return out


def numeric_units(text):
    """The card's numeric claims: each table (one unit) and each other prose line with a number a reader could hold
    us to (a decimal, a percentage, a ratio, a count with a unit)."""
    units, table = [], []
    for _, line in prose_lines(text):
        if line.startswith("|"):
            table.append(line)
            continue
        if table:
            units.append("\n".join(table))
            table = []
        if NUMERIC_CLAIM.search(line):
            units.append(line)
    if table:
        units.append("\n".join(table))
    return [u for u in units if NUMERIC_CLAIM.search(u)]


def unregistered_units(text, claims):
    """Numeric units no registered claim quote covers (a claim covers a unit when its quote is inside it)."""
    return [u for u in numeric_units(text) if not any(c["quote"] in u for c in claims)]


def table_rows(text, header):
    """The body rows of the table whose header line contains `header`, as lists of stripped cells."""
    unit = next(u for u in numeric_units(text) if header in u)
    return [[c.strip() for c in line.strip("|").split("|")] for line in unit.splitlines()[2:]]


QUESTION_TYPES = {"choice", "noul", "score", "box"}


def request_shape_errors(body):
    """The shape every decision request on the cards takes; a list of problems (empty when fine)."""
    errs = []
    if "state" not in body:
        errs.append("no state")
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        return errs + ["questions must be a non-empty object"]
    for name, q in qs.items():
        t = q.get("type")
        if t not in QUESTION_TYPES:
            errs.append(f"{name}: unknown type {t!r}")
        if not isinstance(q.get("instructions"), str):
            errs.append(f"{name}: instructions must be text")
        c = q.get("criteria")
        if t == "choice" and not (isinstance(c, dict) and len(c) >= 2):
            errs.append(f"{name}: choice needs a criteria object with two or more options")
        if t == "score" and not (isinstance(c, list) and len(c) >= 2):
            errs.append(f"{name}: score needs a criteria list with two or more levels")
        if t == "noul" and c is not None and not (isinstance(c, dict) and set(c) <= {"true", "false"}):
            errs.append(f"{name}: noul criteria keys must be true/false")
    return errs


def response_shape_errors(resp, req):
    """The documented answer shape, checked against the request it answers."""
    errs = []
    answers = resp.get("answers")
    if not isinstance(answers, dict):
        return ["answers must be an object"]
    if set(answers) != set(req["questions"]):
        errs.append(f"answers {sorted(answers)} != questions {sorted(req['questions'])}")
    for name, a in answers.items():
        q = req["questions"].get(name, {})
        if a.get("type") != q.get("type"):
            errs.append(f"{name}: type {a.get('type')!r} != {q.get('type')!r}")
        if a.get("type") == "choice":
            p = a.get("probabilities", {})
            if set(p) != set(q["criteria"]):
                errs.append(f"{name}: probability keys {sorted(p)} != options {sorted(q['criteria'])}")
            if abs(sum(p.values()) - 1) > 1e-3:
                errs.append(f"{name}: probabilities sum to {sum(p.values())}")
            if p and a.get("choice") != max(p, key=p.get):
                errs.append(f"{name}: choice is not the most probable option")
        if a.get("type") == "noul" and not 0 <= a.get("noul", -1) <= 1:
            errs.append(f"{name}: noul out of [0, 1]")
        if a.get("type") == "score":
            p = a.get("probabilities", {})
            if len(p) != len(q["criteria"]):
                errs.append(f"{name}: {len(p)} levels != {len(q['criteria'])}")
            ev = sum(int(k) * v for k, v in p.items())
            if abs(ev - a.get("score", -1)) > 2e-3:
                errs.append(f"{name}: score {a.get('score')} != expected level {ev:.4f}")
    return errs


def numbers_close(shown, got, tol, path="$"):
    """Every number in `shown` (a card example) matches `got` within tol; every key shown must exist in `got`."""
    errs = []
    if isinstance(shown, dict):
        for k, v in shown.items():
            if not isinstance(got, dict) or k not in got:
                errs.append(f"{path}.{k} missing")
            else:
                errs += numbers_close(v, got[k], tol, f"{path}.{k}")
    elif isinstance(shown, list):
        if not isinstance(got, list) or len(shown) != len(got):
            errs.append(f"{path}: {got!r} does not match the shown list")
        else:
            for i, (a, b) in enumerate(zip(shown, got)):
                errs += numbers_close(a, b, tol, f"{path}[{i}]")
    elif isinstance(shown, (bool, str)):
        if shown != got:
            errs.append(f"{path}: {got!r} != {shown!r}")
    elif isinstance(shown, (int, float)):
        if not isinstance(got, (int, float)) or abs(got - shown) > tol:
            errs.append(f"{path}: {got!r} not within {tol} of {shown!r}")
    return errs


def argparse_flags(path):
    """Long flags a script declares with add_argument (one-line argparse included)."""
    import ast
    flags = set()
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--"):
                    flags.add(a.value)
    return flags


def command_flags(code, script_suffix):
    """For each command in a bash block that runs a script ending with script_suffix: the long flags passed to it."""
    import shlex
    out = []
    for cmd in re.split(r"\n(?=\S)", code.replace("\\\n", " ")):
        try:
            words = shlex.split(cmd, comments=True)
        except ValueError:
            continue
        for i, w in enumerate(words):
            if w.endswith(script_suffix):
                out.append([x.split("=")[0] for x in words[i + 1:] if x.startswith("--")])
    return out


def frontmatter(text):
    """The card's YAML front matter as a flat dict (lists for "- item" lines); enough for license/base_model/tags."""
    if not text.startswith("---\n"):
        return {}
    fm, key = {}, None
    for raw in text[4:text.index("\n---\n", 4)].splitlines():
        if raw.lstrip().startswith("- "):
            fm.setdefault(key, []).append(raw.split("- ", 1)[1].strip())
        elif ":" in raw:
            key, _, val = raw.partition(":")
            key = key.strip()
            fm[key] = val.strip() or []
    return fm


def referenced_files(text, prefix="", suffixes=(".py", ".sh", ".json", ".gguf", ".safetensors", ".jinja", ".png", ".md")):
    """Repo files the prose names in backticks (the card's download-folder prefix removed) and its relative links."""
    out = set()
    prose = "\n".join(t for _, t in prose_lines(text))
    for span in re.findall(r"`([^`\n]+)`", prose):
        s = span.strip()
        if " " in s or s.startswith(("-", "~", "/", "<", "\u2026")) or "\u2026" in s:
            continue
        if prefix and s.startswith(prefix):
            s = s[len(prefix):]
        if s.endswith(suffixes):
            out.add(s)
    for link in re.findall(r"\]\(([^)\s]+)\)", prose):
        if not re.match(r"^(https?:|#|mailto:)", link):
            out.add(link.split("#")[0])
    return sorted(p for p in out if p)


def bash_curls(text):
    """(block, url, body) for every curl -d example in the card's bash blocks."""
    return [(b, url, body) for b in readme_blocks(text) if b["lang"] == "bash"
            for url, body in curl_json_bodies(b["code"])]


def wait_post_200(url, body, seconds, proc=None, log=None, every=5):
    """POST body until 200 and return the answer; a server process (proc) that exits first fails at once, quoting
    the end of its log."""
    t0, last = time.time(), None
    while time.time() - t0 < seconds:
        if proc is not None and proc.poll() is not None:
            tail = open(log, "rb").read().decode("utf-8", "replace")[-2000:] if log else ""
            raise RuntimeError(f"the server process exited with {proc.returncode} before {url} answered:\n{tail}")
        try:
            code, resp = post_example(url, body, timeout=60)
            if code == 200:
                return resp
            last = (code, resp)
        except OSError as e:
            last = repr(e)
        time.sleep(every)
    raise TimeoutError(f"{url} never answered 200: last {last}")


def ports_free(*ports):
    """True when nothing listens on these loopback ports (so the card's server, not another, will answer)."""
    import socket
    for port in ports:
        with socket.socket() as sk:
            if sk.connect_ex(("127.0.0.1", port)) == 0:
                return False
    return True


# ---------------------------------------------------------------------------------------------- shared checks
def adapter_key_layout(path, prefix="base_model.model.model.language_model.layers."):
    """LoRA keys of a PEFT adapter against the module path the serving engine matches (vLLM's
    Qwen3_5ForConditionalGeneration: base_model.model.model.language_model.layers.N.*). An adapter saved from a
    text-only load carries base_model.model.model.layers.N.* and is accepted silently by vLLM 0.30 while applying
    nothing. Returns (total keys, keys outside the prefix); only the header is read."""
    hdr, _ = safetensors_keys(path)
    bad = [k for k in hdr if not k.startswith(prefix)]
    return len(hdr), bad


def probe_moves(base_top, other_top, eps):
    """A documented check must be able to fail: the top-5 logprob maps of two servings of the same probe prompt
    (the base model and the model under test) must differ by more than eps on the base's top token. Returns
    (top token, difference)."""
    top = max(base_top, key=base_top.get)
    diff = abs(float(base_top[top]) - float(other_top.get(top, -99.0)))
    return top, diff if diff > eps else -diff
