"""Model-card conformance: every command, example, pin and claim in README.md is run or checked against the files of
one exact commit, parsed from the card itself.

    CARD_REV=<40-hex> pytest test_card.py -m static            # CPU, seconds: files, pins, layout, links, claims
    CARD_REV=<40-hex> CARD_LIVE=1 pytest test_card.py -m live  # a GPU: the card's launch commands and examples
    CARD_REV=<40-hex> CARD_BENCH231_DIR=... pytest test_card.py -m slow   # the numeric claims recomputed

CARD_DIR=<folder> uses an existing download of that commit instead of a fresh snapshot. The JSON report
(CARD_REPORT, default card_report.json) lists every file read with its sha256 and every result."""
import ast
import json
import os
import re
import shutil
import subprocess
import time
import unittest

import pytest

import card_common as C

REPO = C.env("CARD_REPO", C.DEFAULT_REPO)
REV = C.env("CARD_REV")
BASE_REPO, BASE_REV = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TOL_PROB = 0.15   # a probability shown in the card's example output vs the served value (other GPU, other build)
TOL_PROBE = 0.10  # the adapter-active probe: the adapter's top logprob must move at least this much from the base

# ---------------------------------------------------------------------------------------------------- the claims
# Registered claims: an exact quote from the card and how it is checked (tier). A quote that is no longer in the
# card fails the static tier, so the card cannot change a number without the test noticing.
CLAIMS = [
    {"quote": "| **v1.3.1** (`main`), two-way reading (the default; `bash serve.sh`) | **203 / 231 (87.9 %)** | 48/48 · 70/72 · 85/111 |",
     "tier": "slow", "check": "bench231", "mode": "two-way", "correct": 203, "tolerance": 2},
    {"quote": "| v1.3.1 (`main`), one-way reading (`TWO_WAY=0`; stock generate-mode vLLM) | **198 / 231 (85.7 %)** | 48/48 · 70/72 · 80/111 |",
     "tier": "slow", "check": "bench231", "mode": "one-way", "correct": 198, "tolerance": 2},
    {"quote": "| v1.3 (`--revision v1.3.0.5`), two-way | 202 / 231 (87.4 %) | 48/48 · 70/72 · 84/111 |", "tier": "static", "check": "quote",
     "note": "v1.3's run, kept as a comparison row"},
    {"quote": "| v1.2.3 (`--revision v1.2.3`), one-way | 205 / 231 (88.7 %) | 48/48 · 71/72 · 86/111 |", "tier": "static", "check": "quote"},
    {"quote": "v1.3.1 text **57.86 two-way / 56.72 one-way** (140,178 scored rows)", "tier": "slow", "check": "di_text",
     "values": {"two-way": 57.86, "one-way": 56.72}, "tolerance": 0.3},
    {"quote": "vision **67.13 two-way / 67.51 one-way** (public", "tier": "slow", "check": "di_vision",   # the card wraps the line after "(public"
     "values": {"two-way": 67.13, "one-way": 67.51}, "tolerance": 0.3},
    {"quote": "| v1.3 (`--revision v1.3.0.5`), two-way | 57.06 | 67.86 |", "tier": "static", "check": "quote", "note": "v1.3's Decision Index two-way re-read through the v1.3.0.5 package's serve.sh"},
    {"quote": "| v1.3 (`--revision v1.3.0.5`), one-way | 55.88 | 67.57 |", "tier": "static", "check": "quote"},
    {"quote": "| v1.2.3 (`--revision v1.2.3`), one-way | 51.41 | 63.09 |", "tier": "static", "check": "quote", "note": "v1.2.3 text and vision through its own package's serve.sh"},
    {"quote": "one-way | 52.25 | 64.27 |", "tier": "static", "check": "quote", "note": "the image tag's own serve.sh, one-way, max_state_tokens 1000000"},
    {"quote": "one-way | 51.35 | no image input in that package |", "tier": "static", "check": "quote", "note": "v1.1 text through its own package; the v1.1 shim takes no images"},
    {"quote": "gave the same top answer on 230 of 231 items", "tier": "static", "check": "quote",
     "note": "v1.2.3 pair (card: v1.2.3 weights and their adapter); the v1.3 pair's agreement is the hybrid claim below"},
    {"quote": "it ranks correct boxes above wrong ones with AUC 0.77, with expected calibration error 0.056",
     "tier": "static", "check": "quote", "note": "300 held-out public COCO/document/UI images; the set is not in this repo"},
    {"quote": "the shipped label rows against the served weights", "tier": "live", "check": "guard"},
    {"quote": "on 230 of 231 argmax answers (per-item max |dp| median 0.0004,", "tier": "static", "check": "quote",
     "note": "one-engine parity on the v1.3.1 weights: flagged read vs the package's serve.sh two-way read (231 items; 204 vs 203 correct)"},
]
HYBRID_CLAIM = re.compile(r"hybrid-mode agreement with the merged v1\.3(?:\.1)? weights on the same 231 items is (\d+) of 231")

# ---------------------------------------------------------------------------------------------------- fixtures
_STATE = {}


def folder():
    if "folder" not in _STATE:
        f, got = C.fresh_snapshot(REPO, REV)
        _STATE["folder"], _STATE["commit"] = f, got
    return _STATE["folder"]


def readme():
    if "readme" not in _STATE:
        # CARD_README_OVERRIDE: a mutated copy of the card for the negative controls (test_card_controls.py); the
        # files under test stay the commit's
        p = C.REPORT.read(C.env("CARD_README_OVERRIDE") or os.path.join(folder(), "README.md"))
        _STATE["readme"] = open(p, encoding="utf-8").read()
        _STATE["blocks"] = C.readme_blocks(_STATE["readme"])
    return _STATE["readme"]


def blocks(lang=None, heading_contains=None, code_contains=None):
    readme()
    out = _STATE["blocks"]
    if lang:
        out = [b for b in out if b["lang"] == lang]
    if heading_contains:
        out = [b for b in out if heading_contains.lower() in b["heading"].lower()]
    if code_contains:
        out = [b for b in out if code_contains in b["code"]]
    return out


def hub_files():
    if "hub_files" not in _STATE:
        from huggingface_hub import HfApi
        info = HfApi().model_info(REPO, revision=REV or _STATE.get("commit"), files_metadata=True)
        _STATE["hub_files"] = {s.rfilename: (s.lfs.sha256 if s.lfs else None, s.size) for s in info.siblings}
        _STATE["hub_commit"] = info.sha
    return _STATE["hub_files"]


def paths_map():
    """The card's relative names -> the snapshot folder (every command runs from the folder's parent)."""
    return {"h2o-lightning-4b/": folder() + "/", "./h2o-lightning-4b": folder(), "cd h2o-lightning-4b": "cd " + folder()}


# ---------------------------------------------------------------------------------------------------- static tier
@pytest.mark.static
class StaticTest(unittest.TestCase):
    def test_commit_is_the_one_asked_for(self):
        folder()
        if REV:
            self.assertTrue(C.HEX40.match(REV), "CARD_REV must be a full 40-hex commit")
            self.assertEqual(hub_files() and _STATE["hub_commit"], REV)

    def test_every_file_the_card_names_exists(self):
        names = C.referenced_paths(readme())
        present = set(os.path.basename(f) for f in hub_files()) | set(hub_files())
        local = set(os.listdir(folder()))
        # not shipped by design: label_rows.safetensors is the stale file the card tells users to delete,
        # .model_sha256.json is the sidecar the shim writes beside the weights at first use
        runtime = {"label_rows.safetensors", ".model_sha256.json", "results.jsonl"}
        missing = [n for n in names if n not in present and n not in local and n not in runtime]
        self.assertIn("adapter/adapter_model.safetensors", hub_files())
        C.REPORT.result("files_named", names)
        self.assertEqual(missing, [])

    def test_pins_and_tags_resolve(self):
        from huggingface_hub import HfApi
        api = HfApi()
        refs = api.list_repo_refs(REPO)
        tags = {t.name for t in refs.tags}
        named = set(re.findall(r"`(v1\.\d(?:\.\d)*(?:\.\d)?)`", readme())) | set(re.findall(r"--revision (v1[\w.]+)", readme()))
        named = {n.rstrip("`;,).") for n in named}
        missing = sorted(n for n in named if n not in tags)
        C.REPORT.result("tags_named", sorted(named))
        self.assertEqual(missing, [], f"tags named on the card but absent on the Hub: {missing}")
        for rev in set(re.findall(r"--revision ([0-9a-f]{40})", readme())):
            self.assertEqual(api.model_info(BASE_REPO, revision=rev).sha, rev)

    def test_layout_only_the_weights_carry_a_weight_suffix(self):
        top = sorted(f for f in hub_files() if "/" not in f and f.endswith((".safetensors", ".bin", ".pt")))
        self.assertEqual(top, ["model.safetensors"])
        self.assertIn("label_rows.lightning", hub_files())
        self.assertNotIn("label_rows.safetensors", hub_files())

    def test_sibling_pins_match_the_shipped_files(self):
        shim = open(C.REPORT.read(os.path.join(folder(), "h2o_lightning_shim.py")), encoding="utf-8").read()
        pins = dict(re.findall(r"'([\w.]+\.py)': '([0-9a-f]{64})'", shim.split("SIBLING_SHA256 = ", 1)[1].split("\n", 1)[0]))
        self.assertEqual(set(pins), {"box.py", "assets.py", "pooled_readout.py"})
        for name, want in pins.items():
            self.assertEqual(C.sha256_file(C.REPORT.read(os.path.join(folder(), name))), want, name)

    def test_adapter_keys_match_vllm_module_names(self):
        p = os.path.join(folder(), "adapter", "adapter_model.safetensors")
        if not os.path.isfile(p):
            self.skipTest("adapter/ not downloaded")
        total, bad = C.adapter_key_layout(C.REPORT.read(p))
        C.REPORT.result("adapter_keys", {"count": total, "without_language_model": len(bad)})
        self.assertEqual(total, 496)
        self.assertEqual(bad[:3], [], f"{len(bad)} keys would not match vLLM's Qwen3_5ForConditionalGeneration")
        cfg = json.load(open(C.REPORT.read(os.path.join(folder(), "adapter", "adapter_config.json"))))
        self.assertEqual(cfg["base_model_name_or_path"], BASE_REPO)

    def test_weights_are_not_the_base_model(self):
        if not os.path.isfile(os.path.join(folder(), "model.safetensors")):
            self.skipTest("model.safetensors not downloaded")
        name, frac = C.weights_differ_from_base(folder(), BASE_REPO, BASE_REV)
        C.REPORT.result("weights_vs_base", {"tensor": name, "equal_fraction": frac})
        self.assertLess(frac, 0.5)

    def test_json_blocks_parse_and_examples_pair_up(self):
        js = blocks("json")
        self.assertGreaterEqual(len(js), 2)
        for b in js:
            json.loads(b["code"])
        self.assertGreaterEqual(len(example_pairs()), 1)   # the ticket example; the image example is a python block

    def test_curl_bodies_parse(self):
        bodies = [x for b in blocks("bash") for x in C.curl_json_bodies(b["code"])]
        self.assertGreaterEqual(len(bodies), 4)
        for url, body in bodies:
            self.assertTrue(url and url.startswith("http://127.0.0.1:"), url)
            self.assertTrue("questions" in body or "messages" in body or "prompt" in body)

    def test_commands_use_flags_the_scripts_accept(self):
        serve = open(C.REPORT.read(os.path.join(folder(), "serve.sh")), encoding="utf-8").read()
        shim = open(os.path.join(folder(), "h2o_lightning_shim.py"), encoding="utf-8").read()
        for b in blocks("bash"):
            for var in re.findall(r"\b([A-Z][A-Z0-9_]{3,})=", b["code"]):
                if var in ("PYTHONPATH", "HF_TOKEN"):
                    continue
                self.assertTrue(var in serve or var in shim or var.startswith("VLLM_"), f"{var} is not read by serve.sh or the shim")
            if "h2o_lightning_shim.py" in b["code"]:
                for flag in re.findall(r"(--[a-z-]+)", b["code"].split("h2o_lightning_shim.py", 1)[1]):
                    self.assertIn(f'"{flag}"', shim, flag)

    def test_links_resolve(self):
        import urllib.request
        if C.env("CARD_OFFLINE"):
            self.skipTest("CARD_OFFLINE")
        urls = sorted(set(u.rstrip(".,)") for u in re.findall(r"https?://[^\s)>\]`\"']+", readme()) if "127.0.0.1" not in u))
        bad = {}
        for u in urls:
            if "wheels.vllm.ai" in u or "download.pytorch.org" in u:
                continue  # index URLs answer 403/404 to a bare GET
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (card-test)"}, method="GET")
                with urllib.request.urlopen(req, timeout=30) as r:
                    if r.status >= 400:
                        bad[u] = r.status
            except Exception as e:
                bad[u] = str(e)[:80]
        C.REPORT.result("links", {"checked": len(urls), "bad": bad})
        self.assertEqual(bad, {})

    def test_registered_claims_are_in_the_card(self):
        missing = C.claims_check(readme(), CLAIMS)
        self.assertEqual(missing, [])
        m = HYBRID_CLAIM.search(readme())
        self.assertTrue(m, "the card must state the v1.3 pair's hybrid agreement as 'N of 231'")
        C.REPORT.result("hybrid_claim", int(m.group(1)))

    def test_no_placeholder_tokens_in_the_card(self):
        """A card under preparation may carry __TOKEN__ placeholders for numbers still being measured; none may ship."""
        found = sorted(set(re.findall(r"__[A-Z0-9_]+__", readme())))
        self.assertEqual(found, [], f"placeholder tokens in README.md: {found}")

    def test_numeric_sentences_are_listed_in_the_report(self):
        """Every numeric sentence of the measured sections goes to the report; the registered ones are marked, so a new
        number on the card shows up as unregistered in the report for the reviewer."""
        rows = C.numeric_sentences(readme(), {"Measured", "Two-way reading (recommended)", "Hybrid mode: decisions and text from one server"})
        reg = [(s, l, any(c["quote"] in l or l in c["quote"] for c in CLAIMS) or bool(HYBRID_CLAIM.search(l))) for s, l in rows]
        C.REPORT.result("numeric_sentences", [{"section": s, "line": l, "registered": r} for s, l, r in reg])
        self.assertGreater(len(rows), 5)


# ---------------------------------------------------------------------------------------------------- live tier
def example_pairs():
    """(request json block, response json block) pairs: a json block followed by 'The response' and a json block."""
    rd = readme()
    pairs = []
    js = blocks("json")
    for a, b in zip(js, js[1:]):
        between = rd[a["pos"]:b["pos"]]
        req, resp = json.loads(a["code"]), json.loads(b["code"])
        # a request block holds "questions", a response block "answers"; a response followed by another response
        # (the python example's reply) is not a pair
        if "The response" in between and "questions" in req and "answers" in resp:
            pairs.append((req, resp))
    return pairs


class Served:
    """One launch command from the card, run as written from the snapshot folder's parent, stopped at the end."""

    def __init__(self, test, code, extra_env=None, ready="ready on http", health=None, seconds=1200, cwd=None):
        self.test = test
        cwd = cwd or os.path.dirname(folder())   # the card's commands run from the download's parent (h2o-lightning-4b/...)
        self.log = os.path.join(cwd, f"card_live_{int(time.time())}.log")
        code = C.substitute(code, paths_map())
        self.proc = C.run_card_command(code, cwd, self.log, extra_env)
        ok = C.wait_log(self.log, ready, seconds, self.proc) if ready else C.wait_http(health, seconds, self.proc)
        if not ok:
            tail = open(self.log, "rb").read()[-3000:].decode("utf-8", "replace")
            C.stop_tree(self.proc)
            test.fail(f"the card's command did not come up: {code[:120]}\n{tail}")

    def stop(self):
        C.stop_tree(self.proc)
        C.REPORT.read(self.log)


def assert_answer_shape(test, got, shown, tol=TOL_PROB):
    """The served answer against the card's shown answer: same keys, same types, same choice, numbers within tol."""
    test.assertEqual(set(got["answers"]), set(shown["answers"]))
    for q, s in shown["answers"].items():
        g = got["answers"][q]
        test.assertEqual(g["type"], s["type"], q)
        for k in s:
            test.assertIn(k, g, f"{q}.{k}")
        if "choice" in s:
            test.assertEqual(g["choice"], s["choice"], q)
        if "probabilities" in s:
            test.assertEqual(set(g["probabilities"]), set(s["probabilities"]))
            for k, v in s["probabilities"].items():
                test.assertAlmostEqual(g["probabilities"][k], v, delta=tol, msg=f"{q}.{k}")
        for k in ("noul", "score", "confidence"):
            if k in s:
                test.assertAlmostEqual(g[k], s[k], delta=tol * (2 if k == "score" else 1), msg=f"{q}.{k}")
    for k in ("probability_source", "effort_used"):
        test.assertEqual(got[k], shown[k], k)
    for k in shown["usage"]:
        test.assertIn(k, got["usage"])


def ticket_block():
    return [b for b in blocks("bash") if "Which team should take it?" in b["code"] and "urgency" in b["code"]][0]


@pytest.mark.live
@pytest.mark.skipif(not C.env("CARD_LIVE"), reason="CARD_LIVE=1 runs the card's launch commands on a GPU")
class LiveTest(unittest.TestCase):
    def run_examples(self, mode):
        url, body = C.curl_json_bodies(ticket_block()["code"])[0]
        st, r = C.post_example(url, body)
        self.assertEqual(st, 200, r)
        self.assertEqual(r["attention_mode"], mode)
        self.assertEqual(set(r["answers"]), set(body["questions"]))
        results = {"ticket": r}
        for i, (req, shown) in enumerate(example_pairs()):
            st, got = C.post_example("http://127.0.0.1:8741/v1/systemone", req)
            self.assertEqual(st, 200, got)
            assert_answer_shape(self, got, shown)
            results[f"example_{i}"] = got
        img = [b for b in blocks("python") if "example_receipt.png" in b["code"]]
        if img:
            code = C.substitute(img[0]["code"], paths_map())
            out = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=900, cwd=os.path.dirname(folder()))
            self.assertEqual(out.returncode, 0, out.stderr[-500:])
            got = json.loads(out.stdout)
            # the card shows the reply as the json block after the python example ("The response (rounded here)")
            shown = [json.loads(b["code"]) for b in blocks("json") if "answers" in json.loads(b["code"]) and "matches" in json.loads(b["code"])["answers"]]
            self.assertEqual(len(shown), 1, "the card must show exactly one reply for the receipt example")
            assert_answer_shape(self, got, shown[0])
            results["example_image"] = got
        return results

    def launch_block(self, needle):
        bs = [b for b in blocks("bash") if needle in b["code"] and "serve.sh" in b["code"]]
        self.assertTrue(bs, needle)
        line = [l for l in bs[0]["code"].splitlines() if needle in l][0].split("#")[0].strip()
        return line

    # numbered: unittest runs methods in name order and the hybrid test needs the one-way reference
    def test_1_default_two_way_serves_the_examples(self):
        s = Served(self, self.launch_block("bash h2o-lightning-4b/serve.sh"))
        try:
            C.REPORT.result("live_two_way", self.run_examples("bidirectional"))
            url, body = C.curl_json_bodies(ticket_block()["code"])[0]
            st, r = C.post_example(url, dict(body, attention_mode="causal"))
            self.assertEqual(st, 400, "a one-way request on the two-way-only deployment must get 400")
        finally:
            s.stop()

    def test_2_one_way_serves_the_examples(self):
        s = Served(self, self.launch_block("TWO_WAY=0 bash"))
        try:
            C.REPORT.result("live_one_way", self.run_examples("causal"))
        finally:
            s.stop()

    def test_3_box_pair_answers_a_box_question(self):
        s = Served(self, self.launch_block("LIGHTNING_BOX_ENABLED=1 bash"))
        try:
            self.assertIn("Box on", open(s.log, "rb").read().decode("utf-8", "replace"))
            import base64
            png = base64.b64encode(open(os.path.join(folder(), "example_receipt.png"), "rb").read()).decode()
            st, r = C.post_example("http://127.0.0.1:8741/v1/systemone", {
                "assets": {"IMG_1": {"type": "image", "format": "png", "data_base64": png}}, "state": {"image": "IMG_1"},
                "questions": {"where": {"type": "box", "instructions": "the total amount line in IMG_1"}}})
            self.assertEqual(st, 200, r)
            a = r["answers"]["where"]
            self.assertEqual(set(a["box"]), {"left", "top", "right", "bottom"})
            self.assertTrue(0 <= a["confidence"] <= 1 and 0 <= a["present"] <= 1)
            C.REPORT.result("live_box", r)
        finally:
            s.stop()

    def test_4_hybrid_mode_adapter_is_active(self):
        """The card's hybrid command (base model + adapter), then its own adapter-active check: the two names must
        differ; then the shim over the adapter answers the ticket like the merged weights."""
        serve = [b for b in blocks("bash") if "--lora-modules" in b["code"]][0]["code"]
        shim = [b for b in blocks("bash", heading_contains="Hybrid") if "h2o_lightning_shim.py" in b["code"]][0]["code"]
        probe = [b for b in blocks("bash") if "/v1/completions" in b["code"] and "for m in" in b["code"]][0]["code"]
        # the card's hybrid steps run from inside the folder (`... && cd h2o-lightning-4b`): ./adapter, serve_config.json
        s = Served(self, serve, ready=None, health="http://127.0.0.1:8000/health", cwd=folder())
        try:
            out = subprocess.run(["bash", "-c", probe], capture_output=True, text=True, timeout=300, cwd=folder())
            self.assertEqual(out.returncode, 0, out.stderr[-500:])
            lines = [l for l in out.stdout.splitlines() if l.strip()]
            self.assertEqual(len(lines), 2, out.stdout)
            vals = {l.split(" ", 1)[0]: ast.literal_eval(l.split(" ", 1)[1]) for l in lines}
            base, ours = vals[BASE_REPO], vals[REPO]
            top, diff = C.probe_moves(base, ours, TOL_PROBE)   # negative when the probe did not move
            C.REPORT.result("hybrid_probe", {"base": base, "adapter": ours, "top_token": top, "diff": abs(diff)})
            self.assertGreater(diff, 0, "the adapter's probe answer equals the base model's: the adapter is not applied")
            sh = Served(self, shim, ready=None, health="http://127.0.0.1:8741/health", seconds=300, cwd=folder())
            try:
                url, body = C.curl_json_bodies(ticket_block()["code"])[0]
                st, r = C.post_example(url, body)
                self.assertEqual(st, 200, r)
                C.REPORT.result("hybrid_ticket", r)
                ref = C.REPORT.data["results"].get("live_one_way", {}).get("ticket")
                self.assertTrue(ref, "the one-way reference is missing: run the live tier whole (test_2 serves it)")
                for q in ref["answers"]:
                    for k, v in (ref["answers"][q].get("probabilities") or {}).items():
                        self.assertAlmostEqual(r["answers"][q]["probabilities"][k], v, delta=0.05, msg=f"{q}.{k}")
                    if "noul" in ref["answers"][q]:
                        self.assertAlmostEqual(r["answers"][q]["noul"], ref["answers"][q]["noul"], delta=0.05, msg=f"{q}.noul")
            finally:
                sh.stop()
        finally:
            s.stop()


    def test_5_one_engine_serves_both_modes(self):
        """The card's ONE_ENGINE=1 line: one generate-mode engine on our vLLM build answers two-way (default), one-way and
        the Box question per request. Needs CARD_ONE_ENGINE_VLLM = a `vllm` executable of the per-request-attention
        build (an overlay venv); skipped otherwise, since stock vLLM cannot serve this layout."""
        wrapper = C.env("CARD_ONE_ENGINE_VLLM")
        if not wrapper:
            self.skipTest("CARD_ONE_ENGINE_VLLM (the per-request-attention vLLM build) runs the one-engine line")
        bindir = os.path.join(os.path.dirname(folder()), "card_one_engine_bin"); os.makedirs(bindir, exist_ok=True)
        link = os.path.join(bindir, "vllm")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(wrapper, link)
        s = Served(self, self.launch_block("ONE_ENGINE=1 bash"), extra_env={"PATH": bindir + os.pathsep + os.environ["PATH"], "LIGHTNING_BOX_ENABLED": "1"})
        try:
            self.assertIn("one engine", open(s.log, "rb").read().decode("utf-8", "replace"))
            url, body = C.curl_json_bodies(ticket_block()["code"])[0]
            got = {}
            for mode, extra in (("default", {}), ("bidirectional", {"attention_mode": "bidirectional"}), ("causal", {"attention_mode": "causal"})):
                st, r = C.post_example(url, dict(body, **extra))
                self.assertEqual(st, 200, (mode, r))
                got[mode] = r
            self.assertEqual(got["default"]["attention_mode"], "bidirectional")
            self.assertEqual(got["causal"]["attention_mode"], "causal")
            # the two modes answer differently on the probe ticket (an engine ignoring the flag would not)
            two, one = got["bidirectional"]["answers"]["outage"]["noul"], got["causal"]["answers"]["outage"]["noul"]
            self.assertGreater(abs(two - one), 0.005, "the flagged and unflagged reads gave the same answer")
            C.REPORT.result("live_one_engine", got)
        finally:
            s.stop()


# ---------------------------------------------------------------------------------------------------- slow tier
@pytest.mark.slow
class SlowTest(unittest.TestCase):
    def test_bench231_public_items(self):
        kit = C.env("CARD_BENCH231_DIR")
        if not (kit and C.env("CARD_LIVE")):
            self.skipTest("CARD_BENCH231_DIR (the benchmark checkout at the card's commit) and CARD_LIVE=1 run the 231 items")
        # the card's harness block: the one with "cli run" and a --results flag (the benchmark's own command string stays as written)
        cmd = [b for b in blocks("bash") if "cli run" in b["code"] and "--results" in b["code"]][0]["code"]
        results = {}
        for mode, launch, claim in (("two-way", "bash h2o-lightning-4b/serve.sh", CLAIMS[0]), ("one-way", "TWO_WAY=0 bash", CLAIMS[1])):
            line = [l for l in [b for b in blocks("bash") if launch in b["code"] and "serve.sh" in b["code"]][0]["code"].splitlines() if launch in l][0].split("#")[0].strip()
            s = Served(self, line)
            try:
                out = os.path.join(os.path.dirname(folder()), f"card_bench231_{mode}")
                shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
                code = re.sub(r"<[A-Z_]+>", kit, cmd, count=1)   # the card's first placeholder: the checkout path
                code = code.replace("<OUT>", out)                 # the card's output placeholder (results, raw, ledger, manifest)
                r = subprocess.run(["bash", "-c", code], capture_output=True, text=True, timeout=7200, cwd=kit)
                self.assertEqual(r.returncode, 0, r.stderr[-800:])
                rows = [json.loads(l) for l in open(os.path.join(out, "results.jsonl"))]
                correct = sum(bool(x.get("correct")) for x in rows)
                results[mode] = {"rows": len(rows), "correct": correct, "claim": claim["correct"]}
                self.assertEqual(len(rows), 231)
                self.assertLessEqual(abs(correct - claim["correct"]), claim["tolerance"], results)
            finally:
                s.stop()
        C.REPORT.result("bench231", results)

    def test_decision_index_public_suite(self):
        self.skipTest("the Decision Index 0.3 public suite (140,178 text rows, 12,210 vision rows) takes hours per mode; "
                      "run the kit's own scorer at the card's settings (max_state_tokens 1000000) and compare with "
                      "57.06 / 55.88 text and 67.86 / 67.57 vision within 0.3")
