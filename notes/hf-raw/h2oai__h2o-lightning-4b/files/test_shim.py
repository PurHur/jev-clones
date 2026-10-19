#!/usr/bin/env python3
"""Tests for h2o_lightning_shim.py against a stand-in for vLLM. No GPU, no network, standard library only.

    python3 test_shim.py

The stand-in speaks vLLM's /v1/models, /tokenize, /detokenize and /v1/completions (with `logprob_token_ids`) over
a toy tokenizer; a second stand-in speaks the pooling runner's /v1/embeddings for the two-way attention tests. The tests check the prompt, the answers (temperature, yes/no floor, score), several questions per
request, more options than one vLLM request reads, state truncation, the label check, and the status codes
an evaluation runner depends on.
"""
import base64
import copy
import json
import math
import os
import re
import sys
import threading
import unittest
from unittest import mock
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h2o_lightning_shim as S

CFG = json.load(open(os.path.join(HERE, "serve_config.json")))
CFG.pop("image", None)            # the text-path tests use the config as a text deployment; ImageTest adds its own block
if isinstance(CFG["temperature"], dict):   # the tests use one T; test_temperature_by_type covers the per-type form
    CFG["temperature"] = CFG["temperature"]["choice"]
PIECE = re.compile(r" ?[^\s]+|\s+")


class FakeVLLM(BaseHTTPRequestHandler):
    """Label j's log-probability is -0.3 * j unless the prompt carries MOCK_LP=[...]; MOCK_OVERFLOW answers vLLM's
    context error and MOCK_500 a server error."""
    vocab, inv, prompts, completions, lock = {}, {}, [], 0, threading.Lock()

    @classmethod
    def reset(cls):
        cls.vocab = {" " + l: 1000 + j for j, l in enumerate(CFG["labels"])}
        cls.inv = {i: p for p, i in cls.vocab.items()}
        cls.prompts, cls.completions = [], 0
        cls.chats, cls.skew = [], False
        cls.flagged, cls.flag_twist = [], 2.0        # the per-request attention flag (the one-engine fork)

    @classmethod
    def ids(cls, text):
        out = []
        with cls.lock:
            for p in PIECE.findall(text):
                if p not in cls.vocab:
                    cls.vocab[p] = 500000 + len(cls.vocab)
                    cls.inv[cls.vocab[p]] = p
                out.append(cls.vocab[p])
        return out

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"object": "list", "data": [{"id": CFG["model"], "max_model_len": 40960}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize" and "messages" in req:
            assert req["continue_final_message"] and not req["add_generation_prompt"]
            return self._send(200, {"tokens": self.ids(self.render(req["messages"]))})
        if self.path == "/tokenize":
            return self._send(200, {"tokens": self.ids(req["prompt"])})
        if self.path == "/v1/chat/completions":
            return self.chat(req)
        if self.path == "/detokenize":
            return self._send(200, {"prompt": "".join(self.inv[i] for i in req["tokens"])})
        text = req["prompt"]
        with FakeVLLM.lock:
            FakeVLLM.prompts.append(text)
            FakeVLLM.completions += 1
        if "MOCK_OVERFLOW" in text:
            return self._send(400, {"error": {"message": "This model's maximum context length is 40960 tokens."}})
        if "MOCK_500" in text:
            return self._send(500, {"error": {"message": "boom"}})
        ids = req["logprob_token_ids"]
        assert req["logprobs"] == len(ids) <= 128 and req["max_tokens"] == 1 and req["return_tokens_as_token_ids"]
        m = re.search(r"MOCK_LP=(\[[^\]]*\])", text)
        lp = json.loads(m.group(1)) if m else None
        twist = FakeVLLM.flag_twist if (req.get("vllm_xargs") or {}).get("attention_mode") == "bidirectional" else 0.0
        with FakeVLLM.lock:
            FakeVLLM.flagged.append((req.get("vllm_xargs") or {}).get("attention_mode"))
        top = {f"token_id:{i}": (lp[i - 1000] if lp else -0.3 * (i - 1000)) - (twist if i == 1000 else 0.0) for i in ids}
        self._send(200, {"choices": [{"text": "", "logprobs": {"top_logprobs": [top]}}],
                         "usage": {"prompt_tokens": len(self.ids(text)), "completion_tokens": 1}})


    # ---- the image path: chat completions, rendered as Qwen3.5's template renders a continued final assistant turn
    IMAGE_TOKENS = 9                    # each image's pad becomes this many tokens
    chats, skew = [], False             # skew: render the assistant turn wrongly (a template that is not the model's)

    @classmethod
    def render(cls, messages):
        out = ""
        for i, m in enumerate(messages):
            c = m["content"]
            if isinstance(c, list):
                c = "".join(S.VISION_SLOT if p["type"] == "image_url" else p["text"] for p in c)
            c = c.strip()
            if m["role"] == "assistant":
                assert i == len(messages) - 1
                out += ("<|im_start|>assistant\n" if cls.skew else "<|im_start|>assistant\n<think>\n\n</think>\n\n") + c
            else:
                out += f"<|im_start|>{m['role']}\n{c}<|im_end|>\n"
        return out

    def chat(self, req):
        text = self.render(req["messages"])
        n_img = sum(p["type"] == "image_url" for m in req["messages"] if isinstance(m["content"], list)
                    for p in m["content"])
        with FakeVLLM.lock:
            FakeVLLM.chats.append(req)
        if any(p.get("image_url", {}).get("url", "").endswith("MOCK_422.png") for m in req["messages"]
               if isinstance(m["content"], list) for p in m["content"]):
            return self._send(422, {"error": {"message": "Failed to fetch the image"}})
        ids = req["logprob_token_ids"]
        assert req["logprobs"] is True and len(ids) <= 128 and req["max_completion_tokens"] == 1
        assert req["continue_final_message"] and not req["add_generation_prompt"] and req["return_tokens_as_token_ids"]
        assert req["chat_template_kwargs"] == {"enable_thinking": False} and req["temperature"] == 0.0
        m = re.search(r"MOCK_LP=(\[[^\]]*\])", text)
        lp = json.loads(m.group(1)) if m else None
        twist = FakeVLLM.flag_twist if (req.get("vllm_xargs") or {}).get("attention_mode") == "bidirectional" else 0.0
        top = [{"token": f"token_id:{i}", "logprob": (lp[i - 1000] if lp else -0.3 * (i - 1000)) - (twist if i == 1000 else 0.0)} for i in ids]
        top.append({"token": "token_id:7", "logprob": -0.01})        # the sampled token, not a label
        self._send(200, {"choices": [{"message": {"content": ""}, "logprobs": {"content": [
            {"token": "token_id:7", "logprob": -0.01, "top_logprobs": top}]}}],
            "usage": {"prompt_tokens": len(self.ids(text)) + n_img * (self.IMAGE_TOKENS - 1), "completion_tokens": 1}})


def serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def post(url, body, path="/v1/systemone"):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def softmax_t(lp, T=CFG["temperature"]):
    e = [math.exp(x / T) for x in lp]
    return [x / sum(e) for x in e]


OK_Q = {"type": "choice", "instructions": "x", "criteria": {"a": "b"}}


class ShimTest(unittest.TestCase):
    def setUp(self):
        FakeVLLM.reset()
        self.vsrv, vurl = serve(FakeVLLM)
        self.shim = S.Shim(S.Contract(copy.deepcopy(CFG), env={}), S.VLLM(vurl, CFG["model"]))
        self.ssrv, self.url = serve(S.make_handler(self.shim))

    def tearDown(self):
        self.ssrv.shutdown()
        self.vsrv.shutdown()

    def test_choice_prompt_and_probabilities(self):
        body = {"state": {"doc": "MOCK_LP=[-0.2,-1.9,-3.0]"}, "questions": {"decision": {
            "type": "choice", "instructions": "Which?", "criteria": {"x": "first", "y": "second", "z": "third"}}}}
        st, r = post(self.url, body)
        self.assertEqual(st, 200, r)
        a = r["answers"]["decision"]
        for k, p in zip("xyz", softmax_t([-0.2, -1.9, -3.0])):
            self.assertAlmostEqual(a["probabilities"][k], p, places=12)
        self.assertEqual(a["choice"], "x")
        self.assertEqual(r["usage"]["output_tokens"], 1)   # one label token per answered question
        text = FakeVLLM.prompts[-1]
        self.assertTrue(text.startswith("<|im_start|>system\n" + CFG["prompt"]["system"] + "<|im_end|>\n"))
        self.assertIn('<|im_start|>user\nrecord: {"doc":"MOCK_LP=[-0.2,-1.9,-3.0]"}\nquestion: Which?\noptions:\n'
                      'A) x: first\nB) y: second\nC) z: third<|im_end|>\n', text)
        self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:"))

    def test_yes_no_floor_and_threshold(self):
        lp = [math.log(0.55), math.log(0.45)]
        q = {"type": "noul", "instructions": "Holds?", "criteria": {"true": "t", "false": "f"}, "threshold": 0.9}
        st, r = post(self.url, {"state": f"MOCK_LP={json.dumps(lp)}", "questions": {"d": q}})
        a = r["answers"]["d"]
        self.assertEqual((st, a["noul"], a["decision"]), (200, CFG["noul_floor"], False))
        lp = [math.log(0.02), math.log(0.98)]        # committed: left alone
        st, r = post(self.url, {"state": f"MOCK_LP={json.dumps(lp)}", "questions": {"d": q}})
        self.assertAlmostEqual(r["answers"]["d"]["noul"], softmax_t(lp)[0], places=12)

    def test_score_and_several_questions(self):
        qs = {"s": {"type": "score", "instructions": "rate", "criteria": ["lo", "mid", "hi"]},
              "c": {"type": "choice", "instructions": "pick", "criteria": {"a": "1", "b": "2"}},
              "n": {"type": "noul", "instructions": "ok?", "criteria": {"true": "y", "false": "n"}}}
        st, r = post(self.url, {"state": "MOCK_LP=[-1.0,-0.5,-2.0]", "questions": qs})
        self.assertEqual((st, set(r["answers"]), FakeVLLM.completions), (200, {"s", "c", "n"}, 3))
        p = softmax_t([-1.0, -0.5, -2.0])
        self.assertAlmostEqual(r["answers"]["s"]["score"], p[1] + 2 * p[2], places=12)
        self.assertEqual(r["answers"]["s"]["legend"], {"0": "lo", "1": "mid", "2": "hi"})
        self.assertEqual(r["answers"]["c"]["choice"], "b")

    def test_usage_counts_the_shared_head_once(self):
        qs = {"s": {"type": "score", "instructions": "rate", "criteria": ["lo", "mid", "hi"]},
              "c": {"type": "choice", "instructions": "pick", "criteria": {"a": "1", "b": "2"}},
              "n": {"type": "noul", "instructions": "ok?", "criteria": {"true": "y", "false": "n"}}}
        state = "MOCK_LP=[-1.0,-0.5,-2.0] " + " ".join(f"w{i}" for i in range(300))
        st, one = post(self.url, {"state": state, "questions": {"c": qs["c"]}})     # one question: its whole prompt
        self.assertEqual((st, one["usage"]["input_tokens"]), (200, len(FakeVLLM.ids(FakeVLLM.prompts[-1]))))
        singles = {}
        for qid, q in qs.items():
            st, r = post(self.url, {"state": state, "questions": {qid: q}})
            singles[qid] = r["answers"][qid]
        FakeVLLM.prompts = []
        st, r = post(self.url, {"state": state, "questions": qs})
        self.assertEqual(st, 200)
        self.assertEqual(json.dumps(r["answers"], sort_keys=True), json.dumps(singles, sort_keys=True))   # answers unchanged
        ids = [FakeVLLM.ids(t) for t in FakeVLLM.prompts]
        shared = S.common_prefix_len(ids)
        self.assertGreater(shared, 300)                   # the system prompt and the whole record
        self.assertEqual(r["usage"]["input_tokens"], shared + sum(len(x) - shared for x in ids))
        self.assertLess(r["usage"]["input_tokens"], sum(len(x) for x in ids))

    def test_common_prefix(self):
        self.assertEqual(S.common_prefix_len([[1, 2, 3], [1, 2, 4], [1, 2]]), 2)
        self.assertEqual(S.common_prefix_len([[1, 2, 3], [9, 2, 3]]), 0)      # differing prompts: the plain sum
        self.assertEqual(S.common_prefix_len([[1, 2, 3]]), 0)                 # one question: nothing shared
        self.assertEqual(S.common_prefix_len([[1, 2], [1, 2]]), 2)

    def test_many_options_are_read_in_chunks(self):
        lp = [-5.0] * 200
        lp[150] = -0.1
        q = {"type": "choice", "instructions": "wide", "criteria": {f"opt{i}": f"d{i}" for i in range(200)}}
        st, r = post(self.url, {"state": f"MOCK_LP={json.dumps(lp)}", "questions": {"decision": q}})
        self.assertEqual((st, FakeVLLM.completions, r["answers"]["decision"]["choice"]), (200, 2, "opt150"))

    def test_long_state_keeps_head_and_tail(self):
        self.shim.c.max_state_tokens = 40
        st, r = post(self.url, {"state": " ".join(f"w{i}" for i in range(100)), "questions": {"d": OK_Q}})
        text = FakeVLLM.prompts[-1]
        self.assertEqual(st, 200)
        self.assertIn("record: w0 w1", text)
        self.assertIn("\n...\n", text)
        self.assertIn(" w99\nquestion: x", text)
        self.assertNotIn(" w50 ", text)

    def test_kept_alive_connection_has_no_delayed_ack_stall(self):
        # one client connection, many requests (as a client with keep-alive sends them): every reply must arrive
        # promptly, not ~40 ms late (Nagle on the reply's second write vs the client's delayed ACK)
        import http.client, time, urllib.parse
        u = urllib.parse.urlsplit(self.url)
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
        body = json.dumps({"state": "s", "questions": {"d": OK_Q}}).encode()
        times = []
        for _ in range(12):
            t0 = time.perf_counter()
            c.request("POST", "/v1/systemone", body=body, headers={"Content-Type": "application/json"})
            r = c.getresponse(); r.read()
            self.assertEqual(r.status, 200)
            times.append(time.perf_counter() - t0)
        c.close()
        times.sort()
        self.assertLess(times[len(times) // 2], 0.02, times)

    def test_status_codes(self):
        self.assertEqual(post(self.url, {"state": "s", "questions": {}})[0], 400)
        self.assertEqual(post(self.url, {"state": "s", "questions": {"d": OK_Q}, "effort": "maximim"})[0], 400)
        self.assertEqual(post(self.url, {"questions": {"d": OK_Q}})[0], 422)
        self.assertEqual(post(self.url, {"state": "s", "questions": {"d": {"type": "rank", "criteria": {"a": 1}}}})[0], 422)
        wide = {"type": "choice", "instructions": "x", "criteria": {f"o{i}": "d" for i in range(256)}}
        self.assertEqual(post(self.url, {"state": "s", "questions": {"d": wide}})[0], 422)
        self.assertEqual(post(self.url, {"state": "MOCK_OVERFLOW", "questions": {"d": OK_Q}})[0], 422)
        self.assertEqual(post(self.url, {"state": "MOCK_500", "questions": {"d": OK_Q}})[0], 502)
        self.assertEqual(post(self.url, {"state": "s", "questions": {"d": OK_Q}}, "/api/alpha/decisions")[0], 200)

    def test_health_and_a_misconfigured_backend(self):
        with urllib.request.urlopen(self.url + "/health", timeout=30) as r:
            h = json.loads(r.read())
        self.assertTrue(h["ok"])
        self.assertEqual(h["noul_floor"]["value"], CFG["noul_floor"])
        self.assertEqual(h["temperature_by_type"]["noul"], CFG["temperature"])
        bad = copy.deepcopy(CFG)
        bad["labels"][3] = "X Y"                      # two tokens: not a label
        shim = S.Shim(S.Contract(bad, env={}), self.shim.v)
        with self.assertRaises(S.UpstreamError) as e:
            shim.check_backend()
        self.assertEqual(e.exception.status, 503)
        wrong = S.Shim(S.Contract(dict(copy.deepcopy(CFG), model="other"), env={}), S.VLLM(self.shim.v.base, "other"))
        with self.assertRaises(S.UpstreamError) as e:
            wrong.check_backend()
        self.assertEqual(e.exception.status, 503)

    def test_temperature_by_type(self):
        cfg = dict(copy.deepcopy(CFG), temperature={"choice": 0.75, "noul": 1.3, "score": 0.65})
        shim = S.Shim(S.Contract(cfg, env={}), self.shim.v)
        srv, url = serve(S.make_handler(shim))
        self.addCleanup(srv.shutdown)
        lp = [-0.2, -1.9, -3.0]
        qs = {"c": {"type": "choice", "instructions": "Which?", "criteria": {"x": "first", "y": "second", "z": "third"}},
              "s": {"type": "score", "instructions": "rate", "criteria": ["lo", "mid", "hi"]},
              "n": {"type": "noul", "instructions": "ok?", "criteria": {"true": "y", "false": "n"}}}
        st, r = post(url, {"state": f"MOCK_LP={json.dumps(lp)}", "questions": qs})
        self.assertEqual(st, 200)
        for k, x in zip("xyz", softmax_t(lp, 0.75)):
            self.assertAlmostEqual(r["answers"]["c"]["probabilities"][k], x, places=12)
        ps = softmax_t(lp, 0.65)
        self.assertAlmostEqual(r["answers"]["s"]["score"], ps[1] + 2 * ps[2], places=12)
        py = softmax_t(lp[:2], 1.3)[0]                     # the yes/no read at T 1.3, then the floor
        self.assertAlmostEqual(r["answers"]["n"]["noul"], CFG["noul_floor"] if max(py, 1 - py) < CFG["noul_floor"] else py,
                               places=12)
        self.assertEqual(shim.health()["temperature_by_type"], {"choice": 0.75, "noul": 1.3, "score": 0.65})
        one = S.Contract(cfg, env={"SHIM_TEMPERATURE": "1"})
        self.assertEqual(one.temperature_by_type, {"choice": 1.0, "noul": 1.0, "score": 1.0})
        with self.assertRaises(ValueError):
            S.Contract(dict(copy.deepcopy(CFG), temperature={"choice": 0.7, "noul": 1.0}), env={})
        self.assertEqual(self.shim.c.temperature_by_type, {t: CFG["temperature"] for t in S.PRIMITIVES})

    def test_a_dead_backend_is_a_502(self):
        self.vsrv.shutdown()
        self.vsrv.server_close()
        dead = S.Shim(S.Contract(copy.deepcopy(CFG), env={}), S.VLLM(self.shim.v.base, CFG["model"]))
        srv, url = serve(S.make_handler(dead))
        try:
            self.assertEqual(post(url, {"state": "s", "questions": {"d": OK_Q}})[0], 502)
        finally:
            srv.shutdown()
        self.vsrv, _ = serve(FakeVLLM)                # for tearDown


def png_bytes(w=2, h=1):
    """A valid w x h grey PNG, made with the standard library."""
    import struct
    import zlib

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = b"".join(b"\x00" + b"\x80" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


PNG = png_bytes()
PNG_B64 = S.base64.b64encode(PNG).decode()
PNG_URI = "data:image/png;base64," + PNG_B64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00\x10JFIF" + bytes(range(64))
JPEG_B64 = S.base64.b64encode(JPEG).decode()
ICFG = dict(copy.deepcopy(CFG), image={"max_images": 4, "max_pixels": 1048576, "temperature": 0.9})
Q3 = {"type": "choice", "instructions": "Which?", "criteria": {"x": "first", "y": "second", "z": "third"}}


class ImageTest(unittest.TestCase):
    """The image path (serve_config.json with an "image" block), and the text path left as it was."""

    def setUp(self):
        FakeVLLM.reset()
        self.vsrv, vurl = serve(FakeVLLM)
        self.shim = S.Shim(S.Contract(copy.deepcopy(ICFG), env={}), S.VLLM(vurl, CFG["model"]))
        self.ssrv, self.url = serve(S.make_handler(self.shim))

    def tearDown(self):
        self.ssrv.shutdown()
        self.vsrv.shutdown()

    def user_text(self, req):
        return req["messages"][1]["content"][-1]["text"]

    def image_urls(self, req):
        return [p["image_url"]["url"] for p in req["messages"][1]["content"] if p["type"] == "image_url"]

    def test_text_requests_take_the_text_path_unchanged(self):
        body = {"state": {"doc": "MOCK_LP=[-0.2,-1.9,-3.0]", "link": "https://example.com/a.png",
                          "kind": {"type": "image", "image": "a sunset over the bay"}, "code": "Qk" + "A" * 80},
                "questions": {"decision": Q3}}
        st, r = post(self.url, body)
        self.assertEqual((st, FakeVLLM.chats, FakeVLLM.completions), (200, [], 1))
        plain = S.Shim(S.Contract(copy.deepcopy(CFG), env={}), self.shim.v)   # the same request, a text deployment
        prompt_img_cfg = FakeVLLM.prompts[-1]
        self.assertEqual(json.dumps(plain.decide(copy.deepcopy(body))["answers"], sort_keys=True),
                         json.dumps(r["answers"], sort_keys=True))
        self.assertEqual(FakeVLLM.prompts[-1], prompt_img_cfg)
        self.assertIn('"link":"https://example.com/a.png"', prompt_img_cfg)
        self.assertIn('"kind":{"type":"image","image":"a sunset over the bay"}', prompt_img_cfg)
        with urllib.request.urlopen(self.url + "/health", timeout=30) as h:
            self.assertEqual(json.loads(h.read())["images"]["max_images"], 4)

    def test_a_text_deployment_ignores_images(self):
        plain = S.Shim(S.Contract(copy.deepcopy(CFG), env={}), self.shim.v)
        plain.decide({"state": {"photo": PNG_URI, "n": "MOCK_LP=[-0.2,-1.9,-3.0]"}, "questions": {"d": Q3}})
        self.assertEqual(FakeVLLM.chats, [])
        self.assertIn(PNG_B64, FakeVLLM.prompts[-1])           # read as text, as before
        self.assertNotIn("images", plain.health())

    def test_data_uri_in_state(self):
        body = {"state": {"photo": PNG_URI, "note": "MOCK_LP=[-0.2,-1.9,-3.0]"}, "questions": {"decision": Q3}}
        st, r = post(self.url, body)
        self.assertEqual((st, FakeVLLM.completions, len(FakeVLLM.chats)), (200, 0, 1), r)
        req = FakeVLLM.chats[0]
        self.assertEqual(req["messages"][0], {"role": "system", "content": CFG["prompt"]["system"]})
        self.assertEqual(req["messages"][2], {"role": "assistant", "content": CFG["prompt"]["prefill"]})
        self.assertEqual(self.image_urls(req), [PNG_URI])
        self.assertEqual(req["mm_processor_kwargs"], {"max_pixels": 1048576})
        self.assertEqual(self.user_text(req), 'record: {"photo":"[image 1]","note":"MOCK_LP=[-0.2,-1.9,-3.0]"}\n'
                                              'question: Which?\noptions:\nA) x: first\nB) y: second\nC) z: third')
        a = r["answers"]["decision"]
        for k, p in zip("xyz", softmax_t([-0.2, -1.9, -3.0], T=0.9)):        # the image temperature
            self.assertAlmostEqual(a["probabilities"][k], p, places=12)
        self.assertEqual((a["choice"], r["usage"]["images"]), ("x", 1))
        # the chat render is the text prompt with the image slot ahead of the user text
        text_prompt = S.render_chat(CFG["prompt"]["system"], self.user_text(req), "Answer:")
        self.assertEqual(FakeVLLM.render(req["messages"]),
                         text_prompt.replace("<|im_start|>user\n", "<|im_start|>user\n" + S.VISION_SLOT))
        self.assertTrue(self.shim.health()["images"]["verified"])

    # an image link is fetched by the shim itself (safe fetch) and reaches vLLM as a data URI
    @mock.patch.object(S, "_fetch_bytes", lambda url, max_bytes=None: base64.b64decode(JPEG_B64))
    def test_every_image_form(self):
        state = {"a": [{"type": "image_url", "image_url": {"url": PNG_URI}}, "keep me"],
                 "b": PNG_B64, "c": {"type": "input_image", "image_url": "data:image/jpeg;base64," + JPEG_B64},
                 "d": "MOCK_LP=[-0.2,-1.9,-3.0]"}
        body = {"state": state, "questions": {"decision": Q3},
                "images": [PNG_URI, "https://example.com/x.jpg", {"url": PNG_B64}],
                "image_data": {"type": "base64", "media_type": "image/jpeg", "data": JPEG_B64}}
        self.shim.c.max_images = 8
        st, r = post(self.url, body)
        self.assertEqual(st, 200, r)
        req = FakeVLLM.chats[0]
        jpeg_uri = "data:image/jpeg;base64," + JPEG_B64
        self.assertEqual(self.image_urls(req), [PNG_URI, jpeg_uri, PNG_URI, jpeg_uri,
                                                PNG_URI, PNG_URI, jpeg_uri])
        self.assertIn('record: {"a":["[image 5]","keep me"],"b":"[image 6]","c":"[image 7]",', self.user_text(req))
        self.assertEqual(r["usage"]["images"], 7)

    def test_multipart(self):
        boundary = "b0undary"
        req = json.dumps({"state": {"listing": {"color": "red"}, "n": "MOCK_LP=[-0.2,-1.9,-3.0]"},
                          "questions": {"decision": Q3}}).encode()
        parts = [(b'Content-Disposition: form-data; name="request"\r\nContent-Type: application/json', req),
                 (b'Content-Disposition: form-data; name="image"; filename="p.png"\r\nContent-Type: image/png', PNG)]
        data = b"".join(b"--" + boundary.encode() + b"\r\n" + h + b"\r\n\r\n" + d + b"\r\n" for h, d in parts)
        data += b"--" + boundary.encode() + b"--\r\n"
        r = urllib.request.Request(self.url + "/v1/systemone", data=data,
                                   headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(r, timeout=30) as resp:
            out = json.loads(resp.read())
        self.assertEqual(out["answers"]["decision"]["choice"], "x")
        self.assertEqual(self.image_urls(FakeVLLM.chats[0]), [PNG_URI])

    def test_limits_and_errors(self):
        self.assertEqual(post(self.url, {"state": {"p": PNG_URI}, "images": [PNG_URI] * 4,
                                         "questions": {"d": Q3}})[0], 422)            # 5 images, at most 4
        self.assertEqual(post(self.url, {"state": {"p": "data:image/png;base64,@@not base64@@"},
                                         "questions": {"d": Q3}})[0], 422)
        self.assertEqual(post(self.url, {"state": "s", "images": ["just words"], "questions": {"d": Q3}})[0], 422)
        self.assertEqual(post(self.url, {"state": "s", "images": [7], "questions": {"d": Q3}})[0], 422)
        self.assertEqual(FakeVLLM.chats, [])
        # an image vLLM cannot fetch is one bad item (422), not a backend failure (502)
        self.assertEqual(post(self.url, {"state": "s", "images": ["https://example.com/MOCK_422.png"],
                                         "questions": {"d": Q3}})[0], 422)

    def test_usage_counts_the_image_and_record_once(self):
        qs = {"c": Q3, "n": {"type": "noul", "instructions": "ok?", "criteria": {"true": "y", "false": "n"}},
              "s": {"type": "score", "instructions": "rate", "criteria": ["lo", "mid", "hi"]}}
        state = {"photo": PNG_URI, "words": " ".join(f"w{i}" for i in range(200))}
        st, r = post(self.url, {"state": state, "questions": qs})
        self.assertEqual((st, len(FakeVLLM.chats)), (200, 3))
        whole = [len(FakeVLLM.ids(FakeVLLM.render(c["messages"]))) + FakeVLLM.IMAGE_TOKENS - 1 for c in FakeVLLM.chats]
        shared = S.common_prefix_len([FakeVLLM.ids(FakeVLLM.render(c["messages"])) for c in FakeVLLM.chats])
        self.assertGreater(shared, 200)
        self.assertEqual(r["usage"]["input_tokens"], sum(whole) - 2 * (shared + FakeVLLM.IMAGE_TOKENS - 1))
        st, one = post(self.url, {"state": state, "questions": {"c": Q3}})
        last = FakeVLLM.chats[-1]["messages"]
        self.assertEqual(one["usage"]["input_tokens"], len(FakeVLLM.ids(FakeVLLM.render(last))) + FakeVLLM.IMAGE_TOKENS - 1)

    def test_many_options_are_read_in_chunks(self):
        lp = [-5.0] * 200
        lp[150] = -0.1
        q = {"type": "choice", "instructions": "wide", "criteria": {f"opt{i}": f"d{i}" for i in range(200)}}
        st, r = post(self.url, {"state": {"p": PNG_URI, "m": f"MOCK_LP={json.dumps(lp)}"}, "questions": {"d": q}})
        self.assertEqual((st, len(FakeVLLM.chats), r["answers"]["d"]["choice"]), (200, 2, "opt150"))

    def test_a_template_that_is_not_the_models_is_a_503(self):
        FakeVLLM.skew = True
        self.assertEqual(post(self.url, {"state": {"p": PNG_URI}, "questions": {"d": Q3}})[0], 503)
        self.assertEqual(FakeVLLM.chats, [])
        self.assertEqual(post(self.url, {"state": "text only", "questions": {"d": Q3}})[0], 200)   # text unaffected

    def test_lift_images(self):
        found = []
        state = {"s": "plain words", "n": 3, "l": ["x", {"y": None}], "u": "https://x.org/p.png",
                 "b64ish": "QUFB" * 40, "png": PNG_B64}
        out = S.lift_images(state, found)
        self.assertEqual(out["png"], "[image 1]")
        self.assertEqual(found, [PNG_URI])
        for k in ("s", "n", "l", "u", "b64ish"):
            self.assertIs(out[k], state[k])
        untouched = {"a": ["b", {"c": "d"}], "e": "A" * 100}
        self.assertIs(S.lift_images(untouched, []), untouched)
        self.assertEqual(S.sniff_image(b"RIFF\x00\x00\x00\x00WEBPVP8 "), "image/webp")
        self.assertEqual(S.extract_images({"state": "x"}), ("x", []))


class ServingFixesTest(unittest.TestCase):
    """A score's confidence is the probability of its most likely level; image decoding gives the same images without
    per-character scans; a kept-alive backend connection the server closed (or left idle) is replaced, a live one is
    reused."""

    @staticmethod
    def _png(w=80, h=60):
        import struct
        import zlib

        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

        rows = b"".join(b"\x00" + bytes((x * 7 + y * 3) % 256 for x in range(w * 3)) for y in range(h))
        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))

    def test_score_confidence_is_the_top_level_probability(self):
        lp = [math.log(x) for x in (0.1, 0.6, 0.3)]
        a = S.format_answer("score", ["0", "1", "2"], ["Low", "Medium", "High"], lp, 1.0, 0.5)
        self.assertAlmostEqual(a["confidence"], max(a["probabilities"].values()), places=12)
        c = S.format_answer("choice", ["a", "b", "c"], ["x", "y", "z"], lp, 1.0, 0.5)
        self.assertEqual(c["confidence"], S.confidence([c["probabilities"][k] for k in "abc"]))

    def test_images_decode_as_before(self):
        import base64

        raw = self._png()
        b = base64.b64encode(raw).decode()
        want = S._data_uri("image/png", raw)
        self.assertEqual(S.image_from_string("data:image/png;base64," + b, True), want)
        self.assertEqual(S.image_from_string(b, True), want)
        self.assertEqual(S.image_from_string(b, False), want)  # bare inside the record: one unbroken token
        self.assertEqual(S.image_from_string("data:image/png;base64," + base64.encodebytes(raw).decode(), True), want)
        self.assertIsNone(S.image_from_string("hello world " * 20, False))
        self.assertIsNone(S.image_from_string(b[:80] + " tail", False))  # whitespace: not one token
        self.assertIsNone(S._b64_bytes("not base64 !!"))
        self.assertEqual(S._b64_bytes(base64.urlsafe_b64encode(b"\xfb\xff\xfe" * 30).decode()), b"\xfb\xff\xfe" * 30)

    def test_backend_connections(self):
        import http.server
        import socket
        import threading
        import time

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
                self.close_connection = self.path == "/close"  # the server closes the kept-alive connection

            def log_message(self, *args):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            v = S.VLLM(f"http://127.0.0.1:{srv.server_address[1]}", "m")
            v.call("POST", "/ok", {})
            live = v.local.conn.sock
            v.call("POST", "/ok", {})
            self.assertIs(v.local.conn.sock, live)  # a live connection is reused
            v.call("POST", "/close", {})
            time.sleep(0.2)
            closed = v.local.conn.sock
            self.assertEqual(v.call("POST", "/ok", {"x": "y" * 2_000_000})[0], 200)  # a large body, fresh connection
            self.assertIsNot(v.local.conn.sock, closed)
            idle, S.IDLE_SECONDS = S.IDLE_SECONDS, 0.05
            try:
                first = v.local.conn.sock
                time.sleep(0.2)
                v.call("POST", "/ok", {})
                self.assertIsNot(v.local.conn.sock, first)  # idle past the limit: fresh
            finally:
                S.IDLE_SECONDS = idle
        finally:
            srv.shutdown()
            srv.server_close()
        a, b = socket.socketpair()
        self.assertFalse(S._peer_closed(a))
        b.close()
        self.assertTrue(S._peer_closed(a))
        a.close()


class OpenAIDecisionsTest(unittest.TestCase):
    """POST /v1/decisions (OpenAI's Decisions API) asks the model exactly what the equivalent /v1/systemone request
    asks, and returns the same probabilities in OpenAI's shape."""

    def setUp(self):
        FakeVLLM.reset()
        self.vsrv, vurl = serve(FakeVLLM)
        self.shim = S.Shim(S.Contract(copy.deepcopy(ICFG), env={}), S.VLLM(vurl, CFG["model"]))
        self.ssrv, self.url = serve(S.make_handler(self.shim))

    def tearDown(self):
        self.ssrv.shutdown()
        self.vsrv.shutdown()

    def test_same_prompts_and_answers_as_systemone(self):
        oai = {"model": "h2o-lightning-4b", "input": "Ticket: checkout is down for every customer.",
               "questions": [{"type": "predicate", "name": "outage", "instructions": "Is this an outage?"},
                             {"type": "choice", "name": "team", "instructions": "Which team?",
                              "choices": [{"value": "payments", "description": "Payments"}, {"value": "web"}]},
                             {"type": "score", "name": "urgency", "instructions": "How urgent?",
                              "levels": [{"label": "low"}, {"label": "medium", "description": "today"},
                                         {"label": "high"}]}]}
        st, r = post(self.url, oai, "/v1/decisions")
        self.assertEqual(st, 200, r)
        n0 = len(FakeVLLM.prompts)
        native = {"state": oai["input"], "questions": {
            "q_0": {"type": "noul", "instructions": "Is this an outage?"},
            "q_1": {"type": "choice", "instructions": "Which team?", "criteria": {"payments": "Payments", "web": None}},
            "q_2": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium: today", "high"]}}}
        st2, n = post(self.url, native)
        self.assertEqual(st2, 200, n)
        self.assertEqual(sorted(FakeVLLM.prompts[:n0]), sorted(FakeVLLM.prompts[n0:]))  # the model reads the same
        a, b = r["answers"], n["answers"]
        self.assertEqual([x["name"] for x in a], ["outage", "team", "urgency"])
        self.assertEqual(a[0]["probability"], b["q_0"]["noul"])
        self.assertEqual(a[1]["choice"], b["q_1"]["choice"])
        self.assertEqual(a[1]["confidence"], b["q_1"]["confidence"])
        self.assertEqual([(p["value"], p["probability"]) for p in a[1]["probabilities"]],
                         [(k, b["q_1"]["probabilities"][k]) for k in ("payments", "web")])
        self.assertEqual((a[2]["score"], a[2]["confidence"]), (b["q_2"]["score"], b["q_2"]["confidence"]))
        self.assertEqual([p["label"] for p in a[2]["probabilities"]], ["low", "medium", "high"])
        self.assertEqual(r["model"], CFG["model"])
        self.assertEqual(r["usage"]["total_tokens"], r["usage"]["input_tokens"] + 3)

    def test_typed_and_colliding_choice_values(self):
        q = [{"type": "choice", "instructions": "Pick", "choices": [{"value": "true"}, {"value": True}]}]
        st, r = post(self.url, {"model": CFG["model"], "input": "x", "questions": q}, "/v1/decisions")
        self.assertEqual(st, 200, r)
        self.assertEqual([p["value"] for p in r["answers"][0]["probabilities"]], ["true", True])
        self.assertIn(r["answers"][0]["choice"], ("true", True))

    def test_image_input(self):
        for detail in ("auto", "original"):
            FakeVLLM.reset()
            st, r = post(self.url, {"model": CFG["model"], "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "A receipt."},
                {"type": "input_image", "image_url": PNG_URI, "detail": detail}]}],
                "questions": [{"type": "predicate", "instructions": "Is it legible?"}]}, "/v1/decisions")
            self.assertEqual(st, 200, r)
            self.assertEqual(self.image_urls(FakeVLLM.chats[0]), [PNG_URI])  # the image reached vLLM
        bad = {"model": CFG["model"], "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": PNG_URI, "detail": "max"}]}],
            "questions": [{"type": "predicate", "instructions": "?"}]}
        self.assertEqual(post(self.url, bad, "/v1/decisions")[0], 400)

    def image_urls(self, req):
        return [p["image_url"]["url"] for m in req["messages"] if isinstance(m["content"], list)
                for p in m["content"] if p.get("type") == "image_url"]

    def test_model_and_shape_are_checked(self):
        q = [{"type": "predicate", "instructions": "ok?"}]
        self.assertEqual(post(self.url, {"model": CFG["model"], "input": "x", "questions": q}, "/v1/decisions")[0], 200)
        for bad in ({"model": "gpt-x", "input": "x", "questions": q},
                    {"model": CFG["model"], "input": "x", "questions": []},
                    {"model": CFG["model"], "input": "x", "questions": [{"type": "rank", "instructions": "?"}]},
                    {"model": CFG["model"], "input": "x", "questions": q, "extra": 1},
                    {"model": CFG["model"], "input": [{"role": "system", "content": "x"}], "questions": q}):
            self.assertEqual(post(self.url, bad, "/v1/decisions")[0], 400, bad)


class UsageTest(unittest.TestCase):
    """usage.output_tokens = the answered questions; usage.input_tokens = the full input even when a long record is cut
    (usage.processed_input_tokens = what vLLM read); the same on /v1/decisions."""

    def run_with(self, max_state_tokens, body, path="/v1/systemone"):
        FakeVLLM.reset()
        vsrv, vurl = serve(FakeVLLM)
        cfg = copy.deepcopy(CFG)
        cfg["prompt"] = dict(cfg["prompt"], max_state_tokens=max_state_tokens)
        shim = S.Shim(S.Contract(cfg, env={}), S.VLLM(vurl, CFG["model"]))
        ssrv, url = serve(S.make_handler(shim))
        try:
            return post(url, body, path)
        finally:
            ssrv.shutdown()
            vsrv.shutdown()

    def test_full_input_and_one_output_token_per_question(self):
        record = {"log": " ".join(f"line {i} ok" for i in range(400)), "error": "disk full on db-7"}
        qs = {"a": {"type": "noul", "instructions": "Is there an error?"},
              "b": {"type": "choice", "instructions": "Which node?", "criteria": {"db": "db-7", "web": "web-2"}}}
        st1, whole = self.run_with(100000, {"state": record, "questions": qs})
        st2, cut = self.run_with(64, {"state": record, "questions": qs})
        self.assertEqual((st1, st2), (200, 200))
        self.assertEqual(whole["usage"]["output_tokens"], 2)
        self.assertEqual(whole["usage"]["input_tokens"], whole["usage"]["processed_input_tokens"])
        self.assertEqual(cut["usage"]["input_tokens"], whole["usage"]["input_tokens"])  # billed as if read whole
        self.assertLess(cut["usage"]["processed_input_tokens"], cut["usage"]["input_tokens"])
        oai = {"model": CFG["model"], "input": "x " * 300,
               "questions": [{"type": "predicate", "instructions": "ok?"}, {"type": "predicate", "instructions": "no?"}]}
        st3, r = self.run_with(16, oai, "/v1/decisions")
        self.assertEqual(st3, 200, r)
        self.assertEqual(r["usage"]["output_tokens"], 2)
        self.assertEqual(r["usage"]["total_tokens"], r["usage"]["input_tokens"] + 2)
        self.assertLess(r["usage"]["processed_input_tokens"], r["usage"]["input_tokens"])


class BoxTest(unittest.TestCase):
    """The Box question (box.py) and `assets` (assets.py): the gate is off by default (the 400); switched on, the
    grounding is a free generation under box.SYSTEM and the two readouts go through the image path at the box
    temperature; an asset reference becomes an image; a bad asset map is a 400."""

    def make(self, enabled, temperature=None):
        cfg = copy.deepcopy(ICFG)
        cfg["box"] = {"enabled": enabled, **({"temperature": temperature} if temperature else {})}
        shim = S.Shim(S.Contract(cfg, env={}), S.VLLM(self.vurl, CFG["model"]))
        shim.check_backend = lambda: None   # a real backend check also fills the label token ids
        shim.check_chat = lambda: None
        shim.c.label_ids = {label: 1000 + i for i, label in enumerate(shim.c.labels)}
        return shim

    def setUp(self):
        FakeVLLM.reset()
        self.vsrv, self.vurl = serve(FakeVLLM)

    def tearDown(self):
        self.vsrv.shutdown()

    def box_body(self):
        return {"state": {"photo": "PHOTO_REF"}, "assets": {"PHOTO_REF": {"type": "image", "format": "png",
                "data_base64": PNG_URI.split(",", 1)[1]}},
                "questions": {"m": {"type": "box", "instructions": "the mug in `photo`"}}}

    def test_the_gate_is_off_by_default(self):
        with self.assertRaises(S.BadRequest) as e:
            self.make(False).decide(self.box_body())
        self.assertIn("unsupported_question_type", str(e.exception))

    def test_box_runs_on_the_chat_path(self):
        import math
        shim = self.make(True, temperature=2.0)
        gen, reads = [], []

        def chat_generate(messages, max_tokens=64, max_pixels=0, json_schema=None):
            gen.append(messages)
            return '{"bbox_2d": [100, 200, 600, 900]}', 300

        def label_logprobs_chat(messages, ids, max_pixels=0):
            reads.append(messages)
            return {ids[0]: math.log(0.8), ids[1]: math.log(0.2)}, {"prompt_tokens": 400}

        shim.v.chat_generate = chat_generate
        shim.v.label_logprobs_chat = label_logprobs_chat
        out = shim.decide(self.box_body())
        a = out["answers"]["m"]
        self.assertEqual(a["box"], {"left": 0.1, "top": 0.2, "right": 0.6, "bottom": 0.9})
        p = 0.8 ** 0.5 / (0.8 ** 0.5 + 0.2 ** 0.5)
        self.assertAlmostEqual(a["confidence"], p, places=9)
        self.assertAlmostEqual(a["present"], p, places=9)
        self.assertEqual((out["usage"]["input_tokens"], out["usage"]["output_tokens"]), (1100, 1))
        self.assertEqual(gen[0][0]["content"], S._sibling("box").SYSTEM)
        self.assertEqual(len(reads), 2)

    def test_mixed_box_and_decision_questions_merge(self):
        import math
        shim = self.make(True)
        shim.v.chat_generate = lambda messages, max_tokens=64, max_pixels=0, json_schema=None: ('{"bbox_2d": [0, 0, 500, 500]}', 10)
        shim.v.label_logprobs_chat = lambda m, ids, max_pixels=0, attention_mode=None: ({i: math.log(0.5) for i in ids}, {"prompt_tokens": 5})
        body = self.box_body()
        body["questions"]["q"] = {"type": "noul", "instructions": "ok?", "criteria": {"true": "yes", "false": "no"}}
        out = shim.decide(body)
        self.assertEqual(set(out["answers"]), {"m", "q"})
        self.assertEqual(out["usage"]["output_tokens"], 2)

    def test_edges_flat_reads_100_ordinal_bins_per_edge(self):
        import json as _json
        import math
        shim = self.make(True, temperature=1.0)
        shim.c.box_mode = "edges"
        box = S._sibling("box")
        hit = {"left": 10, "right": 60, "top": 20, "bottom": 90}
        texts = []

        def label_logprobs_chat(messages, ids, max_pixels=0):
            text = _json.dumps(messages)
            texts.append((text, len(ids)))
            if len(ids) == 2:
                return {ids[0]: math.log(0.8), ids[1]: math.log(0.2)}, {"prompt_tokens": 7}
            edge = next(e for e in box.EDGE_ORDER if f"the {e} edge" in text.lower())
            return {i: (0.0 if j == hit[edge] else -40.0) for j, i in enumerate(ids)}, {"prompt_tokens": 10}

        shim.v.chat_generate = lambda *a, **k: self.fail("edges mode does not generate")
        shim.v.label_logprobs_chat = label_logprobs_chat
        out = shim.decide(self.box_body())
        a = out["answers"]["m"]
        for e, i in hit.items():
            self.assertAlmostEqual(a["box"][e], (i + 0.5) / 100, places=6)
        self.assertEqual([n for _t, n in texts].count(100), 4)
        self.assertEqual(out["usage"]["input_tokens"], 4 * 10 + 2 * 7)

    def test_a_bad_asset_map_is_a_400(self):
        body = self.box_body()
        body["assets"]["PHOTO_REF"]["format"] = "gif"
        with self.assertRaises(S.BadRequest) as e:
            self.make(True).decide(body)
        self.assertIn("unsupported_assets", str(e.exception))


# ------------------------------------------------------------------------------------------- two-way attention
class FakePool(BaseHTTPRequestHandler):
    """A stand-in for the second vLLM (the pooling runner): /v1/embeddings returns a pooled vector whose dot products
    with the label rows reproduce FakeVLLM's log-probabilities (MOCK_LP or -0.3 * j), shifted by TWIST on the first
    label when `twisted` (a two-way engine answers differently from the one-way one); `causal` makes it answer exactly
    as FakeVLLM (an engine started without the override); `normalised` returns unit vectors."""
    HIDDEN, requests, lock = 8, [], threading.Lock()
    twisted, causal, normalised = True, False, False
    TWIST = 2.0

    @classmethod
    def reset(cls):
        cls.requests, cls.twisted, cls.causal, cls.normalised = [], True, False, False

    @classmethod
    def rows(cls):
        """Label j's row is the unit vector e_j (HIDDEN >= the number of labels used in the tests)."""
        return [[1.0 if k == j else 0.0 for k in range(cls.HIDDEN)] for j in range(cls.HIDDEN)]

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"object": "list", "data": [{"id": CFG["model"], "max_model_len": 40960}]})

    def do_POST(self):
        import struct
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize" and "messages" in req:
            return self._send(200, {"tokens": FakeVLLM.ids(FakeVLLM.render(req["messages"]))})
        if self.path == "/tokenize":
            return self._send(200, {"tokens": FakeVLLM.ids(req["prompt"])})
        if self.path == "/detokenize":
            return self._send(200, {"prompt": "".join(FakeVLLM.inv[i] for i in req["tokens"])})
        with FakePool.lock:
            FakePool.requests.append((self.path, req))
        assert self.path == "/v1/embeddings" and req["use_activation"] is False
        assert req["encoding_format"] == "base64" and req["embed_dtype"] == "float32"
        if "messages" in req:
            assert req["continue_final_message"] and not req["add_generation_prompt"]
            assert req["chat_template_kwargs"] == {"enable_thinking": False}
            text = FakeVLLM.render(req["messages"])
            n = len(FakeVLLM.ids(text))
        else:
            assert isinstance(req["input"], list) and all(isinstance(i, int) for i in req["input"])
            assert req["add_special_tokens"] is False
            text = "".join(FakeVLLM.inv[i] for i in req["input"])
            n = len(req["input"])
        m = re.search(r"MOCK_LP=(\[[^\]]*\])", text)
        lp = json.loads(m.group(1)) if m else None
        h = [(lp[j] if lp and j < len(lp) else -0.3 * j) for j in range(self.HIDDEN)]
        if self.twisted and not self.causal:
            h[0] -= self.TWIST
        if self.normalised:
            s = math.sqrt(sum(x * x for x in h))
            h = [x / s for x in h]
        vec = base64.b64encode(struct.pack(f"<{len(h)}f", *h)).decode()
        self._send(200, {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": vec}],
                         "usage": {"prompt_tokens": n, "total_tokens": n}})


class TwoWayTest(unittest.TestCase):
    """The two-way (bidirectional) reading: a request chooses the mode, the pooled state is read with the label rows,
    the two-way calibration applies, the usage reports what the engine read, and the checks refuse a second engine
    that answers like the first (no override), unit vectors, or label rows of another model."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        import pooled_readout as P
        cls.tmp = tempfile.mkdtemp()
        labels = CFG["labels"][:FakePool.HIDDEN]
        groups = [[l, [1000 + j]] for j, l in enumerate(labels)]
        cls.rows_path = os.path.join(cls.tmp, "label_rows.lightning")
        P.write_label_rows(cls.rows_path, FakePool.rows(), [1000 + j for j in range(FakePool.HIDDEN)], groups, 0,
                           FakePool.HIDDEN, {"model_sha256": "0" * 64})

    def setUp(self):
        FakeVLLM.reset()
        FakePool.reset()
        self.vsrv, vurl = serve(FakeVLLM)
        self.psrv, self.purl = serve(FakePool)
        self.vurl = vurl

    def tearDown(self):
        for s in (getattr(self, "ssrv", None), self.vsrv, self.psrv):
            if s is not None:
                s.shutdown()

    def start(self, two_way=None, env=None, with_engine=True, with_one_way=True, model_dir=None):
        cfg = copy.deepcopy(CFG)
        cfg["labels"] = CFG["labels"][:FakePool.HIDDEN]
        cfg["_config_dir"] = self.tmp
        if two_way is not None:
            cfg["two_way"] = two_way
        self.shim = S.Shim(S.Contract(cfg, env=env or {}), S.VLLM(self.vurl, CFG["model"]) if with_one_way else None,
                           two_way_vllm=S.VLLM(self.purl, CFG["model"]) if with_engine else None, model_dir=model_dir)
        self.ssrv, self.url = serve(S.make_handler(self.shim))
        return self.shim

    @staticmethod
    def reference(twist=None):
        """The stand-in's two-way answers to the shim's probes at T = 1 (label j reads -0.3 j, the first label
        shifted by TWIST): the anchors a deployment's guard compares the engine with."""
        twist = FakePool.TWIST if twist is None else twist
        out = []
        for body in S.TWO_WAY_PROBES:
            entry = {}
            for qid, q in body["questions"].items():
                names = S.options_of(q, FakePool.HIDDEN)[0]
                lp = [-0.3 * j - (twist if j == 0 else 0.0) for j in range(len(names))]
                entry[qid] = dict(zip(names, S.probabilities(lp, 1.0)))
            out.append(entry)
        return out

    TW = {"label_rows": "label_rows.lightning", "temperature": 0.5, "noul_floor": 0.9}

    def test_modes_read_the_same_prompt_from_the_two_engines(self):
        self.start(self.TW)
        body = {"state": {"doc": "MOCK_LP=[-0.2,-1.9,-3.0]"}, "questions": {"d": {
            "type": "choice", "instructions": "Which?", "criteria": {"x": "first", "y": "second", "z": "third"}}}}
        st, one = post(self.url, dict(body, attention_mode="causal"))
        self.assertEqual((st, one["attention_mode"]), (200, "causal"))
        st, two = post(self.url, dict(body, attention_mode="bidirectional"))
        self.assertEqual((st, two["attention_mode"]), (200, "bidirectional"))
        # the pooled engine received the exact token ids of the one-way prompt
        ids_sent = [r["input"] for p, r in FakePool.requests if "input" in r and "MOCK_LP=[-0.2" in "".join(FakeVLLM.inv[i] for i in r["input"])]
        want = [t for t in FakeVLLM.prompts if "MOCK_LP=[-0.2" in t][-1]
        self.assertEqual(ids_sent[-1], FakeVLLM.ids(want))
        # one-way: the one-way temperature; two-way: the shifted log-probabilities at the two-way temperature
        for k, p in zip("xyz", softmax_t([-0.2, -1.9, -3.0])):
            self.assertAlmostEqual(one["answers"]["d"]["probabilities"][k], p, places=9)
        for k, p in zip("xyz", softmax_t([-0.2 - FakePool.TWIST, -1.9, -3.0], 0.5)):
            self.assertAlmostEqual(two["answers"]["d"]["probabilities"][k], p, places=6)   # fp32 on the wire
        self.assertEqual((one["answers"]["d"]["choice"], two["answers"]["d"]["choice"]), ("x", "y"))
        # the default mode with the engine configured is two-way; "two-way" / "one-way" are accepted spellings
        st, r = post(self.url, body)
        self.assertEqual((st, r["attention_mode"]), (200, "bidirectional"))
        st, r = post(self.url, dict(body, attention_mode="one-way"))
        self.assertEqual((st, r["attention_mode"]), (200, "causal"))
        st, r = post(self.url, dict(body, attention_mode="sideways"))
        self.assertEqual(st, 400)
        h = json.loads(urllib.request.urlopen(self.url + "/health").read())
        self.assertEqual((h["attention_mode"], h["two_way"]["verified"], h["two_way"]["temperature_by_type"]["choice"]),
                         ("bidirectional", True, 0.5))
        self.assertGreaterEqual(h["two_way"]["probe_separation"], 0.05)

    def test_two_way_floor_and_usage(self):
        self.start(self.TW)
        lp = [math.log(0.6), math.log(0.4)]
        q = {"type": "noul", "instructions": "Holds?", "criteria": {"true": "t", "false": "f"}}
        # one-way: 0.6 at T 1 is under the 0.801 floor -> the floor; two-way: the twist flips it, floor 0.9 applies
        st, r = post(self.url, {"state": f"MOCK_LP={json.dumps(lp)}", "questions": {"d": q}, "attention_mode": "causal"})
        self.assertEqual((st, r["answers"]["d"]["noul"]), (200, CFG["noul_floor"]))
        st, r = post(self.url, {"state": f"MOCK_LP={json.dumps(lp)}", "questions": {"d": q}, "attention_mode": "bidirectional"})
        p_two = S.probabilities([lp[0] - FakePool.TWIST, lp[1]], 0.5)[0]       # the twist flips it; T 0.5, floor 0.9
        self.assertEqual(st, 200, r)
        self.assertAlmostEqual(r["answers"]["d"]["noul"], S.commit_noul(p_two, 0.9), places=6)
        self.assertNotAlmostEqual(r["answers"]["d"]["noul"], S.commit_noul(S.probabilities(lp, 1.0)[0], CFG["noul_floor"]), places=3)
        lp2 = [math.log(0.52), math.log(0.48)]                                   # under the two-way floor: moved to it
        st, r = post(self.url, {"state": f"MOCK_LP={json.dumps(lp2)}", "questions": {"d": q}, "attention_mode": "bidirectional"})
        p_two = S.probabilities([lp2[0] - FakePool.TWIST, lp2[1]], 0.5)[0]
        self.assertAlmostEqual(r["answers"]["d"]["noul"], S.commit_noul(p_two, 0.9), places=6)
        # usage: input_tokens counts the shared record once in both modes; processed_input_tokens says the two-way
        # engine read every prompt whole
        qs = {f"q{i}": {"type": "choice", "instructions": f"Which {i}?", "criteria": {"a": "x", "b": "y"}} for i in range(3)}
        st, one = post(self.url, {"state": "a long shared record " * 20, "questions": qs, "attention_mode": "causal"})
        st, two = post(self.url, {"state": "a long shared record " * 20, "questions": qs, "attention_mode": "bidirectional"})
        self.assertEqual(one["usage"]["input_tokens"], two["usage"]["input_tokens"])
        self.assertEqual(one["usage"]["processed_input_tokens"], one["usage"]["input_tokens"])
        self.assertGreater(two["usage"]["processed_input_tokens"], two["usage"]["input_tokens"])

    def test_default_mode_and_openai_route(self):
        self.start(dict(self.TW, default="causal"))
        body = {"state": "s", "questions": {"d": OK_Q}}
        st, r = post(self.url, body)
        self.assertEqual((st, r["attention_mode"]), (200, "causal"))
        self.start(self.TW, env={"SHIM_ATTENTION_MODE": "one-way"})
        st, r = post(self.url, body)
        self.assertEqual((st, r["attention_mode"]), (200, "causal"))
        oai = {"model": CFG["model"], "input": "s", "attention_mode": "two-way",
               "questions": [{"type": "choice", "name": "d", "instructions": "Which?", "choices": [{"value": "a", "description": "x"}, {"value": "b", "description": "y"}]}]}
        st, r = post(self.url, oai, "/v1/decisions")
        self.assertEqual(st, 200, r)
        self.assertTrue(any("input" in req for _p, req in FakePool.requests))

    def test_without_the_engine_two_way_is_a_400_and_the_default_is_one_way(self):
        self.start(self.TW, with_engine=False)
        body = {"state": "s", "questions": {"d": OK_Q}}
        st, r = post(self.url, body)
        self.assertEqual((st, r["attention_mode"]), (200, "causal"))
        st, r = post(self.url, dict(body, attention_mode="bidirectional"))
        self.assertEqual(st, 400)
        self.assertIn("not running", r["detail"])
        with self.assertRaises(ValueError):          # the engine without the block
            cfg = copy.deepcopy(CFG); cfg["labels"] = CFG["labels"][:FakePool.HIDDEN]; cfg.pop("two_way", None)
            S.Shim(S.Contract(cfg, env={}), S.VLLM(self.vurl, CFG["model"]), two_way_vllm=S.VLLM(self.purl, CFG["model"]))
        # a one-way deployment never reads the label-row file: a config whose file is missing still serves
        cfg = copy.deepcopy(CFG); cfg["labels"] = CFG["labels"][:FakePool.HIDDEN]; cfg["two_way"] = {"label_rows": "missing.safetensors"}
        S.Shim(S.Contract(cfg, env={}), S.VLLM(self.vurl, CFG["model"]))
        with self.assertRaises(OSError):             # with the engine, the file must exist
            S.Shim(S.Contract(cfg, env={}), S.VLLM(self.vurl, CFG["model"]), two_way_vllm=S.VLLM(self.purl, CFG["model"]))

    def test_checks_refuse_a_one_way_engine_unit_vectors_and_foreign_rows(self):
        body = {"state": "s", "questions": {"d": OK_Q}, "attention_mode": "bidirectional"}
        FakePool.causal = True                        # started without the override: answers as the one-way engine
        self.start(self.TW)
        st, r = post(self.url, body)
        self.assertEqual(st, 503, r)
        self.assertIn("answers like the one-way one", r["detail"])
        FakePool.causal = False                       # fixed: refused for TWO_WAY_RETRY_SECONDS, then checked again
        st, r = post(self.url, body)
        self.assertEqual(st, 503, r)
        self.assertIn("checked again", r["detail"])
        with mock.patch.object(S, "TWO_WAY_RETRY_SECONDS", 0.0):
            st, r = post(self.url, body)
        self.assertEqual(st, 200, r)
        self.ssrv.shutdown()
        FakePool.normalised = True
        self.start(self.TW)
        st, r = post(self.url, body)
        self.assertEqual(st, 503, r)
        self.assertIn("L2-normalised", r["detail"])
        self.ssrv.shutdown()
        FakePool.normalised = False
        import pooled_readout as P
        other = os.path.join(self.tmp, "other_rows.safetensors")
        P.write_label_rows(other, FakePool.rows(), [2000 + j for j in range(FakePool.HIDDEN)],
                           [[l, [2000 + j]] for j, l in enumerate(CFG["labels"][:FakePool.HIDDEN])], 0, FakePool.HIDDEN, {})
        self.start(dict(self.TW, label_rows=other))
        st, r = post(self.url, body)
        self.assertEqual(st, 503, r)
        self.assertIn("another model or tokenizer", r["detail"])

    def test_one_engine_two_way_deployment(self):
        """The shipped default: the two-way engine alone. Requests default to two-way; a one-way request is a 400,
        never a silent two-way answer; a box question is a 400 (a generation); the guard rests on the release's
        recorded probe answers (anchors) and refuses an engine that answers differently; the tokenizer calls go to
        the two-way engine."""
        tw = dict(self.TW, probe_reference=self.reference())
        self.start(tw, with_one_way=False)
        body = {"state": "s", "questions": {"d": OK_Q}}
        st, r = post(self.url, body)
        self.assertEqual((st, r["attention_mode"]), (200, "bidirectional"), r)
        self.assertEqual(FakeVLLM.completions, 0)                     # nothing went to a one-way engine
        st, r = post(self.url, dict(body, attention_mode="causal"))
        self.assertEqual(st, 400, r)
        self.assertIn("one-way attention is not running", r["detail"])
        h = json.loads(urllib.request.urlopen(self.url + "/health").read())
        self.assertEqual((h["attention_mode"], h["modes"], h["backend"]), ("bidirectional", ["bidirectional"], None))
        self.assertLess(h["two_way"]["probe_anchor_distance"], 1e-6)
        self.assertIsNone(h["two_way"]["probe_separation"])
        cfg_box = {"enabled": True}
        self.ssrv.shutdown()
        cfg = copy.deepcopy(CFG); cfg["labels"] = CFG["labels"][:FakePool.HIDDEN]; cfg["_config_dir"] = self.tmp
        cfg["two_way"] = tw; cfg["image"] = {"max_images": 2, "max_pixels": 1000000, "temperature": 0.65}; cfg["box"] = cfg_box
        self.shim = S.Shim(S.Contract(cfg, env={}), None, two_way_vllm=S.VLLM(self.purl, CFG["model"]))
        self.ssrv, self.url = serve(S.make_handler(self.shim))
        st, r = post(self.url, {"state": {"image": "[image 1]"}, "images": [PNG_URI],
                                "questions": {"w": {"type": "box", "instructions": "the thing"}}})
        self.assertEqual(st, 400, r)
        self.assertIn("LIGHTNING_BOX_ENABLED=1", r["detail"])
        st, r = post(self.url, {"state": "s", "images": [PNG_URI], "questions": {"d": OK_Q}})   # the image path, two-way
        self.assertEqual((st, r["attention_mode"]), (200, "bidirectional"), r)
        # the anchors: an engine answering differently from the release is refused
        self.ssrv.shutdown()
        self.start(dict(self.TW, probe_reference=self.reference(twist=0.0)), with_one_way=False)
        st, r = post(self.url, body)
        self.assertEqual(st, 503, r)
        self.assertIn("does not answer the probes as the release did", r["detail"])
        self.ssrv.shutdown()
        self.start(self.TW, with_one_way=False)                       # no anchors recorded: refused
        st, r = post(self.url, body)
        self.assertEqual(st, 503, r)
        self.assertIn("probe_reference is missing", r["detail"])

    def test_label_rows_identity_against_the_served_weights(self):
        """With model.safetensors beside the config, the label rows' model_sha256 must be its hash."""
        import hashlib, tempfile
        import pooled_readout as P
        d = tempfile.mkdtemp()
        open(os.path.join(d, "model.safetensors"), "wb").write(b"not really weights " * 1000)
        sha = hashlib.sha256(open(os.path.join(d, "model.safetensors"), "rb").read()).hexdigest()
        labels = CFG["labels"][:FakePool.HIDDEN]
        for name, h in (("rows_same.safetensors", sha), ("rows_other.safetensors", "1" * 64)):
            P.write_label_rows(os.path.join(self.tmp, name), FakePool.rows(), [1000 + j for j in range(FakePool.HIDDEN)],
                               [[l, [1000 + j]] for j, l in enumerate(labels)], 0, FakePool.HIDDEN, {"model_sha256": h})
        body = {"state": "s", "questions": {"d": OK_Q}, "attention_mode": "bidirectional"}
        self.start(dict(self.TW, label_rows="rows_other.safetensors"), model_dir=d)
        st, r = post(self.url, body)
        self.assertEqual(st, 503, r)
        self.assertIn("belong to another checkpoint", r["detail"])
        self.ssrv.shutdown()
        self.start(dict(self.TW, label_rows="rows_same.safetensors"), model_dir=d)
        st, r = post(self.url, body)
        self.assertEqual(st, 200, r)
        self.assertTrue(os.path.isfile(os.path.join(d, ".model_sha256.json")))     # hashed once, kept beside the weights
        st, r = post(self.url, body)
        self.assertEqual(st, 200, r)

    def test_block_validation_and_warnings(self):
        cfg = copy.deepcopy(CFG); cfg["labels"] = CFG["labels"][:FakePool.HIDDEN]; cfg["_config_dir"] = self.tmp
        c = S.Contract(dict(cfg, two_way={"label_rows": "label_rows.lightning"}), env={})
        self.assertEqual(c.two_way.temperature_by_type, c.temperature_by_type)
        self.assertEqual(len(c.two_way.warnings), 2)                   # temperature and floor fall back, with a warning
        for bad in ({"label_rows": "label_rows.lightning", "temperature": 99},
                    {"label_rows": "label_rows.lightning", "noul_floor": 0.3},
                    {"label_rows": "label_rows.lightning", "default": "sideways"},
                    {"label_rows": "label_rows.lightning", "temperature": {"choice": 1}},
                    {"label_rows": "label_rows.lightning", "probe_tolerance": 0.7},
                    {"label_rows": "label_rows.lightning", "probe_reference": [{}]},
                    {"label_rows": "label_rows.lightning", "probe_reference": [{"outage": {"true": 2}}, {}, {}]}):
            with self.assertRaises(ValueError):
                S.Contract(dict(cfg, two_way=bad), env={})
        with self.assertRaises(ValueError):                              # the file's labels are not the deployment's
            S.Contract(dict(cfg, labels=["Z", "Y"], two_way={"label_rows": "label_rows.lightning"}), env={}).two_way.load_rows()

    def test_image_path_two_way(self):
        cfg_img = {"max_images": 2, "max_pixels": 1000000, "temperature": 0.65}
        cfg = copy.deepcopy(CFG); cfg["labels"] = CFG["labels"][:FakePool.HIDDEN]; cfg["_config_dir"] = self.tmp
        cfg["image"] = cfg_img
        cfg["two_way"] = dict(self.TW, image_temperature=0.4)
        self.shim = S.Shim(S.Contract(cfg, env={}), S.VLLM(self.vurl, CFG["model"]), two_way_vllm=S.VLLM(self.purl, CFG["model"]))
        self.ssrv, self.url = serve(S.make_handler(self.shim))
        body = {"state": {"doc": "MOCK_LP=[-0.2,-1.9,-3.0]"}, "images": [PNG_URI], "attention_mode": "bidirectional",
                "questions": {"d": {"type": "choice", "instructions": "Which?", "criteria": {"x": "first", "y": "second", "z": "third"}}}}
        st, r = post(self.url, body)
        self.assertEqual(st, 200, r)
        req = [q for p, q in FakePool.requests if "messages" in q][-1]
        self.assertEqual(req["mm_processor_kwargs"], {"max_pixels": 1000000})
        self.assertEqual([p["image_url"]["url"] for m in req["messages"] if isinstance(m["content"], list)
                          for p in m["content"] if p["type"] == "image_url"], [PNG_URI])
        for k, p in zip("xyz", softmax_t([-0.2 - FakePool.TWIST, -1.9, -3.0], 0.4)):
            self.assertAlmostEqual(r["answers"]["d"]["probabilities"][k], p, places=6)
        self.assertEqual(r["usage"]["images"], 1)


class SiblingPinTest(unittest.TestCase):
    """box.py, assets.py and pooled_readout.py beside the shim are the shipped files (their sha256 is pinned in the
    shim); a file with other bytes is refused with a 503 before it runs."""

    def test_the_shipped_files_match_their_pins(self):
        import hashlib
        for name, want in S.SIBLING_SHA256.items():
            with open(os.path.join(os.path.dirname(os.path.abspath(S.__file__)), name), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), want, name)

    def test_a_changed_file_is_refused(self):
        import sys as _sys
        saved = dict(S.SIBLING_SHA256), _sys.modules.pop("_lightning_box", None), S._SIBLINGS.pop("_lightning_box", None)
        try:
            S.SIBLING_SHA256["box.py"] = "0" * 64
            with self.assertRaises(S.UpstreamError) as e:
                S._sibling("box")
            self.assertEqual(e.exception.status, 503)
        finally:
            S.SIBLING_SHA256.clear()
            S.SIBLING_SHA256.update(saved[0])
            if saved[1] is not None:
                _sys.modules["_lightning_box"] = saved[1]
            if saved[2] is not None:
                S._SIBLINGS["_lightning_box"] = saved[2]

    def test_concurrent_first_loads_get_the_whole_module(self):
        import sys as _sys
        import threading as _th
        saved = dict(S._SIBLINGS), _sys.modules.pop("_lightning_box", None)
        S._SIBLINGS.pop("_lightning_box", None)
        got, start = [], _th.Barrier(8)

        def load():
            start.wait()
            got.append(hasattr(S._sibling("box"), "run"))

        try:
            threads = [_th.Thread(target=load) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(got, [True] * 8)
        finally:
            S._SIBLINGS.clear()
            S._SIBLINGS.update(saved[0])
            if saved[1] is not None:
                _sys.modules["_lightning_box"] = saved[1]

    def test_no_other_safetensors_beside_the_weights(self):
        """vLLM's weight loader globs *.safetensors (and *.bin) in the model folder: nothing but model.safetensors may
        carry those suffixes in the package (the label rows ship as label_rows.lightning for that reason)."""
        import glob
        here = os.path.dirname(os.path.abspath(S.__file__))
        extra = [os.path.basename(p) for pat in ("*.safetensors", "*.bin", "*.pt") for p in glob.glob(os.path.join(here, pat))
                 if os.path.basename(p) != "model.safetensors"]
        self.assertEqual(extra, [])
        self.assertTrue(os.path.isfile(os.path.join(here, "label_rows.lightning")))

    def test_adapter_keys_carry_the_language_model_segment(self):
        """vLLM loads Qwen3.5 as Qwen3_5ForConditionalGeneration and matches LoRA tensors by name under
        base_model.model.model.language_model.layers.N; an adapter saved from a text-only load of the model
        (base_model.model.model.layers.N) is accepted silently and changes nothing, so hybrid mode would serve the
        base model. Every key of the shipped adapter must carry the language_model segment (the header is read, not
        the tensors)."""
        import struct
        here = os.path.dirname(os.path.abspath(S.__file__))
        path = os.path.join(here, "adapter", "adapter_model.safetensors")
        if not os.path.isfile(path):
            self.skipTest("no adapter/adapter_model.safetensors beside the shim (a download without --include adapter/*)")
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            keys = [k for k in json.loads(f.read(n)) if k != "__metadata__"]
        self.assertTrue(keys)
        bad = [k for k in keys if not k.startswith("base_model.model.model.language_model.layers.")]
        self.assertEqual(bad[:3], [], f"{len(bad)} of {len(keys)} adapter keys lack the language_model segment")


class AssetsOnceTest(unittest.TestCase):
    """Each distinct asset reference is bound once: every later use (state, instructions, criteria, inside a word)
    points at the same [image N], so the model reads the image once."""

    @staticmethod
    def png(rgb):
        import base64, struct, zlib
        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
        raw = b"".join(b"\x00" + bytes(rgb) * 4 for _ in range(4))
        return base64.b64encode(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
                                + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")).decode()

    def test_each_distinct_reference_is_bound_once(self):
        body = {"state": {"a": "IMG_1", "b": "see IMG_1 and IMG_2", "c": "xIMG_1x"},
                "assets": {"IMG_1": {"type": "image", "format": "png", "data_base64": self.png([200, 0, 0])},
                           "IMG_2": {"type": "image", "format": "png", "data_base64": self.png([0, 0, 200])}},
                "questions": {"q": {"type": "choice", "instructions": "Is IMG_2 like IMG_1?",
                                    "criteria": {"y": "yes, IMG_1", "n": "no"}}}}
        out = S.bind_assets(body)
        self.assertEqual(len(out["images"]), 2)
        self.assertEqual(out["state"], {"a": "[image 1]", "b": "see [image 1] and [image 2]", "c": "x[image 1]x"})
        self.assertEqual(out["questions"]["q"]["instructions"], "Is [image 2] like [image 1]?")
        self.assertEqual(out["questions"]["q"]["criteria"]["y"], "yes, [image 1]")


class BoxGenerationConfidenceTest(unittest.TestCase):
    """box.confidence "generation": the one grounding generation carries its token logprobs, the box confidence is
    their calibrated value, and no tightness readout is made (presence off: none at all)."""

    @staticmethod
    def tokens(text, p=0.8):
        import math, re
        out = []
        for tok in re.findall(r"\d|[^\d]+", text):
            if tok.isdigit():
                d = int(tok)
                out.append((tok, math.log(p), [(tok, math.log(p)), (str((d + 1) % 10), math.log(0.15)), (str((d + 9) % 10), math.log(0.05))]))
            else:
                out.append((tok, 0.0, [(tok, 0.0)]))
        return out

    def test_generation_confidence_is_calibrated_and_reads_nothing_else(self):
        cfg = copy.deepcopy(ICFG)
        cfg["box"] = {"enabled": True, "confidence": "generation", "presence": False,
                      "generation_calibration": {"a": 0.7088, "b": 1.6955}}
        shim = S.Shim(S.Contract(cfg, env={}), S.VLLM("http://127.0.0.1:9", CFG["model"]))
        box = S._sibling("box")
        conv = shim.c.box_convention
        text = '{"bbox_2d": [100, 200, 600, 800]}' if conv == "qwen1000" else '{"box_2d": [200, 100, 800, 600]}'
        seen = []

        def chat_generate(messages, max_tokens=64, pixels=0, json_schema=None, top_logprobs=0):
            seen.append(top_logprobs)
            return text, 500, self.tokens(text)

        def no_read(*a, **k):
            raise AssertionError("no readout in generation mode with presence off")

        shim.v.chat_generate = chat_generate
        shim.v.label_logprobs_chat = no_read
        shim.v.read = no_read
        if hasattr(shim, "read"):
            shim.read = no_read
        for name in ("check_backend", "check_images", "check_chat"):
            setattr(shim, name, lambda: None)
        body = {"state": {"photo": "IMG_1"}, "assets": {"IMG_1": {"type": "image", "format": "png",
                "data_base64": PNG_URI.split(",", 1)[1]}}, "questions": {"m": {"type": "box", "instructions": "the mug"}}}
        a = shim.decide(body)["answers"]["m"]
        self.assertEqual(seen, [shim.c.box_top_logprobs])
        self.assertAlmostEqual(a["confidence"], box.calibrate(0.8 ** 4, 0.7088, 1.6955), places=6)
        self.assertEqual((a["box"]["left"], a["box"]["top"]), (0.1, 0.2))
        self.assertEqual(a["present"], a["confidence"])

    def test_the_key_digit_and_unicode_digit_alternatives_do_not_count(self):
        import math
        box = S._sibling("box")
        text = '{"bbox_2d": [100, 200, 600, 800]}'
        toks = self.tokens(text)
        base = box.parse_box_logprobs(toks, "qwen1000")
        self.assertEqual(base[0], {"left": 0.1, "top": 0.2, "right": 0.6, "bottom": 0.8})   # the "2" of bbox_2d is no coordinate
        self.assertAlmostEqual(base[1], 0.8 ** 4, places=6)
        odd = list(toks)
        i = next(k for k, t in enumerate(odd) if t[0] == "1")   # the first coordinate's leading digit
        odd[i] = (odd[i][0], odd[i][1], odd[i][2] + [("\u00b9", math.log(0.3))])   # a superscript digit alternative
        self.assertAlmostEqual(box.parse_box_logprobs(odd, "qwen1000")[1], base[1], places=9)

    def test_the_code_default_is_the_tightness_readout(self):
        cfg = copy.deepcopy(ICFG)
        cfg.pop("box", None)   # serve_config.json may choose generation; without a box block the code default holds
        c = S.Contract(cfg, env={})
        self.assertEqual((c.box_confidence, c.box_generation_calibration), ("tightness", (1.0, 0.0)))
        bad = copy.deepcopy(ICFG)
        bad["box"] = {"generation_calibration": {"a": 9.0, "b": 0.0}}
        with self.assertRaises(ValueError):
            S.Contract(bad, env={})
        for t in (0, -1.0):
            bad = copy.deepcopy(ICFG); bad["box"] = {"generation_temperature": t}
            with self.assertRaises(ValueError):
                S.Contract(bad, env={})

    # the guided (schema-constrained) grounding call against a tiny engine stand-in: an engine without structured
    # outputs gets one free generation at the fallback budget, counted and warned once; any other 400 is the item's
    # error; an engine that honours the schema is called once
    def _engine(self, mode):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        bodies = []
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                b = json.loads(self.rfile.read(int(self.headers["Content-Length"]))); bodies.append(b)
                if "response_format" in b and mode != "ok":
                    msg = {"error": {"message": "structured outputs are not supported" if mode == "noguided"
                                     else "This model's maximum context length is 40960 tokens."}}
                    out = json.dumps(msg).encode(); self.send_response(400)
                else:
                    r = {"choices": [{"message": {"content": '{"bbox_2d": [1, 2, 3, 4]}'},
                                      "logprobs": {"content": [{"token": "1", "logprob": -0.1, "top_logprobs": [{"token": "1", "logprob": -0.1}]}]}}],
                         "usage": {"prompt_tokens": 7}}
                    out = json.dumps(r).encode(); self.send_response(200)
                self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out)))
                self.end_headers(); self.wfile.write(out)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        v = S.VLLM(f"http://127.0.0.1:{srv.server_address[1]}", CFG["model"]); v.guided_fallback_max_tokens = 99
        return v, bodies

    def test_guided_fallback_once_other_400s_are_the_items_error(self):
        S.GUIDED_FALLBACKS.count = 0; S.GUIDED_FALLBACKS.warned = False
        v, bodies = self._engine("noguided")
        r = v.chat_generate([{"role": "user", "content": "x"}], 30, 0, json_schema={"type": "object"}, top_logprobs=20)
        self.assertEqual(len(bodies), 2)
        self.assertIn("response_format", bodies[0]); self.assertNotIn("response_format", bodies[1])
        self.assertEqual(bodies[1]["max_tokens"], 99)
        self.assertEqual(S.GUIDED_FALLBACKS.count, 1)
        self.assertEqual(len(r), 3); self.assertEqual(r[1], 7)
        v, bodies = self._engine("context")
        with self.assertRaises(S.Unprocessable):
            v.chat_generate([{"role": "user", "content": "x"}], 30, 0, json_schema={"type": "object"})
        self.assertEqual(len(bodies), 1)
        self.assertEqual(S.GUIDED_FALLBACKS.count, 1)
        v, bodies = self._engine("ok")
        r = v.chat_generate([{"role": "user", "content": "x"}], 30, 0, json_schema={"type": "object"})
        self.assertEqual(len(bodies), 1); self.assertIn("response_format", bodies[0]); self.assertEqual(r, ('{"bbox_2d": [1, 2, 3, 4]}', 7))


class OneEngineFlagTest(unittest.TestCase):
    """The one-engine two-way (the vLLM fork with per-request attention): SHIM_TWO_WAY_FLAG=1 with --vllm only serves
    two-way readouts by sending vllm_xargs attention_mode=bidirectional, one-way without the flag; the start-up check
    compares the flagged probe readouts with the release's recorded answers and refuses an engine that ignores the
    flag; Box runs on the same engine."""

    def setUp(self):
        FakeVLLM.reset()
        self.vsrv, self.vurl = serve(FakeVLLM)

    def tearDown(self):
        for s in (getattr(self, "ssrv", None), self.vsrv):
            if s is not None:
                s.shutdown()

    def start(self, reference, env=None):
        cfg = copy.deepcopy(CFG); cfg["labels"] = CFG["labels"][:8]
        cfg["two_way"] = {"label_rows": "label_rows.safetensors", "temperature": 0.5, "noul_floor": 0.9,
                          "probe_reference": reference}
        self.shim = S.Shim(S.Contract(cfg, env=env or {}), S.VLLM(self.vurl, CFG["model"]))
        self.shim.two_way_flag = True
        self.ssrv, self.url = serve(S.make_handler(self.shim))

    @staticmethod
    def reference(twist):
        out = []
        for body in S.TWO_WAY_PROBES:
            entry = {}
            for qid, q in body["questions"].items():
                names = S.options_of(q, 8)[0]
                lp = [-0.3 * j - (twist if j == 0 else 0.0) for j in range(len(names))]
                entry[qid] = dict(zip(names, S.probabilities(lp, 1.0)))
            out.append(entry)
        return out

    def test_flagged_two_way_and_plain_one_way_on_one_engine(self):
        self.start(self.reference(FakeVLLM.flag_twist))
        body = {"state": {"doc": "MOCK_LP=[-0.2,-1.9,-3.0]"}, "questions": {"d": {
            "type": "choice", "instructions": "Which?", "criteria": {"x": "first", "y": "second", "z": "third"}}}}
        st, two = post(self.url, body)                                  # the default is two-way
        self.assertEqual((st, two["attention_mode"]), (200, "bidirectional"), two)
        for k, p in zip("xyz", softmax_t([-0.2 - FakeVLLM.flag_twist, -1.9, -3.0], 0.5)):
            self.assertAlmostEqual(two["answers"]["d"]["probabilities"][k], p, places=9)
        st, one = post(self.url, dict(body, attention_mode="causal"))
        self.assertEqual((st, one["attention_mode"]), (200, "causal"))
        for k, p in zip("xyz", softmax_t([-0.2, -1.9, -3.0])):
            self.assertAlmostEqual(one["answers"]["d"]["probabilities"][k], p, places=9)
        self.assertIn("bidirectional", FakeVLLM.flagged)
        h = json.loads(urllib.request.urlopen(self.url + "/health").read())
        self.assertEqual(h["attention_mode"], "bidirectional")
        self.assertLess(self.shim.two_way_anchor, 1e-6)
        self.assertGreater(self.shim.two_way_separation, 0.05)

    def test_an_engine_that_ignores_the_flag_is_refused(self):
        FakeVLLM.flag_twist = 0.0                                       # stock vLLM: the flag changes nothing
        self.start(self.reference(2.0))
        st, r = post(self.url, {"state": "s", "questions": {"d": OK_Q}})
        self.assertEqual(st, 503, r)
        self.assertIn("does not match the release's probe answers", r["detail"])
        FakeVLLM.flag_twist = 2.0

    def test_missing_reference_is_refused(self):
        cfg = copy.deepcopy(CFG); cfg["labels"] = CFG["labels"][:8]
        cfg["two_way"] = {"label_rows": "label_rows.safetensors", "temperature": 0.5}
        self.shim = S.Shim(S.Contract(cfg, env={}), S.VLLM(self.vurl, CFG["model"])); self.shim.two_way_flag = True
        self.ssrv, self.url = serve(S.make_handler(self.shim))
        st, r = post(self.url, {"state": "s", "questions": {"d": OK_Q}})
        self.assertEqual(st, 503, r)
        self.assertIn("probe_reference is missing", r["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
