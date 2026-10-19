#!/usr/bin/env python3
import http.server
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

MARKER_ENTRIES = json.loads(
    (pathlib.Path(__file__).resolve().parents[1] / "markers_255.json").read_text()
)
MARKERS = [entry["m"] for entry in MARKER_ENTRIES]
MARKER_IDS = [entry["id"] for entry in MARKER_ENTRIES]


class BackendHandler(http.server.BaseHTTPRequestHandler):
    last_generate_body = None
    marker_id_offset = 0

    def log_message(self, *_):
        return

    def send_json(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send_json({"status": "ok"})

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(size))
        if self.path == "/tokenize":
            index = MARKERS.index(body["prompt"].strip())
            self.send_json({"tokens": [{"id": MARKER_IDS[index] + (
                BackendHandler.marker_id_offset if index == 0 else 0)}]})
            return
        if self.path == "/generate":
            BackendHandler.last_generate_body = body
            outputs = []
            for index, token_ids in enumerate(body["token_ids_logprob"]):
                winning_index = 0 if index % 2 == 0 else len(token_ids) - 1
                values = [(-0.1 if candidate == winning_index else -4.0, token_id, None) for candidate, token_id in enumerate(token_ids)]
                outputs.append({"meta_info": {"output_token_ids_logprobs": [values], "prompt_tokens": 10}})
            self.send_json(outputs)
            return
        self.send_json({"error": "unexpected path"})


class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = http.server.ThreadingHTTPServer(("127.0.0.1", 39000), BackendHandler)
        cls.backend_thread = threading.Thread(target=cls.backend.serve_forever, daemon=True)
        cls.backend_thread.start()
        server_path = pathlib.Path(__file__).resolve().parents[1] / "server.py"
        environment = {
            **os.environ,
            "OPENJEV_SGLANG_URL": "http://127.0.0.1:39000",
            "OPENJEV_MODEL_ID": "xor-1.2",
            "OPENJEV_MODEL_ALIAS": "xor-1.2",
            "OPENJEV_RELEASE_DATE": "2026-09-28",
            "OPENJEV_IMAGES": "1",
        }
        cls.server = subprocess.Popen([sys.executable, str(server_path), "39001"], env=environment)
        for _ in range(50):
            try:
                urllib.request.urlopen("http://127.0.0.1:39001/health", timeout=1).read()
                return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("API server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(timeout=5)
        cls.backend.shutdown()
        cls.backend.server_close()

    def test_all_question_types(self):
        payload = {
            "state": "A customer was charged twice.",
            "model": "xor-1.2",
            "questions": {
                "urgent": {"type": "noul", "instructions": "Is this urgent?"},
                "department": {
                    "type": "choice",
                    "instructions": "Where should this go?",
                    "criteria": {"billing": "Billing", "technical": "Technical", "sales": "Sales"},
                },
                "mood": {"type": "score", "instructions": "How upset?", "criteria": ["Calm", "Upset", "Angry"]},
            },
        }
        request = urllib.request.Request(
            "http://127.0.0.1:39001/v1/systemone",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.load(response)
        self.assertEqual(result["model"], "xor-1.2")
        self.assertGreater(result["answers"]["urgent"]["noul"], 0.5)
        self.assertEqual(result["answers"]["department"]["choice"], "billing")
        self.assertLess(result["answers"]["mood"]["score"], 1.0)
        self.assertEqual(result["usage"], {"input_tokens": 30, "output_tokens": 3})

    def test_image_data_is_forwarded(self):
        image = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        payload = {
            "state": "Look at the attached image.",
            "model": "xor",
            "images": [image],
            "questions": {
                "color": {
                    "type": "choice",
                    "instructions": "What color is the image?",
                    "criteria": {"red": "Red", "blue": "Blue"},
                }
            },
        }
        request = urllib.request.Request(
            "http://127.0.0.1:39001/v1/systemone",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.load(response)
        self.assertEqual(result["answers"]["color"]["choice"], "red")
        self.assertEqual(BackendHandler.last_generate_body["image_data"], [payload["images"], payload["images"]])

    def test_remote_media_urls_are_rejected(self):
        for field, value in (("images", ["https://example.com/image.png"]),
                             ("video", "https://example.com/video.mp4")):
            with self.subTest(field=field):
                payload = {"state": "Check media.", "model": "xor-1.2",
                           "questions": {"q": {"type": "noul", "instructions": "Is it red?"}},
                           field: value}
                request = urllib.request.Request(
                    "http://127.0.0.1:39001/v1/systemone",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request, timeout=5)
                self.assertEqual(raised.exception.code, 422)
                raised.exception.close()

    def test_large_options_and_video(self):
        for count in (150, 255):
            with self.subTest(count=count):
                payload = {
                    "state": "A customer was charged twice.",
                    "model": "xor",
                    "video": "data:video/mp4;base64,AAAA",
                    "questions": {"department": {
                        "type": "choice",
                        "instructions": "Where should this go?",
                        "criteria": {f"option_{i}": f"Category {i}" for i in range(count)},
                    }},
                }
                request = urllib.request.Request(
                    "http://127.0.0.1:39001/v1/systemone",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    result = json.load(response)
                self.assertEqual(len(result["answers"]["department"]["probabilities"]), count)
                self.assertAlmostEqual(sum(result["answers"]["department"]["probabilities"].values()), 1.0)
                self.assertEqual(BackendHandler.last_generate_body["video_data"],
                                 [[payload["video"]], [payload["video"]]])
                self.assertIn("Answer with a single option code only.",
                              BackendHandler.last_generate_body["text"][0])

    def test_model_identity(self):
        with urllib.request.urlopen("http://127.0.0.1:39001/v1/models", timeout=5) as response:
            result = json.load(response)
        self.assertEqual(result["data"][0]["id"], "xor-1.2")
        self.assertEqual(result["data"][0]["underlying"], "xor-1.2")
        self.assertEqual(result["models"][0]["name"], "xor-1.2")
        self.assertEqual(result["models"][0]["release_date"], "2026-09-28")

    def test_unknown_counts_toward_option_limit(self):
        payload = {
            "state": "Choose a category.",
            "model": "xor-1.2",
            "questions": {"category": {
                "type": "choice",
                "instructions": "Which category applies?",
                "criteria": {f"option_{i}": f"Category {i}" for i in range(254)},
                "unknown": True,
            }},
        }
        request = urllib.request.Request(
            "http://127.0.0.1:39001/v1/systemone",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.load(response)
        self.assertEqual(len(result["answers"]["category"]["probabilities"]), 255)

        payload["questions"]["category"]["criteria"]["option_254"] = "Category 254"
        request = urllib.request.Request(
            "http://127.0.0.1:39001/v1/systemone",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(raised.exception.code, 422)
        raised.exception.close()

    def test_marker_token_id_mismatch_is_rejected(self):
        server_path = pathlib.Path(__file__).resolve().parents[1] / "server.py"
        spec = importlib.util.spec_from_file_location("xor_marker_contract_test", server_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._backend_key = lambda: "mismatched-tokenizer"

        def mismatched_tokenize(_, payload, state_chars=0):
            index = MARKERS.index(payload["prompt"].strip())
            token_id = MARKER_IDS[index] + (1 if index == 0 else 0)
            return {"tokens": [{"id": token_id}]}

        module._post = mismatched_tokenize
        with self.assertRaisesRegex(RuntimeError, "expected 357, got 358"):
            module.label_ids()

    def test_marker_mismatch_blocks_server_startup(self):
        server_path = pathlib.Path(__file__).resolve().parents[1] / "server.py"
        BackendHandler.marker_id_offset = 1
        try:
            environment = {**os.environ, "OPENJEV_SGLANG_URL": "http://127.0.0.1:39000"}
            result = subprocess.run([sys.executable, str(server_path), "39002"],
                                    env=environment, capture_output=True,
                                    text=True, timeout=5)
        finally:
            BackendHandler.marker_id_offset = 0
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("backend tokenizer does not match markers_255.json", result.stderr)

    def test_configured_multiway_temperature(self):
        server_path = pathlib.Path(__file__).resolve().parents[1] / "server.py"
        spec = importlib.util.spec_from_file_location("xor_temperature_test", server_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.label_ids = lambda: MARKER_IDS
        module.TEMP = {"choice": 3.0}

        def generate(_, payload, state_chars=0):
            self.assertEqual(len(payload["text"]), 2)
            return [{"meta_info": {"output_token_ids_logprobs": [[
                (lp, token_id, None) for lp, token_id in zip(logprobs, MARKER_IDS[:3])
            ]], "prompt_tokens": 10}} for logprobs in ((0.0, -3.0, -6.0), (-6.0, -3.0, 0.0))]

        module._post = generate
        results = module.read_options_batch("state", [("question", ["a", "b", "c"], "choice")])
        probs = results[0][0]
        self.assertAlmostEqual(probs[0] / probs[1], __import__("math").e)
        self.assertAlmostEqual(probs[0] / probs[2], __import__("math").e ** 2)

    def test_temperature_is_selected_by_question_type(self):
        server_path = pathlib.Path(__file__).resolve().parents[1] / "server.py"
        spec = importlib.util.spec_from_file_location("xor_type_temperature_test", server_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.label_ids = lambda: MARKER_IDS
        module.TEMP = {"noul": 1.0, "choice": 2.0, "score": 4.0}

        def generate(_, payload, state_chars=0):
            return [{"meta_info": {"output_token_ids_logprobs": [[
                (lp, token_id, None) for lp, token_id in zip(
                    (0.0, -4.0) if index % 2 == 0 else (-4.0, 0.0), MARKER_IDS[:2]
                )
            ]], "prompt_tokens": 10}} for index, _ in enumerate(payload["text"])]

        module._post = generate
        jobs = [("n", ["a", "b"], "noul"), ("c", ["a", "b"], "choice"), ("s", ["a", "b"], "score")]
        results = module.read_options_batch("state", jobs)
        odds = [probs[0] / probs[1] for probs, _, _ in results]
        math = __import__("math")
        self.assertAlmostEqual(odds[0], math.e ** 4)
        self.assertAlmostEqual(odds[1], math.e ** 2)
        self.assertAlmostEqual(odds[2], math.e)

    def test_partial_temperature_map_uses_release_defaults(self):
        server_path = pathlib.Path(__file__).resolve().parents[1] / "server.py"
        script = (
            "import importlib.util, json; "
            f"s=importlib.util.spec_from_file_location('xor_partial_temperature', {str(server_path)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(m.TEMP, sort_keys=True))"
        )
        environment = {**os.environ, "OPENJEV_TEMP_JSON": '{"choice":2.5}'}
        result = subprocess.run([sys.executable, "-c", script], env=environment,
                                capture_output=True, text=True, timeout=5, check=True)
        self.assertEqual(json.loads(result.stdout), {"choice": 2.5, "noul": 1.4, "score": 1.0})

    def test_nearly_uniform_255_way_rounding(self):
        server_path = pathlib.Path(__file__).resolve().parents[1] / "server.py"
        spec = importlib.util.spec_from_file_location("xor_rounding_test", server_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        probs = [0.00395] * 200 + [(1 - 0.00395 * 200) / 55] * 55
        rounded = module.rounded_probabilities(range(255), probs)
        self.assertEqual(len(rounded), 255)
        self.assertAlmostEqual(sum(rounded.values()), 1.0)
        self.assertGreaterEqual(min(rounded.values()), 0.0)

    def test_invalid_temperature_configuration_fails_startup(self):
        server_path = pathlib.Path(__file__).resolve().parents[1] / "server.py"
        environment = {**os.environ, "OPENJEV_TEMP_JSON": '{"choice":0}'}
        result = subprocess.run(
            [sys.executable, "-c", "import runpy; runpy.run_path(" + repr(str(server_path)) + ")"],
            env=environment, capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("temperatures must be positive and finite", result.stderr)

    def test_unknown_companion_does_not_shift_following_answers(self):
        server_path = pathlib.Path(__file__).resolve().parents[1] / "server.py"
        spec = importlib.util.spec_from_file_location("xor_question_order_test", server_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.read_options_batch = lambda *_: [
            ([0.8, 0.2], {}, 10),  # first question
            ([0.9, 0.1], {}, 10),  # answerability companion
            ([0.1, 0.9], {}, 10),  # second question
        ]
        handler = object.__new__(module.Handler)
        answers, tokens = handler._eval_questions("state", {
            "first": {"type": "noul", "instructions": "Known?", "unknown": True},
            "second": {"type": "choice", "instructions": "Which?",
                       "criteria": {"a": "A", "b": "B"}},
        })
        self.assertEqual(answers["second"]["choice"], "b")
        self.assertEqual(tokens, 20)


if __name__ == "__main__":
    unittest.main()
