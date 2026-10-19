"""Image links are fetched safely by the shim (python3 -m unittest test_security): public https only, resolved addresses
checked and pinned, redirects re-checked, a hard deadline however slowly a server sends, size capped, links counted
before any is fetched, real images only, nothing kept; vLLM never fetches."""
import base64
import os
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import h2o_lightning_shim as S
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
PUBLIC_IP = "93.184.216.34"


class Origin(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/png"):
            return self._send(200, "image/png", PNG)
        if self.path.startswith("/redirect-internal"):
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            return self.end_headers()
        if self.path.startswith("/html"):
            return self._send(200, "text/html", b"<html></html>")
        if self.path.startswith("/fake"):
            return self._send(200, "image/png", b"not really an image")
        if self.path.startswith("/big"):
            return self._send(200, "image/png", PNG + b"\0" * (3 * 1024 * 1024))
        if self.path.startswith("/drip-headers"):  # one header byte every 0.2 s, for ever
            self.wfile.write(b"HTTP/1.1 200 OK\r\n")
            for _ in range(200):
                self.wfile.write(b"X")
                self.wfile.flush()
                time.sleep(0.2)
            return None
        if self.path.startswith("/drip-body"):  # one body byte every 0.2 s, for ever
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            for b in PNG * 100:
                self.wfile.write(bytes([b]))
                self.wfile.flush()
                time.sleep(0.2)
            return None
        if self.path.startswith("/slow"):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(PNG[:8])
            self.wfile.flush()
            time.sleep(3)
            return None
        return self._send(404, "text/plain", b"no")

    def _send(self, status, ctype, body):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


class SafeFetch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Origin)
        cls.server.daemon_threads = True
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        real = socket.getaddrinfo
        port = self.server.server_port
        self.connected = []

        def fake(host, p, *a, **k):
            if host == "img.test":
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, p))]
            if host == "slowdns.test":
                time.sleep(3)
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, p))]
            if host == "rebind.test":
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, p)),
                        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", p))]
            return real(host, p, *a, **k)

        test = self

        def connect(conn):
            test.connected.append(conn._ip)
            test.assertEqual(conn._ip, PUBLIC_IP)
            conn.raw = conn.sock = socket.create_connection(("127.0.0.1", port), conn.timeout)

        env = {k: v for k, v in os.environ.items() if not k.startswith("LIGHTNING_IMAGE_")}
        env["LIGHTNING_IMAGE_URL_ALLOW_HTTP"] = "true"  # the local test origin speaks plain http
        for p in (mock.patch.object(socket, "getaddrinfo", fake), mock.patch.object(S._PinnedHTTP, "connect", connect),
                  mock.patch.dict(os.environ, env, clear=True)):
            p.start()
            self.addCleanup(p.stop)

    def test_internal_and_rebinding_targets_are_refused_without_connecting(self):
        for url in ("http://169.254.169.254/latest/meta-data/", "http://10.0.0.5:6379/", "https://127.0.0.1/x.png",
                    "https://localhost/x.png", "https://[::1]/", "https://[fd00:ec2::254]/", "https://100.64.1.1/",
                    "https://[::ffff:10.0.0.1]/", "https://u:p@img.test/png", "ftp://img.test/png",
                    "https://rebind.test/png"):
            with self.subTest(url=url), self.assertRaises(S.Unprocessable) as caught:
                S.image_from_string(url, True)
            self.assertNotIn("169.254", str(caught.exception))
        self.assertEqual(self.connected, [])

    def test_a_public_image_link_becomes_a_data_uri_and_leaves_no_trace(self):
        before = set(os.listdir(tempfile.gettempdir()))
        with self.assertLogs("lightning.image_fetch", "INFO") as logs:
            got = S.image_from_string("http://img.test/png?token=SECRETQ", True)
        self.assertEqual(got, "data:image/png;base64," + base64.b64encode(PNG).decode())
        text = "\n".join(logs.output)
        self.assertNotIn("SECRETQ", text)
        self.assertNotIn("img.test", text)
        self.assertEqual(set(os.listdir(tempfile.gettempdir())), before)

    def test_redirect_to_internal_non_images_oversize_and_slow_are_cut(self):
        with mock.patch.object(S, "IMAGE_FETCH_MAX_BYTES", 1024 * 1024), mock.patch.object(S, "IMAGE_FETCH_SECONDS", 1.0):
            for path in ("/redirect-internal", "/html", "/fake", "/big", "/slow", "/missing"):
                t0 = time.monotonic()
                with self.subTest(path=path), self.assertRaises(S.Unprocessable):
                    S.image_from_string("http://img.test" + path, True)
                self.assertLess(time.monotonic() - t0, 2.5)

    def test_the_deadline_holds_against_slow_names_and_dripping_headers_or_bodies(self):
        with mock.patch.object(S, "IMAGE_FETCH_SECONDS", 1.0):
            for url in ("http://img.test/drip-headers", "http://img.test/drip-body", "http://slowdns.test/png"):
                t0 = time.monotonic()
                with self.subTest(url=url), self.assertRaises(S.Unprocessable):
                    S.image_from_string(url, True)
                self.assertLess(time.monotonic() - t0, 2.0)

    def test_links_are_counted_before_any_is_fetched(self):
        body = {"state": "x", "images": ["http://img.test/png?%d" % i for i in range(50)]}
        with mock.patch.object(S, "_fetch_bytes", side_effect=AssertionError("fetched")) as fetch:
            with self.assertRaises(S.Unprocessable) as caught:
                S.extract_images(body, 4)
        self.assertIn("50 images", str(caught.exception))
        self.assertEqual(fetch.call_count, 0)
        self.assertEqual(self.connected, [])
        state, images = S.extract_images({"state": "x", "images": ["http://img.test/png", "http://img.test/png"]}, 4)
        self.assertEqual(len(images), 2)
        self.assertTrue(all(i.startswith("data:image/png;base64,") for i in images))

    def test_a_request_total_size_cap_and_a_concurrency_cap(self):
        with mock.patch.object(S, "IMAGE_FETCH_MAX_TOTAL_BYTES", len(PNG) + 10):
            with self.assertRaises(S.Unprocessable):
                S.extract_images({"state": "x", "images": ["http://img.test/png"] * 3}, 4)
        self.connected.clear()
        busy = threading.BoundedSemaphore(1)  # every fetch slot taken: the next link waits at most the deadline
        busy.acquire()
        with mock.patch.object(S, "_FETCH_SLOTS", busy), mock.patch.object(S, "IMAGE_FETCH_SECONDS", 0.3):
            t0 = time.monotonic()
            with self.assertRaises(S.Unprocessable):
                S.image_from_string("http://img.test/png", True)
            self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(self.connected, [])

    def test_embedded_ipv4_forms_idn_deny_names_ports_and_non_ascii_paths(self):
        for text in ("::127.0.0.1", "::a00:1", "::ffff:0:a00:1", "::ffff:127.0.0.1"):
            self.assertFalse(S._public_address(text), text)
        with mock.patch.dict(os.environ, {"LIGHTNING_IMAGE_URL_DENY_HOSTS": "img.test"}):
            with self.assertRaises(S.Unprocessable):
                S.image_from_string("http://\uff49mg.test/png", True)  # a full-width spelling of img.test
        for url in ("http://img.test:8080/png", "https://img.test:8443/png"):
            with self.subTest(url=url), self.assertRaises(S.Unprocessable):
                S.image_from_string(url, True)
        self.assertEqual(self.connected, [])
        got = S.image_from_string("http://img.test/png/caf\u00e9 \u5199\u771f.png?q=\u00e9", True)
        self.assertTrue(got.startswith("data:image/png;base64,"))

    def test_switch_off_and_plain_http_needs_opt_in(self):
        with mock.patch.dict(os.environ, {"LIGHTNING_IMAGE_URLS": "off"}):
            with self.assertRaises(S.Unprocessable):
                S.image_from_string("http://img.test/png", True)
        with mock.patch.dict(os.environ, {"LIGHTNING_IMAGE_URL_ALLOW_HTTP": "false"}):
            with self.assertRaises(S.Unprocessable):
                S.image_from_string("http://img.test/png", True)

    def test_a_link_inside_a_record_stays_text(self):
        self.assertIsNone(S.image_from_string("https://example.com/a.png", False))


if __name__ == "__main__":
    unittest.main()
