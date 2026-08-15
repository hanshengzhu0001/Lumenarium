"""Local authenticated proxy for the Omniverse DeepSearch API."""

import base64
import http.server
import os
import socketserver
import tempfile
import urllib.error
import urllib.request
from http.cookiejar import CookieJar


OMNI_BASE = os.environ.get("OMNIVERSE_DEEPSEARCH_BASE", "https://ov.qq.com")
TOKEN = os.environ.get("OMNIVERSE_JWT_TOKEN", "").strip()
PORT = int(os.environ.get("OMNIVERSE_PROXY_PORT", "9192"))

token_file = os.environ.get(
    "OMNIVERSE_JWT_TOKEN_FILE",
    os.path.join(tempfile.gettempdir(), "omni.jwt"),
)
if os.path.isfile(token_file):
    with open(token_file, "r", encoding="utf-8") as handle:
        TOKEN = handle.read().strip().strip("*").replace("\\_", "_")
    os.remove(token_file)

if not TOKEN:
    raise SystemExit("OMNIVERSE_JWT_TOKEN is not set")
if len(TOKEN.split(".")) != 3:
    raise SystemExit("OMNIVERSE_JWT_TOKEN is not a three-part JWT")

AUTH = "Basic " + base64.b64encode(
    ("$omni-api-token:" + TOKEN).encode("utf-8")
).decode("ascii")

cookie_jar = CookieJar()
opener = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(cookie_jar),
    urllib.request.HTTPRedirectHandler(),
)


class DeepSearchProxy(http.server.BaseHTTPRequestHandler):
    def _forward(self, body=None):
        try:
            upstream_path = self.path
            method = self.command

            headers = {
                "Authorization": AUTH,
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            request = urllib.request.Request(
                OMNI_BASE + upstream_path,
                data=body,
                headers=headers,
                method=method,
            )
            with opener.open(request, timeout=120) as response:
                payload = response.read()
                self.send_response(response.status)
                self.send_header(
                    "Content-Type",
                    response.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            self.send_header(
                "Content-Type",
                exc.headers.get("Content-Type", "application/json"),
            )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as exc:
            payload = str(exc).encode("utf-8", errors="replace")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def do_GET(self):
        self._forward()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self._forward(self.rfile.read(length))

    def log_message(self, fmt, *args):
        print("[DeepSearch proxy] " + fmt % args, flush=True)


class ReusableThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


with ReusableThreadingServer(("0.0.0.0", PORT), DeepSearchProxy) as server:
    print(f"DeepSearch proxy ready on :{PORT} -> {OMNI_BASE}", flush=True)
    server.serve_forever()
