"""Serve site/ like Cloudflare Pages: apply _headers, serve 404.html for missing paths.

Usage: python3 scripts/serve_site.py site 8765 headers   (or "plain" to skip _headers)
"""
import http.server, os, sys
from pathlib import Path

ROOT = Path(sys.argv[1]); PORT = int(sys.argv[2]); APPLY = sys.argv[3] == "headers"
HEADERS = []
if APPLY:
    for line in (ROOT / "_headers").read_text().splitlines():
        if line.startswith("  ") and ":" in line:
            k, v = line.strip().split(":", 1)
            HEADERS.append((k.strip(), v.strip()))

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)
    def end_headers(self):
        for k, v in HEADERS:
            self.send_header(k, v)
        super().end_headers()
    def send_error(self, code, message=None, explain=None):
        if code == 404 and (ROOT / "404.html").exists():
            body = (ROOT / "404.html").read_bytes()
            self.send_response(404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().send_error(code, message, explain)
    def log_message(self, *a):
        pass

http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
