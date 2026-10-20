"""Fake Bedrock Runtime Converse endpoint returning the live usage from strands#3546."""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# call 1 writes the cache, call 2 reads it (same numbers as the live capture)
USAGES = [
    {"inputTokens": 10, "outputTokens": 4, "totalTokens": 5862, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 5848},
    {"inputTokens": 10, "outputTokens": 4, "totalTokens": 5862, "cacheReadInputTokens": 5848, "cacheWriteInputTokens": 0},
]
count = {"n": 0}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if not self.path.endswith("/converse"):
            self.send_response(404); self.end_headers(); return
        usage = USAGES[count["n"] % 2]; count["n"] += 1
        body = json.dumps({"output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
                           "stopReason": "end_turn", "usage": usage, "metrics": {"latencyMs": 12}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
