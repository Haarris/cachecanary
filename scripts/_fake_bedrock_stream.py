"""A local stand-in for bedrock-runtime that speaks the real Converse and ConverseStream formats.

Lets an agent framework run its real agent loop offline: each request body is saved, and replies follow
a script of turns: "tools:N" (N parallel toolUse blocks for a lookup_order tool), "text" (a short answer)
or "text:N" (an N-line answer). Point a client at it with endpoint_url=Fake().url and fake credentials.
"""
import json, struct, threading, zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def _header(name, value):
    n, v = name.encode(), value.encode()
    return struct.pack(">B", len(n)) + n + b"\x07" + struct.pack(">H", len(v)) + v

def event(event_type, payload):
    headers = _header(":event-type", event_type) + _header(":content-type", "application/json") + _header(":message-type", "event")
    body = json.dumps(payload).encode()
    total = 12 + len(headers) + len(body) + 4
    prelude = struct.pack(">II", total, len(headers))
    prelude += struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
    msg = prelude + headers + body
    return msg + struct.pack(">I", zlib.crc32(msg) & 0xFFFFFFFF)

class Fake:
    def __init__(self):
        self.requests, self.script, self.turn = [], [], 0
        fake = self
        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                path = self.path
                if path.endswith("/count-tokens"):
                    return self._json({"inputTokens": 100})
                fake.requests.append({"path": path, "body": body})
                step = fake.script[fake.turn] if fake.turn < len(fake.script) else "text"
                fake.turn += 1
                if path.endswith("/converse-stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/vnd.amazon.eventstream")
                    data = fake.stream(step, fake.turn)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers(); self.wfile.write(data)
                else:
                    self._json(fake.whole(step, fake.turn))
            def _json(self, obj):
                d = json.dumps(obj).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
            def log_message(self, *a): pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def reset(self, script):
        self.requests, self.script, self.turn = [], list(script), 0

    @staticmethod
    def _calls(step, turn):
        n = int(step.split(":")[1])
        return [(f"t{turn}_{i}", {"id": f"{turn}-{i}"}) for i in range(n)]

    def stream(self, step, turn):
        out = event("messageStart", {"role": "assistant"})
        if step.startswith("tools:"):
            for i, (tid, inp) in enumerate(self._calls(step, turn)):
                out += event("contentBlockStart", {"start": {"toolUse": {"toolUseId": tid, "name": "lookup_order"}}, "contentBlockIndex": i})
                out += event("contentBlockDelta", {"delta": {"toolUse": {"input": json.dumps(inp)}}, "contentBlockIndex": i})
                out += event("contentBlockStop", {"contentBlockIndex": i})
            stop = "tool_use"
        else:
            text = "All done." if step == "text" else " ".join(f"Line {j} of answer {turn}: order {j} shipped on day {j}." for j in range(int(step.split(":")[1])))
            out += event("contentBlockDelta", {"delta": {"text": text}, "contentBlockIndex": 0})
            out += event("contentBlockStop", {"contentBlockIndex": 0})
            stop = "end_turn"
        out += event("messageStop", {"stopReason": stop})
        out += event("metadata", {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}})
        return out

    def whole(self, step, turn):
        if step.startswith("tools:"):
            content = [{"toolUse": {"toolUseId": tid, "name": "lookup_order", "input": inp}} for tid, inp in self._calls(step, turn)]
            stop = "tool_use"
        else:
            text = "All done." if step == "text" else " ".join(f"Line {j} of answer {turn}: order {j} shipped on day {j}." for j in range(int(step.split(":")[1])))
            content, stop = [{"text": text}], "end_turn"
        return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": stop,
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}
