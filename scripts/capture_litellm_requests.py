"""Capture the exact Bedrock requests LiteLLM builds, offline, for tests/fixtures/litellm/.

A small local server stands in for bedrock-runtime and saves every request body. LiteLLM is pointed at it
with aws_bedrock_runtime_endpoint and fake credentials, so nothing leaves the machine.

    pip install litellm==1.104.0
    python scripts/capture_litellm_requests.py tests/fixtures/litellm
"""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPLY = {"output": {"message": {"role": "assistant", "content": [{"text": "OK"}]}}, "stopReason": "end_turn",
         "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}
SONNET_46 = "bedrock/converse/us.anthropic.claude-sonnet-4-6"
SYSTEM = "You are a support agent for Acme. " + " ".join(
    f"Rule {i}: answer politely and cite section {i}." for i in range(150))
TOOLS = [{"type": "function", "function": {"name": "lookup_order", "description": "Look up an order.",
          "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}}]


def serve():
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = json.dumps(REPLY).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}", bodies


def tool_round(n, prefix="t"):
    calls = {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"{prefix}{i}", "type": "function",
         "function": {"name": "lookup_order", "arguments": json.dumps({"id": f"{prefix}{i}"})}} for i in range(n)]}
    results = [{"role": "tool", "tool_call_id": f"{prefix}{i}", "content": f"order {prefix}{i} shipped"} for i in range(n)]
    return [calls] + results


def capture_ttl(out: Path, name: str, local_map: bool) -> None:
    """The cost map is read at import, so each setting needs its own process."""
    env = dict(os.environ)
    if local_map:
        env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    subprocess.run([sys.executable, __file__, "--ttl-only", str(out / name)], env=env, check=True)


def main() -> int:
    os.environ.update(AWS_ACCESS_KEY_ID="AKIAFAKE", AWS_SECRET_ACCESS_KEY="fake", AWS_REGION_NAME="us-west-2")
    endpoint, bodies = serve()
    import litellm

    def capture(path: Path, **kwargs) -> None:
        count = len(bodies)
        litellm.completion(aws_bedrock_runtime_endpoint=endpoint, max_tokens=5, **kwargs)
        assert len(bodies) == count + 1, f"no request captured for {path.name}"
        path.write_text(json.dumps(bodies[-1], indent=2) + "\n")

    if sys.argv[1] == "--ttl-only":
        capture(Path(sys.argv[2]), model="bedrock/converse/us.anthropic.claude-sonnet-5-5",
                messages=[{"role": "system", "content": [{"type": "text", "text": SYSTEM,
                                                          "cache_control": {"type": "ephemeral", "ttl": "1h"}}]},
                          {"role": "user", "content": "Hi"}])
        return 0

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    last_message = [{"location": "message", "role": "system"}, {"location": "message", "index": -1}]

    # Finding 2: an agent step with 12 parallel tool calls.
    turn1 = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "Check orders 1 to 12 and tell me which shipped."}]
    capture(out / "agent-turn1.json", model=SONNET_46, tools=TOOLS, messages=turn1,
            cache_control_injection_points=last_message)
    capture(out / "agent-turn2.json", model=SONNET_46, tools=TOOLS, messages=turn1 + tool_round(12),
            cache_control_injection_points=last_message)

    # Finding 1: role "user" on a chat with more than three user messages.
    by_role = [{"location": "message", "role": "system"}, {"location": "message", "role": "user"}]
    chat = [{"role": "system", "content": SYSTEM}]
    for i in range(5):
        chat += [{"role": "user", "content": f"Question {i} about order {i}."},
                 {"role": "assistant", "content": f"Answer {i}."}]
    capture(out / "chat-turn6.json", model=SONNET_46, messages=chat + [{"role": "user", "content": "Last question."}],
            cache_control_injection_points=by_role)
    capture(out / "chat-turn7.json", model=SONNET_46, cache_control_injection_points=by_role,
            messages=chat + [{"role": "user", "content": "Last question."}, {"role": "assistant", "content": "Answer 5."},
                             {"role": "user", "content": "One more question."}])

    # Finding 4: the same Sonnet 5.5 request with the downloaded and the built-in price list.
    capture_ttl(out, "with-download.json", local_map=False)
    capture_ttl(out, "without-download.json", local_map=True)
    if '"ttl"' not in (out / "with-download.json").read_text():
        sys.exit("LiteLLM couldn't download its price list (no network?), so with-download.json is wrong. Run again online.")
    assert '"ttl"' not in (out / "without-download.json").read_text()
    print(f"Saved {len(list(out.glob('*.json')))} requests to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
