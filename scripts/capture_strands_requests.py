"""Capture the exact Bedrock requests Strands Agents builds, offline, for tests/fixtures/strands/.

Strands runs its real agent loop (tool calls, streaming, conversation trimming) against a local stand-in
for bedrock-runtime (scripts/_fake_bedrock_stream.py) with fake credentials, so nothing leaves the machine.

    pip install strands-agents==1.58.1
    python scripts/capture_strands_requests.py tests/fixtures/strands
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fake_bedrock_stream import Fake  # noqa: E402

os.environ.update(AWS_ACCESS_KEY_ID="AKIAFAKE", AWS_SECRET_ACCESS_KEY="fake", AWS_REGION="us-west-2")

from strands import Agent, tool  # noqa: E402
from strands.models import BedrockModel  # noqa: E402
from strands.models.model import CacheConfig  # noqa: E402

MODEL = "us.anthropic.claude-sonnet-4-6"
SYSTEM = "You are a support agent for Acme. " + " ".join(
    f"Rule {i}: answer politely and cite section {i}." for i in range(150))


@tool
def lookup_order(id: str) -> str:
    """Look up an order."""
    return f"order {id} shipped"


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    fake = Fake()

    def run(script, prompts, cache_config):
        fake.reset(script)
        model = BedrockModel(model_id=MODEL, endpoint_url=fake.url, region_name="us-west-2", cache_config=cache_config)
        agent = Agent(model=model, tools=[lookup_order], system_prompt=SYSTEM, callback_handler=None)
        for prompt in prompts:
            agent(prompt)
        return [request["body"] for request in fake.requests]

    def save(name, body):
        (out / name).write_text(json.dumps(body, indent=2) + "\n")

    auto = CacheConfig(strategy="auto")
    # Finding 2: one agent step with 11 parallel tool calls.
    bodies = run(["tools:11", "text"], ["Check my orders."], auto)
    save("agent-turn1.json", bodies[0])
    save("agent-turn2.json", bodies[1])
    # Finding 3: a long chat with the default sliding window (40 messages); turn 21 trims the first messages.
    bodies = run(["text"] * 22, [f"Question {i} about order {i}." for i in range(22)], auto)
    save("chat-turn20.json", bodies[20])
    save("chat-turn21.json", bodies[21])
    # Finding 5: a 1-hour default lifetime with a 5-minute system prompt point.
    bodies = run(["text"], ["Hi"], CacheConfig(strategy="auto", ttl="1h", system_prompt_ttl="5m"))
    save("ttl-order.json", bodies[0])
    print(f"Saved {len(list(out.glob('*.json')))} requests to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
