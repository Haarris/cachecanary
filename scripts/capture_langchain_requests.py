"""Capture the exact Bedrock requests LangChain builds, offline, for tests/fixtures/langchain/.

LangChain runs its real agent loop (create_agent, tool calls, middleware) against a local stand-in for
bedrock-runtime (scripts/_fake_bedrock_stream.py) with fake credentials, so nothing leaves the machine.

    pip install langchain-aws==1.8.1 langchain==1.4.3 langgraph==1.2.14
    python scripts/capture_langchain_requests.py tests/fixtures/langchain
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fake_bedrock_stream import Fake  # noqa: E402

os.environ.update(AWS_ACCESS_KEY_ID="AKIAFAKE", AWS_SECRET_ACCESS_KEY="fake", AWS_DEFAULT_REGION="us-west-2")

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware, dynamic_prompt  # noqa: E402
from langchain_aws import ChatBedrock, ChatBedrockConverse  # noqa: E402
from langchain_aws.middleware.prompt_caching import BedrockPromptCachingMiddleware  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

MODEL = "us.anthropic.claude-sonnet-4-6"
SYSTEM = "You are a support agent for Acme. " + " ".join(
    f"Rule {i}: answer politely and cite section {i}." for i in range(150))


@tool
def lookup_order(id: str) -> str:
    """Look up an order."""
    return f"order {id} shipped on day {id}. " * 8


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    fake = Fake()

    def converse():
        return ChatBedrockConverse(model=MODEL, region_name="us-west-2", endpoint_url=fake.url, max_tokens=50)

    def run(script, model, prompts, middleware, system=SYSTEM):
        fake.reset(script)
        agent = create_agent(model=model, tools=[lookup_order], system_prompt=system, middleware=middleware)
        for prompt in prompts:
            agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return [request["body"] for request in fake.requests]

    def save(name, body):
        (out / name).write_text(json.dumps(body, indent=2) + "\n")

    # Finding 2: ChatBedrock (InvokeModel) with the caching middleware, 11 parallel tool calls.
    chat_bedrock = ChatBedrock(model=MODEL, region_name="us-west-2", endpoint_url=fake.url, model_kwargs={"max_tokens": 50})
    bodies = run(["tools:11", "text"], chat_bedrock, ["Check my orders."], [BedrockPromptCachingMiddleware()])
    save("invoke-turn1.json", bodies[0])
    save("invoke-turn2.json", bodies[1])

    # Finding 3: context editing that clears a sliding window of old tool results on every call.
    editing = ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=600, keep=3)])
    bodies = run(["tools:1"] * 9 + ["text"], converse(), ["Check my orders."], [editing, BedrockPromptCachingMiddleware()])
    save("edit-call8.json", bodies[8])
    save("edit-call9.json", bodies[9])

    # Smaller things: today's date in a dynamic system prompt.
    day = {"date": "2026-10-07"}

    @dynamic_prompt
    def with_date(request):
        return f"Today is {day['date']}. " + SYSTEM

    fake.reset(["text", "text"])
    agent = create_agent(model=converse(), tools=[lookup_order], middleware=[with_date, BedrockPromptCachingMiddleware()])
    agent.invoke({"messages": [{"role": "user", "content": "Hi"}]})
    day["date"] = "2026-10-08"
    agent.invoke({"messages": [{"role": "user", "content": "Hi"}]})
    save("date-day1.json", fake.requests[0]["body"])
    save("date-day2.json", fake.requests[1]["body"])
    print(f"Saved {len(list(out.glob('*.json')))} requests to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
