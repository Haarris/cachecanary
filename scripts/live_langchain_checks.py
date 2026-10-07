"""Re-run the live numbers on cachecanary.com/langchain/ against real Amazon Bedrock.

Settings checks run LangChain fully live. Agent-loop checks need an exact number of parallel tool calls,
which a live model won't do on request, so LangChain runs its real agent loop against a local stand-in
(scripts/_fake_bedrock_stream.py) and the exact requests it built are then sent, in order, to Bedrock.
Caching depends only on the request, so the numbers are the same.

Synthetic prompts with a fresh random tag per case. About 70 small calls on Claude Sonnet 4.6 (about $1).
Uses your normal AWS credentials, always in the region you pass.

    pip install langchain-aws==1.8.1 langchain==1.4.3 langgraph==1.2.14
    python scripts/live_langchain_checks.py --region us-west-2

To include the application inference profile check, create a profile for Sonnet 4.6 and pass its ARN
(see scripts/live_litellm_checks.py for the create and delete commands):

    python scripts/live_langchain_checks.py --region us-west-2 --profile-arn ARN
"""

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fake_bedrock_stream import Fake  # noqa: E402

from langchain.agents.middleware import ClearToolUsesEdit  # noqa: E402
from langchain_core.messages import ToolMessage  # noqa: E402

MODEL = "us.anthropic.claude-sonnet-4-6"


# Same as on the page (tests/test_site.py checks they match).
@dataclass(slots=True)
class ClearToolUsesInChunks(ClearToolUsesEdit):
    """Clear old tool results in whole batches, not one per call."""

    chunk: int = 10

    def apply(self, messages, *, count_tokens) -> None:
        results = sum(isinstance(m, ToolMessage) for m in messages)
        cleared = max(0, results - self.keep) // self.chunk * self.chunk
        keep, self.keep = self.keep, results - cleared  # whole batches only
        try:
            ClearToolUsesEdit.apply(self, messages, count_tokens=count_tokens)
        finally:
            self.keep = keep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--region", required=True)
    parser.add_argument("--profile-arn", help="an application inference profile for Sonnet 4.6 (optional)")
    args = parser.parse_args()

    import boto3
    from langchain.agents import create_agent
    from langchain.agents.middleware import ContextEditingMiddleware
    from langchain_aws import ChatBedrock, ChatBedrockConverse
    from langchain_aws.middleware.prompt_caching import BedrockPromptCachingMiddleware
    from langchain_core.messages import SystemMessage
    from langchain_core.tools import tool

    bedrock = boto3.client("bedrock-runtime", region_name=args.region)
    assert bedrock.meta.region_name == args.region
    tool_output = {"repeat": 8}

    @tool
    def lookup_order(id: str) -> str:
        """Look up one order by its id. Returns its shipping status."""
        return f"order {id} shipped on day {id}. " * tool_output["repeat"]

    def live_model(model_id=MODEL, **kwargs):
        # boto3 prefers the region in ~/.aws/config over environment variables, so pass it and check.
        model = ChatBedrockConverse(model=model_id, region_name=args.region, max_tokens=kwargs.pop("max_tokens", 50), **kwargs)
        assert model.client.meta.region_name == args.region, model.client.meta.region_name
        return model

    def tag_prompts():
        tag = uuid.uuid4().hex
        system = f"[run {tag}] You are a support agent for Acme. " + " ".join(
            f"Rule {i}: answer politely and cite section {i}." for i in range(150))
        history = f"[doc {tag}] Customer history: " + " ".join(
            f"Order {i} was placed on day {i} and contained item {i * 7}." for i in range(250))
        return system, history

    def usage(messages):
        total = {"read": 0, "write": 0}
        for message in messages:
            details = (getattr(message, "usage_metadata", None) or {}).get("input_token_details", {}) or {}
            total["read"] += details.get("cache_read", 0)
            # LangChain moves writes into the per-lifetime keys when Bedrock reports them.
            total["write"] += (details.get("cache_creation", 0) + details.get("ephemeral_5m_input_tokens", 0)
                               + details.get("ephemeral_1h_input_tokens", 0))
        return total

    def show(row):
        print(json.dumps(row), flush=True)
        return row

    checks = []

    # Settings, fully live: the same system prompt in two fresh agents.
    def twice(case, middleware, **model_kwargs):
        system = tag_prompts()[0] + " ".join(f"Extra rule {i}." for i in range(150))
        rows = []
        for call in (1, 2):
            agent = create_agent(model=live_model(**model_kwargs), tools=[lookup_order], system_prompt=system,
                                 middleware=middleware)
            out = agent.invoke({"messages": [{"role": "user", "content": "Reply with OK."}]})
            rows.append(show({"case": case, "call": call, **usage(out["messages"])}))
            time.sleep(2)
        return rows

    off = twice("no middleware (the default)", [])
    on = twice("BedrockPromptCachingMiddleware", [BedrockPromptCachingMiddleware()])
    checks += [off[1]["read"] == 0 and off[1]["write"] == 0, on[1]["read"] > 1000]
    if args.profile_arn:
        looked_up = twice("profile ARN, provider anthropic", [BedrockPromptCachingMiddleware()],
                          model_id=args.profile_arn, provider="anthropic")
        named = twice("profile ARN, provider anthropic, base_model_id", [BedrockPromptCachingMiddleware()],
                      model_id=args.profile_arn, provider="anthropic", base_model_id="anthropic.claude-sonnet-4-6")
        checks += [looked_up[1]["read"] > 1000, named[1]["read"] > 1000]

    # Agent loops: LangChain's real loop on the stand-in, then the exact requests replayed to Bedrock.
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "unused-by-the-stand-in")
    fake = Fake()

    def capture(script, *, invoke_route=False, middleware=None, manual_point=False):
        system, history = tag_prompts()
        fake.reset(script)
        if invoke_route:
            model = ChatBedrock(model=MODEL, region_name=args.region, endpoint_url=fake.url, model_kwargs={"max_tokens": 50})
        else:
            model = ChatBedrockConverse(model=MODEL, region_name=args.region, endpoint_url=fake.url, max_tokens=50)
        if manual_point:
            system = SystemMessage(content=[{"type": "text", "text": system}, ChatBedrockConverse.create_cache_point()])
        agent = create_agent(model=model, tools=[lookup_order], system_prompt=system,
                             middleware=middleware or [BedrockPromptCachingMiddleware()])
        agent.invoke({"messages": [{"role": "user", "content": history + " Check my orders."}]})
        return [(request["path"], request["body"]) for request in fake.requests]

    def replay(case, requests):
        rows = []
        for i, (path, body) in enumerate(requests):
            if path.endswith("/invoke"):
                out = json.loads(bedrock.invoke_model(modelId=MODEL, body=json.dumps({**body, "max_tokens": 5}))["body"].read())
                u = out["usage"]
                read, write, uncached = u.get("cache_read_input_tokens", 0), u.get("cache_creation_input_tokens", 0), u.get("input_tokens", 0)
            else:
                u = bedrock.converse(modelId=MODEL, **{**body, "inferenceConfig": {"maxTokens": 5}})["usage"]
                read, write, uncached = u.get("cacheReadInputTokens", 0), u.get("cacheWriteInputTokens", 0), u.get("inputTokens", 0)
            rows.append(show({"case": case, "call": i, "read": read, "write": write, "prompt": read + write + uncached}))
            time.sleep(2)
        return rows

    def kept(rows):  # the second call read (almost) the whole first request
        return rows[1]["read"] >= rows[0]["prompt"] - 5

    def kept_with_retry(case, script, **kwargs):
        # With a us. or global. model ID, Bedrock can route two calls to different Regions, and caches
        # are per Region: the second call then reads nothing, not even the system prompt. That is not
        # LangChain, so try once more before counting it.
        for attempt in (1, 2):
            rows = replay(case if attempt == 1 else f"{case} (retry after a full miss)", capture(script, **kwargs))
            if rows[1]["read"] > 0:
                return kept(rows)
        return False

    checks.append(kept_with_retry("ChatBedrockConverse, 21 parallel calls", ["tools:21", "text"]))
    checks.append(not kept(replay("ChatBedrockConverse, 22 parallel calls", capture(["tools:22", "text"]))))
    checks.append(kept_with_retry("ChatBedrock, 10 parallel calls", ["tools:10", "text"], invoke_route=True))
    invoke_11 = replay("ChatBedrock, 11 parallel calls", capture(["tools:11", "text"], invoke_route=True))
    checks.append(invoke_11[1]["read"] == 0)  # nothing read: no separate system prompt marker on this route
    checks.append(kept_with_retry("middleware only, 11 parallel calls", ["tools:11", "text"]))
    checks.append(not kept(replay("manual system cache point + middleware, 11 parallel calls",
                                  capture(["tools:11", "text"], manual_point=True))))

    # Context editing: stock clearing vs clearing in batches vs none, 13 calls with long tool results.
    tool_output["repeat"] = 40

    def cost(row):  # in full-price input tokens: reads 0.1x, 5-minute writes 1.25x
        return row["read"] * 0.1 + row["write"] * 1.25 + (row["prompt"] - row["read"] - row["write"])

    stock = replay("ClearToolUsesEdit (stock)", capture(["tools:1"] * 12 + ["text"], middleware=[
        ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=5800, keep=3)]), BedrockPromptCachingMiddleware()]))
    chunked = replay("ClearToolUsesInChunks(chunk=5)", capture(["tools:1"] * 12 + ["text"], middleware=[
        ContextEditingMiddleware(edits=[ClearToolUsesInChunks(trigger=5800, keep=3, chunk=5)]), BedrockPromptCachingMiddleware()]))
    none = replay("no context editing", capture(["tools:1"] * 12 + ["text"]))
    show({"case": "cost of call 12", "stock": round(cost(stock[12])), "chunks": round(cost(chunked[12])), "none": round(cost(none[12]))})
    checks += [cost(stock[12]) > 3 * cost(none[12]), cost(chunked[12]) < cost(none[12])]

    print(f"\n{sum(checks)}/{len(checks)} checks matched the page.")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
