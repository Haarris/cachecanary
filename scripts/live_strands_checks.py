"""Re-run the live numbers on cachecanary.com/strands/ against real Amazon Bedrock.

Two kinds of checks. Settings checks run Strands Agents fully live. Agent-loop checks need an exact number
of parallel tool calls, which a live model won't do on request, so Strands runs its real agent loop
against a local stand-in (scripts/_fake_bedrock_stream.py) and the exact requests it built are then sent,
in order, to Bedrock. Caching depends only on the request, so the numbers are the same.

Synthetic prompts with a fresh random tag per case. About 45 small calls on Claude Sonnet 4.6 (under $1).
Uses your normal AWS credentials, always in the region you pass.

    pip install strands-agents==1.58.1
    python scripts/live_strands_checks.py --region us-west-2

To include the application inference profile check, create a profile for Sonnet 4.6 and pass its ARN
(see scripts/live_litellm_checks.py for the create and delete commands):

    python scripts/live_strands_checks.py --region us-west-2 --profile-arn ARN
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fake_bedrock_stream import Fake  # noqa: E402

from strands.agent.conversation_manager import SlidingWindowConversationManager  # noqa: E402

MODEL = "us.anthropic.claude-sonnet-4-6"


# Same conversation manager as on the page (tests/test_site.py checks they match).
class TrimInChunks(SlidingWindowConversationManager):
    """Trim old messages in bulk, so the cached start rarely changes."""

    def __init__(self, window_size: int = 40, trim_to: int = 20, **kwargs):
        super().__init__(window_size=window_size, **kwargs)
        self.trim_to = trim_to

    def apply_management(self, agent, **kwargs) -> None:
        if len(agent.messages) <= self.window_size:
            return
        full, self.window_size = self.window_size, self.trim_to
        try:
            self.reduce_context(agent)
        finally:
            self.window_size = full

    def restore_from_session(self, state):
        # Let sessions saved with the default manager switch to this one.
        if state.get("__name__") == "SlidingWindowConversationManager":
            state = {**state, "__name__": type(self).__name__}
        return super().restore_from_session(state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--region", required=True)
    parser.add_argument("--profile-arn", help="an application inference profile for Sonnet 4.6 (optional)")
    args = parser.parse_args()

    import boto3
    from strands import Agent, tool
    from strands.models import BedrockModel
    from strands.models.model import CacheConfig

    bedrock = boto3.client("bedrock-runtime", region_name=args.region)
    assert bedrock.meta.region_name == args.region

    def live_model(**kwargs):
        # boto3 prefers the region in ~/.aws/config over AWS_REGION, so pass it explicitly and check.
        model = BedrockModel(region_name=args.region, **kwargs)
        assert model.client.meta.region_name == args.region, model.client.meta.region_name
        return model

    def tag_prompts():
        tag = uuid.uuid4().hex
        system = f"[run {tag}] You are a support agent for Acme. " + " ".join(
            f"Rule {i}: answer politely and cite section {i}." for i in range(150))
        history = f"[doc {tag}] Customer history: " + " ".join(
            f"Order {i} was placed on day {i} and contained item {i * 7}." for i in range(250))
        return system, history

    def totals(agent):
        u = agent.event_loop_metrics.accumulated_usage
        return {"read": u.get("cacheReadInputTokens", 0), "write": u.get("cacheWriteInputTokens", 0),
                "uncached": u.get("inputTokens", 0)}

    def show(row):
        print(json.dumps(row), flush=True)
        return row

    checks = []

    @tool
    def lookup_order(id: str) -> str:
        """Look up an order."""
        return f"order {id} shipped on day {id}. " * 8

    # Settings, fully live: the same system prompt twice in two fresh agents.
    def twice(case, **model_kwargs):
        system, _ = tag_prompts()
        rows = []
        for call in (1, 2):
            agent = Agent(model=live_model(**model_kwargs), system_prompt=system, callback_handler=None)
            try:
                agent("Reply with OK.")
            except Exception as exc:  # noqa: BLE001 - a rejected request is a result here
                show({"case": case, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                return [{"case": case, "error": str(exc)}]
            rows.append(show({"case": case, "call": call, **totals(agent)}))
            time.sleep(2)
        return rows

    off = twice("no cache_config (the default)", model_id=MODEL)
    on = twice("cache_config auto", model_id=MODEL, cache_config=CacheConfig(strategy="auto"))
    checks += [off[1]["read"] == 0 and off[1]["write"] == 0, on[1]["read"] > 1000]
    rejected = twice("ttl 1h with system_prompt_ttl 5m", model_id=MODEL,
                     cache_config=CacheConfig(strategy="auto", ttl="1h", system_prompt_ttl="5m"))
    checks.append("must not come after" in rejected[0].get("error", ""))
    try:  # the same rule with a 5-minute tools point (needs a tool so there is a tools section)
        Agent(model=live_model(model_id=MODEL, cache_config=CacheConfig(strategy="auto", ttl="1h", tools_ttl="5m")),
              tools=[lookup_order], system_prompt=tag_prompts()[0], callback_handler=None)("Reply with OK.")
        checks.append(False)
        show({"case": "ttl 1h with tools_ttl 5m", "result": "accepted"})
    except Exception as exc:  # noqa: BLE001
        show({"case": "ttl 1h with tools_ttl 5m", "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
        checks.append("must not come after" in str(exc))
    if args.profile_arn:
        arn_auto = twice("profile ARN, strategy auto", model_id=args.profile_arn, cache_config=CacheConfig(strategy="auto"))
        arn_anthropic = twice("profile ARN, strategy anthropic", model_id=args.profile_arn,
                              cache_config=CacheConfig(strategy="anthropic"))
        checks += [arn_auto[1]["read"] == 0 and arn_auto[1]["write"] == 0, arn_anthropic[1]["read"] > 1000]

    # Agent loops: Strands' real loop on the stand-in, then the exact requests replayed to Bedrock.
    fake = Fake()

    def capture(script, prompts, *, manager=None):
        system, history = tag_prompts()
        fake.reset(script)
        # The stand-in needs no real credentials; a full fake pair keeps them out of the live checks.
        stand_in = boto3.Session(aws_access_key_id="unused-by-the-stand-in",
                                 aws_secret_access_key="unused-by-the-stand-in", region_name=args.region)
        model = BedrockModel(model_id=MODEL, endpoint_url=fake.url, boto_session=stand_in,
                             cache_config=CacheConfig(strategy="auto"))
        agent = Agent(model=model, tools=[lookup_order], callback_handler=None, conversation_manager=manager,
                      system_prompt=system)
        for i, prompt in enumerate(prompts):
            agent((history + " " if i == 0 else "") + prompt)
        return [request["body"] for request in fake.requests]

    def replay(case, bodies, only=None):
        rows = []
        for i, body in enumerate(bodies):
            if only is not None and i not in only:
                continue
            body = {**body, "inferenceConfig": {"maxTokens": 5}}
            usage = {}
            for event in bedrock.converse_stream(modelId=MODEL, **body)["stream"]:
                usage = event.get("metadata", {}).get("usage", usage)
            read, write = usage.get("cacheReadInputTokens", 0), usage.get("cacheWriteInputTokens", 0)
            rows.append(show({"case": case, "call": i, "read": read, "write": write,
                              "prompt": usage.get("inputTokens", 0) + read + write}))
            time.sleep(2)
        return rows

    ten = replay("auto, 10 parallel calls", capture(["tools:10", "text"], ["Check my orders."]))
    eleven = replay("auto, 11 parallel calls", capture(["tools:11", "text"], ["Check my orders."]))
    # Since Oct 2026 (by Oct 28) Bedrock counts a group of tool calls (and their results) as one lookback
    # position, so auto mode keeps the whole conversation even after a wide step.
    checks += [ten[1]["read"] >= ten[0]["prompt"] - 5, eleven[1]["read"] >= eleven[0]["prompt"] - 5]

    session = replay("auto session", capture(["tools:12", "tools:1", "tools:20", "text", "tools:3", "tools:40", "text"],
                                             ["Check my orders.", "Now refund order 3-4."]))
    for previous, row in zip(session[:6], session[1:7]):  # every step reads the whole previous request
        checks.append(row["read"] >= previous["prompt"] - 5)

    chat = [f"Question {i}: summarise order batch {i}." for i in range(26)]
    default = replay("default window, long chat", capture(["text:40"] * 26, chat), only=set(range(19, 26)))
    chunked = replay("TrimInChunks, long chat", capture(["text:40"] * 26, chat, manager=TrimInChunks()),
                     only=set(range(19, 26)))
    checks += [all(r["read"] < r["prompt"] / 4 for r in default[2:]),  # after the trim, every turn rewrites
               all(r["read"] >= p["prompt"] - 5 for p, r in zip(chunked[2:], chunked[3:]))]

    # A fully live agent in auto mode: the model chooses its own tool calls.
    system, _ = tag_prompts()
    agent = Agent(model=live_model(model_id=MODEL, cache_config=CacheConfig(strategy="auto")), tools=[lookup_order],
                  callback_handler=None, system_prompt=system)
    before = totals(agent)
    reads = []
    for prompt in ("Look up orders 1, 2, 3, 4, 5 and 6 at the same time (call the tool for each in parallel), "
                   "then say which shipped.", "Now look up orders 7 and 8 in parallel and summarise."):
        agent(prompt)
        now = totals(agent)
        delta = {k: now[k] - before[k] for k in now}
        before = now
        calls = [sum("toolUse" in b for b in m["content"]) for m in agent.messages if m["role"] == "assistant"]
        show({"case": "live agent, auto mode", "prompt": prompt[:28], **delta, "tool_calls_per_turn": calls})
        reads.append(delta["read"])
    checks.append(reads[1] > reads[0] > 0)

    print(f"\n{sum(checks)}/{len(checks)} checks matched the page.")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
