"""Re-run the live numbers on cachecanary.com/litellm/ through LiteLLM against real Amazon Bedrock.

Synthetic prompts with a fresh random tag per case, so old cache entries can't help. About 20 small calls
on Claude Sonnet 4.6 and Sonnet 5.5 (a few cents). Uses your normal AWS credentials.

    pip install litellm==1.104.0
    python scripts/live_litellm_checks.py --region us-west-2

To include the application inference profile check, create a profile for Sonnet 4.6 and pass its ARN:

    aws bedrock create-inference-profile --inference-profile-name cachecanary-test \
        --model-source copyFrom=arn:aws:bedrock:REGION:ACCOUNT:inference-profile/us.anthropic.claude-sonnet-4-6
    python scripts/live_litellm_checks.py --region us-west-2 --profile-arn ARN
    aws bedrock delete-inference-profile --inference-profile-identifier ARN
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid

MODEL = "bedrock/converse/us.anthropic.claude-sonnet-4-6"
TOOLS = [{"type": "function", "function": {"name": "lookup_order", "description": "Look up an order.",
          "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}}]
SYSTEM_AND_LAST = [{"location": "message", "role": "system"}, {"location": "message", "index": -1}]


# Same helper as on the page (tests/test_site.py checks they match).
def cache_points(messages):
    """Bedrock cache points for this call (Bedrock allows 4)."""
    # The system prompt and the end of the conversation.
    points = [{"location": "message", "role": "system"},
              {"location": "message", "index": -1}]
    users = [i for i, m in enumerate(messages) if m["role"] == "user"]
    rounds = [i for i, m in enumerate(messages)
              if m["role"] == "assistant" and m.get("tool_calls")]
    # The newest user message.
    if users:
        points.append({"location": "message", "index": users[-1]})
    # The first tool result of the newest round of tool calls.
    if rounds and (not users or rounds[-1] > users[-1]) \
            and rounds[-1] + 1 < len(messages):
        points.append({"location": "message", "index": rounds[-1] + 1})
    return points


def lifetime_check() -> int:
    """One call asking for a 1-hour cache; prints the lifetime Bedrock reports it used."""
    model, extra = sys.argv[2], json.loads(sys.argv[3])
    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    seen = {}

    class Raw(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            seen["raw"] = kwargs.get("original_response")

    litellm.callbacks = [Raw()]
    system = f"[run {uuid.uuid4().hex}] " + " ".join(f"Rule {i}: cite section {i}." for i in range(250))
    litellm.completion(model=model, max_tokens=5, **extra,
                       messages=[{"role": "system", "content": system}, {"role": "user", "content": "Reply OK."}],
                       cache_control_injection_points=[{"location": "message", "role": "system",
                                                        "control": {"type": "ephemeral", "ttl": "1h"}}])
    time.sleep(1)
    raw = seen["raw"]
    details = (json.loads(raw) if isinstance(raw, str) else raw)["usage"].get("cacheDetails") or []
    print(json.dumps({"lifetimes": [d["ttl"] for d in details]}))
    return 0


def main() -> int:
    if sys.argv[1:2] == ["--lifetime-check"]:
        return lifetime_check()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--region", required=True)
    parser.add_argument("--profile-arn", help="an application inference profile for Sonnet 4.6 (optional)")
    args = parser.parse_args()
    os.environ["AWS_REGION_NAME"] = args.region
    import litellm

    def call(case, messages, points, tools=None):
        r = litellm.completion(model=MODEL, max_tokens=5, messages=messages, tools=tools,
                               cache_control_injection_points=points)
        u = r.usage
        row = {"case": case, "read": u.cache_read_input_tokens, "write": u.cache_creation_input_tokens,
               "uncached": u.prompt_tokens - u.cache_read_input_tokens - u.cache_creation_input_tokens,
               "prompt": u.prompt_tokens}
        print(json.dumps(row), flush=True)
        time.sleep(2)
        return row

    def fresh():
        tag = uuid.uuid4().hex
        system = f"[run {tag}] You are a support agent for Acme. " + " ".join(
            f"Rule {i}: answer politely and cite section {i}." for i in range(150))
        history = f"[doc {tag}] Customer history: " + " ".join(
            f"Order {i} was placed on day {i} and contained item {i * 7}." for i in range(250))
        return [{"role": "system", "content": system}, {"role": "user", "content": history + " Check my orders."}]

    def tool_round(r, n):
        calls = {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"r{r}t{i}", "type": "function",
             "function": {"name": "lookup_order", "arguments": json.dumps({"id": f"{r}-{i}"})}} for i in range(n)]}
        return [calls] + [{"role": "tool", "tool_call_id": f"r{r}t{i}", "content": f"order {r}-{i} shipped on day {i}. " * 8}
                          for i in range(n)]

    checks = []

    # Finding 1: role "user" vs the last message, turn 10 of a chat.
    for label, points in (("role user", [{"location": "message", "role": "system"}, {"location": "message", "role": "user"}]),
                          ("last message", SYSTEM_AND_LAST)):
        chat = fresh()[:1]
        for i in range(8):
            chat += [{"role": "user", "content": f"Question {i}: summarise order batch {i}."},
                     {"role": "assistant", "content": " ".join(f"Batch {i} line {j}: order {j} shipped on day {j}." for j in range(60))}]
        call(f"{label}, turn 9", chat + [{"role": "user", "content": "Question 8?"}], points)
        chat += [{"role": "user", "content": "Question 8?"}, {"role": "assistant", "content": "Answer 8. " * 200}]
        checks.append((label, call(f"{label}, turn 10", chat + [{"role": "user", "content": "Question 9?"}], points)))
    checks_ok = [checks[0][1]["uncached"] > 5000, checks[1][1]["uncached"] < 20]

    # Finding 2: 4 vs 12 parallel tool calls with points on the system prompt and the last message.
    start = fresh()
    call("agent start", start, SYSTEM_AND_LAST, TOOLS)
    twelve = call("12 parallel calls", start + tool_round(1, 12), SYSTEM_AND_LAST, TOOLS)
    four = call("4 parallel calls", start + tool_round(1, 4), SYSTEM_AND_LAST, TOOLS)
    checks_ok += [twelve["read"] < four["read"] / 2, four["write"] < 1000]

    # The helper through a 7-step session, then a 21-call round.
    steps = [fresh()]
    for r, n in ((1, 12), (2, 1), (3, 20)):
        steps.append(steps[-1] + tool_round(r, n))
    steps.append(steps[-1] + [{"role": "assistant", "content": "All checked."}, {"role": "user", "content": "Refund order 3-4."}])
    steps.append(steps[-1] + tool_round(4, 3))
    previous = None
    for i, messages in enumerate(steps):
        row = call(f"helper step {i}", messages, cache_points(messages), TOOLS)
        if previous is not None:
            checks_ok.append(row["read"] >= previous["prompt"] - 5)  # read almost the whole previous request
        previous = row
    wide = call("helper, 21 parallel calls", steps[-1] + tool_round(5, 21), cache_points(steps[-1] + tool_round(5, 21)), TOOLS)
    checks_ok.append(wide["read"] < previous["prompt"] - 5)

    # Finding 4: the 1-hour lifetime. The price list is read at import, so each case runs in its own process.
    def lifetime(case, model, extra=None, built_in_prices=False):
        env = dict(os.environ)
        if built_in_prices:
            env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        out = subprocess.run([sys.executable, __file__, "--lifetime-check", model, json.dumps(extra or {})],
                             env=env, capture_output=True, text=True, check=True).stdout
        got = json.loads(out.strip().splitlines()[-1])["lifetimes"]
        print(json.dumps({"case": case, "bedrock_cache_lifetime": got}), flush=True)
        return got

    checks_ok.append(lifetime("Sonnet 5.5, downloaded price list", "bedrock/converse/us.anthropic.claude-sonnet-5-5") == ["1h"])
    checks_ok.append(lifetime("Sonnet 5.5, built-in price list", "bedrock/converse/us.anthropic.claude-sonnet-5-5",
                              built_in_prices=True) == ["5m"])
    checks_ok.append(lifetime("Sonnet 4.6, built-in price list", MODEL, built_in_prices=True) == ["1h"])
    if args.profile_arn:
        checks_ok.append(lifetime("profile ARN as the model", f"bedrock/converse/{args.profile_arn}") == ["5m"])
        checks_ok.append(lifetime("model name + model_id=ARN", MODEL, {"model_id": args.profile_arn}) == ["1h"])

    print(f"\n{sum(checks_ok)}/{len(checks_ok)} checks matched the page.")
    return 0 if all(checks_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
