"""Check CacheCanary 0.3's rules against live Amazon Bedrock, case by case.

For each case it sends real requests, records what Bedrock did (error, hit, miss, partial), runs
cachecanary lint or diff on the same requests, and prints whether the two agree. Synthetic prompts
only, fresh salts so old cache entries can't fake a hit. About 30 small calls on Sonnet 4.6.

    python scripts/live_v03_checks.py --region us-west-2
"""

import argparse
import copy
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from cachecanary import diff, lint
from cachecanary.request import normalize

M = "us.anthropic.claude-sonnet-4-6"
CP = {"cachePoint": {"type": "default"}}
TOOL = {"toolSpec": {"name": "lookup_order", "description": "Look up an order.",
                     "inputSchema": {"json": {"type": "object", "properties": {"id": {"type": "string"}}}}}}
TOOL2 = {"toolSpec": {"name": "refund_order", "description": "Refund an order.",
                      "inputSchema": {"json": {"type": "object", "properties": {"id": {"type": "string"}}}}}}


def long(salt, n=160):
    return f"[run {salt}] " + " ".join(f"Rule {i}: answer politely and cite section {i}." for i in range(n))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--region", required=True)
    args = parser.parse_args()
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    session = boto3.Session(region_name=args.region)
    client = session.client("bedrock-runtime")

    def call(req):
        try:
            u = client.converse(modelId=M, **req)["usage"]
            return {"read": u.get("cacheReadInputTokens", 0), "write": u.get("cacheWriteInputTokens", 0),
                    "uncached": u.get("inputTokens", 0)}
        except Exception as exc:  # noqa: BLE001 - we report every failure as data
            return {"error": f"{type(exc).__name__}: {exc}"}

    def raw_call(req):
        """Plain signed HTTP, the way gateways send requests (boto3 refuses some shapes)."""
        url = f"https://bedrock-runtime.{args.region}.amazonaws.com/model/{urllib.parse.quote(M, safe='')}/converse"
        signed = AWSRequest(method="POST", url=url, data=json.dumps(req), headers={"Content-Type": "application/json"})
        SigV4Auth(session.get_credentials(), "bedrock", args.region).add_auth(signed)
        body = signed.body if isinstance(signed.body, bytes) else signed.body.encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=dict(signed.headers),
                                                               method="POST"), timeout=60) as resp:
                u = json.load(resp)["usage"]
            return {"read": u.get("cacheReadInputTokens", 0), "write": u.get("cacheWriteInputTokens", 0)}
        except urllib.error.HTTPError as exc:
            return {"error": f"HTTP {exc.code}: {exc.read().decode()[:200]}"}

    def lint_rules(req):
        return {f.rule for f in lint.lint(normalize(req, M))}

    def diff_codes(a, b):
        return [r.code for r in diff.explain(normalize(a, M), normalize(b, M))]

    rows = []

    def record(name, live, tool, ok):
        rows.append((name, ok, live, tool))
        print(f"{'PASS' if ok else 'FAIL'}  {name}\n      live: {live}\n      tool: {tool}", flush=True)

    def pair(a, b):
        first = call(a)
        time.sleep(2)
        return first, call(b)

    user = [{"role": "user", "content": [{"text": "Reply with OK."}]}]
    base = {"inferenceConfig": {"maxTokens": 5}}

    # 1. Cache points with nothing before them in their own list: Bedrock rejects, lint says error.
    shapes = {
        "cache point first in a later message": {"messages": [
            {"role": "user", "content": [{"text": long(uuid.uuid4().hex)}]},
            {"role": "assistant", "content": [{"text": "Understood."}]},
            {"role": "user", "content": [CP, {"text": "Reply with OK."}]}]},
        "message with only a cache point": {"messages": [
            {"role": "user", "content": [{"text": long(uuid.uuid4().hex)}]},
            {"role": "assistant", "content": [CP]}, {"role": "user", "content": [{"text": "q"}]}]},
        "system starting with a cache point": {"system": [CP, {"text": long(uuid.uuid4().hex)}], "messages": user},
        "tools starting with a cache point": {"toolConfig": {"tools": [CP, TOOL]}, "messages": user},
    }
    for name, req in shapes.items():
        req = {**base, **req}
        live = call(req)
        rejected = "nothing available to cache" in live.get("error", "")
        rules = lint_rules(req)
        record(name, live, sorted(rules), rejected and "orphan-checkpoint" in rules)

    # 2. Valid placements: Bedrock caches, lint has no placement error.
    for name, req in {
        "cache point after content in a message": {"messages": [{"role": "user", "content": [{"text": long(uuid.uuid4().hex)}, CP]}]},
        "two cache points in a row": {"system": [{"text": long(uuid.uuid4().hex)}, CP, CP], "messages": user},
    }.items():
        req = {**base, **req}
        live = call(req)
        rules = lint_rules(req)
        record(name, live, sorted(rules), "error" not in live and live.get("write", 0) > 0
               and not rules & {"orphan-checkpoint", "nested-checkpoint"})

    # 3. cachePoint nested in toolResult.content over plain HTTP: accepted, caches nothing; lint error.
    salt = uuid.uuid4().hex
    nested = {**base, "toolConfig": {"tools": [TOOL]}, "messages": [
        {"role": "user", "content": [{"text": long(salt)}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "lookup_order", "input": {"id": "1"}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "order 1 shipped"}, CP]}}]}]}
    live = raw_call(nested)
    rules = lint_rules(nested)
    record("cache point nested in a tool result (plain HTTP)", live, sorted(rules),
           "error" not in live and live.get("read") == 0 and live.get("write") == 0 and "nested-checkpoint" in rules)

    # 4. Settings between two identical prompts.
    def settings_case(name, extra_a, extra_b, expect_codes, judge):
        s = uuid.uuid4().hex
        req = {"system": [{"text": long(s, 200)}, CP], "messages": user, "inferenceConfig": {"maxTokens": 30}}
        a, b = copy.deepcopy(req), copy.deepcopy(req)
        for target, extra in ((a, extra_a), (b, extra_b)):
            if extra.get("amrf"):
                target["additionalModelRequestFields"] = extra["amrf"]
            if extra.get("inference"):
                target["inferenceConfig"] = extra["inference"]
            if extra.get("thinking_tokens"):
                target["inferenceConfig"] = {"maxTokens": 2048}
        first, second = pair(a, b)
        codes = diff_codes(a, b)
        record(name, {"first": first, "second": second}, codes, judge(first, second) and codes == expect_codes)

    hit = lambda f, s: "error" not in s and s.get("read", 0) > 0 and s.get("write", 0) == 0  # noqa: E731
    miss = lambda f, s: "error" not in s and s.get("read", 0) == 0 and s.get("write", 0) > 0  # noqa: E731
    settings_case("thinking off -> enabled", {}, {"amrf": {"thinking": {"type": "enabled", "budget_tokens": 1024}},
                  "thinking_tokens": True}, ["thinking-changed"], miss)
    settings_case("thinking absent -> disabled", {}, {"amrf": {"thinking": {"type": "disabled"}}}, ["prefix-identical"], hit)
    settings_case("effort absent -> low", {}, {"amrf": {"output_config": {"effort": "low"}}}, ["effort-changed"], miss)
    settings_case("effort absent -> high", {}, {"amrf": {"output_config": {"effort": "high"}}}, ["prefix-identical"], hit)
    settings_case("temperature and max tokens change", {"inference": {"maxTokens": 5, "temperature": 0.0}},
                  {"inference": {"maxTokens": 40, "temperature": 1.0}}, ["prefix-identical"], hit)

    # 5. Tool choice groups: auto -> any is a partial miss, any -> tool is a hit.
    s = uuid.uuid4().hex

    def tc(choice):
        return {"toolConfig": {"tools": [TOOL, TOOL2, CP], "toolChoice": choice},
                "system": [{"text": long(s + "sys", 120)}, CP],
                "messages": [{"role": "user", "content": [{"text": long(s + "msg", 120)}, CP]}],
                "inferenceConfig": {"maxTokens": 50}}
    auto = call(tc({"auto": {}}))
    time.sleep(2)
    anyc = call(tc({"any": {}}))
    time.sleep(2)
    tool = call(tc({"tool": {"name": "refund_order"}}))
    codes = diff_codes(tc({"auto": {}}), tc({"any": {}}))
    record("tool choice auto -> any (partial miss)", {"auto": auto, "any": anyc}, codes,
           anyc.get("read", 0) > 0 and anyc.get("write", 0) > 0 and codes == ["tool-choice-changed"])
    codes = diff_codes(tc({"any": {}}), tc({"tool": {"name": "refund_order"}}))
    record("tool choice any -> tool (hit)", {"tool": tool}, codes,
           tool.get("read", 0) > 0 and tool.get("write", 0) == 0 and codes == ["prefix-identical"])

    # 6. A conversation cache point that doesn't move: the prefix is read, the new turn is not cached.
    s = uuid.uuid4().hex
    a = {"toolConfig": {"tools": [TOOL]}, "inferenceConfig": {"maxTokens": 5},
         "messages": [{"role": "user", "content": [{"text": long(s)}, CP]}]}
    b = copy.deepcopy(a)
    b["messages"] += [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "lookup_order", "input": {"id": "1"}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "order 1 shipped " * 40}]}}]}]
    first, second = pair(a, b)
    codes = diff_codes(a, b)
    record("cache point not moved after a tool round", {"first": first, "second": second}, codes,
           second.get("read", 0) > 0 and second.get("write", 0) == 0 and second.get("uncached", 0) > 50
           and codes == ["prefix-identical", "checkpoint-not-moved"])

    passed = sum(1 for r in rows if r[1])
    print(f"\n{passed}/{len(rows)} cases: CacheCanary agreed with live Bedrock.")
    with open("live_v03_results.json", "w") as fh:
        json.dump([{"case": n, "agree": ok, "live": live, "tool": tool} for n, ok, live, tool in rows], fh, indent=2)
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
