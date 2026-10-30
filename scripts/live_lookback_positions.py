"""Check CacheCanary's lookback counting against live Bedrock.

Request A caches one long user turn. Request B is the same turn plus some added blocks, with a
checkpoint on B's last block. B reads A's entry only if the added blocks fit in Bedrock's lookback.
For each shape the script asks diff.lookback_positions how many positions were added, predicts hit
(<= MAX_LOOKBACK_POSITIONS) or miss, sends both requests and reports whether Bedrock agreed.

Measured Oct 30 2026 in us-west-2: every shape below matched on Sonnet 4.6 (Converse and InvokeModel, two
runs each, 52/52) and on Sonnet 5.5 (one run each, 24/24; with adaptive thinking the model made no
reasoning block, so the thinking shape was skipped there). The thinking shape uses a real reasoning block. Roughly
$1.50 for the default run (Sonnet 4.6, both APIs, two runs).

    python scripts/live_lookback_positions.py --region us-west-2
    python scripts/live_lookback_positions.py --region us-west-2 --model us.anthropic.claude-sonnet-5-5 --repeats 1
"""

import argparse
import copy
import json
import sys
import time
import uuid

from cachecanary import diff, probe
from cachecanary.models import MAX_LOOKBACK_POSITIONS
from cachecanary.request import normalize

SONNET = "us.anthropic.claude-sonnet-4-6"


def _text(n):
    return [("assistant", ["text"]), ("user", ["text"] * n)]


def _parallel(n, tail=0, lead=0):
    return [("assistant", ["text"] * lead + ["tu"] * n), ("user", ["tr"] * n + ["text"] * tail)]


def _sequential(rounds):
    return [turn for _ in range(rounds) for turn in (("assistant", ["tu"]), ("user", ["tr"]))]


def _mixed(m):
    return [("assistant", ["text", "tu"] * m), ("user", ["tr"] * m)]


SHAPES = [
    ("20 text blocks", _text(20)), ("21 text blocks", _text(21)),
    ("10 sequential calls", _sequential(10)), ("11 sequential calls", _sequential(11)),
    ("11 parallel calls", _parallel(11)), ("40 parallel calls", _parallel(40)),
    ("5 parallel + 19 text", _parallel(5, tail=19)), ("5 parallel + 20 text", _parallel(5, tail=20)),
    ("text before each of 9 calls", _mixed(9)), ("text before each of 10 calls", _mixed(10)),
    ("18 text, 10 parallel", _parallel(10, lead=18)), ("19 text, 10 parallel", _parallel(10, lead=19)),
]


def first_turn(salt: str) -> str:
    return f"[run {salt}] Background for this ticket: " + " ".join(
        f"Rule {i}: answer politely and cite section {i}." for i in range(180))


def converse_pair(salt, shape):
    tool = {"toolSpec": {"name": "lookup_order", "description": "Look up an order.",
                         "inputSchema": {"json": {"type": "object", "properties": {"id": {"type": "string"}}}}}}
    cp = {"cachePoint": {"type": "default"}}

    def block(kind, i):
        if kind == "text":
            return {"text": f"note {i}"}
        if kind == "tu":
            return {"toolUse": {"toolUseId": f"tu{i}", "name": "lookup_order", "input": {"id": str(i)}}}
        return {"toolResult": {"toolUseId": f"tu{i}", "content": [{"text": f"order {i}: shipped"}]}}
    turn = first_turn(salt)
    a = {"toolConfig": {"tools": [tool]}, "system": [{"text": "You are a support agent."}],
         "messages": [{"role": "user", "content": [{"text": turn}, cp]}], "inferenceConfig": {"maxTokens": 5}}
    b = copy.deepcopy(a)
    msgs = [{"role": "user", "content": [{"text": turn}]}]
    texts, calls, pending = 0, 0, []
    for role, kinds in shape:
        content = []
        for k in kinds:
            if k == "text":
                content.append(block("text", texts))
                texts += 1
            elif k == "tu":
                content.append(block("tu", calls))
                pending.append(calls)
                calls += 1
            else:  # results answer the previous assistant turn's calls, in order
                content.append(block("tr", pending.pop(0)))
        msgs.append({"role": role, "content": content})
    msgs[-1]["content"].append(cp)
    b["messages"] = msgs
    return a, b


def invoke_pair(salt, shape):
    a, b = converse_pair(salt, shape)
    eph = {"type": "ephemeral"}

    def conv(item):
        if "text" in item:
            return {"type": "text", "text": item["text"]}
        if "toolUse" in item:
            u = item["toolUse"]
            return {"type": "tool_use", "id": f"toolu_{u['toolUseId']}", "name": u["name"], "input": u["input"]}
        r = item["toolResult"]
        return {"type": "tool_result", "tool_use_id": f"toolu_{r['toolUseId']}", "content": r["content"][0]["text"]}

    def to_invoke(req):
        msgs = [{"role": m["role"], "content": [conv(c) for c in m["content"] if "cachePoint" not in c]}
                for m in req["messages"]]
        msgs[-1]["content"][-1]["cache_control"] = eph
        tool = req["toolConfig"]["tools"][0]["toolSpec"]
        return {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 5,
                "tools": [{"name": tool["name"], "description": tool["description"], "input_schema": tool["inputSchema"]["json"]}],
                "system": [{"type": "text", "text": req["system"][0]["text"]}], "messages": msgs}
    return to_invoke(a), to_invoke(b)


def thinking_pair(client, model, salt, tail):
    """A real reasoning block: ask the model (thinking on) to call the tool five times at once."""
    tool = {"toolSpec": {"name": "lookup_order", "description": "Look up an order by id.",
                         "inputSchema": {"json": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}}}
    cp = {"cachePoint": {"type": "default"}}
    turn = first_turn(salt) + " Now look up orders 1, 2, 3, 4 and 5. Call the tool for all five at once."
    a = {"toolConfig": {"tools": [tool]}, "system": [{"text": "You are a support agent."}],
         "messages": [{"role": "user", "content": [{"text": turn}, cp]}], "inferenceConfig": {"maxTokens": 2000},
         "additionalModelRequestFields": {"thinking": {"type": "enabled", "budget_tokens": 1024}}}
    try:
        reply = client.converse(modelId=model, **a)["output"]["message"]["content"]
    except Exception as exc:  # Claude 5.x only takes adaptive thinking
        if "adaptive" not in str(exc):
            raise
        a["additionalModelRequestFields"] = {"thinking": {"type": "adaptive"}}
        reply = client.converse(modelId=model, **a)["output"]["message"]["content"]
    asst = [c for c in reply if "text" not in c]  # reasoning + tool calls only
    if not any("toolUse" in c for c in asst):
        return a, None
    results = [{"toolResult": {"toolUseId": c["toolUse"]["toolUseId"], "content": [{"text": "shipped"}]}}
               for c in asst if "toolUse" in c]
    b = copy.deepcopy(a)
    b["messages"] = [{"role": "user", "content": [{"text": turn}]}, {"role": "assistant", "content": asst},
                     {"role": "user", "content": results + [{"text": f"extra {j}"} for j in range(tail)] + [cp]}]
    return a, b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--region", required=True)
    parser.add_argument("--model", default=SONNET)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--apis", default="converse,invoke")
    parser.add_argument("--pause", type=float, default=2.0)
    args = parser.parse_args()
    client = probe.make_client(args.region)
    rows, wrong = [], 0

    def check(name, api, a, b):
        nonlocal wrong
        na, nb = normalize(a, args.model), normalize(b, args.model)
        pos = diff.lookback_positions(nb.blocks, na.checkpoint_indexes[-1], nb.checkpoint_indexes[-1])
        predict_hit = pos <= MAX_LOOKBACK_POSITIONS
        probe._call(client, a, args.model, False)
        time.sleep(args.pause)
        u = probe._call(client, b, args.model, False)
        hit = u.cache_read > 2000  # A's long first turn is ~2,900 tokens
        ok = hit == predict_hit
        wrong += not ok
        rows.append({"shape": name, "api": api, "model": args.model, "positions": pos,
                     "blocks_added": nb.checkpoint_indexes[-1] - na.checkpoint_indexes[-1],
                     "b_read": u.cache_read, "b_write": u.cache_write, "hit": hit, "prediction_ok": ok})
        print(f"{api:8} {name:30} positions={pos:3} predict={'hit ' if predict_hit else 'miss'} "
              f"read={u.cache_read:5} -> {'hit ' if hit else 'miss'} {'OK' if ok else 'WRONG'}", flush=True)

    for name, shape in SHAPES:
        for api in args.apis.split(","):
            for _ in range(args.repeats):
                a, b = (converse_pair if api == "converse" else invoke_pair)(uuid.uuid4().hex[:10], shape)
                check(name, api, a, b)
    for tail in (19, 20):
        for _ in range(args.repeats):
            a, b = thinking_pair(client, args.model, uuid.uuid4().hex[:10], tail)
            if b is None or not any("reasoningContent" in c for m in b["messages"] for c in m["content"]):
                print("thinking: the model made no tool call or no reasoning block, skipped")
                continue
            check(f"thinking, 5 parallel, {tail} text", "converse", a, b)

    with open("live_lookback_positions.json", "w") as fh:
        json.dump({"region": args.region, "model": args.model, "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": rows}, fh, indent=2)
    print(f"\n{len(rows) - wrong}/{len(rows)} matched CacheCanary's prediction. Raw data: live_lookback_positions.json")
    return 0 if not wrong else 1


if __name__ == "__main__":
    sys.exit(main())
