"""Measure where Bedrock's ~20-block cache lookback actually ends.

Request A: tools + a short uncached system prompt + one long user turn ending in a cache point.
Request B: the same history with that cache point removed, then an assistant turn with N parallel
tool calls (optionally preceded by a text block) and a user turn with the N results, ending in a
new cache point. Bedrock can only reuse A's entry if it finds it by looking back from B's point.

Only the conversation is cached, so a hit reads the long first turn and a miss reads 0.
Each gap size runs twice with fresh salts. About 24 small calls on Sonnet 4.6 (roughly $0.25).

    python scripts/live_lookback_boundary.py --region us-west-2
"""

import argparse
import copy
import json
import sys
import time
import uuid

from cachecanary import probe
from cachecanary.request import normalize

SONNET = "us.anthropic.claude-sonnet-4-6"
TOOL = {"toolSpec": {"name": "lookup_order", "description": "Look up an order.",
                     "inputSchema": {"json": {"type": "object", "properties": {"id": {"type": "string"}}}}}}
CP = {"cachePoint": {"type": "default"}}

# gap (blocks between the two cache points) -> (parallel tool calls, leading text block?)
GAPS = {19: (9, True), 20: (10, False), 21: (10, True), 22: (11, False), 23: (11, True), 24: (12, False)}


def pair(salt: str, n_calls: int, with_text: bool) -> tuple[dict, dict]:
    turn = f"[run {salt}] Background for this ticket: " + " ".join(
        f"Rule {i}: answer politely and cite section {i}." for i in range(180))
    a = {"toolConfig": {"tools": [TOOL]},
         "system": [{"text": "You are a support agent."}],
         "messages": [{"role": "user", "content": [{"text": turn}, CP]}],
         "inferenceConfig": {"maxTokens": 5}}
    calls = [{"toolUse": {"toolUseId": f"tu{i}", "name": "lookup_order", "input": {"id": str(i)}}} for i in range(n_calls)]
    results = [{"toolResult": {"toolUseId": f"tu{i}", "content": [{"text": f"order {i}: shipped"}]}} for i in range(n_calls)]
    b = copy.deepcopy(a)
    b["messages"] = [
        {"role": "user", "content": [{"text": turn}]},
        {"role": "assistant", "content": ([{"text": "Checking those orders."}] if with_text else []) + calls},
        {"role": "user", "content": results + [CP]},
    ]
    return a, b


def gap_of(a: dict, b: dict) -> int:
    na, nb = normalize(a, SONNET), normalize(b, SONNET)
    return nb.checkpoint_indexes[-1] - na.checkpoint_indexes[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--region", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--pause", type=float, default=2.0)
    args = parser.parse_args()
    client = probe.make_client(args.region)
    rows = []
    for gap, (n_calls, with_text) in GAPS.items():
        for rep in range(args.repeats):
            a, b = pair(uuid.uuid4().hex[:10], n_calls, with_text)
            assert gap_of(a, b) == gap, (gap, gap_of(a, b))
            u1 = probe._call(client, a, SONNET, False)
            time.sleep(args.pause)
            u2 = probe._call(client, b, SONNET, False)
            rows.append({"gap": gap, "repeat": rep + 1, "a_write": u1.cache_write, "b_read": u2.cache_read,
                         "b_write": u2.cache_write, "hit": u2.cache_read > 0})
            print(f"gap {gap:2} #{rep + 1}: A wrote {u1.cache_write:5}  B read {u2.cache_read:5}  B wrote {u2.cache_write:5}"
                  f"  -> {'HIT' if u2.cache_read > 0 else 'miss'}", flush=True)
    with open("live_lookback_results.json", "w") as fh:
        json.dump({"region": args.region, "model": SONNET, "rows": rows}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
