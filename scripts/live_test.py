"""Live verification of cachecanary's assumptions against real Amazon Bedrock.

Uses only synthetic prompts. Makes roughly 30 small calls (well under $1).
Run in a Region with no production traffic, e.g.:

    pip install -e ".[aws]"
    python scripts/live_test.py --region us-west-2

Prints, per scenario, the expected outcome, what Bedrock actually returned, and whether
cachecanary's lint/diff gave the right explanation. Saves raw usage to live_results.json.
"""

import argparse
import copy
import json
import sys
import time
import uuid

from cachecanary import diff, lint, probe
from cachecanary.request import normalize

SONNET = "us.anthropic.claude-sonnet-4-6"
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def policy(n_sentences: int, salt: str) -> str:
    # A unique salt per run so earlier runs' cache entries can't make a "miss" look like a hit.
    return f"[run {salt}] " + " ".join(f"Rule {i}: answer politely and cite section {i}." for i in range(n_sentences))


def converse_req(system_text: str, ttl: str | None = None, question: str = "Reply with OK.") -> dict:
    cp = {"type": "default", **({"ttl": ttl} if ttl else {})}
    return {
        "system": [{"text": system_text}, {"cachePoint": cp}],
        "messages": [{"role": "user", "content": [{"text": question}]}],
        "inferenceConfig": {"maxTokens": 5},
    }


def invoke_body(system_text: str, ttl: str | None = None) -> dict:
    cc = {"type": "ephemeral", **({"ttl": ttl} if ttl else {})}
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 5,
        "system": [{"type": "text", "text": system_text, "cache_control": cc}],
        "messages": [{"role": "user", "content": "Reply with OK."}],
    }


TOOL_A = {"toolSpec": {"name": "lookup_order", "description": "Look up an order.",
                       "inputSchema": {"json": {"type": "object", "properties": {"id": {"type": "string"}}}}}}
TOOL_B = {"toolSpec": {"name": "refund_order", "description": "Refund an order.",
                       "inputSchema": {"json": {"type": "object", "properties": {"id": {"type": "string"}}}}}}


def with_tools(req: dict, tools: list) -> dict:
    out = copy.deepcopy(req)
    out["toolConfig"] = {"tools": tools}
    return out


def tool_history(system_text: str, first_turn_text: str, n_calls: int, intermediate_cp: bool) -> tuple[dict, dict]:
    """Reproduces the 'rolling checkpoint' pattern from pydantic-ai #9404.

    Request A: system (checkpoint) + a long first user turn with a checkpoint at its end.
    Request B: the same history WITHOUT the old turn checkpoint (it rolls to the end), plus one
    assistant turn with n parallel tool calls and their results. 2n new blocks sit between A's
    conversation checkpoint and B's, so with n=12 Bedrock's ~20-block lookback cannot reach A's
    entry and only the system cache is read. intermediate_cp adds a checkpoint after the assistant
    turn (within 20 blocks of A's), which should recover the conversation cache.
    """
    cp = {"cachePoint": {"type": "default"}}
    system = [{"text": system_text}, cp]
    a = with_tools({"system": system,
                    "messages": [{"role": "user", "content": [{"text": first_turn_text}, cp]}],
                    "inferenceConfig": {"maxTokens": 5}}, [TOOL_A])
    calls = [{"toolUse": {"toolUseId": f"tu{i}", "name": "lookup_order", "input": {"id": str(i)}}} for i in range(n_calls)]
    results = [{"toolResult": {"toolUseId": f"tu{i}", "content": [{"text": f"order {i}: shipped"}]}} for i in range(n_calls)]
    b = copy.deepcopy(a)
    b["messages"] = [
        {"role": "user", "content": [{"text": first_turn_text}]},
        {"role": "assistant", "content": calls + ([cp] if intermediate_cp else [])},
        {"role": "user", "content": results + [cp]},
    ]
    return a, b


def call(client, payload, model, stream=False):
    return probe._call(client, payload, model, stream)


def run(args) -> int:
    try:
        client = probe.make_client(args.region)
    except probe.ProbeError as exc:
        print(f"Could not create a Bedrock client: {exc}", file=sys.stderr)
        return 2
    salt = uuid.uuid4().hex[:8]
    long_text = policy(220, salt)      # ~2.5k tokens: above Sonnet 4.6's 1,024, below Haiku 4.5's 4,096
    results, rows = [], []

    def record(name, expected, ok, detail, explain=""):
        rows.append((name, expected, "PASS" if ok else "FAIL", detail, explain))
        results.append({"scenario": name, "expected": expected, "ok": ok, "detail": detail, "explain": explain})

    def pair(name, a, b, model, expect_hit, stream=False, expect_reason=None):
        try:
            u1 = call(client, a, model, stream)
            time.sleep(args.pause)
            u2 = call(client, b, model, stream)
        except Exception as exc:  # report and continue with the other scenarios
            record(name, "hit" if expect_hit else "miss", False, f"ERROR {probe._translate(exc)}")
            return
        hit = bool(u2 and u2.cache_read > 0)
        detail = f"1st read/write={u1.cache_read if u1 else None}/{u1.cache_write if u1 else None} " \
                 f"2nd read/write={u2.cache_read if u2 else None}/{u2.cache_write if u2 else None}"
        explain = ""
        if expect_reason:
            codes = [r.code for r in diff.explain(normalize(a, model), normalize(b, model))]
            explain = f"diff={codes} (want {expect_reason})"
            ok = hit == expect_hit and expect_reason in codes
        else:
            ok = hit == expect_hit
        record(name, "hit" if expect_hit else "miss", ok, detail, explain)

    # 1. Basic hits on both APIs, both TTLs, streaming.
    r = converse_req(long_text)
    pair("converse 5m", r, r, SONNET, True)
    r = converse_req(policy(220, salt + "b"), ttl="1h")
    pair("converse 1h ttl", r, r, SONNET, True)
    b = invoke_body(policy(220, salt + "c"))
    pair("invoke 5m", b, b, SONNET, True)
    r = converse_req(policy(220, salt + "d"))
    pair("converse stream", r, r, SONNET, True, stream=True)
    b = invoke_body(policy(220, salt + "e"))
    pair("invoke stream", b, b, SONNET, True, stream=True)

    # 2. Minimum prefix size: same ~2.5k-token prompt hits on Sonnet 4.6 but not Haiku 4.5 (needs 4,096).
    r = converse_req(policy(220, salt + "f"))
    findings = {f.rule for f in lint.lint(normalize(r, HAIKU))}
    pair("haiku below 4096 min", r, r, HAIKU, False)
    rows[-1] = rows[-1][:4] + (f"lint prefix-too-short={'prefix-too-short' in findings}",)
    r = converse_req(policy(40, salt + "g"))   # ~450 tokens, below Sonnet's 1,024
    pair("sonnet below 1024 min", r, r, SONNET, False)

    # 3. Prefix changes that must miss, with diff naming the cause.
    base = policy(220, salt + "h")
    a = converse_req(base + " Today is 2026-10-05 09:14.")
    b2 = converse_req(base + " Today is 2026-10-05 09:19.")
    pair("date in system", a, b2, SONNET, False, expect_reason="system-changed")
    a = with_tools(converse_req(policy(220, salt + "i")), [TOOL_A, TOOL_B, {"cachePoint": {"type": "default"}}])
    b2 = with_tools(converse_req(policy(220, salt + "i")), [TOOL_B, TOOL_A, {"cachePoint": {"type": "default"}}])
    pair("tools reordered", a, b2, SONNET, False, expect_reason="tools-reordered")
    r = converse_req(policy(220, salt + "j"))
    try:
        u1 = call(client, r, SONNET)
        time.sleep(args.pause)
        u2 = call(client, r, "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        codes = [x.code for x in diff.explain(normalize(r, SONNET), normalize(r, "us.anthropic.claude-sonnet-4-5-20250929-v1:0"))]
        record("model switched", "miss", (u2.cache_read == 0) and "model-changed" in codes,
               f"2nd read={u2.cache_read}", f"diff={codes}")
    except Exception as exc:
        record("model switched", "miss", False, f"ERROR {probe._translate(exc)}")

    # 4. 20-block lookback: 12 parallel tool calls = 24 new blocks between conversation checkpoints.
    #    Both variants still read the SYSTEM cache, so judge by how many tokens were read: without a
    #    middle checkpoint the long first turn (~1.5k tokens) must NOT come from cache; with one it must.
    reads = {}
    for variant, mid in (("no middle checkpoint", False), ("middle checkpoint", True)):
        sys_text = policy(220, salt + ("k" if not mid else "l"))
        turn = "Background for this ticket: " + policy(130, salt + ("m" if not mid else "n"))
        a, b2 = tool_history(sys_text, turn, 12, intermediate_cp=mid)
        try:
            call(client, a, SONNET)
            time.sleep(args.pause)
            u2 = call(client, b2, SONNET)
            reads[variant] = u2.cache_read if u2 else 0
            codes = [x.code for x in diff.explain(normalize(a, SONNET), normalize(b2, SONNET))]
            reads[variant + " diff"] = codes
        except Exception as exc:
            record(f"lookback: {variant}", "-", False, f"ERROR {probe._translate(exc)}")
    if "no middle checkpoint" in reads and "middle checkpoint" in reads:
        no_mid, mid = reads["no middle checkpoint"], reads["middle checkpoint"]
        recovered = mid - no_mid > 1000  # the first turn is ~1.5k tokens
        record("lookback >20 blocks (rolling checkpoint)", "partial miss",
               recovered and "lookback-exceeded" in reads["no middle checkpoint diff"],
               f"read without middle cp={no_mid}, with middle cp={mid}",
               f"diff(no mid)={reads['no middle checkpoint diff']} diff(mid)={reads['middle checkpoint diff']}")

    # Print the table.
    width = max(len(r[0]) for r in rows)
    print(f"\nRegion {args.region}, run {salt}\n")
    for name, expected, status, detail, explain in rows:
        print(f"{status:4}  {name:<{width}}  expect {expected:<4}  {detail}  {explain}")
    with open("live_results.json", "w") as fh:
        json.dump({"region": args.region, "run": salt, "results": results}, fh, indent=2)
    failed = [r for r in rows if r[2] != "PASS"]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} scenarios matched expectations. Raw data: live_results.json")
    print("Note: on 'lookback >20 blocks' the 2nd call may still hit the SYSTEM cache (read>0) while missing the "
          "conversation cache; compare read tokens with the 'middle checkpoint' row to see the difference.")
    return 0 if not failed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, help="use a Region without production traffic")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between the two calls of a pair")
    sys.exit(run(parser.parse_args()))
