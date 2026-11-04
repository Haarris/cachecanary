"""Live check behind the `logs --ttl-check` README example: 5-minute vs 1-hour cache with real gaps.

Two agents with a ~21,000-token system prompt, one on the 5-minute cache and one on the 1-hour cache,
call Claude Sonnet 4.5 at 0, 2, 7 and 20 minutes. Each result line is appended to ttl_test.jsonl;
the November 4 2026 run is tests/fixtures/live_ttl_check_2026-11-04.jsonl, and
scripts/build_ttl_fixture.py turns it into the invocation-log fixture the README uses.

Same request shape Claude Code sends on Bedrock (InvokeModel, Anthropic Messages body, cache_control
breakpoints on system and the last message). Calls at t = 0, +2 min, +7 min, +20 min.
Records cache read / write (with the 5m/1h split) and latency for each call.

    python scripts/live_ttl_check.py --region us-west-2   (uses your default AWS credentials; under $1)
"""
import argparse, json, sys, threading, time, uuid
import boto3

MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
GAPS_MIN = [0, 2, 7, 20]  # minutes after the first call


def prefix(salt):
    rules = [f"[{salt}] You are an operations agent for a small company workspace."]
    for i in range(440):
        rules.append(f"Skill note {i}: when a task touches area {i % 17}, read the linked runbook first, keep changes "
                     f"reversible, log what you did, and hand off to a human if the request is ambiguous or risky.")
    return "\n".join(rules)


def body(sys_text, question, ttl):
    cc = {"type": "ephemeral"} if ttl == "5m" else {"type": "ephemeral", "ttl": "1h"}
    return {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 20,
            "system": [{"type": "text", "text": sys_text, "cache_control": cc}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}]}


def run_arm(br, ttl, out, t0):
    sys_text = prefix(uuid.uuid4().hex[:10])
    for k, gap in enumerate(GAPS_MIN):
        wait = t0 + gap * 60 - time.time()
        if wait > 0:
            time.sleep(wait)
        start = time.time()
        r = br.invoke_model(modelId=MODEL, body=json.dumps(body(sys_text, f"Task {k}: reply OK.", ttl)))
        latency_ms = round((time.time() - start) * 1000)
        u = json.loads(r["body"].read())["usage"]
        row = {"ttl": ttl, "call": k, "minute": gap, "latency_ms": latency_ms,
               "read": u.get("cache_read_input_tokens", 0), "write": u.get("cache_creation_input_tokens", 0),
               "uncached": u.get("input_tokens", 0), "cache_creation": u.get("cache_creation")}
        out.write(json.dumps(row) + "\n"); out.flush()
        print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    a = ap.parse_args()
    br = boto3.client("bedrock-runtime", region_name=a.region)
    assert br.meta.region_name == a.region
    out = open("ttl_test.jsonl", "a")
    t0 = time.time() + 5
    arms = [threading.Thread(target=run_arm, args=(br, ttl, out, t0)) for ttl in ("5m", "1h")]
    for t in arms:
        t.start()
    for t in arms:
        t.join()
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
