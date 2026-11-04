"""Build tests/fixtures/bedrock_invocation_logs_ttl_2026-11.json from the live run of scripts/live_ttl_check.py.

The live run records the usage Bedrock returned for each call and the call's minute in the schedule,
not a clock time, so the fixture's timestamps follow the schedule (16:03 + minute; the 1-hour agent one
second later). Prompts are replaced with a placeholder of the same role; account and request IDs are
dummies; inferenceRegion is left out because the run did not record it.

    python scripts/build_ttl_fixture.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARN = "arn:aws:bedrock:us-west-2:111122223333:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def build(rows):
    recs = []
    for i, r in enumerate(sorted(rows, key=lambda r: (r["minute"], r["ttl"]))):
        cc = {"type": "ephemeral"} if r["ttl"] == "5m" else {"type": "ephemeral", "ttl": "1h"}
        usage = {"input_tokens": r["uncached"], "cache_read_input_tokens": r["read"],
                 "cache_creation_input_tokens": r["write"], "cache_creation": r["cache_creation"], "output_tokens": 2}
        recs.append({
            "schemaType": "ModelInvocationLog", "schemaVersion": "1.0",
            "timestamp": f"2026-11-04T16:{3 + r['minute']:02d}:{10 if r['ttl'] == '5m' else 11}Z",
            "accountId": "111122223333", "region": "us-west-2", "requestId": f"00000000-0000-0000-0000-{i:012d}",
            "operation": "InvokeModel", "modelId": ARN,
            "identity": {"arn": f"arn:aws:sts::111122223333:assumed-role/ExampleRole/agent-{r['ttl']}"},
            "input": {"inputContentType": "application/json", "inputTokenCount": r["uncached"],
                      "inputBodyJson": {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 20,
                                        "system": [{"type": "text", "cache_control": cc,
                                                    "text": f"[{r['ttl']} agent: 21,000-token system prompt, redacted]"}],
                                        "messages": [{"role": "user", "content": [
                                            {"type": "text", "text": f"Task {r['call']}: reply OK."}]}]}},
            "output": {"outputContentType": "application/json", "outputTokenCount": 2,
                       "outputBodyJson": {"type": "message", "role": "assistant",
                                          "content": [{"type": "text", "text": "OK"}], "usage": usage}}})
    return recs


if __name__ == "__main__":
    rows = [json.loads(line) for line in (ROOT / "tests/fixtures/live_ttl_check_2026-11-04.jsonl").read_text().splitlines()
            if line.strip()]
    out = ROOT / "tests/fixtures/bedrock_invocation_logs_ttl_2026-11.json"
    out.write_text(json.dumps(build(rows), indent=1))
    print(f"wrote {out} ({len(rows)} calls)")
