"""Turn raw records saved by live_logs_test.py into a redacted regression fixture.

Writes tests/fixtures/bedrock_invocation_logs_2026-10.json only if NO sensitive value remains,
and never deletes the raw file. On failure it prints the JSON paths that still leak.
"""

import json
import sys
from pathlib import Path

RAW = Path("live_log_records.json")
OUT = Path("tests/fixtures/bedrock_invocation_logs_2026-10.json")


def redact(rec: dict, idx: int, account: str) -> dict:
    r = json.loads(json.dumps(rec).replace(account, "111122223333")) if account else json.loads(json.dumps(rec))
    r["accountId"] = "111122223333"
    r["identity"] = {"arn": "arn:aws:sts::111122223333:assumed-role/ExampleRole/example-user"}
    r["requestId"] = f"00000000-0000-0000-0000-{idx:012d}"
    r["input"] = {k: v for k, v in (r.get("input") or {}).items() if k in ("inputContentType", "inputTokenCount")}
    out = r.get("output") or {}
    body = out.get("outputBodyJson")
    if isinstance(body, dict):
        if "content" in body:
            body["content"] = [{"type": "text", "text": "OK"}]
        if "output" in body:
            body["output"] = {"message": {"role": "assistant", "content": [{"text": "OK"}]}}
        if "id" in body:
            body["id"] = f"msg_redacted_{idx}"
    elif isinstance(body, list):
        for ev in body:
            if isinstance(ev, dict):
                if ev.get("type") == "content_block_delta":
                    ev["delta"] = {"type": "text_delta", "text": "OK"}
                msg = ev.get("message")
                if isinstance(msg, dict) and "id" in msg:
                    msg["id"] = f"msg_redacted_{idx}"
    r["output"] = {k: v for k, v in out.items() if k in ("outputContentType", "outputBodyJson", "outputTokenCount")}
    return r


def leaks(obj, needles: list[str], path="$") -> list[str]:
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += leaks(v, needles, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += leaks(v, needles, f"{path}[{i}]")
    elif isinstance(obj, str):
        low = obj.lower()
        found += [f"{path} contains '{n}'" for n in needles if n.lower() in low]
    return found


def main() -> int:
    recs = json.loads(RAW.read_text())
    account = recs[0].get("accountId", "")
    user = (recs[0].get("identity") or {}).get("arn", "").rsplit("/", 1)[-1]
    needles = [n for n in (account, user, "logtest", "Rule 1:") if n]
    redacted = [redact(r, i, account) for i, r in enumerate(recs)]
    problems = leaks(redacted, needles)
    if problems:
        print("NOT written; still sensitive:\n  " + "\n  ".join(problems[:20]), file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(redacted, indent=1))
    print(f"Wrote {OUT} ({len(redacted)} records). Raw file kept at {RAW}; delete it yourself when done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
