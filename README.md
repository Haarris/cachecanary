# CacheCanary 🐤

**Know the moment your Claude prompt cache breaks on Amazon Bedrock — not when the bill arrives.**

When the cache breaks, every request still returns 200 with a correct answer. The only symptoms are a collapsing cache-read count and a bill that doubles. Anthropic's cache diagnostics only covers the Claude API, not Bedrock. `cachecanary` fills that gap.

## Install
```bash
pip install cachecanary            # lint / diff / logs
pip install "cachecanary[aws]"     # + live probe via boto3
```

## Commands
| Command | What it does | Exit 1 when |
|---|---|---|
| `cachecanary lint request.json` | Static checks: prefix below the model's minimum tokens, dates/IDs inside the cached prefix, too many checkpoints, TTL order, profile ARNs and unknown model IDs that libraries skip | any error-level finding |
| `cachecanary diff prev.json next.json` | Explains why `next` missed `prev`'s cache: system/tools/history changed, tools reordered, model changed, TTL changed, >20-block lookback | the prefix changed |
| `cachecanary probe request.json --model ID [--stream]` | Sends the request twice with your AWS credentials and checks the 2nd call reads from cache (streaming APIs too), and explains likely causes on failure | no cache read |
| `cachecanary logs *.json.gz --min-hit 0.5` | Hit rate per model / IAM principal from Bedrock model-invocation logs | a group is below the threshold |

Exit codes: `0` ok, `1` caching problem found, `2` could not run (bad input, AWS error such as missing model access or credentials).

Requests can be Converse-shaped (`system`/`toolConfig`/`cachePoint`) or InvokeModel Claude bodies (`anthropic_version`, `cache_control`). Add `--json` before the command for machine-readable output.

## CI example (GitHub Actions)
```yaml
- run: pip install "cachecanary[aws]"
- run: cachecanary lint tests/fixtures/agent_request.json
- run: cachecanary probe tests/fixtures/agent_request.json --model us.anthropic.claude-sonnet-4-6
```

## Notes
- Token counts in `lint` are estimates (~4 chars/token); they catch prefixes far below a minimum.
- `logs` reads S3-delivered records (JSONL or JSON array), CloudWatch Logs exports (`<timestamp> <json>`) and `aws logs filter-log-events` output, optionally gzipped; streamed response bodies are supported.
- Invocation logs only include cache counts inside the response body, which is inline when ≤ 100 KB. Larger ones are reported as `unparsed`.
- Model limits are in `src/cachecanary/models.py` (from the AWS prompt-caching docs, Oct 2026).

## Live verification
`scripts/live_test.py --region <region-without-production-traffic>` runs ~30 small synthetic calls and checks every assumption above against real Bedrock (hits, TTLs, streaming, per-model minimums, prefix changes, model switch, 20-block lookback).
