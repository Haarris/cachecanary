# CacheCanary

Know the moment your Claude prompt cache breaks on Amazon Bedrock, not when the bill arrives.

[![CI](https://github.com/Haarris/cachecanary/actions/workflows/ci.yml/badge.svg)](https://github.com/Haarris/cachecanary/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Haarris/cachecanary/blob/main/LICENSE)

Cache reads on Bedrock cost about a tenth of normal input tokens, so a long system prompt or tool list gets a lot cheaper once it's cached. The catch is that caching fails quietly. Bump a library, switch to a new model ID, put today's date in the system prompt, or let your tools come out in a different order, and the cache stops hitting. Requests still return 200 and the answers are still right. You find out from the bill.

LiteLLM hit this in [July 2026](https://docs.litellm.ai/blog/bedrock-invoke-prompt-caching-incident). After an upgrade, Claude Code traffic on Bedrock went from about 90% cache hits to 25-45%, and daily spend went up 2-3x for six days before anyone noticed. Anthropic has a cache diagnostics feature, but it only works on their own API, not on Bedrock.

What that costs: the cached part of each request becomes about 10x more expensive, because cache reads are billed at a tenth of the input price. For one agent with a 10,000-token prompt and 100,000 requests a month, that's roughly $2,700 a month extra at Anthropic's list price for Claude Sonnet 4.6 ($3 per million input tokens; Bedrock prices vary by region).

CacheCanary is a small CLI and GitHub Action for this. It checks your requests in CI, tells you why a request missed the cache, and reads your Bedrock logs to show hit rates in production.

## Install

```bash
pip install cachecanary          # lint, diff, logs (no AWS needed)
pip install "cachecanary[aws]"   # adds probe, which calls Bedrock
```

## Try it

Save a request your app sends to Bedrock as JSON (Converse or InvokeModel format) and run:

```text
$ cachecanary lint examples/request_bad.json
[warn] dynamic-in-prefix: Found ISO date text inside the cached prefix. If it changes per request, every call misses. (system[0])

$ cachecanary diff examples/request_bad.json examples/request_next.json
system-changed: The system prompt changed inside the cached prefix (look for dates, IDs or per-user text). (system[0])

$ cachecanary lint examples/request_good.json --model us.anthropic.claude-haiku-4-5-20251001-v1:0
[error] prefix-too-short: Prefix up to this checkpoint is ~1892 tokens; claude-haiku-4-5 needs at least 4096. The request succeeds but nothing is cached. (system[0])
```

The last one is easy to miss. The same prompt caches fine on Sonnet 4.6 (minimum 1,024 tokens) and never caches on Haiku 4.5 (minimum 4,096).

## Commands

| Command | What it does | Fails (exit 1) when |
|---|---|---|
| `lint request.json [--model ID]` | Checks one request for things that stop caching | it finds an error |
| `diff previous.json next.json` | Explains why the second request missed the first one's cache | the cached part changed |
| `probe request.json --model ID [--region R] [--stream]` | Sends the request twice and checks the second one read from cache | nothing was read from cache |
| `logs files... [--by model\|principal] [--min-hit 0.5]` | Hit rate per model or IAM principal from Bedrock invocation logs | a group is under the threshold |

Exit code 2 means it couldn't run at all (bad file, no model access, no AWS credentials). `--json` and `--github` go before the command.

## What it looks for

- Dates, times, UUIDs or user-specific text before the cache checkpoint.
- A cached prefix shorter than the model's minimum (512 to 4,096 tokens depending on the model).
- Tools listed in a different order, or a tool definition that changed. Tools come first, so this throws away the whole cache.
- More than about 20 new content blocks since the last checkpoint, which is common after many parallel tool calls. Bedrock only looks back that far.
- Model IDs or application inference profile ARNs that your library may not recognize, so it quietly stops sending cache markers.
- A switch between the Converse and InvokeModel APIs, which builds a different prompt.
- More than 4 checkpoints, an unknown TTL, or a 1-hour checkpoint after a 5-minute one.
- A plain-string `system` field in InvokeModel, which can't carry `cache_control`.

## GitHub Action

Problems show up as annotations on the pull request, plus a short table in the job summary.

```yaml
- uses: actions/checkout@v7

# Checks a saved request. No AWS needed.
- uses: Haarris/cachecanary@v0
  with:
    command: lint
    args: tests/fixtures/agent_request.json --model us.anthropic.claude-sonnet-4-6

# Calls Bedrock. Set up AWS credentials first (for example with aws-actions/configure-aws-credentials and OIDC).
- uses: Haarris/cachecanary@v0
  with:
    command: probe
    args: tests/fixtures/agent_request.json --model us.anthropic.claude-sonnet-4-6 --region us-west-2
```

`args` takes the same arguments as the CLI, separated by spaces. Paths with spaces aren't supported.

## Production hit rates

Turn on [Bedrock model invocation logging](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html) to S3 or CloudWatch, then point `logs` at the files:

```text
$ cachecanary logs tests/fixtures/bedrock_invocation_logs_2026-10.json --min-hit 0.4
us.anthropic.claude-sonnet-4-6: hit 50% over 8 calls (read 11496, write 11496, uncached 80, unparsed 0)
```

It reads S3 deliveries, CloudWatch exports and `aws logs filter-log-events` output, gzipped or not, including streamed responses. Bedrock logs InvokeModel calls under an inference profile ARN and Converse calls under the short model ID, so CacheCanary merges the two.

## How it's tested

The checks follow AWS's documented prompt-caching rules. The core behaviour was verified against live Amazon Bedrock: Converse and InvokeModel, streaming, both cache TTLs, model minimums, prompt changes and the 20-block lookback. Every check also has unit tests. The log reader is tested against real Bedrock invocation logs (redacted copies are in [tests/fixtures](https://github.com/Haarris/cachecanary/tree/main/tests/fixtures)).

To run the live checks yourself, use an AWS Region without production traffic: `python scripts/live_test.py --region <region>`.

## Privacy

Everything runs on your machine or CI runner. Nothing is sent anywhere. `probe` only calls your own Bedrock endpoint with your own credentials.

## Limits

- Token counts in `lint` are estimates. On English text Bedrock counted about 11% more than CacheCanary did, so near a model's minimum it warns you and suggests running `probe`.
- Bedrock keeps cache counts inside the logged response body, and only bodies up to 100 KB are stored inline. Bigger ones show up as `unparsed`.
- Bedrock only for now.

## What's next

A hosted dashboard: cache hit rate and wasted spend per app, an alert when the rate drops, and the reason for each miss. It would run inside your own AWS account, so prompts never leave it. If you'd use that, email [hello@cachecanary.com](mailto:hello@cachecanary.com).

Vertex AI and LiteLLM support are also planned.

## License

Apache 2.0. See [LICENSE](https://github.com/Haarris/cachecanary/blob/main/LICENSE). Copyright 2026 Haris Farooq.
