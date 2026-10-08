# CacheCanary

Know the moment your Claude prompt cache breaks on Amazon Bedrock, not when the bill arrives.

[![CI](https://github.com/Haarris/cachecanary/actions/workflows/ci.yml/badge.svg)](https://github.com/Haarris/cachecanary/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Haarris/cachecanary/blob/main/LICENSE)

Cache reads on Bedrock cost about a tenth of normal input tokens, so a long system prompt or tool list gets a lot cheaper once it's cached. The catch is that caching fails quietly. Bump a library, switch to a new model ID, put today's date in the system prompt, or let your tools come out in a different order, and the cache stops hitting. Requests still return 200 and the answers are still right. You find out from the bill.

LiteLLM hit this in [July 2026](https://docs.litellm.ai/blog/bedrock-invoke-prompt-caching-incident). After an upgrade, Claude Code traffic on Bedrock went from about 90% cache hits to 25-45%, and daily spend went up 2-3x for six days before anyone noticed. Anthropic has a cache diagnostics feature, but it only works on their own API, not on Bedrock.

I built CacheCanary after running into these failures while running Claude agents in production at [Ruby](https://heyruby.io).

I wrote up the five ways I've seen caching break on Bedrock, with the bug reports behind each one and the lookback limit I measured: [Five ways Claude prompt caching quietly breaks on Amazon Bedrock](https://harisfarooq.substack.com/p/five-ways-claude-prompt-caching-quietly).

Website: [cachecanary.com](https://cachecanary.com). It has guides that show exactly what a framework sends to Bedrock, where the cache is lost, and a tested setup that keeps it: [LiteLLM](https://cachecanary.com/litellm/), [Strands Agents](https://cachecanary.com/strands/) and [LangChain](https://cachecanary.com/langchain/).

What that costs: the cached part of each request becomes about 10x more expensive, because cache reads are billed at a tenth of the input price. For one agent with a 10,000-token prompt and 100,000 requests a month, that's roughly $2,700 a month extra at Anthropic's list price for Claude Sonnet 4.6 ($3 per million input tokens; Bedrock prices vary by region).

CacheCanary is a small CLI and GitHub Action for this. It checks your requests in CI, tells you why a request missed the cache, and reads your Bedrock logs to show hit rates in production.

## Install

```bash
pip install cachecanary          # lint, diff, logs (no AWS needed)
pip install "cachecanary[aws]"   # adds probe, which calls Bedrock
```

## Try it

Save a request your app sends to Bedrock as JSON (Converse or InvokeModel format). With boto3's Converse API, dump the same kwargs you pass to `client.converse(**kwargs)`:

```python
json.dump(kwargs, open("request.json", "w"))
```

For InvokeModel, save the JSON body and pass `--model`. Then run:

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
| `logs files-or-folders... [--by model\|principal] [--min-hit 0.5] [--price 3]` | Hit rate and input cost per model or IAM principal from Bedrock invocation logs, plus how much the misses cost | a group is under the threshold |

Exit code 2 means it couldn't run at all (bad file, no model access, no AWS credentials). `--json` and `--github` go before the command.

## What it looks for

- Dates, times, UUIDs or user-specific text before the cache checkpoint.
- A cached prefix shorter than the model's minimum (512 to 4,096 tokens depending on the model).
- Tools listed in a different order, or a tool definition that changed. Tools come first, so this throws away the whole cache.
- More than 21 new content blocks between the old checkpoint and the new one, which is common after many parallel tool calls. Bedrock only looks back that far: measured live, 21 added blocks still hit and 22 always missed (`scripts/live_lookback_boundary.py`).
- Model IDs or application inference profile ARNs that your library may not recognize, so it quietly stops sending cache markers. Legacy models (Sonnet 4, Opus 4 and 4.1, 3.5 Haiku) are checked against Anthropic's documented minimums, since AWS's caching table doesn't list them.
- A switch between the Converse and InvokeModel APIs, which builds a different prompt.
- More than 4 checkpoints, an unknown cache lifetime (TTL), or a 1-hour checkpoint after a 5-minute one.
- A plain-string `system` field in InvokeModel, which can't carry `cache_control`.
- A cache point with nothing before it in its own list (a message, `system` or `tools`), for example at the start of a message. Bedrock rejects the request: "There is nothing available to cache."
- A Converse `cachePoint` inside a `toolResult`'s content. boto3 refuses to send it; over plain HTTP, as gateways send it, Bedrock accepts the request and caches nothing. It belongs after the `toolResult`.
- A conversation cache point stuck several messages back in an agent loop, so every new tool call and result is paid at full price (`lint`), or one that didn't move while new blocks were added (`diff`).
- Between two requests (`diff`): thinking settings changed or a different effort level, which on Bedrock throws away the whole cache including the system prompt; or tool choice switched between auto/none and any/tool, which rewrites the conversation part. Temperature and max tokens don't matter.

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

## Production hit rates and cost

Turn on [Bedrock model invocation logging](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html) (it's off by default; you pay normal S3 or CloudWatch storage for the logs). Copy the log files to your machine and point `logs` at the folder. It reads every log file inside it, including the dated subfolders S3 creates:

```bash
aws s3 sync s3://YOUR-BUCKET/AWSLogs/YOUR-ACCOUNT-ID/BedrockModelInvocationLogs/ bedrock-logs/
cachecanary logs bedrock-logs/ --min-hit 0.8
```

(If you set a key prefix for the logs, put it before `AWSLogs/`.) Here is the output on the redacted real logs in this repo:

```text
$ cachecanary logs tests/fixtures/bedrock_invocation_logs_2026-10.json --min-hit 0.8
us.anthropic.claude-sonnet-4-6: hit 50% over 8 calls (read 11496, write 11496, uncached 80, unparsed 0)  <-- below threshold
  input cost $0.05 at list price; about $0.03 of it lost to cache misses (target: 80% hit rate)
List prices: Amazon Bedrock on-demand, Oct 2026 (global. IDs at list, others +10%). Use --price for your own rate.
```

How the dollars work:

- **Input cost** is what these calls paid for input: uncached tokens at the input price, cache reads at the read price (0.1x for most models, 0.05x on Opus 5.5, 0.025x on Fable 5.1 and Mythos 5.1), and cache writes at 1.25x (5-minute) or 2x (1-hour).
- **Lost to cache misses** compares that with the same calls at your target hit rate (`--min-hit`, or 90% if you don't set it). The tokens that would have been cache reads were paid for at full price instead. If you're at or above the target, it's zero.
- **Prices** come from the Amazon Bedrock price list (on-demand, checked October 2026; the same in every Region we checked). `global.` model IDs are billed at list price, and regional or `us.`/`eu.`-style IDs cost 10% more. Claude 3.x models and application inference profile ARNs show no dollars. If you have a discount or a private price, pass `--price` with your input price per million tokens.
- Output tokens aren't included. Caching doesn't change them.

It reads S3 deliveries, CloudWatch exports and `aws logs filter-log-events` output, gzipped or not, including streamed responses. Bedrock logs InvokeModel calls under an inference profile ARN and Converse calls under the short model ID, so CacheCanary merges the two.

## How it's tested

The checks follow AWS's documented prompt-caching rules. The core behaviour was verified against live Amazon Bedrock: Converse and InvokeModel, streaming, both cache lifetimes (5 minutes and 1 hour), model minimums, prompt changes and the exact lookback limit (21 added blocks hit, 22 miss). Every check added in 0.3 was measured live first, and `scripts/live_v03_checks.py` re-runs those cases against Bedrock and compares CacheCanary's verdict with what Bedrock did (15 of 15 agreed in October 2026). Every check also has unit tests. The log reader is tested against real Bedrock invocation logs (redacted copies are in [tests/fixtures](https://github.com/Haarris/cachecanary/tree/main/tests/fixtures)).

To run the live checks yourself, use an AWS Region without production traffic: `python scripts/live_test.py --region <region>`.

## Privacy

Everything runs on your machine or CI runner. Nothing is sent anywhere. `probe` only calls your own Bedrock endpoint with your own credentials.

## Limits

- Token counts in `lint` are estimates. On English text Bedrock counted about 11% more than CacheCanary did, so near a model's minimum it warns you and suggests running `probe`.
- Bedrock keeps cache counts inside the logged response body, and only bodies up to 100 KB are stored inline. Bigger ones show up as `unparsed` and aren't in the dollar figures.
- Dollar figures use list prices, which change. Batch, provisioned throughput and priority tiers aren't covered; use `--price` for those.
- Bedrock only for now.

## What's next

A hosted dashboard: cache hit rate and wasted spend per app, an alert when the rate drops, and the reason for each miss. It would run inside your own AWS account, so prompts never leave it.

Using CacheCanary? I'd love to hear how it's going, good or bad, and whether you'd want the dashboard. Email me at [hello@cachecanary.com](mailto:hello@cachecanary.com). I read every message.

Vertex AI support is also planned.

## License

Apache 2.0. See [LICENSE](https://github.com/Haarris/cachecanary/blob/main/LICENSE). Copyright 2026 Haris Farooq.

Made by [Haris Farooq](https://www.linkedin.com/in/haris-farooq), an engineer who runs Claude agents on Bedrock in production.
