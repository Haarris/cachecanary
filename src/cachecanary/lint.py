"""Static checks on a single Bedrock request for patterns that silently defeat caching."""

import re
from dataclasses import dataclass

from cachecanary.models import is_profile_arn, lookup
from cachecanary.request import NormalizedRequest, estimate_tokens

# Values that usually change per request. Matching them inside the cached prefix is
# the classic "cache never hits" bug (e.g. "Today's date: ..." in the system prompt).
DYNAMIC_PATTERNS = {
    "ISO date": re.compile(r"\b20\d{2}-[01]\d-[0-3]\d\b"),
    "clock time": re.compile(r"\b[0-2]?\d:[0-5]\d(:[0-5]\d)?\b"),
    "UUID": re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
    "unix timestamp": re.compile(r"\b1[6-9]\d{8}(\d{3})?\b"),
}

# Conversation blocks after the last checkpoint before we suggest caching the history too.
MIN_UNCACHED_TAIL = 4

# Token estimates are approximate: live Bedrock runs (Oct 2026) counted ~11% MORE tokens than
# estimate_tokens() for English prompts. Only call a prefix "too short" when it is clearly below
# the minimum; inside the uncertainty band, warn and point to `probe` for a definitive answer.
CLEARLY_BELOW = 0.80
UNCERTAIN_UP_TO = 1.20


@dataclass
class Finding:
    rule: str
    severity: str   # "error" = caching cannot work as written; "warn" = likely miss / money left on table
    message: str
    location: str | None = None


def lint(req: NormalizedRequest) -> list[Finding]:
    findings: list[Finding] = []
    cps = req.checkpoint_indexes
    key, limits = lookup(req.model_id)

    if not req.model_id:
        findings.append(Finding(
            "no-model", "warn",
            "No model ID given (use --model). Model-specific checks such as the minimum prefix size were skipped.",
        ))
    elif is_profile_arn(req.model_id):
        findings.append(Finding(
            "profile-arn", "warn",
            "Model is an application inference profile ARN. Many libraries match model names to "
            "decide whether to send cache checkpoints and silently skip this case. Verify the "
            "response shows cache reads/writes. Minimum-size checks were skipped (model unknown).",
        ))
    elif key is None:
        findings.append(Finding(
            "unknown-model", "warn",
            f"'{req.model_id}' is not in the known Claude caching table. New model IDs are a common "
            "reason libraries stop adding checkpoints. Verify cache usage in responses.",
        ))

    for loc in req.orphan_checkpoints:
        findings.append(Finding(
            "orphan-checkpoint", "warn", "Cache checkpoint with no content before it; it caches nothing.", loc,
        ))

    if req.system_is_string and not any(req.blocks[i].section in ("tools", "system") for i in cps):
        findings.append(Finding(
            "string-system", "warn",
            "'system' is a plain string, which cannot carry cache_control. Send it as a list of "
            "text blocks and put cache_control on the last one to cache the system prompt.",
            "system",
        ))

    if not cps:
        findings.append(Finding(
            "no-checkpoint", "warn",
            "No explicit cache checkpoint. Only best-effort implicit caching applies; add a "
            "checkpoint after the static content (tools/system) for reliable hits.",
        ))
        return findings

    max_cp = limits.max_checkpoints if limits else 4
    if req.checkpoint_markers > max_cp:
        findings.append(Finding(
            "too-many-checkpoints", "error",
            f"{req.checkpoint_markers} cache markers; the maximum is {max_cp}. Bedrock rejects the request or ignores the extras.",
        ))
    for loc in req.duplicate_checkpoints:
        findings.append(Finding(
            "duplicate-checkpoint", "warn", "Two cache markers in a row; the second adds nothing but counts toward the limit.", loc,
        ))

    if limits:
        cumulative = 0
        clearly_short: list[tuple[int, int]] = []
        borderline: list[tuple[int, int]] = []
        cached_any = False
        for i, block in enumerate(req.blocks[: cps[-1] + 1]):
            cumulative += estimate_tokens(block.content)
            if not block.checkpoint:
                continue
            if cumulative < limits.min_tokens * CLEARLY_BELOW:
                clearly_short.append((i, cumulative))
            elif cumulative < limits.min_tokens * UNCERTAIN_UP_TO:
                borderline.append((i, cumulative))
                cached_any = cached_any or cumulative >= limits.min_tokens
            else:
                cached_any = True
        for i, prefix in clearly_short:
            consequence = (
                "The request succeeds but nothing is cached."
                if not cached_any and not borderline
                else "This checkpoint is ignored; later ones may still cache."
            )
            findings.append(Finding(
                "prefix-too-short", "error" if not cached_any and not borderline else "warn",
                f"Prefix up to this checkpoint is ~{prefix} tokens; {key} needs at least {limits.min_tokens}. {consequence}",
                req.blocks[i].location,
            ))
        for i, prefix in borderline:
            findings.append(Finding(
                "prefix-near-minimum", "warn",
                f"Prefix up to this checkpoint is ~{prefix} tokens (estimate), close to {key}'s minimum of "
                f"{limits.min_tokens}. Run `cachecanary probe` to confirm it is actually cached.",
                req.blocks[i].location,
            ))

    ttls = [req.blocks[i].ttl for i in cps]
    unknown_ttls = sorted({t for t in ttls if t not in ("5m", "1h")})
    if unknown_ttls:
        findings.append(Finding(
            "ttl-invalid", "error", f"Unsupported TTL value(s) {unknown_ttls}; only '5m' and '1h' exist.",
        ))
    if limits and not limits.supports_1h and "1h" in ttls:
        findings.append(Finding(
            "ttl-unsupported", "error", f"{key} only supports the 5-minute TTL; 'ttl: 1h' can raise a ValidationException.",
        ))
    if "5m" in ttls and "1h" in ttls and ttls.index("5m") < max(i for i, t in enumerate(ttls) if t == "1h"):
        findings.append(Finding(
            "ttl-order", "error", "A 1h checkpoint appears after a 5m checkpoint. Longer TTLs must come first.",
        ))

    last = cps[-1]
    for block in req.blocks[: last + 1]:
        for label, pattern in DYNAMIC_PATTERNS.items():
            if pattern.search(block.content):
                findings.append(Finding(
                    "dynamic-in-prefix", "warn",
                    f"Found {label} text inside the cached prefix. If it changes per request, every call misses.",
                    block.location,
                ))
                break

    tail = len(req.blocks) - 1 - last
    # Short tails are normal (one new question); only flag history that is re-billed every turn.
    if tail >= MIN_UNCACHED_TAIL and req.blocks[last].section != "messages":
        findings.append(Finding(
            "no-conversation-checkpoint", "warn",
            f"The last checkpoint is in {req.blocks[last].section}; {tail} conversation blocks after it are "
            "re-billed every turn. Add a checkpoint near the end of the conversation history.",
            req.blocks[last].location,
        ))

    return findings
