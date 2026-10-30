"""Explicit prompt-caching limits per Claude model on Amazon Bedrock.

Source: AWS docs "Prompt caching for faster model inference" (checked 2026-10-15).
Keep this table current; new model IDs are the most common reason libraries
silently stop sending cache checkpoints.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheLimits:
    min_tokens: int          # cumulative prefix tokens required before a checkpoint
    max_checkpoints: int
    supports_1h: bool | None  # None = not documented for Bedrock
    legacy: bool = False      # Bedrock legacy model, not in AWS's prompt-caching table


# Keyed by the base model name as it appears inside Bedrock model IDs / profile IDs.
CLAUDE_LIMITS: dict[str, CacheLimits] = {
    "claude-sonnet-5-5": CacheLimits(512, 4, True),
    "claude-haiku-5-5": CacheLimits(512, 4, True),
    "claude-opus-5-5": CacheLimits(512, 4, True),
    "claude-fable-5-1": CacheLimits(512, 4, True),
    "claude-mythos-5-1": CacheLimits(512, 4, True),
    "claude-fable-5": CacheLimits(512, 4, True),
    "claude-mythos-5": CacheLimits(512, 4, True),
    "claude-opus-5": CacheLimits(512, 4, True),
    "claude-opus-4-8": CacheLimits(1024, 4, True),
    "claude-opus-4-7": CacheLimits(4096, 4, True),
    "claude-opus-4-6": CacheLimits(4096, 4, True),
    "claude-opus-4-5": CacheLimits(4096, 4, True),
    "claude-sonnet-5": CacheLimits(1024, 4, True),
    "claude-sonnet-4-6": CacheLimits(1024, 4, True),
    "claude-sonnet-4-5": CacheLimits(1024, 4, True),
    "claude-3-7-sonnet": CacheLimits(1024, 4, False),
    "claude-3-5-sonnet": CacheLimits(1024, 4, False),
    "claude-haiku-4-5": CacheLimits(4096, 4, True),
    # Legacy on Bedrock: AWS's caching table doesn't list them and Bedrock refuses accounts that haven't
    # used them in 30 days (checked live Oct 2026). Minimums are Anthropic's documented values; AWS's
    # price list has cache read/write prices for them but no 1-hour write price.
    "claude-sonnet-4": CacheLimits(1024, 4, None, legacy=True),
    "claude-opus-4-1": CacheLimits(1024, 4, None, legacy=True),
    "claude-opus-4": CacheLimits(1024, 4, None, legacy=True),
    "claude-3-5-haiku": CacheLimits(2048, 4, None, legacy=True),
}

# How far back a checkpoint finds an earlier cache entry, in lookback positions (see
# diff.lookback_positions for how blocks turn into positions). Measured on live Bedrock, us-west-2,
# Oct 30 2026 (scripts/live_lookback_positions.py): Sonnet 4.6 two runs per shape (Converse and
# InvokeModel, plus real thinking turns), Sonnet 5.5 one run (no thinking shape). 21 positions added
# still hit, 22 always missed. On Oct 7 2026 Bedrock still counted every block, so many parallel tool
# calls lost the cache; by Oct 28 it counted a run of tool calls (or tool results) once, as Anthropic's
# own API documents: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
MAX_LOOKBACK_POSITIONS = 21


def lookup(model_id: str | None) -> tuple[str | None, CacheLimits | None]:
    """Match a Bedrock model ID, cross-region profile ID or ARN to its limits.

    The name must end where the model name ends (end of ID, a version suffix like '-v1:0', or a
    date like '-20250514'), so 'claude-opus-4' never matches a newer 'claude-opus-4-9'.
    """
    if not model_id:
        return None, None
    lowered = model_id.lower()
    for key in sorted(CLAUDE_LIMITS, key=len, reverse=True):
        if re.search(re.escape(key) + r"(?=$|[:/]|-v\d|-\d{8})", lowered):
            return key, CLAUDE_LIMITS[key]
    return None, None


def is_profile_arn(model_id: str | None) -> bool:
    """APPLICATION inference profile ARNs hide the model name; many libraries then skip caching.
    System profile ARNs (…:inference-profile/us.anthropic.claude-…) still name the model."""
    return bool(model_id) and model_id.startswith("arn:") and ":application-inference-profile/" in model_id


def canonical_model_id(model_id: str | None) -> str | None:
    """Collapse ARN forms to the ID callers actually send.

    Observed live (Oct 2026): invocation logs record Converse calls as
    'us.anthropic.claude-sonnet-4-6' but InvokeModel calls for the same model as
    'arn:aws:bedrock:<region>:<account>:inference-profile/us.anthropic.claude-sonnet-4-6'.
    Foundation-model ARNs ('arn:aws:bedrock:<region>::foundation-model/<id>') are collapsed too.
    Application inference profile ARNs are opaque and left unchanged.
    """
    if not model_id or not model_id.startswith("arn:"):
        return model_id
    for marker in (":inference-profile/", ":foundation-model/"):
        if marker in model_id and ":application-inference-profile/" not in model_id:
            return model_id.split(marker, 1)[1]
    return model_id
