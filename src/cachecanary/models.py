"""Explicit prompt-caching limits per Claude model on Amazon Bedrock.

Source: AWS docs "Prompt caching for faster model inference" (checked 2026-10-05).
Keep this table current; new model IDs are the most common reason libraries
silently stop sending cache checkpoints.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheLimits:
    min_tokens: int          # cumulative prefix tokens required before a checkpoint
    max_checkpoints: int
    supports_1h: bool


# Keyed by the base model name as it appears inside Bedrock model IDs / profile IDs.
CLAUDE_LIMITS: dict[str, CacheLimits] = {
    "claude-sonnet-5-5": CacheLimits(512, 4, True),
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
}

# Bedrock's automatic prefix check only looks back ~20 content blocks from a checkpoint.
LOOKBACK_BLOCKS = 20


def lookup(model_id: str | None) -> tuple[str | None, CacheLimits | None]:
    """Match a Bedrock model ID, cross-region profile ID or ARN to its limits.

    Longest key wins so 'claude-opus-5-5' is not mistaken for 'claude-opus-5'.
    """
    if not model_id:
        return None, None
    lowered = model_id.lower()
    for key in sorted(CLAUDE_LIMITS, key=len, reverse=True):
        if key in lowered:
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
