"""Explain why request B missed the cache that request A should have created.

This rebuilds, for Bedrock, what Anthropic's cache diagnostics reports on the Claude API:
compare consecutive requests and name the first thing that changed inside the cached prefix.
"""

from dataclasses import dataclass

from cachecanary.models import LOOKBACK_BLOCKS
from cachecanary.request import NormalizedRequest


@dataclass
class MissReason:
    code: str
    message: str
    location: str | None = None


def explain(a: NormalizedRequest, b: NormalizedRequest) -> list[MissReason]:
    reasons: list[MissReason] = []

    if a.api != b.api:
        reasons.append(MissReason(
            "api-changed",
            f"Requests switched API ({a.api} -> {b.api}). Converse and InvokeModel build different prompts, "
            "so the cache entry cannot be reused (common after a gateway/library upgrade).",
        ))
        return reasons

    if (a.model_id or "") != (b.model_id or ""):
        reasons.append(MissReason("model-changed", f"Model changed from {a.model_id} to {b.model_id}. Caches are per model."))
        return reasons

    a_cps = a.checkpoint_indexes
    if not a_cps:
        reasons.append(MissReason("no-checkpoint-in-previous", "The previous request had no checkpoint, so it wrote nothing to read back."))
        return reasons
    a_last = a_cps[-1]

    if not b.checkpoint_indexes:
        reasons.append(MissReason(
            "no-checkpoint-in-next",
            "The new request has no cache checkpoint, so it never asks to read the cache "
            "(a library or code path stopped sending cache markers).",
        ))
        return reasons

    for i in range(min(a_last + 1, len(b.blocks))):
        if a.blocks[i].content != b.blocks[i].content:
            reasons.append(_classify_change(a, b, i))
            return reasons
    if len(b.blocks) <= a_last:
        reasons.append(MissReason("prefix-truncated", "The new request is shorter than the previous cached prefix (history was trimmed or compacted)."))
        return reasons

    a_ttls = [a.blocks[i].ttl for i in a_cps]
    b_ttls = [b.blocks[i].ttl for i in b.checkpoint_indexes]
    if a_ttls and b_ttls and a_ttls[0] != b_ttls[0]:
        reasons.append(MissReason("ttl-changed", f"Checkpoint TTL changed ({a_ttls[0]} -> {b_ttls[0]}); entries are keyed by TTL."))

    b_cps = b.checkpoint_indexes
    if b_cps and b_cps[-1] - a_last > LOOKBACK_BLOCKS and not any(a_last <= i < b_cps[-1] for i in b_cps[:-1]):
        reasons.append(MissReason(
            "lookback-exceeded",
            f"{b_cps[-1] - a_last} blocks were added between the previous checkpoint and the new one. Bedrock only "
            f"looks back ~{LOOKBACK_BLOCKS} blocks, so it cannot find the earlier cache entry. Add an intermediate "
            "checkpoint (common after many parallel tool calls).",
            b.blocks[b_cps[-1]].location,
        ))

    if not reasons:
        reasons.append(MissReason(
            "prefix-identical",
            "The cached prefix is identical. If B still missed, the entry likely expired (TTL elapsed between calls) "
            "or cross-region inference routed to a Region without the entry.",
        ))
    return reasons


def _classify_change(a: NormalizedRequest, b: NormalizedRequest, i: int) -> MissReason:
    section = a.blocks[i].section
    if section == "tools":
        a_tools = sorted(x.content for x in a.blocks if x.section == "tools")
        b_tools = sorted(x.content for x in b.blocks if x.section == "tools")
        if a_tools == b_tools:
            return MissReason("tools-reordered", "Same tools, different order. Sort tools deterministically.", b.blocks[i].location)
        return MissReason("tools-changed", "A tool definition changed; tools come first, so the whole cache is invalidated.", b.blocks[i].location)
    if section == "system":
        return MissReason("system-changed", "The system prompt changed inside the cached prefix (look for dates, IDs or per-user text).", b.blocks[i].location)
    return MissReason("history-changed", "Earlier conversation history was edited (not just appended).", b.blocks[i].location)
