"""Explain why request B missed the cache that request A should have created.

This rebuilds, for Bedrock, what Anthropic's cache diagnostics reports on the Claude API:
compare consecutive requests and name the first thing that changed inside the cached prefix.
"""

import json
from dataclasses import dataclass

from cachecanary.models import MAX_BLOCKS_ADDED
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

    # Verified live on Bedrock (Oct 2026): either change makes the next call read nothing from the
    # cache, not even the system prompt, on both Converse and InvokeModel.
    if a.thinking != b.thinking:
        reasons.append(MissReason(
            "thinking-changed",
            f"Thinking settings changed ({_thinking_label(a.thinking)} -> {_thinking_label(b.thinking)}). On Bedrock "
            "this throws away the whole cache, system prompt included. Keep thinking settings fixed within a session.",
        ))
        return reasons
    if a.effort != b.effort:
        reasons.append(MissReason(
            "effort-changed",
            f"Effort changed ({a.effort} -> {b.effort}; no setting means high). Each effort level has its own "
            "cache on Bedrock, so this call reads nothing from it. Keep effort fixed within a session.",
        ))
        return reasons
    # Bedrock adds a different tool instruction for auto/none than for any/tool, so the conversation
    # part is written again while tools and system are still read (verified live, Oct 2026).
    if a.tool_choice and b.tool_choice and a.tool_choice != b.tool_choice:
        reasons.append(MissReason(
            "tool-choice-changed",
            f"Tool choice switched between auto/none and any/tool ({a.tool_choice} -> {b.tool_choice}). Bedrock "
            "then writes the conversation part of the cache again; tools and system are still read.",
        ))

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

    # B reads A's entry only if one of B's checkpoints sits at A's last checkpoint or within reach after it.
    later = [i for i in b.checkpoint_indexes if i >= a_last]
    if later and later[0] - a_last > MAX_BLOCKS_ADDED:
        added = later[0] - a_last
        reasons.append(MissReason(
            "lookback-exceeded",
            f"{added} blocks were added between the previous checkpoint and the next one. Bedrock only finds an "
            f"earlier cache entry up to {MAX_BLOCKS_ADDED} blocks back ({MAX_BLOCKS_ADDED} added still hits, "
            f"{MAX_BLOCKS_ADDED + 1} misses), so this call writes the cache again. Add a checkpoint in between "
            "(common after many parallel tool calls).",
            b.blocks[later[0]].location,
        ))

    if not reasons:
        reasons.append(MissReason(
            "prefix-identical",
            "The cached prefix is identical. If B still missed, the entry likely expired (TTL elapsed between calls) "
            "or cross-region inference routed to a Region without the entry.",
        ))
        # Not a miss, but money: the conversation grew and the cache point stayed where it was.
        added = len(b.blocks) - 1 - a_last
        if (b.checkpoint_indexes[-1] == a_last and a.blocks[a_last].section == "messages"
                and added >= STUCK_MIN_ADDED):
            reasons.append(MissReason(
                "checkpoint-not-moved",
                f"The cached part was read, but the last cache point didn't move while {added} new blocks were added "
                "after it. Those are paid at full price on every call. Put the cache point at the end of the "
                "newest message.",
                b.blocks[a_last].location,
            ))
    return reasons


# New blocks after an unmoved conversation checkpoint before it is worth pointing out
# (one agent round trip is at least a tool call and its result).
STUCK_MIN_ADDED = 2


def _thinking_label(key: str | None) -> str:
    if key is None:
        return "off"
    try:
        cfg = json.loads(key)
    except ValueError:
        return key
    if not isinstance(cfg, dict):
        return key
    budget = cfg.get("budget_tokens")
    return str(cfg.get("type", "on")) + (f", budget {budget}" if budget is not None else "")


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
