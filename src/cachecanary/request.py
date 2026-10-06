"""Normalize Bedrock Converse and InvokeModel (Claude) requests into one ordered block list.

Bedrock processes cacheable content in the order tools -> system -> messages, and a
change anywhere invalidates everything after it. Flattening both API shapes into the
same sequence lets lint and diff reason about "the prefix" uniformly.
"""

import json
import re
from dataclasses import dataclass, field


class RequestError(ValueError):
    """The payload is not a request shape we understand."""


@dataclass
class Block:
    section: str        # "tools" | "system" | "messages"
    location: str       # human-readable path, e.g. "messages[3].content[1]"
    content: str        # canonical JSON of the block without cache markers
    checkpoint: bool = False
    ttl: str | None = None  # "5m" | "1h" when checkpoint


@dataclass
class NormalizedRequest:
    api: str                       # "converse" | "invoke"
    model_id: str | None
    blocks: list[Block] = field(default_factory=list)
    # Converse cachePoints with no block before them in the same list (a message's content, system
    # or tools). Bedrock rejects these: "There is nothing available to cache" (verified live, Oct 2026).
    orphan_checkpoints: list[str] = field(default_factory=list)
    # Converse cachePoints nested inside toolResult.content. boto3 refuses to send them; over plain
    # HTTP Bedrock accepts the request and caches nothing for them (verified live, Oct 2026).
    nested_checkpoints: list[str] = field(default_factory=list)
    # Extra markers on a block that was already a checkpoint (count toward Bedrock's limit).
    duplicate_checkpoints: list[str] = field(default_factory=list)
    # InvokeModel "system": "<string>" cannot carry cache_control.
    system_is_string: bool = False
    # Request settings that change the cache key on Bedrock (verified live, Oct 2026):
    thinking: str | None = None   # canonical thinking config; None when absent or {"type": "disabled"}
    effort: str = "high"          # output_config.effort; absent behaves exactly like "high"
    tool_choice: str | None = None  # "auto" (auto/none/absent) or "forced" (any/tool); None without tools

    @property
    def checkpoint_indexes(self) -> list[int]:
        return [i for i, b in enumerate(self.blocks) if b.checkpoint]

    @property
    def checkpoint_markers(self) -> int:
        """Markers as Bedrock counts them toward the per-request maximum."""
        return len(self.checkpoint_indexes) + len(self.duplicate_checkpoints) + len(self.orphan_checkpoints)


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _ttl(marker) -> str:
    if isinstance(marker, dict) and marker.get("ttl"):
        return str(marker["ttl"])
    return "5m"


def _items(value, path: str) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RequestError(f"'{path}' must be a list, got {type(value).__name__}")
    return value


def _mark_previous(req: NormalizedRequest, marker, location: str, list_start: int) -> None:
    """Converse uses a standalone cachePoint item: it closes the block before it in the same list.

    list_start is the index of the first block of the current list (message content, system or
    tools). A cachePoint with no block of its own list before it caches nothing, and Bedrock rejects
    it, even when an earlier list ends with a block.
    """
    if len(req.blocks) <= list_start:
        req.orphan_checkpoints.append(location)
        return
    prev = req.blocks[-1]
    if prev.checkpoint:
        req.duplicate_checkpoints.append(location)
        return
    prev.checkpoint = True
    prev.ttl = _ttl(marker)


def _thinking_key(thinking) -> str | None:
    if thinking is None or (isinstance(thinking, dict) and thinking.get("type") == "disabled"):
        return None
    return _canon(thinking)


def _effort(output_config) -> str:
    effort = output_config.get("effort") if isinstance(output_config, dict) else None
    return str(effort).lower() if effort else "high"


def _choice_group(kind) -> str:
    return "forced" if kind in ("any", "tool") else "auto"


def _normalize_converse(payload: dict, req: NormalizedRequest) -> None:
    tool_config = payload.get("toolConfig")
    if tool_config is None:
        tool_config = {}
    if not isinstance(tool_config, dict):
        raise RequestError("'toolConfig' must be an object")
    start = len(req.blocks)
    tools = _items(tool_config.get("tools"), "toolConfig.tools")
    for i, tool in enumerate(tools):
        loc = f"toolConfig.tools[{i}]"
        if isinstance(tool, dict) and "cachePoint" in tool:
            _mark_previous(req, tool["cachePoint"], loc, start)
        else:
            req.blocks.append(Block("tools", loc, _canon(tool)))
    if any(not (isinstance(t, dict) and "cachePoint" in t) for t in tools):
        choice = tool_config.get("toolChoice")
        req.tool_choice = _choice_group(next(iter(choice), None) if isinstance(choice, dict) else None)
    start = len(req.blocks)
    for i, item in enumerate(_items(payload.get("system"), "system")):
        loc = f"system[{i}]"
        if isinstance(item, dict) and "cachePoint" in item:
            _mark_previous(req, item["cachePoint"], loc, start)
        else:
            req.blocks.append(Block("system", loc, _canon(item)))
    for m, msg in enumerate(_items(payload.get("messages"), "messages")):
        if not isinstance(msg, dict):
            raise RequestError(f"messages[{m}] must be an object")
        start = len(req.blocks)
        for c, item in enumerate(_items(msg.get("content"), f"messages[{m}].content")):
            loc = f"messages[{m}].content[{c}]"
            if isinstance(item, dict) and "cachePoint" in item:
                _mark_previous(req, item["cachePoint"], loc, start)
                continue
            result = item.get("toolResult") if isinstance(item, dict) else None
            if isinstance(result, dict) and isinstance(result.get("content"), list):
                for k, part in enumerate(result["content"]):
                    if isinstance(part, dict) and "cachePoint" in part:
                        req.nested_checkpoints.append(f"{loc}.toolResult.content[{k}]")
            body = item if isinstance(item, dict) else {"value": item}
            req.blocks.append(Block("messages", loc, _canon({"role": msg.get("role"), **body})))
    extra = payload.get("additionalModelRequestFields")
    if isinstance(extra, dict):
        req.thinking = _thinking_key(extra.get("thinking"))
        req.effort = _effort(extra.get("output_config"))


def _invoke_block(section: str, location: str, item) -> Block:
    """InvokeModel attaches cache_control to the block itself."""
    if isinstance(item, dict) and "cache_control" in item:
        rest = {k: v for k, v in item.items() if k != "cache_control"}
        return Block(section, location, _canon(rest), checkpoint=True, ttl=_ttl(item["cache_control"]))
    return Block(section, location, _canon(item))


def _normalize_invoke(payload: dict, req: NormalizedRequest) -> None:
    for i, tool in enumerate(_items(payload.get("tools"), "tools")):
        req.blocks.append(_invoke_block("tools", f"tools[{i}]", tool))
    system = payload.get("system")
    if isinstance(system, str):
        req.system_is_string = True
        # Same canonical form as a one-block list, so switching shapes is not reported as a change.
        req.blocks.append(Block("system", "system", _canon({"type": "text", "text": system})))
    else:
        for i, item in enumerate(_items(system, "system")):
            req.blocks.append(_invoke_block("system", f"system[{i}]", item))
    for m, msg in enumerate(_items(payload.get("messages"), "messages")):
        if not isinstance(msg, dict):
            raise RequestError(f"messages[{m}] must be an object")
        content = msg.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for c, item in enumerate(_items(content, f"messages[{m}].content")):
            # A tool_result can carry cache_control on a block inside its content; Bedrock caches
            # up to that point (verified live, Oct 2026), so it counts as a checkpoint here. The
            # inner marker is left out of the compared content, like a top-level one.
            inner = []
            if isinstance(item, dict) and isinstance(item.get("content"), list):
                inner = [p for p in item["content"] if isinstance(p, dict) and "cache_control" in p]
                if inner:
                    item = {**item, "content": [{k: v for k, v in p.items() if k != "cache_control"}
                                                if isinstance(p, dict) else p for p in item["content"]]}
            block = _invoke_block("messages", f"messages[{m}].content[{c}]", item)
            if inner and not block.checkpoint:
                block.checkpoint, block.ttl = True, _ttl(inner[-1]["cache_control"])
            block.content = _canon({"role": msg.get("role"), "block": json.loads(block.content)})
            req.blocks.append(block)
    req.thinking = _thinking_key(payload.get("thinking"))
    req.effort = _effort(payload.get("output_config"))
    if payload.get("tools"):
        choice = payload.get("tool_choice")
        req.tool_choice = _choice_group(choice.get("type") if isinstance(choice, dict) else None)


def normalize(payload, model_id: str | None = None) -> NormalizedRequest:
    """Accept a Converse request (system/toolConfig/cachePoint shapes) or an InvokeModel
    Claude body (has anthropic_version). `model_id` overrides any modelId in the payload."""
    if not isinstance(payload, dict):
        raise RequestError(f"request must be a JSON object, got {type(payload).__name__}")
    api = "invoke" if "anthropic_version" in payload else "converse"
    req = NormalizedRequest(api, model_id or payload.get("modelId"))
    if api == "invoke":
        _normalize_invoke(payload, req)
    else:
        _normalize_converse(payload, req)
    return req


# Long base64 runs (inline images/documents). Their token cost has nothing to do with
# their character length, so they are replaced with a flat estimate.
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{512,}")
TOKENS_PER_BINARY = 1500


def _looks_like_base64(run: str) -> bool:
    """Real base64 mixes upper case, lower case and digits; a long word-like run does not."""
    return any(c.isupper() for c in run) and any(c.islower() for c in run) and any(c.isdigit() for c in run)


def estimate_tokens(text: str) -> int:
    """Rough estimate (~4 chars/token, binary blobs at a flat rate). Good enough to warn
    about prefixes far below a model minimum; not a billing-grade count."""
    binaries = 0

    def _strip(match: re.Match) -> str:
        nonlocal binaries
        if _looks_like_base64(match.group(0)):
            binaries += 1
            return ""
        return match.group(0)

    stripped = _BASE64_RUN.sub(_strip, text)
    return max(1, len(stripped) // 4 + binaries * TOKENS_PER_BINARY)
