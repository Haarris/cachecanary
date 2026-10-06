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
    # Cache markers that had no block before them (Converse cachePoint as the very first item).
    orphan_checkpoints: list[str] = field(default_factory=list)
    # Extra markers on a block that was already a checkpoint (count toward Bedrock's limit).
    duplicate_checkpoints: list[str] = field(default_factory=list)
    # InvokeModel "system": "<string>" cannot carry cache_control.
    system_is_string: bool = False

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


def _mark_previous(req: NormalizedRequest, marker, location: str) -> None:
    """Converse uses a standalone cachePoint item: it closes the block before it."""
    if not req.blocks:
        req.orphan_checkpoints.append(location)
        return
    prev = req.blocks[-1]
    if prev.checkpoint:
        req.duplicate_checkpoints.append(location)
        return
    prev.checkpoint = True
    prev.ttl = _ttl(marker)


def _normalize_converse(payload: dict, req: NormalizedRequest) -> None:
    tool_config = payload.get("toolConfig")
    if tool_config is None:
        tool_config = {}
    if not isinstance(tool_config, dict):
        raise RequestError("'toolConfig' must be an object")
    for i, tool in enumerate(_items(tool_config.get("tools"), "toolConfig.tools")):
        loc = f"toolConfig.tools[{i}]"
        if isinstance(tool, dict) and "cachePoint" in tool:
            _mark_previous(req, tool["cachePoint"], loc)
        else:
            req.blocks.append(Block("tools", loc, _canon(tool)))
    for i, item in enumerate(_items(payload.get("system"), "system")):
        loc = f"system[{i}]"
        if isinstance(item, dict) and "cachePoint" in item:
            _mark_previous(req, item["cachePoint"], loc)
        else:
            req.blocks.append(Block("system", loc, _canon(item)))
    for m, msg in enumerate(_items(payload.get("messages"), "messages")):
        if not isinstance(msg, dict):
            raise RequestError(f"messages[{m}] must be an object")
        for c, item in enumerate(_items(msg.get("content"), f"messages[{m}].content")):
            loc = f"messages[{m}].content[{c}]"
            if isinstance(item, dict) and "cachePoint" in item:
                _mark_previous(req, item["cachePoint"], loc)
            else:
                body = item if isinstance(item, dict) else {"value": item}
                req.blocks.append(Block("messages", loc, _canon({"role": msg.get("role"), **body})))


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
            block = _invoke_block("messages", f"messages[{m}].content[{c}]", item)
            block.content = _canon({"role": msg.get("role"), "block": json.loads(block.content)})
            req.blocks.append(block)


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
