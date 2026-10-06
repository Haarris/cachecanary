"""Read cache token usage out of Bedrock responses (Converse or InvokeModel Claude),
including streamed responses."""

import json
from dataclasses import dataclass


@dataclass
class Usage:
    uncached_input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0
    cache_write_1h: int = 0  # the part of cache_write that went to the 1-hour cache, when reported

    @property
    def total_input(self) -> int:
        return self.uncached_input + self.cache_read + self.cache_write

    @property
    def hit_ratio(self) -> float:
        return self.cache_read / self.total_input if self.total_input else 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.uncached_input + other.uncached_input,
            self.cache_read + other.cache_read,
            self.cache_write + other.cache_write,
            self.output + other.output,
            self.cache_write_1h + other.cache_write_1h,
        )


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def from_usage_dict(usage) -> Usage | None:
    """Converse: {inputTokens, cacheReadInputTokens, cacheWriteInputTokens, outputTokens} where
    inputTokens is already the non-cached part, plus cacheDetails [{ttl: "5m"|"1h", inputTokens}].
    Claude Messages: {input_tokens, cache_read_input_tokens, cache_creation_input_tokens,
    output_tokens}, plus cache_creation {ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}."""
    if not isinstance(usage, dict):
        return None
    if any(k in usage for k in ("inputTokens", "cacheReadInputTokens", "cacheWriteInputTokens")):
        details = usage.get("cacheDetails")
        write_1h = sum(_int(d.get("inputTokens")) for d in details
                       if isinstance(d, dict) and d.get("ttl") == "1h") if isinstance(details, list) else 0
        return Usage(_int(usage.get("inputTokens")), _int(usage.get("cacheReadInputTokens")),
                     _int(usage.get("cacheWriteInputTokens")), _int(usage.get("outputTokens")), write_1h)
    if any(k in usage for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")):
        creation = usage.get("cache_creation")
        write_1h = _int(creation.get("ephemeral_1h_input_tokens")) if isinstance(creation, dict) else 0
        return Usage(_int(usage.get("input_tokens")), _int(usage.get("cache_read_input_tokens")),
                     _int(usage.get("cache_creation_input_tokens")), _int(usage.get("output_tokens")), write_1h)
    return None


def from_response(resp) -> Usage | None:
    """A non-streamed response body. A list is treated as a stream of events."""
    if isinstance(resp, list):
        return from_stream_events(resp)
    if not isinstance(resp, dict):
        return None
    return from_usage_dict(resp.get("usage"))


def _decode_event(event):
    """InvokeModelWithResponseStream wraps each event as {"chunk": {"bytes": b"<json>"}}."""
    if isinstance(event, dict) and isinstance(event.get("chunk"), dict):
        raw = event["chunk"].get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
    return event


def from_stream_events(events) -> Usage | None:
    """ConverseStream: a {"metadata": {"usage": {...}}} event near the end.
    Claude Messages stream: message_start.message.usage carries input and cache counts,
    message_delta.usage carries the final output_tokens."""
    found = None
    for raw in events or []:
        event = _decode_event(raw)
        if not isinstance(event, dict):
            continue
        meta = event.get("metadata")
        if isinstance(meta, dict) and from_usage_dict(meta.get("usage")):
            return from_usage_dict(meta.get("usage"))
        kind = event.get("type")
        if kind == "message_start":
            found = from_usage_dict((event.get("message") or {}).get("usage")) or found
        elif kind == "message_delta" and found is not None:
            out = (event.get("usage") or {}).get("output_tokens")
            if out is not None:
                found.output = _int(out)
    return found
