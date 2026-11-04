"""Why calls had to write the cache again, from Bedrock invocation logs (`logs --ttl-check`).

A call that read nothing from the cache but wrote to it is a re-write: no cached copy was found.
Each re-write is matched to the previous call that started the same way:
- with the request body in the logs and a system prompt or tools in it: the same model and the
  same system prompt and tools (cache markers ignored);
- otherwise: the same model and the same caller (IAM principal), which can mix different prompts,
  so those rows say "from this caller".
Only calls that read or wrote the cache count as the previous call.

Then it reports how long before the re-write that call happened and whether it was served in a
different region. The default cache lifetime is 5 minutes, reset by every cache hit; the optional
1-hour cache keeps it for an hour. AWS: a cache entry is stored on the server that processed the
request, so a request served by a different server (for example in another region), or sent before
an identical request has started its response, writes a new entry.

Calls that read from the cache and also wrote are not classified: in the logs, new text at the end
of the prompt looks the same as a conversation's own history expiring while a shared start stays
cached. Dollar figures are prices of what the logs show, not predictions.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime

from cachecanary.logs import _body, _is_invocation_record
from cachecanary.models import canonical_model_id, lookup
from cachecanary.pricing import Rates, rates_for
from cachecanary.usage import Usage, from_response

FIVE_MINUTES = 300
ONE_HOUR = 3600

# Re-write classes, in the order they are printed.
FIRST = "first"
FIVE_TO_60M = "5_to_60m"
OVER_1H = "over_1h"
ON_1H_CACHE = "on_1h_cache"
WITHIN_5M_OTHER_REGION = "within_5m_other_region"
WITHIN_5M_REGION_UNKNOWN = "within_5m_region_unknown"
WITHIN_5M = "within_5m"
CLASSES = (FIRST, FIVE_TO_60M, OVER_1H, ON_1H_CACHE, WITHIN_5M_OTHER_REGION, WITHIN_5M_REGION_UNKNOWN, WITHIN_5M)


@dataclass
class Call:
    when: datetime
    model: str  # canonical ID: used for grouping and pricing (keeps the global./us. prefix)
    caller: str
    region: str | None
    prompt_start: str | None  # hash of system prompt + tools, when the body is logged and has either
    body: str  # "inline", "s3" (over 100 KB, stored as a separate object) or "none"
    usage: Usage

    @property
    def touched_cache(self) -> bool:
        return self.usage.cache_read + self.usage.cache_write > 0


@dataclass
class Group:
    calls: int = 0
    calls_body_in_s3: int = 0
    calls_no_body: int = 0
    # Keyed "<class>" when matched by system prompt and tools, "<class>/caller" when matched by caller.
    rewrites: dict[str, int] = field(default_factory=dict)
    rewrite_tokens: dict[str, int] = field(default_factory=dict)
    same_second: int = 0  # re-writes less than a second after the previous call
    write_tokens: int = 0  # every cache write in the group (re-writes and other writes)
    write_tokens_1h: int = 0
    supports_1h: bool | None = None
    priced: bool = True
    cost_writes_now: float = 0.0  # all cache writes, as logged (5-minute / 1-hour split)
    cost_writes_at_1h: float = 0.0  # the same write tokens at the 1-hour write price
    cost_5to60_now: float = 0.0  # the 5-60 minute re-writes, as logged
    cost_5to60_as_reads: float = 0.0  # the same tokens at the cache read price


def _parse_time(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else None


def _is_marker(key, value) -> bool:
    return key == "cache_control" and isinstance(value, dict) and value.get("type") == "ephemeral"


def _strip_cache_markers(value):
    """Cache markers don't change what is cached, so leave them out of the fingerprint."""
    if isinstance(value, dict):
        if set(value) == {"cachePoint"}:
            return None
        return {k: _strip_cache_markers(v) for k, v in value.items() if not _is_marker(k, v)}
    if isinstance(value, list):
        return [x for x in (_strip_cache_markers(v) for v in value) if x is not None]
    return value


def _request_body(rec: dict):
    inp = rec.get("input")
    body = inp.get("inputBodyJson") if isinstance(inp, dict) else None
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return None
    return body if isinstance(body, dict) and "messages" in body else None


def _body_state(rec: dict) -> str:
    if _request_body(rec) is not None:
        return "inline"
    inp = rec.get("input")
    if isinstance(inp, dict) and inp.get("inputBodyS3Path"):
        return "s3"
    return "none"


def prompt_start(rec: dict) -> str | None:
    """Fingerprint of the system prompt and tools in the logged request body (Converse or Claude Messages).

    None when there is no usable body, or when it has neither a system prompt nor tools (then the
    fingerprint would match every unrelated conversation on the model).
    """
    body = _request_body(rec)
    if body is None:
        return None
    tools = body.get("tools")
    if tools is None and isinstance(body.get("toolConfig"), dict):
        tools = body["toolConfig"].get("tools")
    system = _strip_cache_markers(body.get("system"))
    tools = _strip_cache_markers(tools)
    if not system and not tools:
        return None
    start = {"system": system, "tools": tools}
    return hashlib.sha256(json.dumps(start, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def collect(records) -> tuple[list[Call], int]:
    """Calls with usage and a timestamp, and how many invocation records were left out."""
    calls, skipped = [], 0
    for rec in records:
        if not _is_invocation_record(rec):
            continue
        usage = from_response(_body(rec))
        when = _parse_time(rec.get("timestamp"))
        if usage is None or when is None:
            skipped += 1
            continue
        identity = rec.get("identity")
        caller = identity.get("arn") if isinstance(identity, dict) else None
        region = rec.get("inferenceRegion")
        calls.append(Call(when, str(canonical_model_id(rec.get("modelId")) or "?"), str(caller or "?"),
                          region if isinstance(region, str) and region else None,
                          prompt_start(rec), _body_state(rec), usage))
    return calls, skipped


def classify(gap_seconds: float | None, region: str | None, previous_region: str | None,
             previous_all_1h: bool = False) -> str:
    if gap_seconds is None:
        return FIRST
    if gap_seconds > ONE_HOUR:
        return OVER_1H
    if gap_seconds > FIVE_MINUTES:
        return ON_1H_CACHE if previous_all_1h else FIVE_TO_60M
    if not region or not previous_region:
        return WITHIN_5M_REGION_UNKNOWN
    if region != previous_region:
        return WITHIN_5M_OTHER_REGION
    return WITHIN_5M


def _write_cost(u: Usage, rates: Rates) -> float:
    write_1h = min(u.cache_write_1h, u.cache_write)
    return ((u.cache_write - write_1h) * rates.cache_write_5m + write_1h * rates.cache_write_1h) / 1_000_000


def check(calls: list[Call], custom_price: float | None = None) -> dict[str, Group]:
    """Re-writes per model, each classified by the time since the previous call that started the same way."""
    groups: dict[str, Group] = {}
    last: dict[tuple, Call] = {}
    last_write_all_1h: dict[tuple, bool] = {}  # was the latest write in this stream entirely 1-hour?
    for call in sorted(calls, key=lambda c: c.when):
        g = groups.get(call.model)
        if g is None:
            _, limits = lookup(call.model)
            g = groups[call.model] = Group(supports_1h=limits.supports_1h if limits else None)
        g.calls += 1
        g.calls_body_in_s3 += call.body == "s3"
        g.calls_no_body += call.body == "none"
        u = call.usage
        exact = call.prompt_start is not None
        key = (call.model, "prompt", call.prompt_start) if exact else (call.model, "caller", call.caller)
        previous = last.get(key)
        priced = rates_for(call.model, custom_price)
        if priced is None:
            g.priced = False
        if u.cache_write > 0:
            g.write_tokens += u.cache_write
            g.write_tokens_1h += min(u.cache_write_1h, u.cache_write)
            if priced:
                g.cost_writes_now += _write_cost(u, priced.rates)
                g.cost_writes_at_1h += u.cache_write * priced.rates.cache_write_1h / 1_000_000
        if u.cache_write > 0 and u.cache_read == 0:
            gap = (call.when - previous.when).total_seconds() if previous else None
            cls = classify(gap, call.region, previous.region if previous else None,
                           last_write_all_1h.get(key, False))
            row = cls if exact else f"{cls}/caller"
            g.rewrites[row] = g.rewrites.get(row, 0) + 1
            g.rewrite_tokens[row] = g.rewrite_tokens.get(row, 0) + u.cache_write
            if gap is not None and gap < 1:
                g.same_second += 1
            if cls == FIVE_TO_60M and priced:
                g.cost_5to60_now += _write_cost(u, priced.rates)
                g.cost_5to60_as_reads += u.cache_write * priced.rates.cache_read / 1_000_000
        if call.touched_cache:
            # Calls with a prompt fingerprint also update their caller's timeline, so a caller-matched
            # call (for example one whose body grew past 100 KB) sees every earlier call from that caller.
            for k in {key, (call.model, "caller", call.caller)}:
                if u.cache_write > 0:
                    last_write_all_1h[k] = u.cache_write_1h >= u.cache_write
                last[k] = call
    return groups


_WHEN = {
    FIRST: "no earlier call {who} read or wrote the cache in these logs",
    FIVE_TO_60M: "5 to 60 minutes after the previous call {who}",
    OVER_1H: "more than 1 hour after the previous call {who}",
    ON_1H_CACHE: "5 to 60 minutes after the previous call {who}; the latest cache write before it was all "
                 "on the 1-hour cache",
    WITHIN_5M_OTHER_REGION: "within 5 minutes of the previous call {who}, served in a different region",
    WITHIN_5M_REGION_UNKNOWN: "within 5 minutes of the previous call {who} (region not in the logs)",
    WITHIN_5M: "within 5 minutes of the previous call {who}, same region",
}
ROWS = list(CLASSES) + [f"{c}/caller" for c in CLASSES]
LABELS = {**{c: t.format(who="that started the same way") for c, t in _WHEN.items()},
          **{f"{c}/caller": t.format(who="from this caller (matched by caller and model, not by prompt)")
             for c, t in _WHEN.items()}}
