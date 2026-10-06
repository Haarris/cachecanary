"""Aggregate cache hit rates from Bedrock model-invocation logs.

Invocation log records only carry inputTokenCount/outputTokenCount at the top level; the
cache counts live in output.outputBodyJson (the model response), which is inline only when
the body is <= 100 KB. Records whose body was offloaded to S3 are counted as 'unparsed'.

Accepted inputs (optionally .gz):
- S3 delivery: one JSON record per line, or a JSON array of records
- CloudWatch Logs export to S3: "<ISO timestamp> <json record>" per line
- `aws logs filter-log-events` output: {"events": [{"message": "<json record>"}, ...]}
"""

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from cachecanary.models import canonical_model_id
from cachecanary.usage import Usage, from_response


@dataclass
class Group:
    calls: int = 0
    unparsed: int = 0
    usage: Usage = field(default_factory=Usage)
    # Usage split by model ID, so a group of several models can be priced per model.
    by_model: dict[str, Usage] = field(default_factory=dict)


@dataclass
class ReadStats:
    bad_lines: int = 0


def _iter_concatenated(text: str, stats: ReadStats | None):
    """Records glued together ('{...}{...}') or one per line; skips unreadable fragments."""
    decoder = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            return
        if text[i] != "{":
            # CloudWatch export: "<timestamp> {json}" -> skip to the next object start.
            nxt = text.find("{", i)
            if nxt == -1:
                if stats:
                    stats.bad_lines += 1
                return
            i = nxt
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            if stats:
                stats.bad_lines += 1
            nl = text.find("\n", i)
            if nl == -1:
                return
            i = nl + 1
            continue
        yield obj
        i = end


def iter_records(path: Path, stats: ReadStats | None = None):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        return
    if text.startswith("["):
        yield from json.loads(text)
        return
    if text.startswith("{"):
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            doc = None
        if isinstance(doc, dict):
            if isinstance(doc.get("events"), list):
                for event in doc["events"]:
                    try:
                        yield json.loads(event.get("message", ""))
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        if stats:
                            stats.bad_lines += 1
                return
            yield doc
            return
    yield from _iter_concatenated(text, stats)


def _body(rec: dict):
    body = (rec.get("output") or {}).get("outputBodyJson")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return None
    return body


def _is_invocation_record(rec) -> bool:
    """Real log records, not offloaded request/response bodies stored next to them in S3."""
    if not isinstance(rec, dict):
        return False
    if rec.get("schemaType") is not None:
        return rec.get("schemaType") == "ModelInvocationLog"
    return "modelId" in rec and ("output" in rec or "operation" in rec)


def aggregate(records, by: str = "model") -> dict[str, Group]:
    """Group by 'model', 'principal' or 'model+principal'."""
    groups: dict[str, Group] = defaultdict(Group)
    for rec in records:
        if not _is_invocation_record(rec):
            continue
        model = canonical_model_id(rec.get("modelId")) or "?"
        principal = (rec.get("identity") or {}).get("arn") or "?"
        key = {"model": model, "principal": principal}.get(by, f"{model} | {principal}")
        g = groups[key]
        g.calls += 1
        usage = from_response(_body(rec))
        if usage is None:
            g.unparsed += 1
            continue
        g.usage = g.usage + usage
        g.by_model[model] = g.by_model.get(model, Usage()) + usage
    return dict(groups)
