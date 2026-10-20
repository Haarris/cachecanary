"""Cache hit rates from the generations Langfuse already records, for teams without Bedrock
invocation logging.

Langfuse keeps each generation's token counts in `usageDetails`, a free-form map whose key names
depend on how the data was sent (checked against Langfuse, langfuse-python, Strands, LiteLLM and
the Vercel AI SDK source, Oct 2026):

- OpenTelemetry (Strands, LiteLLM OTel, OpenInference, Pydantic AI): `input` (Langfuse subtracts
  the cache counts from the reported input), `input_cached_tokens`, `input_cache_creation`, `total`.
- Vercel AI SDK: `input_cached_tokens` or `input_cache_read`, and `input_cache_creation` /
  `input_cache_write`, with `input_cache_creation_5m` / `_1h` split out of the aggregate.
- LangChain: `input_cache_read` / `input_cache_creation`, or Anthropic's `cache_read_input_tokens` /
  `cache_creation_input_tokens` plus `cache_creation_ephemeral_5m_input_tokens` / `_1h` inside it.
- LiteLLM's `langfuse` callback and hand-written SDK calls: `cache_read_input_tokens` /
  `cache_creation_input_tokens`.
- Strands also sends `cache_write_input_tokens`, which Langfuse stores unchanged (before Strands
  1.54.0 it is the only cache write count).

Two known sources end up with stored numbers that don't add up (Langfuse only logs a warning, issue #10592):
Strands before 1.54.0 reported Bedrock's already-uncached input, which Langfuse then reduced by the
cache counts again (often to 0, so the cache looks perfect), and LiteLLM's `langfuse` callback
leaves cache writes inside `input` as well as reporting them separately. When a generation carries
a total that was really sent (not one Langfuse added up itself), CacheCanary checks the parts
against it and rebuilds the uncached input as total - output - cache read - cache write; Bedrock
totals include the cache counts.
"""

import base64
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cachecanary import __version__
from cachecanary.logs import Group
from cachecanary.models import canonical_model_id
from cachecanary.usage import Usage

DEFAULT_HOST = "https://cloud.langfuse.com"
V2_PATH = "/api/public/v2/observations"
V1_PATH = "/api/public/observations"
V2_PAGE_SIZE = 1000  # API maximum
V1_PAGE_SIZE = 100   # API maximum
# Only what the report needs: the v2 API returns no prompts, outputs or metadata. (The v1 fallback can't select
# fields and returns whole observations; only type, model, name, environment and usageDetails are read.)
V2_FIELDS = "core,basic,model,usage"

READ_KEYS = ("input_cache_read", "input_cached_tokens", "cache_read_input_tokens")
# Vercel AI SDK: input_cache_creation is what is left after the TTL split, so the parts add up.
SPLIT_WRITE_KEYS = ("input_cache_creation", "input_cache_creation_5m", "input_cache_creation_1h")
# LangChain + Anthropic: the aggregate already contains the TTL parts.
ANTHROPIC_WRITE_KEY = "cache_creation_input_tokens"
# Names Langfuse does not recognise and stores unchanged. Strands sends its deprecated
# gen_ai.usage.cache_write_input_tokens attribute as this key; before 1.54 it is the only write count.
PASSTHROUGH_WRITE_KEYS = ("cache_write_input_tokens",)
ANTHROPIC_TTL_KEYS = ("cache_creation_ephemeral_5m_input_tokens", "cache_creation_ephemeral_1h_input_tokens")
WRITE_1H_KEYS = ("input_cache_creation_1h", "cache_creation_ephemeral_1h_input_tokens")
CACHE_KEYS = frozenset(READ_KEYS + SPLIT_WRITE_KEYS + (ANTHROPIC_WRITE_KEY, "input_cache_write")
                       + ANTHROPIC_TTL_KEYS + PASSTHROUGH_WRITE_KEYS)


class LangfuseError(Exception):
    pass


@dataclass
class Normalized:
    usage: Usage | None
    status: str  # "ok" | "corrected" | "total-mismatch" | "no-usage"
    has_cache_fields: bool = False


def _num(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value >= 0 and value == value and value != float("inf"):
        return int(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return _num(parsed)
    return None


def normalize_usage(details) -> Normalized:
    """Turn one generation's usageDetails into uncached / read / write / output counts."""
    if not isinstance(details, dict):
        return Normalized(None, "no-usage")
    values = {k: n for k, v in details.items() if isinstance(k, str) and (n := _num(v)) is not None}
    if not any(v for k, v in values.items() if k != "total"):
        return Normalized(None, "no-usage")

    has_cache = any(k in values for k in CACHE_KEYS)
    read = max((values.get(k, 0) for k in READ_KEYS), default=0)
    split_write = sum(values.get(k, 0) for k in SPLIT_WRITE_KEYS)
    anthropic_write = max(values.get(ANTHROPIC_WRITE_KEY, 0), sum(values.get(k, 0) for k in ANTHROPIC_TTL_KEYS))
    # One integration sends one family; the largest guards against the same writes under two names.
    write = max(split_write, anthropic_write, values.get("input_cache_write", 0),
                *(values.get(k, 0) for k in PASSTHROUGH_WRITE_KEYS))
    write_1h = min(max((values.get(k, 0) for k in WRITE_1H_KEYS), default=0), write)

    output = sum(v for k, v in values.items() if k == "output" or k.startswith("output_"))
    # Other input categories (e.g. input_audio) are still uncached input. Priority-tier and
    # modality keys repeat tokens already counted elsewhere.
    extra_input = sum(v for k, v in values.items()
                      if k.startswith("input_") and k not in CACHE_KEYS
                      and "priority" not in k and "modality" not in k)
    uncached = values.get("input", 0) + extra_input

    total = values.get("total")
    # Langfuse fills in total = sum of every value when none was sent; such a total proves nothing.
    sent_total = total is not None and total != sum(v for k, v in values.items() if k != "total")
    status = "ok"
    if sent_total:
        derived = total - output - read - write
        if derived < 0:
            status = "total-mismatch"
        elif derived != uncached:
            uncached, status = derived, "corrected"
    return Normalized(Usage(uncached, read, write, output, write_1h), status, has_cache)


@dataclass
class Quality:
    generations: int = 0
    skipped_other_types: int = 0
    no_usage: int = 0
    no_cache_fields: int = 0
    corrected: int = 0
    total_mismatch: int = 0


# LiteLLM's Bedrock routing prefixes (litellm.llms.bedrock.common_utils.strip_bedrock_routing_prefix) and its
# optional Region path ("bedrock/us-west-2/<model>"). LiteLLM's `langfuse` callback logs e.g.
# "converse/us.anthropic.claude-sonnet-4-6"; left in place, the prefix splits one model into two groups and hides
# a "global." endpoint from pricing.
_ROUTE_PREFIXES = ("bedrock/", "chat_completions/", "converse/", "invoke/", "openai/", "mantle/")
_REGION_PATH = re.compile(r"^[a-z]{2}(?:-gov)?-[a-z]+-\d+/")


def model_id(raw) -> str | None:
    """The Bedrock model ID behind a Langfuse model name: LiteLLM prefixes and ARN forms removed."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    name = raw.strip()
    changed = True
    while changed:
        changed = False
        for prefix in _ROUTE_PREFIXES:
            if name.startswith(prefix) and len(name) > len(prefix):
                name, changed = name[len(prefix):], True
    name = _REGION_PATH.sub("", name, count=1) or name
    return canonical_model_id(name)


def group_key(obs: dict, by: str) -> str:
    if by == "name":
        return obs.get("name") or "(no name)"
    if by == "environment":
        return obs.get("environment") or "default"
    return model_id(obs.get("model")) or "(no model)"


def aggregate(observations, by: str = "model") -> tuple[dict[str, Group], Quality]:
    groups: dict[str, Group] = defaultdict(Group)
    quality = Quality()
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        if (obs.get("type") or "GENERATION").upper() != "GENERATION":
            # Agent and chain spans can repeat the summed usage of the calls under them.
            quality.skipped_other_types += 1
            continue
        quality.generations += 1
        g = groups[group_key(obs, by)]
        g.calls += 1
        result = normalize_usage(obs.get("usageDetails"))
        if result.usage is None:
            quality.no_usage += 1
            g.unparsed += 1
            continue
        quality.corrected += result.status == "corrected"
        quality.total_mismatch += result.status == "total-mismatch"
        quality.no_cache_fields += not result.has_cache_fields
        model = model_id(obs.get("model")) or "(no model)"
        g.usage = g.usage + result.usage
        g.by_model[model] = g.by_model.get(model, Usage()) + result.usage
    return dict(groups), quality


# ---------------------------------------------------------------- reading saved API responses

def iter_file(path: Path):
    """Observations from a saved API response ({"data": [...]}), a list, or JSON Lines of either."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise LangfuseError(f"file not found: {path}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise LangfuseError(f"could not read {path}: {exc}") from exc
    if not text:
        return
    try:
        docs = [json.loads(text)]
    except json.JSONDecodeError:
        docs = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise LangfuseError(f"{path} line {n} is not valid JSON: {exc}") from exc
    for doc in docs:
        if isinstance(doc, dict) and isinstance(doc.get("data"), list):
            yield from doc["data"]
        elif isinstance(doc, list):
            yield from doc
        elif isinstance(doc, dict):
            yield doc
        else:
            raise LangfuseError(f"{path} does not hold Langfuse observations")


# ---------------------------------------------------------------- the API

@dataclass
class FetchResult:
    observations: list = field(default_factory=list)
    api: str = ""          # "v2" | "v1"
    capped: bool = False


class Client:
    def __init__(self, host: str, public_key: str, secret_key: str, *, timeout: float = 60,
                 sleep=None, opener=None, progress=None):
        host = (host or DEFAULT_HOST).strip().rstrip("/")
        if not re.match(r"^https?://[^/\s]+", host):
            raise LangfuseError(f"Langfuse host must start with https:// (got {host!r})")
        if not public_key or not secret_key:
            raise LangfuseError("set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (a project's API keys) "
                                "to read from Langfuse, or pass --file with a saved API response")
        self.host = host
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self.headers = {"Authorization": f"Basic {token}", "Accept": "application/json",
                        "User-Agent": f"cachecanary/{__version__}"}
        self.timeout, self.progress = timeout, progress
        self.sleep = sleep or time.sleep
        self.opener = opener or urllib.request.urlopen

    def get(self, path: str, params: dict) -> tuple[int, dict | None]:
        """(status, JSON body). 404 is returned to the caller; other failures raise."""
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
        url = f"{self.host}{path}?{query}"
        rate_waits = server_retries = 0
        while True:
            try:
                with self.opener(urllib.request.Request(url, headers=self.headers), timeout=self.timeout) as resp:
                    raw = resp.read()
                break
            except urllib.error.HTTPError as exc:
                status = exc.code
                if status == 404:
                    return 404, _json_or_none(exc.read())
                if status in (401, 403):
                    raise LangfuseError(
                        f"Langfuse rejected the API keys (HTTP {status}) at {self.host}. Check LANGFUSE_PUBLIC_KEY / "
                        "LANGFUSE_SECRET_KEY and that the host matches the project's region "
                        "(EU https://cloud.langfuse.com, US https://us.cloud.langfuse.com).") from None
                if status == 429 and rate_waits < 6:
                    rate_waits += 1
                    self.sleep(_retry_after(exc.headers))
                    continue
                if status >= 500 and server_retries < 3:
                    server_retries += 1
                    self.sleep(2 ** server_retries)
                    continue
                raise LangfuseError(f"Langfuse returned HTTP {status} for {path}: {_message(exc.read())}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if server_retries < 3:
                    server_retries += 1
                    self.sleep(2 ** server_retries)
                    continue
                reason = getattr(exc, "reason", exc)
                raise LangfuseError(f"could not reach Langfuse at {self.host}: {reason}") from None
        body = _json_or_none(raw)
        if not isinstance(body, dict):
            raise LangfuseError(f"Langfuse sent a response that is not JSON for {path}; is {self.host} a Langfuse host?")
        return 200, body

    def fetch_generations(self, from_time: str, to_time: str | None, environments: list[str] | None,
                          max_calls: int) -> FetchResult:
        result = FetchResult()
        common = {"type": "GENERATION", "fromStartTime": from_time, "toStartTime": to_time,
                  "environment": environments or None}
        status, body = self.get(V2_PATH, {**common, "fields": V2_FIELDS, "limit": V2_PAGE_SIZE})
        if status == 200:
            result.api = "v2"
            seen_cursors = set()
            while True:
                result.capped = self._take(result, body, max_calls)
                cursor = (body.get("meta") or {}).get("cursor")
                if result.capped or not cursor or not body.get("data"):
                    return result
                if cursor in seen_cursors:
                    raise LangfuseError("Langfuse returned the same page cursor twice; stopping to avoid a loop")
                seen_cursors.add(cursor)
                status, body = self.get(V2_PATH, {**common, "fields": V2_FIELDS, "limit": V2_PAGE_SIZE,
                                                  "cursor": cursor})
                if status != 200:
                    raise LangfuseError(f"Langfuse returned HTTP {status} part-way through the results")
        # Langfuse v3, or v4 before its write mode is switched on, has only the v1 list.
        v2_message = _message(body)
        result.api, page = "v1", 1
        while True:
            status, body = self.get(V1_PATH, {**common, "page": page, "limit": V1_PAGE_SIZE})
            if status != 200:
                raise LangfuseError(
                    f"this Langfuse host serves neither observations API (v2: {v2_message}; v1: {_message(body)})")
            result.capped = self._take(result, body, max_calls)
            total_pages = _num((body.get("meta") or {}).get("totalPages")) or 0
            if result.capped or not body.get("data") or page >= total_pages:
                return result
            page += 1

    def _take(self, result: FetchResult, body: dict, max_calls: int) -> bool:
        data = body.get("data")
        if not isinstance(data, list):
            raise LangfuseError("Langfuse response has no 'data' list; unexpected API response")
        room = max_calls - len(result.observations)
        result.observations.extend(data[:room])
        if self.progress:
            self.progress(len(result.observations))
        return len(data) > room or (len(result.observations) >= max_calls and _more(body))


def _more(body: dict) -> bool:
    meta = body.get("meta") or {}
    return bool(meta.get("cursor")) or (_num(meta.get("totalPages")) or 0) > (_num(meta.get("page")) or 0)


def _retry_after(headers) -> float:
    value = headers.get("Retry-After") if headers else None
    seconds = _num(value)
    return float(min(max(seconds if seconds is not None else 5, 1), 60))


def _json_or_none(raw):
    try:
        return json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
    except (ValueError, UnicodeDecodeError, AttributeError, TypeError):
        return None


def _message(body) -> str:
    if isinstance(body, (bytes, bytearray)):
        body = _json_or_none(body) or body.decode("utf-8", "replace")
    if isinstance(body, dict):
        body = body.get("message") or body.get("error") or json.dumps(body)
    text = str(body or "no details").strip().replace("\n", " ")
    return text[:200]


# ---------------------------------------------------------------- time window

_SINCE = re.compile(r"^(\d+)\s*([mhdw])$")


def parse_since(value: str) -> timedelta:
    match = _SINCE.match(value.strip().lower())
    if not match or int(match.group(1)) == 0:
        raise ValueError("use a number and a unit, e.g. 30m, 24h, 7d or 2w")
    amount, unit = int(match.group(1)), match.group(2)
    return {"m": timedelta(minutes=amount), "h": timedelta(hours=amount),
            "d": timedelta(days=amount), "w": timedelta(weeks=amount)}[unit]


def parse_time(value: str) -> datetime:
    """A date (2026-10-01, midnight UTC) or an ISO 8601 time; times without an offset are UTC."""
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError("use a date like 2026-10-01 or a time like 2026-10-01T09:00:00Z") from None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def stderr_progress(count: int) -> None:
    if sys.stderr.isatty():
        print(f"\rRead {count:,} generations from Langfuse...", end="", file=sys.stderr, flush=True)
