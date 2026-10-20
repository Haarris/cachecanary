"""`cachecanary langfuse`: usageDetails shapes as each integration stores them, the API client, and the CLI.

Every usage shape below is derived from the source of the integration that produces it (Oct 2026):
Langfuse's OTel ingestion (OtelIngestionProcessor.extractGenericGenAiUsageDetails and the AI SDK branch),
langfuse-python's LangChain CallbackHandler, LiteLLM's `langfuse` callback, Strands' tracer before and
after 1.54.0, and Langfuse filling in `total` when none is sent (IngestionService.getUsageUnits).
The token numbers are the live Bedrock capture from strands-agents/sdk-python#3546:
inputTokens 10, outputTokens 4, cache 5848, totalTokens 5862.
"""

import base64
import io
import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from cachecanary import cli, langfuse
from cachecanary.langfuse import normalize_usage

SONNET = "us.anthropic.claude-sonnet-4-6"


def with_langfuse_total(details):
    """Langfuse adds total = sum of every value when the integration sent none."""
    return {**details, "total": sum(details.values())}


# ---------------------------------------------------------------- one usageDetails shape per integration

def test_strands_current_otel_shape_is_taken_as_is():
    # Strands >= 1.54 sends input_tokens = 10 + 5848 (whole prompt); Langfuse subtracts the cache counts.
    n = normalize_usage({"input": 10, "output": 4, "total": 5862, "input_cached_tokens": 5848, "input_cache_creation": 0})
    assert n.status == "ok" and n.has_cache_fields
    assert (n.usage.uncached_input, n.usage.cache_read, n.usage.cache_write, n.usage.output) == (10, 5848, 0, 4)


def test_strands_before_1_54_hit_is_rebuilt_from_total():
    # Old Strands sent input_tokens = 10 (Bedrock's uncached count); Langfuse took 5848 off again -> 0.
    n = normalize_usage({"input": 0, "output": 4, "total": 5862, "input_cached_tokens": 5848, "input_cache_creation": 0})
    assert n.status == "corrected"
    assert (n.usage.uncached_input, n.usage.cache_read) == (10, 5848)


def test_strands_before_1_54_write_is_rebuilt_from_total():
    n = normalize_usage({"input": 0, "output": 4, "total": 5862, "input_cached_tokens": 0, "input_cache_creation": 5848})
    assert n.status == "corrected"
    assert (n.usage.uncached_input, n.usage.cache_write) == (10, 5848)


def test_strands_before_1_54_would_otherwise_look_like_a_perfect_cache():
    n = normalize_usage({"input": 0, "output": 4, "total": 5862, "input_cached_tokens": 5848, "input_cache_creation": 0})
    assert n.usage.hit_ratio == pytest.approx(5848 / 5858)  # not 100%


def test_litellm_langfuse_callback_write_counted_twice_is_fixed():
    # prompt_tokens = 10 + 0 + 5848; the callback sends input = prompt - cache read, so the write stays inside.
    n = normalize_usage({"input": 5858, "output": 4, "total": 5862,
                         "cache_creation_input_tokens": 5848, "cache_read_input_tokens": 0})
    assert n.status == "corrected"
    assert (n.usage.uncached_input, n.usage.cache_write, n.usage.total_input) == (10, 5848, 5858)


def test_litellm_langfuse_callback_hit_already_adds_up():
    n = normalize_usage({"input": 10, "output": 4, "total": 5862,
                         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 5848})
    assert n.status == "ok" and n.usage.uncached_input == 10


def test_langchain_bedrock_converse_shape():
    # usage_metadata input_tokens includes the cache; the handler subtracts input_token_details.
    n = normalize_usage({"input": 10, "output": 4, "total": 5862, "input_cache_read": 5848, "input_cache_creation": 0})
    assert n.status == "ok"
    assert (n.usage.uncached_input, n.usage.cache_read) == (10, 5848)


def test_langchain_anthropic_ttl_parts_are_not_added_to_the_aggregate():
    details = with_langfuse_total({"input": 10, "output": 4, "cache_read_input_tokens": 5000,
                                   "cache_creation_input_tokens": 848,
                                   "cache_creation_ephemeral_5m_input_tokens": 0,
                                   "cache_creation_ephemeral_1h_input_tokens": 848})
    n = normalize_usage(details)
    assert n.status == "ok"
    assert (n.usage.uncached_input, n.usage.cache_write, n.usage.cache_write_1h) == (10, 848, 848)


def test_langchain_anthropic_ttl_parts_without_aggregate():
    n = normalize_usage(with_langfuse_total({"input": 10, "output": 4,
                                             "cache_creation_ephemeral_5m_input_tokens": 300,
                                             "cache_creation_ephemeral_1h_input_tokens": 200}))
    assert (n.usage.cache_write, n.usage.cache_write_1h) == (500, 200)


def test_ai_sdk_anthropic_ttl_split_adds_up():
    # Langfuse leaves input_cache_creation = aggregate - 5m - 1h.
    n = normalize_usage({"input": 7, "output": 50, "total": 18545, "input_cached_tokens": 16399,
                         "input_cache_creation": 0, "input_cache_creation_5m": 2089})
    assert n.status == "ok"
    assert (n.usage.uncached_input, n.usage.cache_read, n.usage.cache_write, n.usage.cache_write_1h) == (7, 16399, 2089, 0)


def test_ai_sdk_bedrock_metadata_names():
    n = normalize_usage({"input": 10, "output": 4, "total": 4496, "input_cache_read": 4482,
                         "input_cache_write": 0, "input_cache_creation": 0})
    assert n.status == "ok" and n.usage.cache_read == 4482


def test_ai_sdk_bedrock_write_under_two_names_counts_once():
    n = normalize_usage(with_langfuse_total({"input": 10, "output": 4, "input_cache_write": 700,
                                             "input_cache_creation": 700}))
    assert n.usage.cache_write == 700


def test_hand_written_sdk_call_without_total():
    # e.g. usage_details={"input": ..., "output": ..., "cache_read_input_tokens": ...}; Langfuse sums a total.
    n = normalize_usage(with_langfuse_total({"input": 10, "output": 4, "cache_read_input_tokens": 5848}))
    assert n.status == "ok"
    assert (n.usage.uncached_input, n.usage.cache_read) == (10, 5848)


def test_openai_style_subset_shape():
    # prompt_tokens 1000 incl. 900 cached; Langfuse stores input 100 + input_cached_tokens 900.
    n = normalize_usage({"input": 100, "input_cached_tokens": 900, "output": 10, "total": 1010})
    assert n.status == "ok" and n.usage.hit_ratio == pytest.approx(0.9)


def test_reasoning_output_detail_is_part_of_output():
    # OTel keeps total raw and splits reasoning out of output.
    n = normalize_usage({"input": 0, "output": 4, "output_reasoning_tokens": 20, "total": 5882,
                         "input_cached_tokens": 5848, "input_cache_creation": 0})
    assert n.status == "corrected"
    assert (n.usage.uncached_input, n.usage.output) == (10, 24)


def test_other_input_categories_count_as_uncached():
    n = normalize_usage(with_langfuse_total({"input": 90, "input_audio": 10, "output": 4}))
    assert n.usage.uncached_input == 100


def test_priority_and_modality_keys_are_not_double_counted():
    n = normalize_usage(with_langfuse_total({"input": 100, "input_priority": 100, "input_modality_text": 100,
                                             "output": 4}))
    assert n.usage.uncached_input == 100


def test_two_read_names_with_same_value_count_once():
    n = normalize_usage(with_langfuse_total({"input": 10, "output": 4, "input_cached_tokens": 500,
                                             "cache_read_input_tokens": 500}))
    assert n.usage.cache_read == 500


def test_total_smaller_than_the_parts_keeps_sent_counts():
    n = normalize_usage({"input": 10, "output": 4, "total": 100, "input_cached_tokens": 5848})
    assert n.status == "total-mismatch"
    assert (n.usage.uncached_input, n.usage.cache_read) == (10, 5848)


def test_no_cache_fields_is_flagged():
    n = normalize_usage({"input": 100, "output": 5, "total": 105})
    assert n.status == "ok" and not n.has_cache_fields


def test_zero_cache_fields_still_count_as_reported():
    n = normalize_usage({"input": 100, "output": 5, "total": 105, "input_cached_tokens": 0, "input_cache_creation": 0})
    assert n.has_cache_fields


@pytest.mark.parametrize("details", [None, {}, [], "x", {"total": 0}, {"input": 0, "output": 0, "total": 0},
                                     {"total": 50}, {"input": -5}, {"input": True}, {"input": "abc"},
                                     {"input": float("nan")}, {"input": float("inf")}, {"input": None}])
def test_missing_or_unusable_usage(details):
    assert normalize_usage(details).status == "no-usage"


def test_numeric_strings_and_floats_are_read():
    n = normalize_usage({"input": "10", "output": 4.0, "input_cached_tokens": "5848", "total": "5862"})
    assert (n.usage.uncached_input, n.usage.cache_read, n.usage.output) == (10, 5848, 4)


def test_huge_counts():
    big = 10 ** 12
    n = normalize_usage({"input": big, "output": 1, "input_cached_tokens": big * 9, "total": big * 10 + 1})
    assert n.status == "ok" and n.usage.hit_ratio == pytest.approx(0.9)


def test_non_string_keys_are_ignored():
    n = normalize_usage({1: 500, "input": 10, "output": 4})
    assert n.usage.uncached_input == 10


# ---------------------------------------------------------------- aggregation

def gen(model=SONNET, name="agent", env="production", type_="GENERATION", **usage):
    return {"id": "x", "type": type_, "model": model, "name": name, "environment": env,
            "usageDetails": usage or {"input": 10, "output": 4, "input_cached_tokens": 5848, "total": 5862}}


def test_only_generations_are_counted():
    obs = [gen(), gen(type_="SPAN"), gen(type_="AGENT"), gen(type_="generation")]
    groups, q = langfuse.aggregate(obs)
    assert groups[SONNET].calls == 2 and q.skipped_other_types == 2 and q.generations == 2


def test_missing_type_counts_as_generation_and_non_dicts_are_skipped():
    obs = [{"model": SONNET, "usageDetails": {"input": 5, "output": 1}}, "junk", None, 3]
    groups, q = langfuse.aggregate(obs)
    assert groups[SONNET].calls == 1 and q.generations == 1


def test_grouping_and_quality_counts():
    obs = [gen(), gen(input=0, output=4, total=5862, input_cached_tokens=5848),
           gen(input=100, output=5), {"type": "GENERATION", "model": SONNET, "usageDetails": {}},
           gen(model="arn:aws:bedrock:us-east-1:1:inference-profile/us.anthropic.claude-sonnet-4-6")]
    groups, q = langfuse.aggregate(obs)
    g = groups[SONNET]
    assert g.calls == 5 and g.unparsed == 1
    assert (q.corrected, q.no_cache_fields, q.no_usage) == (1, 1, 1)
    assert set(g.by_model) == {SONNET}


def test_group_by_name_and_environment():
    obs = [gen(name="a", env="production"), gen(name="b", env="staging"), gen(name=None, env=None)]
    by_name, _ = langfuse.aggregate(obs, by="name")
    by_env, _ = langfuse.aggregate(obs, by="environment")
    assert set(by_name) == {"a", "b", "(no name)"}
    assert set(by_env) == {"production", "staging", "default"}


def test_missing_model_is_its_own_group():
    groups, _ = langfuse.aggregate([gen(model=None)])
    assert "(no model)" in groups


# ---------------------------------------------------------------- saved API responses

def test_file_shapes(tmp_path):
    one = gen()
    (tmp_path / "resp.json").write_text(json.dumps({"data": [one, one], "meta": {}}))
    (tmp_path / "list.json").write_text(json.dumps([one]))
    (tmp_path / "lines.jsonl").write_text(json.dumps({"data": [one]}) + "\n\n" + json.dumps(one) + "\n")
    (tmp_path / "empty.json").write_text("  ")
    assert len(list(langfuse.iter_file(tmp_path / "resp.json"))) == 2
    assert len(list(langfuse.iter_file(tmp_path / "list.json"))) == 1
    assert len(list(langfuse.iter_file(tmp_path / "lines.jsonl"))) == 2
    assert list(langfuse.iter_file(tmp_path / "empty.json")) == []


@pytest.mark.parametrize("content,match", [("{bad\n{worse", "line 1 is not valid JSON"), ("42", "does not hold")])
def test_file_errors(tmp_path, content, match):
    p = tmp_path / "x.json"
    p.write_text(content)
    with pytest.raises(langfuse.LangfuseError, match=match):
        list(langfuse.iter_file(p))


def test_file_missing(tmp_path):
    with pytest.raises(langfuse.LangfuseError, match="file not found"):
        list(langfuse.iter_file(tmp_path / "nope.json"))


# ---------------------------------------------------------------- API client with a fake transport

class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(url, code, body=b"{}", headers=None):
    return urllib.error.HTTPError(url, code, "err", headers or {}, io.BytesIO(body))


class Transport:
    """Plays back a list of (status, body[, headers]) per call and records each request."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, request, timeout):
        self.requests.append(request)
        status, body, *headers = self.replies.pop(0)
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        if status != 200:
            raise http_error(request.full_url, status, raw, headers[0] if headers else None)
        return FakeResponse(raw)

    def params(self, i):
        url = urlparse(self.requests[i].full_url)
        return url.path, parse_qs(url.query)


def client(transport, sleeps=None):
    return langfuse.Client("https://us.cloud.langfuse.com/", "pk-lf-1", "sk-lf-2", opener=transport,
                           sleep=(sleeps.append if sleeps is not None else lambda s: None))


def page(n, cursor=None):
    return {"data": [gen() for _ in range(n)], "meta": {"cursor": cursor} if cursor else {}}


def test_v2_request_shape_and_auth():
    t = Transport([(200, page(1))])
    res = client(t).fetch_generations("2026-10-13T00:00:00.000Z", None, ["production", "staging"], 100)
    path, q = t.params(0)
    assert path == "/api/public/v2/observations" and res.api == "v2"
    assert q["type"] == ["GENERATION"] and q["fields"] == ["core,basic,model,usage"] and q["limit"] == ["1000"]
    assert q["fromStartTime"] == ["2026-10-13T00:00:00.000Z"] and "toStartTime" not in q
    assert q["environment"] == ["production", "staging"]
    auth = t.requests[0].get_header("Authorization")
    assert base64.b64decode(auth.split()[1]).decode() == "pk-lf-1:sk-lf-2"
    assert "io" not in q["fields"][0] and "metadata" not in q["fields"][0]  # never asks for prompts or outputs


def test_v2_follows_cursors_to_the_end():
    t = Transport([(200, page(2, "c1")), (200, page(2, "c2")), (200, page(1))])
    res = client(t).fetch_generations("t", None, None, 100)
    assert len(res.observations) == 5 and not res.capped
    assert t.params(1)[1]["cursor"] == ["c1"] and t.params(2)[1]["cursor"] == ["c2"]


def test_v2_repeated_cursor_stops():
    t = Transport([(200, page(1, "same")), (200, page(1, "same"))])
    with pytest.raises(langfuse.LangfuseError, match="same page cursor twice"):
        client(t).fetch_generations("t", None, None, 100)


def test_v2_empty_page_with_cursor_stops():
    t = Transport([(200, {"data": [], "meta": {"cursor": "c"}})])
    assert client(t).fetch_generations("t", None, None, 100).observations == []


def test_falls_back_to_v1_when_v2_is_missing():
    v2_404 = (404, {"message": "The observations v2 API is only available in a Langfuse v4 write mode."})
    t = Transport([v2_404, (200, {"data": [gen()] * 100, "meta": {"page": 1, "totalPages": 2}}),
                   (200, {"data": [gen()] * 30, "meta": {"page": 2, "totalPages": 2}})])
    res = client(t).fetch_generations("t", "u", None, 1000)
    assert res.api == "v1" and len(res.observations) == 130
    path, q = t.params(1)
    assert path == "/api/public/observations" and q["page"] == ["1"] and q["limit"] == ["100"]
    assert q["toStartTime"] == ["u"] and "fields" not in q
    assert t.params(2)[1]["page"] == ["2"]


def test_neither_api_available():
    t = Transport([(404, {"message": "v2 off"}), (404, b"<html>not found</html>")])
    with pytest.raises(langfuse.LangfuseError, match="neither observations API.*v2 off"):
        client(t).fetch_generations("t", None, None, 10)


def test_max_calls_caps_mid_page():
    t = Transport([(200, page(5, "c1"))])
    res = client(t).fetch_generations("t", None, None, 3)
    assert len(res.observations) == 3 and res.capped and len(t.requests) == 1


def test_max_calls_exactly_at_last_page_is_not_capped():
    t = Transport([(200, page(3))])
    res = client(t).fetch_generations("t", None, None, 3)
    assert len(res.observations) == 3 and not res.capped


def test_max_calls_at_page_boundary_with_more_pages_is_capped():
    t = Transport([(200, page(3, "c1"))])
    res = client(t).fetch_generations("t", None, None, 3)
    assert res.capped and len(t.requests) == 1


def test_rate_limit_waits_for_retry_after():
    sleeps = []
    t = Transport([(429, {}, {"Retry-After": "7"}), (429, {}, {}), (200, page(1))])
    res = client(t, sleeps).fetch_generations("t", None, None, 10)
    assert len(res.observations) == 1 and sleeps == [7.0, 5.0]


def test_rate_limit_wait_is_capped_and_gives_up():
    sleeps = []
    t = Transport([(429, {}, {"Retry-After": "9999"})] * 7)
    with pytest.raises(langfuse.LangfuseError, match="HTTP 429"):
        client(t, sleeps).fetch_generations("t", None, None, 10)
    assert sleeps == [60.0] * 6


def test_server_errors_retry_then_succeed():
    sleeps = []
    t = Transport([(502, b"bad gateway"), (500, {}), (200, page(1))])
    assert len(client(t, sleeps).fetch_generations("t", None, None, 10).observations) == 1
    assert sleeps == [2, 4]


def test_server_errors_give_up_with_message():
    t = Transport([(500, {"message": "boom"})] * 4)
    with pytest.raises(langfuse.LangfuseError, match="HTTP 500.*boom"):
        client(t).fetch_generations("t", None, None, 10)


@pytest.mark.parametrize("code", [401, 403])
def test_bad_keys(code):
    t = Transport([(code, {"message": "Invalid credentials"})])
    with pytest.raises(langfuse.LangfuseError, match=f"rejected the API keys \\(HTTP {code}\\).*us.cloud.langfuse.com"):
        client(t).fetch_generations("t", None, None, 10)


def test_other_client_error_is_reported():
    t = Transport([(400, {"message": "Invalid request data"})])
    with pytest.raises(langfuse.LangfuseError, match="HTTP 400.*Invalid request data"):
        client(t).fetch_generations("t", None, None, 10)


def test_network_errors_retry_then_report():
    calls = []

    def down(request, timeout):
        calls.append(1)
        raise urllib.error.URLError("nodename nor servname provided")

    with pytest.raises(langfuse.LangfuseError, match="could not reach Langfuse.*nodename"):
        langfuse.Client("https://lf.example.com", "a", "b", opener=down, sleep=lambda s: None).get("/x", {})
    assert len(calls) == 4


def test_non_json_success_body():
    t = Transport([(200, b"<html>login</html>")])
    with pytest.raises(langfuse.LangfuseError, match="not JSON"):
        client(t).fetch_generations("t", None, None, 10)


def test_response_without_data_list():
    t = Transport([(200, {"items": []})])
    with pytest.raises(langfuse.LangfuseError, match="no 'data' list"):
        client(t).fetch_generations("t", None, None, 10)


def test_failure_part_way_through_pages():
    t = Transport([(200, page(1, "c1")), (404, {})])
    with pytest.raises(langfuse.LangfuseError, match="part-way"):
        client(t).fetch_generations("t", None, None, 10)


@pytest.mark.parametrize("host", ["cloud.langfuse.com", "ftp://x", "", "https://"])
def test_host_validation(host):
    if host == "":
        assert langfuse.Client(host, "a", "b").host == langfuse.DEFAULT_HOST
        return
    with pytest.raises(langfuse.LangfuseError, match="must start with https://"):
        langfuse.Client(host, "a", "b")


@pytest.mark.parametrize("pk,sk", [("", "b"), ("a", ""), ("", "")])
def test_missing_keys(pk, sk):
    with pytest.raises(langfuse.LangfuseError, match="LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"):
        langfuse.Client("https://x.io", pk, sk)


# ---------------------------------------------------------------- time window parsing

@pytest.mark.parametrize("text,hours", [("30m", 0.5), ("24h", 24), ("7d", 168), ("2w", 336), (" 3D ", 72)])
def test_parse_since(text, hours):
    assert langfuse.parse_since(text).total_seconds() == hours * 3600


@pytest.mark.parametrize("text", ["0d", "7", "d", "-1d", "1y", "1.5d", ""])
def test_parse_since_rejects(text):
    with pytest.raises(ValueError):
        langfuse.parse_since(text)


def test_parse_time_and_iso():
    assert langfuse.iso(langfuse.parse_time("2026-10-01")) == "2026-10-01T00:00:00.000Z"
    assert langfuse.iso(langfuse.parse_time("2026-10-01T09:30:00Z")) == "2026-10-01T09:30:00.000Z"
    assert langfuse.iso(langfuse.parse_time("2026-10-01T14:30:00+05:00")) == "2026-10-01T09:30:00.000Z"
    with pytest.raises(ValueError):
        langfuse.parse_time("yesterday")


# ---------------------------------------------------------------- CLI

@pytest.fixture
def saved(tmp_path):
    obs = ([gen(input=10, output=4, input_cached_tokens=5848, total=5862)] * 8
           + [gen(input=0, output=4, input_cache_creation=5848, input_cached_tokens=0, total=5862)] * 2
           + [gen(model="gpt-5", input=100, output=5)] + [gen(type_="AGENT")]
           + [{"type": "GENERATION", "model": SONNET, "usageDetails": {}}])
    p = tmp_path / "obs.json"
    p.write_text(json.dumps({"data": obs, "meta": {}}))
    return p


def test_cli_file_report(saved, capsys):
    assert cli.main(["langfuse", "--file", str(saved)]) == 0
    out = capsys.readouterr().out
    assert f"{SONNET}: hit 80% over 11 calls (read 46784, write 11696, uncached 100, no usage 1)" in out
    assert "input cost" in out and "lost to cache misses" in out
    assert "gpt-5: hit 0% over 1 calls" in out and "no price for this model" in out
    assert "Read 12 generations." in out
    assert "2 generation(s): Langfuse's input count did not add up" in out
    assert "1 generation(s) have no cache fields at all" in out
    assert "1 generation(s) have no token counts." in out
    assert "Ignored 1 non-generation observation(s)" in out


def test_cli_min_hit_exit_code(saved, capsys):
    assert cli.main(["langfuse", "--file", str(saved), "--min-hit", "0.85"]) == 1
    assert "below threshold" in capsys.readouterr().out


def test_cli_json(saved, capsys):
    assert cli.main(["--json", "langfuse", "--file", str(saved), "--by", "name"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["groups"]["agent"]["calls"] == 11 and data["groups"]["(no name)"]["calls"] == 1 and any("did not add up" in n for n in data["notes"])


def test_cli_github_summary(saved, tmp_path, monkeypatch, capsys):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert cli.main(["--github", "langfuse", "--file", str(saved), "--min-hit", "0.85"]) == 1
    text = summary.read_text()
    assert "### CacheCanary langfuse" in text and "| no usage |" in text and "did not add up" in text
    assert "::error" in capsys.readouterr().out


def test_cli_empty(tmp_path, capsys):
    p = tmp_path / "e.json"
    p.write_text(json.dumps({"data": []}))
    assert cli.main(["langfuse", "--file", str(p)]) == 0
    assert "No generations found" in capsys.readouterr().out


def test_cli_bad_file_exit_2(tmp_path, capsys):
    assert cli.main(["langfuse", "--file", str(tmp_path / "nope.json")]) == 2
    assert "file not found" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--since", "7"], ["--from", "soon"], ["--max-calls", "0"],
                                  ["--since", "1d", "--from", "2026-10-01"], ["--by", "trace"]])
def test_cli_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(["langfuse", *argv])
    assert exc.value.code == 2


def test_cli_from_after_to(monkeypatch, capsys):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "a")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "b")
    assert cli.main(["langfuse", "--from", "2026-10-10", "--to", "2026-10-01"]) == 2
    assert "--from must be earlier" in capsys.readouterr().err


def test_cli_without_keys(monkeypatch, capsys):
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert cli.main(["langfuse"]) == 2
    assert "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY" in capsys.readouterr().err


# ---------------------------------------------------------------- end to end against a local HTTP server

class FakeLangfuse(BaseHTTPRequestHandler):
    v2 = True
    hits: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        FakeLangfuse.hits.append((url.path, q, self.headers.get("Authorization")))
        if self.headers.get("Authorization") != "Basic " + base64.b64encode(b"pk:sk").decode():
            return self._send(401, {"message": "Invalid credentials"})
        if url.path == "/api/public/v2/observations":
            if not FakeLangfuse.v2:
                return self._send(404, {"message": "only available in a Langfuse v4 write mode"})
            if len(FakeLangfuse.hits) == 1:
                return self._send(429, {"message": "rate limited"}, {"Retry-After": "0"})
            if "cursor" not in q:
                return self._send(200, {"data": [gen()] * 3, "meta": {"cursor": "p2"}})
            return self._send(200, {"data": [gen(input=0, output=4, total=5862, input_cached_tokens=5848)], "meta": {}})
        if url.path == "/api/public/observations":
            pg = int(q["page"][0])
            return self._send(200, {"data": [gen()] * (100 if pg == 1 else 1),
                                    "meta": {"page": pg, "limit": 100, "totalItems": 101, "totalPages": 2}})
        self._send(404, {"message": "no route"})

    def _send(self, code, body, headers=None):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def server(monkeypatch):
    FakeLangfuse.hits, FakeLangfuse.v2 = [], True
    httpd = HTTPServer(("127.0.0.1", 0), FakeLangfuse)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(langfuse.time, "sleep", lambda s: None)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.setenv("LANGFUSE_BASE_URL", f"http://127.0.0.1:{httpd.server_port}")
    yield httpd
    httpd.shutdown()


def test_e2e_v2_with_rate_limit_and_cursor(server, capsys):
    assert cli.main(["langfuse", "--since", "24h", "--environment", "production"]) == 0
    out = capsys.readouterr().out
    assert f"{SONNET}: hit" in out and "over 4 calls" in out
    assert "(Langfuse API v2)" in out and "1 generation(s): Langfuse's input count did not add up" in out
    paths = [h[0] for h in FakeLangfuse.hits]
    assert paths == ["/api/public/v2/observations"] * 3
    assert FakeLangfuse.hits[-1][1]["environment"] == ["production"]


def test_e2e_v1_fallback(server, capsys):
    FakeLangfuse.v2 = False
    assert cli.main(["langfuse", "--from", "2026-10-01", "--to", "2026-10-08"]) == 0
    out = capsys.readouterr().out
    assert "over 101 calls" in out and "(Langfuse API v1)" in out
    assert "from 2026-10-01T00:00:00.000Z to 2026-10-08T00:00:00.000Z" in out


def test_e2e_host_flag_beats_env_and_bad_keys(server, monkeypatch, capsys):
    monkeypatch.setenv("LANGFUSE_HOST", "https://wrong.example.com")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "wrong")
    assert cli.main(["langfuse", "--host", f"http://127.0.0.1:{server.server_port}"]) == 2
    assert "rejected the API keys (HTTP 401)" in capsys.readouterr().err


def test_e2e_langfuse_host_beats_base_url(server, monkeypatch, capsys):
    monkeypatch.setenv("LANGFUSE_HOST", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://wrong.example.com")
    assert cli.main(["langfuse"]) == 0


# ---------------------------------------------------------------- records captured from a real Langfuse 4.64.0
# Each pair was produced by the real library calling a local stand-in for Bedrock Converse that returned the
# strands#3546 usage (call 1 writes 5848 tokens, call 2 reads them), traced into a local Langfuse 4.64.0
# (docker compose, Oct 20 2026) and read back through /api/public/v2/observations. The right answer for
# every pair is read 5848, write 5848, uncached 20, hit 50%.

REAL_RECORDS = {
    "strands 1.61.1": [
        {"cache_write_input_tokens": 5848, "input": 10, "input_cache_creation": 5848, "input_cached_tokens": 0, "output": 4, "total": 5862},
        {"cache_write_input_tokens": 0, "input": 10, "input_cache_creation": 0, "input_cached_tokens": 5848, "output": 4, "total": 5862},
    ],
    # Before 1.54 the write is only in the key Langfuse does not recognise, and the read call's input is 0.
    "strands 1.53.0": [
        {"cache_write_input_tokens": 5848, "input": 10, "input_cached_tokens": 0, "output": 4, "total": 5862},
        {"cache_write_input_tokens": 0, "input": 0, "input_cached_tokens": 5848, "output": 4, "total": 5862},
    ],
    # LiteLLM 1.106.0 `langfuse` callback (langfuse SDK 4.18.0): the write is also left inside input.
    "litellm langfuse callback": [
        {"cache_creation_input_tokens": 5848, "cache_read_input_tokens": 0, "input": 5858, "output": 4, "total": 5862},
        {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 5848, "input": 10, "output": 4, "total": 5862},
    ],
    "litellm langfuse_otel": [
        {"input": 10, "input_cache_creation": 5848, "output": 4, "total": 5862},
        {"input": 10, "input_cached_tokens": 5848, "output": 4, "total": 5862},
    ],
    # langchain-aws 1.9.0 ChatBedrockConverse + langfuse 5.0.3 CallbackHandler
    "langchain bedrock converse": [
        {"input": 10, "input_cache_creation": 5848, "input_cache_read": 0, "output": 4, "total": 5862},
        {"input": 10, "input_cache_creation": 0, "input_cache_read": 5848, "output": 4, "total": 5862},
    ],
}


@pytest.mark.parametrize("source", sorted(REAL_RECORDS))
def test_real_records_give_the_true_hit_rate(source):
    groups, _ = langfuse.aggregate([{"type": "GENERATION", "model": SONNET, "usageDetails": u}
                                    for u in REAL_RECORDS[source]])
    u = groups[SONNET].usage
    assert (u.cache_read, u.cache_write, u.uncached_input, u.output) == (5848, 5848, 20, 8)
    assert u.hit_ratio == pytest.approx(0.5, abs=0.001)


def test_real_strands_1_53_without_the_passthrough_key_would_show_a_perfect_cache():
    # The bug the live run caught: ignoring cache_write_input_tokens drops the write and shows 100%.
    write_call = dict(REAL_RECORDS["strands 1.53.0"][0])
    assert normalize_usage(write_call).usage.cache_write == 5848


@pytest.mark.parametrize("raw,expected", [
    ("converse/us.anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-4-6"),  # real LiteLLM callback record
    ("bedrock/converse/global.anthropic.claude-sonnet-5-5", "global.anthropic.claude-sonnet-5-5"),
    ("bedrock/invoke/us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-5-5"),
    ("bedrock/us-west-2/anthropic.claude-haiku-4-5-20251001-v1:0", "anthropic.claude-haiku-4-5-20251001-v1:0"),
    ("bedrock/us-gov-west-1/anthropic.claude-sonnet-4-6", "anthropic.claude-sonnet-4-6"),
    ("invoke/arn:aws:bedrock:us-east-1:111:inference-profile/global.anthropic.claude-opus-5-5", "global.anthropic.claude-opus-5-5"),
    ("arn:aws:bedrock:us-east-2:111:inference-profile/global.anthropic.claude-sonnet-5-5", "global.anthropic.claude-sonnet-5-5"),
    ("arn:aws:bedrock:us-east-1:111:application-inference-profile/abc", "arn:aws:bedrock:us-east-1:111:application-inference-profile/abc"),
    ("bedrock/", "bedrock/"), ("gpt-5", "gpt-5"), ("  us.anthropic.claude-sonnet-4-6 ", "us.anthropic.claude-sonnet-4-6"),
    (None, None), ("", None), ("   ", None), (42, None),
])
def test_model_id(raw, expected):
    assert langfuse.model_id(raw) == expected


def test_litellm_prefix_merges_groups_and_keeps_global_price():
    from cachecanary.pricing import rates_for
    obs = [gen(model="converse/global.anthropic.claude-sonnet-5-5"), gen(model="global.anthropic.claude-sonnet-5-5")]
    groups, _ = langfuse.aggregate(obs)
    assert list(groups) == ["global.anthropic.claude-sonnet-5-5"] and groups["global.anthropic.claude-sonnet-5-5"].calls == 2
    assert rates_for(next(iter(groups["global.anthropic.claude-sonnet-5-5"].by_model))).rates.input == 2  # global, no +10%


def test_real_fixture_end_to_end(capsys):
    assert cli.main(["langfuse", "--file", "tests/fixtures/langfuse_observations_2026-10.json"]) == 0
    out = capsys.readouterr().out
    assert "us.anthropic.claude-sonnet-4-6: hit 50% over 10 calls (read 29240, write 29240, uncached 100, no usage 0)" in out
    assert "2 generation(s): Langfuse's input count did not add up" in out
