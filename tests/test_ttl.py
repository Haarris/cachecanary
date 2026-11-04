"""`logs --ttl-check`: why calls had to write the cache again."""

import json

import pytest

from cachecanary import cli, ttl

MODEL = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"  # $3 input, $0.30 read, $3.75 5m write, $6 1h write


def rec(ts, read, write, *, uncached=10, system="S", tools=None, caller="role/a", region="us-east-1",
        model=MODEL, body=True, write_1h=0, body_as_string=False, s3_body=False):
    usage = {"inputTokens": uncached, "cacheReadInputTokens": read, "cacheWriteInputTokens": write, "outputTokens": 5}
    if write_1h:
        usage["cacheDetails"] = [{"ttl": "1h", "inputTokens": write_1h}]
    r = {"schemaType": "ModelInvocationLog", "timestamp": ts, "modelId": model, "identity": {"arn": caller},
         "inferenceRegion": region, "input": {"inputTokenCount": uncached},
         "output": {"outputBodyJson": {"usage": usage}}}
    if s3_body:
        r["input"]["inputBodyS3Path"] = "s3://bucket/AWSLogs/data/x.json"
    elif body:
        b = {"system": [{"text": system}, {"cachePoint": {"type": "default"}}], "messages": [{"role": "user"}]}
        if tools is not None:
            b["toolConfig"] = {"tools": tools}
        r["input"]["inputBodyJson"] = json.dumps(b) if body_as_string else b
    return r


def run(records):
    calls, skipped = ttl.collect(records)
    return ttl.check(calls), skipped


def write_file(tmp_path, records):
    path = tmp_path / "l.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


# --- what counts as a re-write ------------------------------------------------

def test_only_calls_that_read_nothing_are_rewrites():
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000),
                     rec("2026-11-01T10:01:00Z", 20000, 1500),  # read and wrote: not classified
                     rec("2026-11-01T10:30:00Z", 2000, 20000),  # partial read 29 min later: not classified either
                     rec("2026-11-01T10:31:00Z", 21500, 0)])
    g = groups[MODEL]
    assert g.rewrites == {ttl.FIRST: 1}
    assert g.write_tokens == 20000 + 1500 + 20000


@pytest.mark.parametrize("second,region,expected", [
    ("2026-11-01T10:04:59Z", "us-east-1", ttl.WITHIN_5M),
    ("2026-11-01T10:05:00Z", "us-east-1", ttl.WITHIN_5M),  # exactly 5 minutes is still within
    ("2026-11-01T10:05:01Z", "us-east-1", ttl.FIVE_TO_60M),
    ("2026-11-01T11:00:00Z", "us-east-1", ttl.FIVE_TO_60M),  # exactly 1 hour
    ("2026-11-01T11:00:01Z", "us-east-1", ttl.OVER_1H),
    ("2026-11-01T10:03:00Z", "us-east-2", ttl.WITHIN_5M_OTHER_REGION),
    ("2026-11-01T10:03:00Z", None, ttl.WITHIN_5M_REGION_UNKNOWN),
    ("2026-11-01T10:20:00Z", "us-east-2", ttl.FIVE_TO_60M),  # region only matters inside 5 minutes
])
def test_gap_classes(second, region, expected):
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000), rec(second, 0, 20000, region=region)])
    assert groups[MODEL].rewrites == {ttl.FIRST: 1, expected: 1}


def test_gap_counts_from_any_call_that_touched_the_cache():
    # A read at 10:04 resets the 5 minutes, so a re-write at 10:08 is 4 minutes after it.
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000), rec("2026-11-01T10:04:00Z", 20000, 0),
                     rec("2026-11-01T10:08:00Z", 0, 20000)])
    assert groups[MODEL].rewrites == {ttl.FIRST: 1, ttl.WITHIN_5M: 1}


def test_a_call_that_never_touched_the_cache_does_not_count_as_previous():
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000),
                     rec("2026-11-01T10:20:00Z", 0, 0, uncached=20000),
                     rec("2026-11-01T10:22:00Z", 0, 20000)])
    assert groups[MODEL].rewrites == {ttl.FIRST: 1, ttl.FIVE_TO_60M: 1}


def test_same_second_rewrites_are_noted():
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000), rec("2026-11-01T10:00:00.400Z", 0, 20000)])
    assert groups[MODEL].same_second == 1


# --- matching ------------------------------------------------------------------

def test_prompts_from_one_caller_are_tracked_separately():
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000, system="A"),
                     rec("2026-11-01T10:01:00Z", 0, 20000, system="B"),
                     rec("2026-11-01T10:30:00Z", 20000, 0, system="A"),
                     rec("2026-11-01T10:31:00Z", 0, 20000, system="B")])
    g = groups[MODEL]
    assert g.rewrites == {ttl.FIRST: 2, ttl.FIVE_TO_60M: 1}
    assert g.calls_no_body == g.calls_body_in_s3 == 0


def test_without_request_bodies_matching_is_by_caller_and_says_so():
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000, body=False),
                     rec("2026-11-01T10:10:00Z", 0, 20000, body=False),
                     rec("2026-11-01T10:11:00Z", 0, 20000, body=False, caller="role/b")])
    g = groups[MODEL]
    assert g.calls_no_body == 3
    assert g.rewrites == {f"{ttl.FIRST}/caller": 2, f"{ttl.FIVE_TO_60M}/caller": 1}
    assert "matched by caller and model, not by prompt" in ttl.LABELS[f"{ttl.FIVE_TO_60M}/caller"]
    assert "started the same way" in ttl.LABELS[ttl.FIVE_TO_60M]


def test_body_in_s3_is_counted_separately_and_sees_the_callers_earlier_calls():
    # The conversation grew past 100 KB: its next call has no inline body. It is matched by caller,
    # and the caller's earlier (inline) call at 10:00 is its previous call, so it is not "first".
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000), rec("2026-11-01T10:20:00Z", 0, 30000, s3_body=True)])
    g = groups[MODEL]
    assert g.calls_body_in_s3 == 1 and g.calls_no_body == 0
    assert g.rewrites == {ttl.FIRST: 1, f"{ttl.FIVE_TO_60M}/caller": 1}


def test_a_newer_bodyless_call_does_not_override_the_matched_previous_call():
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000),
                     rec("2026-11-01T10:28:00Z", 0, 5000, body=False),
                     rec("2026-11-01T10:30:00Z", 0, 20000)])
    rows = groups[MODEL].rewrites
    assert rows[ttl.FIVE_TO_60M] == 1  # 30 minutes after the call that started the same way
    assert ttl.WITHIN_5M not in rows


def test_no_system_and_no_tools_is_matched_by_caller_not_merged():
    a = rec("2026-11-01T10:00:00Z", 0, 20000, caller="role/a")
    b = rec("2026-11-01T10:01:00Z", 0, 20000, caller="role/b")
    for r in (a, b):
        r["input"]["inputBodyJson"] = {"messages": [{"role": "user", "content": [{"text": r["identity"]["arn"]}]}]}
    groups, _ = run([a, b])
    assert ttl.prompt_start(a) is None
    assert groups[MODEL].rewrites == {f"{ttl.FIRST}/caller": 2}


def test_invokemodel_arn_and_short_id_are_one_stream():
    arn = "arn:aws:bedrock:us-east-1:111122223333:inference-profile/" + MODEL
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000, model=arn), rec("2026-11-01T10:20:00Z", 0, 20000)])
    assert list(groups) == [MODEL]
    assert groups[MODEL].rewrites == {ttl.FIRST: 1, ttl.FIVE_TO_60M: 1}


# --- the 1-hour cache in the logs ------------------------------------------------

def test_previous_write_all_on_1h_is_not_called_an_expiry():
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000, write_1h=20000),
                     rec("2026-11-01T10:20:00Z", 0, 20000)])
    assert groups[MODEL].rewrites == {ttl.FIRST: 1, ttl.ON_1H_CACHE: 1}


def test_previous_write_partly_on_1h_still_counts_as_possible_expiry():
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 50000, write_1h=10000),
                     rec("2026-11-01T10:20:00Z", 0, 50000)])
    assert groups[MODEL].rewrites == {ttl.FIRST: 1, ttl.FIVE_TO_60M: 1}


# --- fingerprint -----------------------------------------------------------------

def test_fingerprint_ignores_cache_markers_but_not_content():
    base = rec("t", 0, 1)
    moved = json.loads(json.dumps(base))
    moved["input"]["inputBodyJson"]["system"] = [{"text": "S", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
    changed = rec("t", 0, 1, system="S plus today's date")
    as_string = rec("t", 0, 1, body_as_string=True)
    with_tools = rec("t", 0, 1, tools=[{"toolSpec": {"name": "x"}}])
    assert ttl.prompt_start(base) == ttl.prompt_start(moved) == ttl.prompt_start(as_string)
    assert ttl.prompt_start(base) != ttl.prompt_start(changed)
    assert ttl.prompt_start(base) != ttl.prompt_start(with_tools)


def test_schema_property_named_cache_control_is_part_of_the_fingerprint():
    a = rec("t", 0, 1, tools=[{"toolSpec": {"name": "x", "inputSchema": {"json": {"properties": {
        "cache_control": {"type": "string"}}}}}}])
    b = rec("t", 0, 1, tools=[{"toolSpec": {"name": "x", "inputSchema": {"json": {"properties": {}}}}}])
    assert ttl.prompt_start(a) != ttl.prompt_start(b)


def test_claude_messages_body_is_fingerprinted():
    r = rec("t", 0, 1)
    r["input"]["inputBodyJson"] = {"anthropic_version": "bedrock-2023-05-31", "system": "S",
                                   "tools": [{"name": "x"}], "messages": [{"role": "user"}]}
    assert ttl.prompt_start(r) is not None


@pytest.mark.parametrize("body", [None, "not json", {"no": "messages"}, ["list"]])
def test_unusable_bodies_give_no_fingerprint(body):
    r = rec("t", 0, 1, body=False)
    r["input"]["inputBodyJson"] = body
    assert ttl.prompt_start(r) is None


# --- records --------------------------------------------------------------------

def test_records_without_usage_or_timestamp_are_counted_not_guessed():
    no_time = rec("", 0, 1)
    bad_time = rec("yesterday", 0, 1)
    naive_time = rec("2026-11-01T10:00:00", 0, 1)  # no timezone: can't be ordered against UTC records
    no_usage = rec("2026-11-01T10:00:00Z", 0, 1)
    no_usage["output"] = {"outputBodyS3Path": "s3://b/k"}
    calls, skipped = ttl.collect([no_time, bad_time, naive_time, no_usage, {"some": "other file"},
                                  rec("2026-11-01T10:00:00.123+00:00", 0, 1)])
    assert len(calls) == 1 and skipped == 4


def test_odd_record_shapes_do_not_crash():
    weird = [rec("2026-11-01T10:00:00Z", 0, 20000), rec("2026-11-01T10:01:00Z", 0, 20000)]
    weird[0]["input"] = "x"
    weird[1]["identity"] = {"arn": {"not": "a string"}}
    weird[1]["input"].pop("inputBodyJson")
    weird.append({**rec("2026-11-01T10:02:00Z", 0, 5), "identity": "role/c", "inferenceRegion": 7})
    calls, skipped = ttl.collect(weird)
    assert len(calls) == 3 and skipped == 0
    ttl.check(calls)


# --- dollars: prices of what the logs show ----------------------------------------

def test_dollar_figures_by_hand():
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000, uncached=0),
                     rec("2026-11-01T10:02:00Z", 20000, 1000, uncached=0),
                     rec("2026-11-01T10:20:00Z", 0, 30000, uncached=0, write_1h=10000)])
    g = groups[MODEL]
    assert g.write_tokens == 51000 and g.write_tokens_1h == 10000
    assert g.cost_writes_now == pytest.approx((21000 * 3.75 + 20000 * 3.75 + 10000 * 6) / 1e6)
    assert g.cost_writes_at_1h == pytest.approx(51000 * 6 / 1e6)
    assert g.cost_5to60_now == pytest.approx((20000 * 3.75 + 10000 * 6) / 1e6)
    assert g.cost_5to60_as_reads == pytest.approx(30000 * 0.30 / 1e6)


def test_arn_and_global_ids_are_priced_at_the_global_rate():
    arn = "arn:aws:bedrock:us-east-1:111122223333:inference-profile/" + MODEL
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000, model=arn)])
    assert groups[MODEL].cost_writes_now == pytest.approx(20000 * 3.75 / 1e6)


def test_regional_ids_cost_10_percent_more():
    regional = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000, model=regional)])
    assert groups[regional].cost_writes_now == pytest.approx(20000 * 3.75 * 1.1 / 1e6)


def test_unknown_models_are_not_priced():
    app = "arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/abc"
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000, model=app)])
    assert not groups[app].priced


# --- CLI -------------------------------------------------------------------------

def test_cli_without_flag_is_unchanged(tmp_path, capsys):
    path = write_file(tmp_path, [rec("2026-11-01T10:00:00Z", 0, 20000), rec("2026-11-01T10:20:00Z", 0, 20000)])
    assert cli.main(["logs", str(path)]) == 0
    assert "re-write" not in capsys.readouterr().out
    assert cli.main(["--json", "logs", str(path)]) == 0
    assert set(json.loads(capsys.readouterr().out)) == {MODEL}  # same shape as before: groups only


def test_cli_with_flag_text_and_json(tmp_path, capsys):
    path = write_file(tmp_path, [rec("2026-11-01T10:00:00Z", 0, 30000), rec("2026-11-01T10:20:00Z", 0, 30000),
                                 rec("2026-11-01T10:21:00Z", 30000, 300)])
    assert cli.main(["logs", str(path), "--ttl-check"]) == 0
    out = capsys.readouterr().out
    assert "2 re-write(s) in 3 calls" in out
    assert "5 to 60 minutes after the previous call that started the same way" in out
    assert "The 5-to-60-minute re-writes cost $0.11 as cache writes" in out  # 30,000 x $3.75/M
    assert "the same tokens at the 1-hour write price would cost $0.36" in out  # 60,300 x $6/M
    assert "at least" not in out
    assert cli.main(["--json", "logs", str(path), "--ttl-check"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data) == {"groups", "notes", "ttl_check", "ttl_check_skipped"}
    t = data["ttl_check"][MODEL]
    assert t["rewrites"] == {ttl.FIRST: 1, ttl.FIVE_TO_60M: 1}
    assert t["cache_writes_at_1h_rate_usd"] == pytest.approx(60300 * 6 / 1e6)


@pytest.mark.parametrize("model,show", [(MODEL, True),
                                        ("us.anthropic.claude-3-7-sonnet-20250219-v1:0", False),
                                        ("us.anthropic.claude-sonnet-4-20250514-v1:0", False)])
def test_1h_price_line_only_for_models_with_documented_1h_cache(tmp_path, capsys, model, show):
    path = write_file(tmp_path, [rec("2026-11-01T10:00:00Z", 0, 20000, model=model)])
    cli.main(["logs", str(path), "--ttl-check", "--price", "3"])
    out = capsys.readouterr().out
    assert ("1-hour write price" in out) is show
    assert ("not documented for this model" in out) is (not show)


def test_cli_body_wording_tells_s3_from_not_logged(tmp_path, capsys):
    path = write_file(tmp_path, [rec("2026-11-01T10:00:00Z", 0, 20000, body=False),
                                 rec("2026-11-01T10:20:00Z", 0, 20000, s3_body=True)])
    cli.main(["logs", str(path), "--ttl-check"])
    out = capsys.readouterr().out
    assert "1 with no body logged" in out
    assert "1 with the body stored as a separate S3 object (over 100 KB, or binary)" in out


def test_cli_flag_exit_code_still_follows_min_hit(tmp_path):
    path = write_file(tmp_path, [rec("2026-11-01T10:00:00Z", 0, 20000)])
    assert cli.main(["logs", str(path), "--ttl-check", "--min-hit", "0.5"]) == 1
    assert cli.main(["logs", str(path), "--ttl-check"]) == 0


def test_github_summary_unchanged_without_flag(tmp_path, capsys, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    path = write_file(tmp_path, [rec("2026-11-01T10:00:00Z", 0, 20000)])
    assert cli.main(["--github", "logs", str(path)]) == 0
    assert "re-write" not in capsys.readouterr().out
    assert "re-write" not in summary.read_text()


# --- round-3 review cases ----------------------------------------------------------

def test_first_means_no_earlier_cache_touching_call_not_first_call():
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 0, uncached=20000), rec("2026-11-01T10:01:00Z", 0, 20000)])
    assert groups[MODEL].rewrites == {ttl.FIRST: 1}
    assert ttl.LABELS[ttl.FIRST].startswith("no earlier call that started the same way read or wrote the cache")


def test_1h_label_follows_the_latest_write_even_when_the_previous_call_only_read():
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000, write_1h=20000),
                     rec("2026-11-01T10:10:00Z", 20000, 0),
                     rec("2026-11-01T10:20:00Z", 0, 20000)])
    assert groups[MODEL].rewrites == {ttl.FIRST: 1, ttl.ON_1H_CACHE: 1}
    assert "the latest cache write before it was all on the 1-hour cache" in ttl.LABELS[ttl.ON_1H_CACHE]


def test_within_5_minutes_region_is_reported_even_on_the_1h_cache():
    groups, _ = run([rec("2026-11-01T10:00:00Z", 0, 20000, write_1h=20000),
                     rec("2026-11-01T10:03:00Z", 0, 20000, region="us-east-2")])
    assert groups[MODEL].rewrites == {ttl.FIRST: 1, ttl.WITHIN_5M_OTHER_REGION: 1}


def test_unknown_model_says_no_1h_information(tmp_path, capsys):
    path = write_file(tmp_path, [rec("2026-11-01T10:00:00Z", 0, 20000, model="anthropic.claude-future-9")])
    cli.main(["logs", str(path), "--ttl-check", "--price", "3"])
    out = capsys.readouterr().out
    assert "CacheCanary has no 1-hour cache information for this model" in out
    assert "not documented" not in out


def test_skipped_count_is_kept_when_nothing_could_be_checked(tmp_path, capsys):
    path = write_file(tmp_path, [rec("", 0, 20000)])
    cli.main(["logs", str(path), "--ttl-check"])
    out = capsys.readouterr().out
    assert "no calls with usage and a timestamp" in out
    assert "Left out of the re-write check: 1 record(s)" in out


# --- real invocation logs ---------------------------------------------------------

def test_real_eu_west_1_logs_nov_2026():
    """Real Bedrock invocation logs (S3 delivery, text data on), eu-west-1, November 4 2026, redacted:
    account, request IDs and caller replaced; every text replaced by a placeholder, the same text by the
    same placeholder. Calls: InvokeModel prompt A at 0, 2 and 8 min; Converse with tools prompt B at 0 and
    1 min and with a changed date line at 3 min; a ~130 KB Converse request (body stored in S3); InvokeModel
    prompt D twice at 4 min; ConverseStream prompt S at 5 min."""
    from pathlib import Path
    from cachecanary import logs
    path = Path(__file__).parent / "fixtures" / "bedrock_invocation_logs_eu_ttl_2026-11.jsonl"
    calls, skipped = ttl.collect(list(logs.iter_records(path)))
    assert skipped == 0 and len(calls) == 10
    g = ttl.check(calls)["eu.anthropic.claude-sonnet-4-6"]
    assert g.rewrites == {ttl.FIRST: 5, ttl.FIVE_TO_60M: 1, f"{ttl.WITHIN_5M}/caller": 1}
    assert g.rewrite_tokens[ttl.FIVE_TO_60M] == 5511
    assert g.calls_body_in_s3 == 1 and g.calls_no_body == 0
    assert all(c.region == "eu-west-1" for c in calls)
    assert sum(c.prompt_start is not None for c in calls) == 9
