"""Dollar figures in `logs`: price table, cost math, 1-hour writes, --price, folders and output."""

import gzip
import json

import pytest

from cachecanary import cli, logs, pricing
from cachecanary.models import CLAUDE_LIMITS
from cachecanary.usage import Usage, from_response, from_stream_events

SONNET_US = "us.anthropic.claude-sonnet-4-6"
SONNET_GLOBAL = "global.anthropic.claude-sonnet-4-6"


def _rec(read, write, uncached, model=SONNET_US, arn="arn:role/a", details=None):
    usage = {"inputTokens": uncached, "cacheReadInputTokens": read, "cacheWriteInputTokens": write}
    if details is not None:
        usage["cacheDetails"] = details
    return {"schemaType": "ModelInvocationLog", "modelId": model, "identity": {"arn": arn},
            "output": {"outputBodyJson": {"usage": usage}}}


def _write(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in recs))
    return path


# --- price table ------------------------------------------------------------

def test_every_priced_model_is_a_known_model():
    assert set(pricing.GLOBAL_PRICES) <= set(CLAUDE_LIMITS)


def test_only_claude_3_models_are_unpriced():
    assert {k for k in CLAUDE_LIMITS if k not in pricing.GLOBAL_PRICES} == {"claude-3-7-sonnet", "claude-3-5-sonnet"}


# Rows copied from the AWS Price List (AmazonBedrockFoundationModels, us-east-1, on-demand, 2026-10-06),
# USD per million tokens: (input, cache read, 5m write, 1h write).
AWS_ROWS = {
    (SONNET_GLOBAL, "global"): (3, 0.3, 3.75, 6),
    (SONNET_US, "regional"): (3.3, 0.33, 4.125, 6.6),
    ("global.anthropic.claude-opus-5-5", "global"): (4, 0.2, 5, 8),
    ("us.anthropic.claude-opus-5-5", "regional"): (4.4, 0.22, 5.5, 8.8),
    ("global.anthropic.claude-fable-5-1", "global"): (10, 0.25, 12.5, 20),
    ("us.anthropic.claude-fable-5-1", "regional"): (11, 0.275, 13.75, 22),
    ("us.anthropic.claude-fable-5", "regional"): (11, 1.1, 13.75, 22),
    ("us.anthropic.claude-mythos-5", "regional"): (11, 1.1, 13.75, 22),
    ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "regional"): (1.1, 0.11, 1.375, 2.2),
    ("global.anthropic.claude-haiku-4-5-20251001-v1:0", "global"): (1, 0.1, 1.25, 2),
    ("global.anthropic.claude-sonnet-5-5", "global"): (2, 0.2, 2.5, 4),
    ("us.anthropic.claude-sonnet-5", "regional"): (2.2, 0.22, 2.75, 4.4),
    ("us.anthropic.claude-opus-4-7", "regional"): (5.5, 0.55, 6.875, 11),
    ("global.anthropic.claude-opus-4-5-20251101-v1:0", "global"): (5, 0.5, 6.25, 10),
    ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "regional"): (3.3, 0.33, 4.125, 6.6),
}


@pytest.mark.parametrize("model,kind", list(AWS_ROWS), ids=[m for m, _ in AWS_ROWS])
def test_prices_match_the_aws_price_list(model, kind):
    priced = pricing.rates_for(model)
    assert priced.source == "list"
    r = priced.rates
    assert (r.input, r.cache_read, r.cache_write_5m, r.cache_write_1h) == pytest.approx(AWS_ROWS[(model, kind)])


@pytest.mark.parametrize("model", [
    None, "", "anthropic.claude-3-7-sonnet-20250219-v1:0", "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/abc123", "amazon.nova-pro-v1:0", "?",
])
def test_unpriced_models(model):
    assert pricing.rates_for(model) is None


def test_bare_model_id_is_regional():
    assert pricing.rates_for("anthropic.claude-sonnet-4-6").rates.input == pytest.approx(3.3)


def test_custom_price_keeps_the_model_read_multiplier_and_skips_the_premium():
    opus = pricing.rates_for("us.anthropic.claude-opus-5-5", 3).rates
    assert (opus.input, opus.cache_read, opus.cache_write_5m, opus.cache_write_1h) == pytest.approx((3, 0.15, 3.75, 6))
    unknown = pricing.rates_for("arn:aws:bedrock:us-east-1:1:application-inference-profile/x", 2)
    assert unknown.source == "custom" and unknown.rates.cache_read == pytest.approx(0.2)


# --- cost math --------------------------------------------------------------

RATES = pricing.Rates(input=3, cache_read=0.3, cache_write_5m=3.75, cache_write_1h=6)


def test_input_cost_by_hand():
    u = Usage(uncached_input=1_000_000, cache_read=2_000_000, cache_write=1_000_000, cache_write_1h=400_000)
    # 1M x 3 + 2M x 0.3 + 0.6M x 3.75 + 0.4M x 6
    assert pricing.input_cost(u, RATES) == pytest.approx(3 + 0.6 + 2.25 + 2.4)


def test_more_1h_writes_than_writes_is_clamped():
    assert pricing.input_cost(Usage(0, 0, 100, 0, 500), RATES) == pytest.approx(100 * 6 / 1e6)


def test_nothing_lost_at_or_above_target_or_without_traffic():
    assert pricing.lost_to_misses(Usage(10, 90, 0), RATES, 0.9) == 0
    assert pricing.lost_to_misses(Usage(0, 100, 0), RATES, 0.9) == 0
    assert pricing.lost_to_misses(Usage(), RATES, 0.9) == 0


def test_dropped_cache_marker_counts_the_full_loss():
    # Nothing cached at all (LiteLLM-style regression): 90% of 1M tokens should have been reads.
    lost = pricing.lost_to_misses(Usage(uncached_input=1_000_000), RATES, 0.9)
    assert lost == pytest.approx(0.9 * (3 - 0.3))


def test_lost_uses_the_mix_actually_paid():
    u = Usage(uncached_input=100_000, cache_read=500_000, cache_write=400_000)
    paid_per_mtok = (100_000 * 3 + 400_000 * 3.75) / 500_000
    expected = (0.9 * 1_000_000 - 500_000) * (paid_per_mtok - 0.3) / 1e6
    assert pricing.lost_to_misses(u, RATES, 0.9) == pytest.approx(expected)
    # Cost at target plus the loss equals what was actually paid.
    at_target = (900_000 * 0.3 + 100_000 * paid_per_mtok) / 1e6
    assert at_target + pricing.lost_to_misses(u, RATES, 0.9) == pytest.approx(pricing.input_cost(u, RATES))


# --- 1-hour writes in responses --------------------------------------------

def test_converse_cache_details():
    u = from_response({"usage": {"inputTokens": 5, "cacheWriteInputTokens": 300, "cacheReadInputTokens": 0,
                                 "cacheDetails": [{"ttl": "1h", "inputTokens": 200}, {"ttl": "5m", "inputTokens": 100}]}})
    assert (u.cache_write, u.cache_write_1h) == (300, 200)


def test_converse_cache_details_malformed_is_ignored():
    u = from_response({"usage": {"inputTokens": 5, "cacheWriteInputTokens": 300,
                                 "cacheDetails": ["junk", {"ttl": "1h", "inputTokens": "x"}, {"ttl": "1h"}]}})
    assert (u.cache_write, u.cache_write_1h) == (300, 0)
    assert from_response({"usage": {"inputTokens": 5, "cacheDetails": "nope"}}).cache_write_1h == 0


def test_messages_cache_creation_1h_including_stream():
    body = {"usage": {"input_tokens": 5, "cache_creation_input_tokens": 300, "cache_read_input_tokens": 0,
                      "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200}}}
    assert from_response(body).cache_write_1h == 200
    stream = [{"type": "message_start", "message": body}, {"type": "message_delta", "usage": {"output_tokens": 7}}]
    u = from_stream_events(stream)
    assert (u.cache_write_1h, u.output) == (200, 7)


def test_1h_writes_add_up_across_calls():
    assert (Usage(0, 0, 10, 0, 4) + Usage(0, 0, 10, 0, 6)).cache_write_1h == 10


# --- CLI output -------------------------------------------------------------

def test_logs_shows_cost_and_loss_with_default_target(tmp_path, capsys):
    path = _write(tmp_path / "l.jsonl", [_rec(0, 0, 1_000_000)])
    assert cli.main(["logs", str(path)]) == 0  # no --min-hit: dollars shown, never fails
    out = capsys.readouterr().out
    assert "input cost $3.30 at list price; about $2.67 of it lost to cache misses (target: 90% hit rate)" in out
    assert "List prices: Amazon Bedrock on-demand" in out


def test_logs_at_target_says_nothing_lost(tmp_path, capsys):
    path = _write(tmp_path / "l.jsonl", [_rec(950_000, 0, 50_000, model=SONNET_GLOBAL)])
    assert cli.main(["logs", str(path), "--min-hit", "0.9"]) == 0
    assert "at or above the 90% target hit rate, nothing lost to cache misses" in capsys.readouterr().out


def test_logs_custom_price(tmp_path, capsys):
    path = _write(tmp_path / "l.jsonl", [_rec(0, 0, 1_000_000)])
    assert cli.main(["logs", str(path), "--price", "2"]) == 0
    out = capsys.readouterr().out
    assert "input cost $2.00 at your price" in out and "List prices" not in out


def test_logs_unpriced_model_hint(tmp_path, capsys):
    path = _write(tmp_path / "l.jsonl", [_rec(0, 0, 1000, model="anthropic.claude-3-7-sonnet-20250219-v1:0")])
    assert cli.main(["logs", str(path)]) == 0
    out = capsys.readouterr().out
    assert "no price for this model; add --price" in out and "input cost" not in out and "List prices" not in out


def test_logs_group_with_priced_and_unpriced_models(tmp_path, capsys):
    path = _write(tmp_path / "l.jsonl", [_rec(0, 0, 1_000_000), _rec(0, 0, 10, model="amazon.nova-pro-v1:0")])
    assert cli.main(["logs", str(path), "--by", "principal"]) == 0
    out = capsys.readouterr().out
    assert "input cost $3.30" in out and "not priced: amazon.nova-pro-v1:0 (add --price to include)" in out


def test_logs_total_line_for_several_groups(tmp_path, capsys):
    path = _write(tmp_path / "l.jsonl", [_rec(0, 0, 1_000_000), _rec(0, 0, 1_000_000, model=SONNET_GLOBAL)])
    assert cli.main(["logs", str(path)]) == 0
    assert "Total: input cost $6.30; about $5.10 lost to cache misses (target: 90% hit rate)" in capsys.readouterr().out


def test_logs_tiny_amounts(tmp_path, capsys):
    path = _write(tmp_path / "l.jsonl", [_rec(0, 0, 100)])
    cli.main(["logs", str(path)])
    assert "input cost under $0.01" in capsys.readouterr().out


def test_logs_json_fields(tmp_path, capsys):
    path = _write(tmp_path / "l.jsonl", [_rec(0, 0, 1_000_000, details=[]),
                                         _rec(0, 1000, 0, details=[{"ttl": "1h", "inputTokens": 1000}]),
                                         _rec(0, 0, 5, model="anthropic.claude-3-7-sonnet-20250219-v1:0")])
    cli.main(["--json", "logs", str(path), "--min-hit", "0.5"])
    data = json.loads(capsys.readouterr().out)
    sonnet = data[SONNET_US]
    assert sonnet["input_cost_usd"] == pytest.approx(3.3 + 1000 * 6.6 / 1e6, abs=1e-6)
    assert sonnet["cache_write_1h"] == 1000 and sonnet["target_hit"] == 0.5 and sonnet["price"] == "list"
    unpriced = data["anthropic.claude-3-7-sonnet-20250219-v1:0"]
    assert unpriced["input_cost_usd"] is None and unpriced["lost_usd"] is None


def test_logs_github_summary_has_money_columns(tmp_path, capsys, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    path = _write(tmp_path / "l.jsonl", [_rec(0, 0, 1_000_000)])
    assert cli.main(["--github", "logs", str(path), "--min-hit", "0.8"]) == 1
    text = summary.read_text()
    assert "| input cost | lost vs 80% hit |" in text and "| $3.30 | $2.38 |" in text


@pytest.mark.parametrize("bad", ["0", "-1", "nan", "inf", "abc", "1000"])
def test_bad_price_is_rejected(bad, tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(["logs", "x.jsonl", "--price", bad])
    assert exc.value.code == 2


# --- folders ----------------------------------------------------------------

def test_logs_reads_a_folder_recursively(tmp_path, capsys):
    root = tmp_path / "bedrock-logs"
    deep = root / "AWSLogs" / "111122223333" / "BedrockModelInvocationLogs" / "us-east-1" / "2026" / "10" / "06" / "10"
    deep.mkdir(parents=True)
    with gzip.open(deep / "batch.json.gz", "wt") as fh:
        fh.write(json.dumps(_rec(900, 0, 100)))
    _write(root / "flat.jsonl", [_rec(900, 0, 100)])
    _write(root / ".hidden" / "skip.json", [_rec(0, 0, 1)])
    (root / "notes.md").write_text("not a log")
    assert cli.main(["logs", str(root)]) == 0
    assert f"{SONNET_US}: hit 90% over 2 calls" in capsys.readouterr().out


def test_logs_empty_folder_is_an_error(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert cli.main(["logs", str(tmp_path / "empty")]) == 2
    assert "no log files" in capsys.readouterr().err


def test_logs_corrupt_gzip_is_an_error_not_a_crash(tmp_path, capsys):
    (tmp_path / "bad.json.gz").write_bytes(b"\x1f\x8bnot really gzip")
    assert cli.main(["logs", str(tmp_path)]) == 2
    assert "could not read" in capsys.readouterr().err


def test_by_model_split_inside_a_principal_group(tmp_path):
    path = _write(tmp_path / "l.jsonl", [_rec(1, 0, 0), _rec(2, 0, 0, model=SONNET_GLOBAL)])
    groups = logs.aggregate(logs.iter_records(path), by="principal")
    assert {m: u.cache_read for m, u in groups["arn:role/a"].by_model.items()} == {SONNET_US: 1, SONNET_GLOBAL: 2}


# --- legacy models and stricter model matching (0.2.1) ----------------------

from cachecanary import lint as lint_mod  # noqa: E402
from cachecanary.models import lookup  # noqa: E402
from cachecanary.request import normalize  # noqa: E402


@pytest.mark.parametrize("model_id,key", [
    ("us.anthropic.claude-sonnet-4-20250514-v1:0", "claude-sonnet-4"),
    ("global.anthropic.claude-sonnet-4-20250514-v1:0", "claude-sonnet-4"),
    ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"),
    ("us.anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("us.anthropic.claude-opus-4-1-20250805-v1:0", "claude-opus-4-1"),
    ("anthropic.claude-opus-4-20250514-v1:0", "claude-opus-4"),
    ("anthropic.claude-opus-4-6-v1", "claude-opus-4-6"),
    ("us.anthropic.claude-3-5-haiku-20241022-v1:0", "claude-3-5-haiku"),
    ("anthropic.claude-3-5-sonnet-20241022-v2:0", "claude-3-5-sonnet"),
    ("us.anthropic.claude-opus-5-5", "claude-opus-5-5"),
    ("us.anthropic.claude-opus-5", "claude-opus-5"),
    ("arn:aws:bedrock:us-west-2:1:inference-profile/us.anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("us.anthropic.claude-opus-4-9", None),          # a future model must not borrow Opus 4's rules
    ("us.anthropic.claude-sonnet-4-7-20270101-v1:0", None),
    ("us.anthropic.claude-opus-5-9", None),
])
def test_lookup_matches_whole_model_names(model_id, key):
    assert lookup(model_id)[0] == key


def _lint_codes(model, ttl=None):
    cp = {"cachePoint": {"type": "default", **({"ttl": ttl} if ttl else {})}}
    req = {"system": [{"text": "Rule. " * 900}, cp], "messages": [{"role": "user", "content": [{"text": "q"}]}]}
    return {f.rule: f for f in lint_mod.lint(normalize(req, model))}


def test_legacy_model_note_instead_of_unknown():
    codes = _lint_codes("us.anthropic.claude-sonnet-4-20250514-v1:0")
    assert "legacy-model" in codes and "unknown-model" not in codes
    assert codes["legacy-model"].severity == "warn" and "1024-token minimum" in codes["legacy-model"].message


def test_legacy_min_size_is_checked():
    small = {"system": [{"text": "Rule one."}, {"cachePoint": {"type": "default"}}],
             "messages": [{"role": "user", "content": [{"text": "q"}]}]}
    rules = {f.rule for f in lint_mod.lint(normalize(small, "us.anthropic.claude-3-5-haiku-20241022-v1:0"))}
    assert "prefix-too-short" in rules


def test_1h_on_legacy_is_a_warning_and_on_3_7_an_error():
    assert _lint_codes("us.anthropic.claude-opus-4-1-20250805-v1:0", "1h")["ttl-unverified"].severity == "warn"
    assert _lint_codes("us.anthropic.claude-3-7-sonnet-20250219-v1:0", "1h")["ttl-unsupported"].severity == "error"
    assert "ttl-unverified" not in _lint_codes("us.anthropic.claude-sonnet-4-6", "1h")


# AWS Price List rows (us-east-1, on-demand, 2026-10-06): legacy models, same price on every endpoint.
@pytest.mark.parametrize("model,expected", [
    ("us.anthropic.claude-sonnet-4-20250514-v1:0", (3, 0.3, 3.75)),
    ("global.anthropic.claude-sonnet-4-20250514-v1:0", (3, 0.3, 3.75)),
    ("us.anthropic.claude-opus-4-1-20250805-v1:0", (15, 1.5, 18.75)),
    ("anthropic.claude-opus-4-20250514-v1:0", (15, 1.5, 18.75)),
    ("us.anthropic.claude-3-5-haiku-20241022-v1:0", (0.8, 0.08, 1.0)),
])
def test_legacy_prices_match_aws_with_no_regional_premium(model, expected):
    r = pricing.rates_for(model).rates
    assert (r.input, r.cache_read, r.cache_write_5m) == pytest.approx(expected)
