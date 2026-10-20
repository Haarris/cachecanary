"""The browser cache checker (site/check/): its prices and maths must match the CLI exactly.

site/check.js is run with Node and compared with cachecanary.pricing on the same inputs.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from cachecanary import cli, pricing
from cachecanary.models import CLAUDE_LIMITS
from cachecanary.usage import Usage

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
CHECK_JS = SITE / "check.js"
PAGE = (SITE / "check" / "index.html").read_text()
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is needed to run site/check.js")


def js(expr: str):
    """Evaluate a JavaScript expression with check.js loaded as `c`, return the JSON result."""
    script = f"const c = require({json.dumps(str(CHECK_JS))}); process.stdout.write(JSON.stringify({expr}));"
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


# ---------------------------------------------------------------- the price table

@needs_node
def test_price_table_matches_the_cli():
    models = js("c.MODELS")
    assert list(models) == list(pricing.GLOBAL_PRICES)  # same models, same order
    for key, rates in pricing.GLOBAL_PRICES.items():
        m = models[key]
        assert m["input"] == rates.input, key
        assert m["read"] * m["input"] == pytest.approx(rates.cache_read), key
        assert m["input"] * js("c.WRITE_5M") == pytest.approx(rates.cache_write_5m), key
        assert m["min"] == CLAUDE_LIMITS[key].min_tokens, key
        assert bool(m.get("legacy")) == (key in pricing.NO_REGIONAL_PREMIUM) == CLAUDE_LIMITS[key].legacy, key
    assert js("c.REGIONAL_PREMIUM") == pricing.REGIONAL_PREMIUM


@needs_node
def test_rates_match_rates_for_every_model_endpoint_and_custom_price():
    cases = [(k, e, p) for k in pricing.GLOBAL_PRICES for e in ("global", "regional") for p in (None, 2.5)]
    got = js("[" + ",".join(f"c.rates({json.dumps(k)}, {json.dumps(e)}, {json.dumps(p)})" for k, e, p in cases) + "]")
    for (key, endpoint, price), r in zip(cases, got):
        model_id = f"{'global' if endpoint == 'global' else 'us'}.anthropic.{key}"
        want = pricing.rates_for(model_id, price).rates
        assert r["input"] == pytest.approx(want.input), (key, endpoint, price)
        assert r["read"] == pytest.approx(want.cache_read), (key, endpoint, price)
        assert r["write"] == pytest.approx(want.cache_write_5m), (key, endpoint, price)


# ---------------------------------------------------------------- the maths

COUNTS = [(0, 0, 1), (1, 0, 0), (0, 1, 0), (100, 900, 0), (100, 0, 900), (1_250_000, 8_400_000, 2_100_000),
          (21_971_480, 24_063_678, 14_464_604), (10, 5848, 5848), (5, 1_000_000, 1), (10**13, 10**13, 10**13),
          (3_000, 4_000, 1_000), (999_999_999, 1, 999_999_999)]


@needs_node
@pytest.mark.parametrize("key", ["claude-sonnet-4-6", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1",
                                 "claude-haiku-5-5", "claude-3-5-haiku"])
@pytest.mark.parametrize("endpoint", ["global", "regional"])
def test_cost_loss_and_hit_rate_match_the_cli(key, endpoint):
    targets = (0.5, 0.8, 0.9, 1.0)
    calls = [f"[c.inputCost({u},{r},{w},rt), c.lostToMisses({u},{r},{w},rt,{t}), c.hitRate({u},{r},{w})]"
             for u, r, w in COUNTS for t in targets]
    got = js(f"(() => {{ const rt = c.rates({json.dumps(key)}, {json.dumps(endpoint)}); return [{','.join(calls)}]; }})()")
    rates = pricing.rates_for(f"{'global' if endpoint == 'global' else 'us'}.anthropic.{key}").rates
    i = 0
    for u, r, w in COUNTS:
        usage = Usage(u, r, w)
        for t in targets:
            cost, lost, hit = got[i]
            i += 1
            assert cost == pytest.approx(pricing.input_cost(usage, rates), rel=1e-12, abs=1e-12)
            assert lost == pytest.approx(pricing.lost_to_misses(usage, rates, t), rel=1e-9, abs=1e-12)
            assert hit == pytest.approx(usage.hit_ratio)


@needs_node
@pytest.mark.parametrize("usd", [0, 0.001, 0.009999, 0.01, 0.5, 1, 12.345, 1234.5, 1_000_000, 82.51])
def test_money_matches_the_cli(usd):
    assert js(f"c.money({usd})") == cli._money(usd)


# ---------------------------------------------------------------- parsing what people paste

@needs_node
@pytest.mark.parametrize("text,expected", [
    ("1250000", 1250000), ("1,250,000", 1250000), (" 1 250 000 ", 1250000), ("1_250_000", 1250000),
    ("2.5M", 2500000), ("830k", 830000), ("1.2b", 1200000000), ("0", 0), ("12.7", 13), (".5k", 500),
    ("", None), ("  ", None), ("-5", None), ("abc", None), ("1e6", None), ("5 tokens", None),
    ("1.2.3", None), ("$100", None), ("1,25,0000", 1250000), ("Infinity", None), ("NaN", None),
])
def test_parse_count(text, expected):
    assert js(f"c.parseCount({json.dumps(text)})") == expected


@needs_node
@pytest.mark.parametrize("text,expected", [("90", 0.9), ("90%", 0.9), (" 80 % ", 0.8), ("100", 1.0), ("0.5", 0.005),
                                           ("0", None), ("101", None), ("-10", None), ("ninety", None), ("", None)])
def test_parse_percent(text, expected):
    got = js(f"c.parsePercent({json.dumps(text)})")
    assert got == (pytest.approx(expected) if expected is not None else None)


@needs_node
@pytest.mark.parametrize("text,expected", [("", "missing"), ("  ", "missing"), ("3", 3), ("$2.5", 2.5),
                                           ("0", None), ("1000", None), ("abc", None), ("-1", None)])
def test_parse_price(text, expected):
    got = js(f"(() => {{ const v = c.parsePrice({json.dumps(text)}); return v === undefined ? 'missing' : v; }})()")
    assert got == expected


# ---------------------------------------------------------------- verdicts and errors

def evaluate(**kw):
    base = {"model": "claude-sonnet-4-6", "endpoint": "regional", "uncached": "0", "read": "0", "write": "0",
            "target": "90", "price": ""}
    return js(f"c.evaluate({json.dumps({**base, **kw})})")


@needs_node
def test_verdicts():
    assert evaluate(uncached="100", read="1000", write="10")["verdict"] == "healthy"  # 90.1%
    assert evaluate(uncached="100", read="950", write="10")["verdict"] == "leaking"  # 89.6%, just under 90%
    assert evaluate(uncached="90", read="900", write="10")["verdict"] == "healthy"  # exactly 90% counts as healthy
    assert evaluate(uncached="500", read="500", write="0")["verdict"] == "leaking"  # exactly 50%
    assert evaluate(uncached="100", read="0", write="5000", target="50")["verdict"] == "never-read"
    assert evaluate(uncached="100", read="600", write="100")["verdict"] == "leaking"
    assert evaluate(uncached="500", read="300", write="200")["verdict"] == "poor"
    assert evaluate(uncached="100", read="0", write="5000")["verdict"] == "never-read"
    assert evaluate(uncached="5000", read="0", write="0")["verdict"] == "none"
    assert evaluate(uncached="100", read="500", write="900")["writesExceedReads"] is True
    assert evaluate(uncached="100", read="0", write="900")["writesExceedReads"] is False  # covered by never-read


@needs_node
def test_result_matches_the_cli_for_a_real_week():
    # The Langfuse-verified Sonnet 5.5 week from the CLI's real-data check: $82.51 input, 40% hit.
    res = evaluate(model="claude-sonnet-5-5", endpoint="global", uncached="21,971,480", read="24,063,678",
                   write="14,464,604")
    assert js(f"c.money({res['cost']})") == "$82.51"
    assert round(res["hit"] * 100) == 40 and res["verdict"] == "poor"


@needs_node
def test_custom_price_and_target():
    res = evaluate(uncached="1000000", read="0", write="0", price="2", target="80")
    assert res["price"] == "custom" and res["cost"] == pytest.approx(2.0) and res["target"] == pytest.approx(0.8)
    assert res["lost"] == pytest.approx(0.8 * 1_000_000 * (2 - 0.2) / 1e6)


@needs_node
@pytest.mark.parametrize("kw,message", [
    ({"uncached": ""}, "InputTokenCount"), ({"read": "x"}, "CacheReadInputTokenCount"),
    ({"write": "-1"}, "CacheWriteInputTokenCount"), ({"uncached": "1", "target": "0"}, "target hit rate"),
    ({"uncached": "1", "price": "free"}, "Your price"), ({"uncached": "1", "model": "gpt-5"}, "Pick a model"),
    ({}, "All three numbers are 0"),
])
def test_errors(kw, message):
    assert message in evaluate(**kw)["error"]


@needs_node
@pytest.mark.parametrize("key,expected", [("claude-sonnet-4-6", "Claude Sonnet 4.6"), ("claude-3-5-haiku", "Claude 3.5 Haiku"),
                                          ("claude-opus-5", "Claude Opus 5"), ("claude-fable-5-1", "Claude Fable 5.1")])
def test_labels(key, expected):
    assert js(f"c.label({json.dumps(key)})") == expected


# ---------------------------------------------------------------- the page

def test_page_basics_and_links():
    assert 'rel="canonical" href="https://cachecanary.com/check/"' in PAGE and "—" not in PAGE
    assert '<script src="/check.js" defer></script>' in PAGE and '<script src="/main.js" defer></script>' in PAGE
    for el in ('id="checker"', 'id="model"', 'id="endpoint"', 'id="uncached"', 'id="read"', 'id="write"',
               'id="target"', 'id="price"', 'id="result"', 'id="error"', 'id="r-causes"'):
        assert el in PAGE, el
    for metric in ("InputTokenCount", "CacheReadInputTokenCount", "CacheWriteInputTokenCount"):
        assert metric in PAGE
    assert "https://docs.aws.amazon.com/bedrock/latest/userguide/quotas-token-burndown.html" in PAGE
    index = (SITE / "index.html").read_text().split("<h2>What a broken cache costs</h2>")[0]
    assert 'href="/check/"' in index  # reachable from the top of the home page
    assert "https://cachecanary.com/check/" in (SITE / "llms.txt").read_text()


def test_script_has_no_network_calls():
    code = CHECK_JS.read_text()
    for call in ("fetch(", "XMLHttpRequest", "sendBeacon", "WebSocket", "import(", "eval(", "new Function"):
        assert call not in code, call
    # innerHTML only ever gets the fixed strings in CAUSES, never typed input
    assert re.findall(r"innerHTML = ([^;]+);", code) == ['""', "c"]
