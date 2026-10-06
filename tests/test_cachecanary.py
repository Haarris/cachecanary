import copy
import gzip
import io
import json

import pytest

from cachecanary import cli, diff, lint, logs, probe
from cachecanary.models import lookup
from cachecanary.request import normalize
from cachecanary.usage import from_response

LONG_SYSTEM = "You are a careful assistant. " * 400  # ~2.8k tokens, above the 1,024 minimum
SONNET = "us.anthropic.claude-sonnet-4-6"
HAIKU = "anthropic.claude-haiku-4-5-20251001-v1:0"


def converse_request(system_text=LONG_SYSTEM, tools=None, messages=None, ttl=None):
    cp = {"type": "default", **({"ttl": ttl} if ttl else {})}
    req = {
        "modelId": SONNET,
        "system": [{"text": system_text}, {"cachePoint": cp}],
        "messages": messages or [{"role": "user", "content": [{"text": "hi"}]}],
    }
    if tools is not None:
        req["toolConfig"] = {"tools": tools}
    return req


def invoke_body(system_text=LONG_SYSTEM, messages=None):
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "system": [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
        "messages": messages or [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
    }


def rules(findings):
    return {f.rule for f in findings}


# --- models ---------------------------------------------------------------

def test_lookup_prefers_longest_key():
    assert lookup("global.anthropic.claude-opus-5-5")[0] == "claude-opus-5-5"
    assert lookup("anthropic.claude-opus-5")[0] == "claude-opus-5"
    assert lookup("anthropic.claude-haiku-4-5-20251001-v1:0")[1].min_tokens == 4096
    assert lookup("meta.llama3") == (None, None)


# --- normalize ------------------------------------------------------------

def test_converse_cachepoint_marks_previous_block():
    req = normalize(converse_request(ttl="1h"))
    assert req.api == "converse"
    assert req.checkpoint_indexes == [0]
    assert req.blocks[0].section == "system" and req.blocks[0].ttl == "1h"


def test_invoke_cache_control_marks_its_own_block():
    req = normalize(invoke_body())
    assert req.api == "invoke"
    assert req.checkpoint_indexes == [0]
    assert req.blocks[0].ttl == "5m"
    assert req.blocks[-1].section == "messages"


def test_checkpoint_marker_does_not_change_block_content():
    with_cp = normalize(invoke_body())
    body = invoke_body()
    del body["system"][0]["cache_control"]
    without_cp = normalize(body)
    assert with_cp.blocks[0].content == without_cp.blocks[0].content


# --- lint -----------------------------------------------------------------

def test_clean_request_has_no_errors():
    findings = lint.lint(normalize(converse_request()))
    assert not [f for f in findings if f.severity == "error"]


def test_no_checkpoint_warns():
    req = converse_request()
    req["system"] = [{"text": LONG_SYSTEM}]
    assert "no-checkpoint" in rules(lint.lint(normalize(req)))


def test_prefix_too_short_is_error_for_model_minimum():
    req = converse_request(system_text="short prompt")
    findings = lint.lint(normalize(req))
    assert "prefix-too-short" in rules(findings)


def test_haiku_needs_4096_tokens():
    req = converse_request(system_text="You are a careful assistant. " * 200)  # ~1.4k tokens
    assert "prefix-too-short" not in rules(lint.lint(normalize(req)))
    assert "prefix-too-short" in rules(lint.lint(normalize(req, HAIKU)))


def test_dynamic_date_in_prefix_warns():
    req = converse_request(system_text=LONG_SYSTEM + " Today's date is 2026-10-05.")
    assert "dynamic-in-prefix" in rules(lint.lint(normalize(req)))


def test_dynamic_content_after_checkpoint_is_fine():
    req = converse_request(messages=[{"role": "user", "content": [{"text": "It is 2026-10-05 10:30"}]}])
    assert "dynamic-in-prefix" not in rules(lint.lint(normalize(req)))


def test_profile_arn_and_unknown_model_warn():
    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
    assert "profile-arn" in rules(lint.lint(normalize(converse_request(), arn)))
    assert "unknown-model" in rules(lint.lint(normalize(converse_request(), "anthropic.claude-zeta-9")))


def test_too_many_checkpoints():
    req = converse_request()
    req["messages"] = [{"role": "user", "content": [{"text": f"m{i}"}, {"cachePoint": {"type": "default"}}]} for i in range(4)]
    assert "too-many-checkpoints" in rules(lint.lint(normalize(req)))


def test_ttl_order_and_unsupported():
    req = converse_request(ttl=None)
    req["messages"] = [{"role": "user", "content": [{"text": "x"}, {"cachePoint": {"type": "default", "ttl": "1h"}}]}]
    assert "ttl-order" in rules(lint.lint(normalize(req)))
    old = converse_request(ttl="1h")
    assert "ttl-unsupported" in rules(lint.lint(normalize(old, "anthropic.claude-3-7-sonnet-20250219-v1:0")))


def test_long_history_without_conversation_checkpoint_warns():
    msgs = [{"role": "user", "content": [{"text": f"turn {i}"}]} for i in range(5)]
    assert "no-conversation-checkpoint" in rules(lint.lint(normalize(converse_request(messages=msgs))))


# --- diff -----------------------------------------------------------------

def test_identical_prefix_points_to_ttl_or_routing():
    a = normalize(converse_request())
    b = normalize(converse_request(messages=[{"role": "user", "content": [{"text": "different question"}]}]))
    assert diff.explain(a, b)[0].code == "prefix-identical"


def test_system_change_detected():
    a = normalize(converse_request(system_text=LONG_SYSTEM + " User: Ali"))
    b = normalize(converse_request(system_text=LONG_SYSTEM + " User: Sara"))
    assert diff.explain(a, b)[0].code == "system-changed"


def test_tools_reordered_vs_changed():
    t1 = {"toolSpec": {"name": "a", "inputSchema": {"json": {}}}}
    t2 = {"toolSpec": {"name": "b", "inputSchema": {"json": {}}}}
    a = normalize(converse_request(tools=[t1, t2, {"cachePoint": {"type": "default"}}]))
    b = normalize(converse_request(tools=[t2, t1, {"cachePoint": {"type": "default"}}]))
    assert diff.explain(a, b)[0].code == "tools-reordered"
    t3 = {"toolSpec": {"name": "a", "description": "new", "inputSchema": {"json": {}}}}
    c = normalize(converse_request(tools=[t3, t2, {"cachePoint": {"type": "default"}}]))
    assert diff.explain(a, c)[0].code == "tools-changed"


def test_model_change():
    a = normalize(converse_request())
    b = normalize(converse_request(), "us.anthropic.claude-opus-5")
    assert diff.explain(a, b)[0].code == "model-changed"


def test_lookback_exceeded_after_many_tool_blocks():
    base = [{"role": "user", "content": [{"text": "start"}, {"cachePoint": {"type": "default"}}]}]
    a_req = converse_request(messages=base)
    calls = [{"toolUse": {"toolUseId": f"t{i}", "name": "x", "input": {}}} for i in range(12)]
    results = [{"toolResult": {"toolUseId": f"t{i}", "content": [{"text": "ok"}]}} for i in range(12)]
    b_msgs = [
        {"role": "user", "content": [{"text": "start"}]},
        {"role": "assistant", "content": calls},
        {"role": "user", "content": results + [{"cachePoint": {"type": "default"}}]},
    ]
    b_req = converse_request(messages=b_msgs)
    reasons = diff.explain(normalize(a_req), normalize(b_req))
    assert "lookback-exceeded" in {r.code for r in reasons}


def test_history_edit_detected():
    a = normalize(converse_request(messages=[{"role": "user", "content": [{"text": "q1"}, {"cachePoint": {"type": "default"}}]}]))
    b = normalize(converse_request(messages=[{"role": "user", "content": [{"text": "q1 edited"}, {"cachePoint": {"type": "default"}}]}]))
    assert diff.explain(a, b)[0].code == "history-changed"


# --- usage ----------------------------------------------------------------

def test_usage_converse_and_invoke_shapes():
    c = from_response({"usage": {"inputTokens": 10, "cacheReadInputTokens": 900, "cacheWriteInputTokens": 0, "outputTokens": 5}})
    assert c.total_input == 910 and round(c.hit_ratio, 3) == 0.989
    i = from_response({"usage": {"input_tokens": 10, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 900}})
    assert i.cache_write == 900 and i.hit_ratio == 0
    assert from_response({"nope": 1}) is None


# --- logs -----------------------------------------------------------------

def _log(model, arn, usage):
    return {"schemaType": "ModelInvocationLog", "modelId": model, "identity": {"arn": arn},
            "output": {"outputBodyJson": {"usage": usage}}}


def test_logs_aggregate_and_unparsed(tmp_path):
    recs = [
        _log("m1", "role/a", {"inputTokens": 100, "cacheReadInputTokens": 900, "cacheWriteInputTokens": 0}),
        _log("m1", "role/a", {"inputTokens": 100, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 900}),
        {"schemaType": "ModelInvocationLog", "modelId": "m1", "identity": {"arn": "role/b"},
         "output": {"outputBodyS3Path": "s3://bucket/key"}},
    ]
    groups = logs.aggregate(recs)
    g = groups["m1"]
    assert g.calls == 3 and g.unparsed == 1
    assert round(g.usage.hit_ratio, 2) == 0.45

    path = tmp_path / "logs.json.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(json.dumps(r) for r in recs))
    assert len(list(logs.iter_records(path))) == 3


def test_logs_cli_threshold_exit_code(tmp_path, capsys):
    path = tmp_path / "l.jsonl"
    path.write_text(json.dumps(_log("m1", "r", {"inputTokens": 1000, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0})))
    assert cli.main(["logs", str(path), "--min-hit", "0.5"]) == 1
    assert "below threshold" in capsys.readouterr().out


# --- probe ----------------------------------------------------------------

class FakeClient:
    def __init__(self, reads):
        self.reads = list(reads)

    def converse(self, modelId, **kwargs):
        assert "modelId" not in kwargs
        return {"usage": {"inputTokens": 10, "cacheReadInputTokens": self.reads.pop(0), "cacheWriteInputTokens": 0}}

    def invoke_model(self, modelId, body):
        payload = {"usage": {"input_tokens": 10, "cache_read_input_tokens": self.reads.pop(0), "cache_creation_input_tokens": 0}}
        return {"body": io.BytesIO(json.dumps(payload).encode())}


@pytest.mark.parametrize("reads,expected", [([0, 900], True), ([0, 0], False)])
def test_probe_converse(reads, expected):
    res = probe.probe(converse_request(), SONNET, client=FakeClient(reads), pause_s=0)
    assert res.passed is expected


def test_probe_invoke():
    res = probe.probe(invoke_body(), SONNET, client=FakeClient([0, 500]), pause_s=0)
    assert res.passed and res.second.cache_read == 500


# --- cli ------------------------------------------------------------------

def test_cli_lint_exit_codes(tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(converse_request()))
    assert cli.main(["lint", str(good)]) == 0
    capsys.readouterr()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(converse_request(system_text="tiny")))
    assert cli.main(["--json", "lint", str(bad)]) == 1
    findings = json.loads(capsys.readouterr().out)
    assert any(f["rule"] == "prefix-too-short" and f["severity"] == "error" for f in findings)


def test_cli_diff_exit_code(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(converse_request(system_text=LONG_SYSTEM + " x")))
    b.write_text(json.dumps(converse_request(system_text=LONG_SYSTEM + " y")))
    assert cli.main(["diff", str(a), str(b)]) == 1
    b.write_text(json.dumps(copy.deepcopy(converse_request(system_text=LONG_SYSTEM + " x"))))
    assert cli.main(["diff", str(a), str(b)]) == 0


def test_single_new_question_is_not_flagged():
    assert "no-conversation-checkpoint" not in rules(lint.lint(normalize(converse_request())))
