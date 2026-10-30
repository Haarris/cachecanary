"""Edge cases: malformed input, unusual request shapes, streaming, log formats, AWS errors."""

import gzip
import json

import pytest

from cachecanary import cli, diff, lint, logs, probe
from cachecanary.request import RequestError, estimate_tokens, normalize
from cachecanary.usage import from_response, from_stream_events

LONG = "Policy text that never changes. " * 400  # ~3.2k tokens
SONNET = "us.anthropic.claude-sonnet-4-6"
CP = {"cachePoint": {"type": "default"}}


def rules(findings):
    return {f.rule for f in findings}


def conv(system=None, messages=None, tools=None, model=SONNET):
    req = {"modelId": model, "messages": messages or [{"role": "user", "content": [{"text": "q"}]}]}
    if system is not None:
        req["system"] = system
    if tools is not None:
        req["toolConfig"] = {"tools": tools}
    return req


# --- malformed input ------------------------------------------------------

@pytest.mark.parametrize("payload", [[], "text", 3, None])
def test_non_object_payload_rejected(payload):
    with pytest.raises(RequestError):
        normalize(payload)


def test_wrong_field_types_rejected():
    with pytest.raises(RequestError):
        normalize({"messages": "hello"})
    with pytest.raises(RequestError):
        normalize({"messages": [{"role": "user", "content": "not a list in converse"}]})
    with pytest.raises(RequestError):
        normalize({"anthropic_version": "x", "system": 5})
    with pytest.raises(RequestError):
        normalize({"toolConfig": []})


def test_empty_request_is_handled():
    req = normalize({})
    assert req.blocks == [] and "no-checkpoint" in rules(lint.lint(req))


# --- unusual shapes -------------------------------------------------------

def test_orphan_checkpoint_first_item():
    req = normalize(conv(tools=[CP], system=[{"text": LONG}, CP]))
    assert req.orphan_checkpoints == ["toolConfig.tools[0]"]
    assert "orphan-checkpoint" in rules(lint.lint(req))


def test_duplicate_markers_count_toward_limit():
    msgs = [{"role": "user", "content": [{"text": "a"}, CP, CP]}, {"role": "user", "content": [{"text": "b"}, CP]}]
    req = normalize(conv(system=[{"text": LONG}, CP, CP], messages=msgs))
    assert len(req.duplicate_checkpoints) == 2
    assert req.checkpoint_markers == 5
    found = rules(lint.lint(req))
    assert {"duplicate-checkpoint", "too-many-checkpoints"} <= found


def test_invoke_string_system_cannot_be_cached():
    body = {"anthropic_version": "bedrock-2023-05-31", "system": LONG, "messages": [{"role": "user", "content": "q"}]}
    req = normalize(body, SONNET)
    assert req.system_is_string
    assert "string-system" in rules(lint.lint(req))


def test_invoke_string_and_list_system_normalize_identically():
    a = normalize({"anthropic_version": "v", "system": LONG})
    b = normalize({"anthropic_version": "v", "system": [{"type": "text", "text": LONG}]})
    assert a.blocks[0].content == b.blocks[0].content


def test_image_bytes_use_flat_estimate():
    import base64, os
    blob = base64.b64encode(os.urandom(150_000)).decode()
    assert estimate_tokens(blob) == 1500
    assert estimate_tokens("abcd" * 1000) == 1000  # long letters-only run is text, not binary
    assert estimate_tokens("x" * 400) == 100


def test_partial_short_prefix_is_warning_not_error():
    tools = [{"toolSpec": {"name": "t", "inputSchema": {"json": {}}}}, CP]
    req = normalize(conv(tools=tools, system=[{"text": LONG}, CP]))
    findings = [f for f in lint.lint(req) if f.rule == "prefix-too-short"]
    assert len(findings) == 1 and findings[0].severity == "warn"
    assert findings[0].location == "toolConfig.tools[0]"


def test_invalid_ttl_value():
    req = normalize(conv(system=[{"text": LONG}, {"cachePoint": {"type": "default", "ttl": "10m"}}]))
    assert "ttl-invalid" in rules(lint.lint(req))


def test_no_model_and_profile_arn_skip_minimum_check():
    short = conv(system=[{"text": "tiny"}, CP], model=None)
    found = rules(lint.lint(normalize(short)))
    assert "no-model" in found and "prefix-too-short" not in found
    arn = "arn:aws:bedrock:us-west-2:111122223333:application-inference-profile/xyz"
    found = rules(lint.lint(normalize(short, arn)))
    assert "profile-arn" in found and "prefix-too-short" not in found


@pytest.mark.parametrize("snippet", [
    "session 3f2b8c1e-9d4a-4e2b-8f6a-1c2d3e4f5a6b",
    "generated at 1759651200",
    "current time 14:05",
])
def test_dynamic_patterns(snippet):
    req = normalize(conv(system=[{"text": LONG + " " + snippet}, CP]))
    assert "dynamic-in-prefix" in rules(lint.lint(req))


def test_static_numbers_are_not_flagged():
    req = normalize(conv(system=[{"text": LONG + " Refunds within 30 days. Ratio 3 to 1. Version 4.6."}, CP]))
    assert "dynamic-in-prefix" not in rules(lint.lint(req))


# --- diff edge cases ------------------------------------------------------

def test_api_switch_detected():
    a = normalize(conv(system=[{"text": LONG}, CP]))
    b = normalize({"anthropic_version": "v", "modelId": SONNET,
                   "system": [{"type": "text", "text": LONG, "cache_control": {"type": "ephemeral"}}]})
    assert diff.explain(a, b)[0].code == "api-changed"


def test_next_request_without_checkpoint():
    a = normalize(conv(system=[{"text": LONG}, CP]))
    b = normalize(conv(system=[{"text": LONG}]))
    assert diff.explain(a, b)[0].code == "no-checkpoint-in-next"


def test_truncated_history():
    msgs = [{"role": "user", "content": [{"text": f"t{i}"}]} for i in range(3)]
    msgs[-1]["content"].append(CP)
    a = normalize(conv(system=[{"text": LONG}, CP], messages=msgs))
    b = normalize(conv(system=[{"text": LONG}, CP], messages=[]))
    b.blocks = b.blocks[:1]
    assert diff.explain(a, b)[0].code == "prefix-truncated"


def test_ttl_change_detected():
    a = normalize(conv(system=[{"text": LONG}, CP]))
    b = normalize(conv(system=[{"text": LONG}, {"cachePoint": {"type": "default", "ttl": "1h"}}]))
    assert "ttl-changed" in {r.code for r in diff.explain(a, b)}


def test_intermediate_checkpoint_avoids_lookback_miss():
    base = [{"role": "user", "content": [{"text": "start"}, CP]}]
    calls = [{"toolUse": {"toolUseId": f"t{i}", "name": "x", "input": {}}} for i in range(12)]
    results = [{"toolResult": {"toolUseId": f"t{i}", "content": [{"text": "ok"}]}} for i in range(12)]
    b_msgs = [{"role": "user", "content": [{"text": "start"}, CP]},
              {"role": "assistant", "content": calls + [CP]},
              {"role": "user", "content": results + [CP]}]
    a = normalize(conv(system=[{"text": LONG}, CP], messages=base))
    b = normalize(conv(system=[{"text": LONG}, CP], messages=b_msgs))
    assert "lookback-exceeded" not in {r.code for r in diff.explain(a, b)}


# --- usage & streaming ----------------------------------------------------

def test_converse_stream_metadata():
    events = [{"messageStart": {"role": "assistant"}}, {"contentBlockDelta": {"delta": {"text": "hi"}}},
              {"metadata": {"usage": {"inputTokens": 5, "cacheReadInputTokens": 800, "cacheWriteInputTokens": 0, "outputTokens": 2}}}]
    u = from_stream_events(events)
    assert u.cache_read == 800 and u.output == 2


def test_invoke_stream_chunks():
    def chunk(obj):
        return {"chunk": {"bytes": json.dumps(obj).encode()}}
    events = [
        chunk({"type": "message_start", "message": {"usage": {"input_tokens": 7, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 900, "output_tokens": 1}}}),
        {"chunk": {"bytes": b"not json"}},
        chunk({"type": "content_block_delta", "delta": {"text": "x"}}),
        chunk({"type": "message_delta", "usage": {"output_tokens": 42}}),
    ]
    u = from_stream_events(events)
    assert u.cache_write == 900 and u.output == 42


def test_usage_without_cache_fields_and_string_numbers():
    u = from_response({"usage": {"inputTokens": "120", "outputTokens": 3}})
    assert u.uncached_input == 120 and u.cache_read == 0
    assert from_response("garbage") is None
    assert from_stream_events(None) is None


# --- logs formats ---------------------------------------------------------

def _rec(read, write, uncached, arn="arn:role/a", model="m"):
    return {"schemaType": "ModelInvocationLog", "modelId": model, "identity": {"arn": arn},
            "output": {"outputBodyJson": {"usage": {"inputTokens": uncached, "cacheReadInputTokens": read, "cacheWriteInputTokens": write}}}}


def test_cloudwatch_export_lines(tmp_path):
    p = tmp_path / "cw.txt"
    p.write_text("\n".join(f"2026-10-05T10:00:0{i}.000Z {json.dumps(_rec(900, 0, 100))}" for i in range(3)))
    groups = logs.aggregate(logs.iter_records(p))
    assert groups["m"].calls == 3 and groups["m"].usage.cache_read == 2700


def test_filter_log_events_json(tmp_path):
    p = tmp_path / "events.json"
    p.write_text(json.dumps({"events": [{"message": json.dumps(_rec(0, 0, 50))}, {"message": "broken"}]}))
    stats = logs.ReadStats()
    recs = list(logs.iter_records(p, stats))
    assert len(recs) == 1 and stats.bad_lines == 1


def test_json_array_and_string_body_and_stream_body(tmp_path):
    string_body = _rec(0, 0, 0)
    string_body["output"]["outputBodyJson"] = json.dumps({"usage": {"input_tokens": 10, "cache_read_input_tokens": 90}})
    stream_body = {"schemaType": "ModelInvocationLog", "modelId": "m", "identity": {"arn": "r"},
                   "output": {"outputBodyJson": [{"type": "message_start", "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 90}}}]}}
    other = {"schemaType": "SomethingElse"}
    p = tmp_path / "arr.json.gz"
    with gzip.open(p, "wt") as fh:
        json.dump([string_body, stream_body, other], fh)
    groups = logs.aggregate(logs.iter_records(p))
    assert groups["m"].calls == 2 and groups["m"].unparsed == 0 and groups["m"].usage.cache_read == 180


def test_group_by_principal_and_bad_lines(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text("\n".join([json.dumps(_rec(1, 0, 0, arn="a")), "{oops", json.dumps(_rec(1, 0, 0, arn="b"))]))
    stats = logs.ReadStats()
    groups = logs.aggregate(logs.iter_records(p, stats), by="principal")
    assert set(groups) == {"a", "b"} and stats.bad_lines == 1


def test_empty_log_file(tmp_path, capsys):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert cli.main(["logs", str(p)]) == 0
    assert "No invocation log records" in capsys.readouterr().out


# --- probe: streaming and errors ------------------------------------------

class StreamClient:
    def __init__(self, reads):
        self.reads = list(reads)

    def converse_stream(self, modelId, **kw):
        return {"stream": [{"metadata": {"usage": {"inputTokens": 1, "cacheReadInputTokens": self.reads.pop(0), "cacheWriteInputTokens": 0}}}]}

    def invoke_model_with_response_stream(self, modelId, body):
        ev = {"type": "message_start", "message": {"usage": {"input_tokens": 1, "cache_read_input_tokens": self.reads.pop(0)}}}
        return {"body": [{"chunk": {"bytes": json.dumps(ev).encode()}}]}


def test_probe_streaming_both_apis():
    assert probe.probe(conv(system=[{"text": LONG}, CP]), SONNET, client=StreamClient([0, 700]), pause_s=0, stream=True).passed
    body = {"anthropic_version": "v", "messages": []}
    assert not probe.probe(body, SONNET, client=StreamClient([0, 0]), pause_s=0, stream=True).passed


class FakeClientError(Exception):
    def __init__(self, code):
        super().__init__(f"An error occurred ({code})")
        self.response = {"Error": {"Code": code, "Message": "nope"}}


class NoCredentialsError(Exception):
    pass


class FailingClient:
    def __init__(self, exc):
        self.exc = exc

    def converse(self, **kw):
        raise self.exc


@pytest.mark.parametrize("exc,needle", [
    (FakeClientError("AccessDeniedException"), "model access"),
    (FakeClientError("ResourceNotFoundException"), "not found"),
    (FakeClientError("ThrottlingException"), "Throttled"),
    (NoCredentialsError(), "AWS_PROFILE"),
])
def test_probe_errors_are_translated(exc, needle):
    with pytest.raises(probe.ProbeError) as info:
        probe.probe(conv(), SONNET, client=FailingClient(exc), pause_s=0)
    assert needle in str(info.value)


def test_cli_probe_error_exit_code(monkeypatch, tmp_path, capsys):
    req = tmp_path / "r.json"
    req.write_text(json.dumps(conv()))
    monkeypatch.setattr(probe, "make_client", lambda region: object())
    monkeypatch.setattr(probe, "_call", lambda *a, **k: (_ for _ in ()).throw(FakeClientError("AccessDeniedException")))
    assert cli.main(["probe", str(req), "--model", SONNET]) == 2
    assert "AccessDeniedException" in capsys.readouterr().err


def test_cli_probe_explains_nothing_written(monkeypatch, tmp_path, capsys):
    from cachecanary.usage import Usage
    req = tmp_path / "r.json"
    req.write_text(json.dumps(conv()))
    monkeypatch.setattr(probe, "make_client", lambda region: object())
    monkeypatch.setattr(probe, "_call", lambda *a, **k: Usage(100, 0, 0, 1))
    assert cli.main(["probe", str(req), "--model", SONNET]) == 1
    assert "wrote nothing to the cache" in capsys.readouterr().out


def test_cli_probe_cross_region_hint(monkeypatch, tmp_path, capsys):
    from cachecanary.usage import Usage
    calls = iter([Usage(10, 0, 900, 1), Usage(10, 0, 900, 1)])
    req = tmp_path / "r.json"
    req.write_text(json.dumps(conv()))
    monkeypatch.setattr(probe, "make_client", lambda region: object())
    monkeypatch.setattr(probe, "_call", lambda *a, **k: next(calls))
    assert cli.main(["probe", str(req), "--model", SONNET]) == 1
    assert "Cross-region" in capsys.readouterr().out


# --- cli input errors -----------------------------------------------------

def test_cli_missing_and_invalid_files(tmp_path, capsys):
    assert cli.main(["lint", str(tmp_path / "missing.json")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert cli.main(["lint", str(bad)]) == 2
    arr = tmp_path / "arr.json"
    arr.write_text("[]")
    assert cli.main(["lint", str(arr)]) == 2
    assert cli.main(["logs", str(tmp_path / "nope.jsonl")]) == 2
    err = capsys.readouterr().err
    assert "file not found" in err and "not valid JSON" in err and "JSON object" in err


def test_cli_min_hit_range_validated():
    with pytest.raises(SystemExit) as info:
        cli.main(["logs", "x.jsonl", "--min-hit", "5"])
    assert info.value.code == 2


def test_missing_boto3_gives_clear_error(monkeypatch, tmp_path, capsys):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("no boto3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    req = tmp_path / "r.json"
    req.write_text(json.dumps(conv()))
    assert cli.main(["probe", str(req), "--model", SONNET]) == 2
    assert "pip install 'cachecanary[aws]'" in capsys.readouterr().err


class MissingDependencyException(Exception):
    pass


def test_missing_crt_dependency_is_translated(monkeypatch):
    import types
    fake = types.SimpleNamespace(client=lambda *a, **k: (_ for _ in ()).throw(MissingDependencyException("Missing Dependency: login provider")))
    monkeypatch.setitem(__import__("sys").modules, "boto3", fake)
    with pytest.raises(probe.ProbeError) as info:
        probe.make_client("us-west-2")
    assert "boto3[crt]" in str(info.value)


# --- estimate margin (calibrated on live Bedrock, Oct 2026) ----------------

def _sys_of_tokens(n_est):
    return "abcd" * n_est  # estimate_tokens = len // 4


@pytest.mark.parametrize("n_est,expected", [
    (700, "prefix-too-short"),     # 68% of 1,024: clearly below -> error
    (900, "prefix-near-minimum"),  # 88%: real count may be above (estimate runs ~11% low)
    (1100, "prefix-near-minimum"), # 107%: probably fine, still verify
    (1300, None),                  # 127%: clearly above
])
def test_minimum_bands_for_sonnet(n_est, expected):
    req = normalize(conv(system=[{"text": _sys_of_tokens(n_est)}, CP]))
    found = {f.rule: f for f in lint.lint(req)}
    assert ("prefix-too-short" in found) == (expected == "prefix-too-short")
    assert ("prefix-near-minimum" in found) == (expected == "prefix-near-minimum")
    if expected == "prefix-too-short":
        assert found["prefix-too-short"].severity == "error"


def test_live_calibration_case_is_not_a_false_error():
    # Live run (us-west-2, Oct 5 2026): this prompt was ~2,587 estimated / 2,870 real tokens and was
    # cached on Sonnet 4.6 but not on Haiku 4.5 (min 4,096). lint must agree with both outcomes.
    text = "[run c4db70ed] " + " ".join(f"Rule {i}: answer politely and cite section {i}." for i in range(220))
    req = conv(system=[{"text": text}, CP])
    sonnet = {f.rule for f in lint.lint(normalize(req))}
    haiku = {f.rule for f in lint.lint(normalize(req, "us.anthropic.claude-haiku-4-5-20251001-v1:0"))}
    assert not {"prefix-too-short", "prefix-near-minimum"} & sonnet
    assert "prefix-too-short" in haiku


# --- lookback boundary, measured on live Bedrock Oct 30 2026 (scripts/live_lookback_positions.py) ----
# Shapes are lists of (role, [block kinds]); "tu"/"tr"/"text"/"think". Expected hit/miss is what live
# Bedrock did (Sonnet 4.6, two runs each; most shapes also on InvokeModel and on Sonnet 5.5, one run).

def _render_converse(kind, i):
    if kind == "text":
        return {"text": f"note {i}"}
    if kind == "tu":
        return {"toolUse": {"toolUseId": f"t{i}", "name": "x", "input": {}}}
    if kind == "tr":
        return {"toolResult": {"toolUseId": f"t{i}", "content": [{"text": "ok"}]}}
    return {"reasoningContent": {"reasoningText": {"text": "thinking", "signature": "sig"}}}


def _render_invoke(kind, i):
    if kind == "text":
        return {"type": "text", "text": f"note {i}"}
    if kind == "tu":
        return {"type": "tool_use", "id": f"toolu_{i}", "name": "x", "input": {}}
    if kind == "tr":
        return {"type": "tool_result", "tool_use_id": f"toolu_{i}", "content": "ok"}
    return {"type": "thinking", "thinking": "thinking", "signature": "sig"}


def _lookback_shape_pair(shape, api="converse"):
    """A caches one user turn; B is the same turn plus `shape`, with a checkpoint on B's last block."""
    model = "us.anthropic.claude-sonnet-4-6"
    if api == "converse":
        a = {"messages": [{"role": "user", "content": [{"text": "start " * 50}, CP]}]}
        msgs = [{"role": "user", "content": [{"text": "start " * 50}]}]
        for role, kinds in shape:
            msgs.append({"role": role, "content": [_render_converse(k, i) for i, k in enumerate(kinds)]})
        msgs[-1]["content"].append(CP)
    else:
        eph = {"type": "ephemeral"}
        first = {"type": "text", "text": "start " * 50}
        a = {"anthropic_version": "bedrock-2023-05-31", "messages": [{"role": "user", "content": [{**first, "cache_control": eph}]}]}
        msgs = [{"role": "user", "content": [first]}]
        for role, kinds in shape:
            msgs.append({"role": role, "content": [_render_invoke(k, i) for i, k in enumerate(kinds)]})
        msgs[-1]["content"][-1]["cache_control"] = eph
        b = {"anthropic_version": "bedrock-2023-05-31", "messages": msgs}
        return normalize(a, model), normalize(b, model)
    return normalize(a, model), normalize({"messages": msgs}, model)


def _text(n):
    return [("assistant", ["text"]), ("user", ["text"] * n)]


def _parallel(n, tail=0, lead=0):
    return [("assistant", ["text"] * lead + ["tu"] * n), ("user", ["tr"] * n + ["text"] * tail)]


def _sequential(rounds):
    return [turn for _ in range(rounds) for turn in (("assistant", ["tu"]), ("user", ["tr"]))]


def _mixed(m):
    return [("assistant", ["text", "tu"] * m), ("user", ["tr"] * m)]


LIVE_LOOKBACK = [
    # name, shape, positions, live result (Sonnet 4.6 Converse, two runs; most also on InvokeModel and Sonnet 5.5)
    ("20 text blocks", _text(20), 21, "hit"),
    ("21 text blocks", _text(21), 22, "miss"),
    ("23 text blocks", _text(23), 24, "miss"),
    ("10 sequential calls", _sequential(10), 20, "hit"),
    ("11 sequential calls", _sequential(11), 22, "miss"),
    ("11 parallel calls", _parallel(11), 3, "hit"),
    ("40 parallel calls", _parallel(40), 3, "hit"),
    ("5 parallel + 19 text", _parallel(5, tail=19), 21, "hit"),
    ("5 parallel + 20 text", _parallel(5, tail=20), 22, "miss"),
    ("text, 11 parallel, 18 text", _parallel(11, tail=18, lead=1), 21, "hit"),
    ("text, 11 parallel, 19 text", _parallel(11, tail=19, lead=1), 22, "miss"),
    ("text before each of 9 calls", _mixed(9), 20, "hit"),
    ("text before each of 10 calls", _mixed(10), 22, "miss"),   # results run ends on the checkpoint: 2
    ("18 text, 10 parallel", _parallel(10, lead=18), 21, "hit"),
    ("19 text, 10 parallel", _parallel(10, lead=19), 22, "miss"),
    ("18 text, 2 parallel", _parallel(2, lead=18), 21, "hit"),
    ("19 text, 2 parallel", _parallel(2, lead=19), 22, "miss"),
    ("thinking, 5 parallel, 19 text", [("assistant", ["think"] + ["tu"] * 5), ("user", ["tr"] * 5 + ["text"] * 19)], 21, "hit"),
    ("thinking, 5 parallel, 20 text", [("assistant", ["think"] + ["tu"] * 5), ("user", ["tr"] * 5 + ["text"] * 20)], 22, "miss"),
]


@pytest.mark.parametrize("api", ["converse", "invoke"])
@pytest.mark.parametrize("name,shape,positions,live", LIVE_LOOKBACK, ids=[c[0] for c in LIVE_LOOKBACK])
def test_lookback_matches_live_bedrock(api, name, shape, positions, live):
    a, b = _lookback_shape_pair(shape, api)
    assert diff.lookback_positions(b.blocks, a.checkpoint_indexes[-1], b.checkpoint_indexes[-1]) == positions
    reasons = {r.code: r for r in diff.explain(a, b)}
    assert ("lookback-exceeded" in reasons) is (live == "miss")
    if live == "miss":
        assert reasons["lookback-exceeded"].message.startswith(f"{positions} lookback positions were added")


def test_lookback_message_names_blocks_and_the_counting_rule():
    a, b = _lookback_shape_pair(_sequential(11))
    msg = {r.code: r for r in diff.explain(a, b)}["lookback-exceeded"].message
    assert "(22 blocks)" in msg and "21 added still hits, 22 misses" in msg
    assert "thinking blocks don't count" in msg and "parallel tool calls" not in msg


def test_extra_checkpoint_within_reach_avoids_the_miss():
    # 30 text blocks would be 31 positions; a checkpoint after the 10th is 11 positions from A's.
    shape = [("assistant", ["text"]), ("user", ["text"] * 30)]
    a, b = _lookback_shape_pair(shape)
    raw = {"messages": [{"role": "user", "content": [{"text": "start " * 50}]}, {"role": "assistant", "content": [{"text": "note 0"}]},
                        {"role": "user", "content": [{"text": f"note {i}"} for i in range(10)] + [CP]
                         + [{"text": f"note {i}"} for i in range(10, 30)] + [CP]}]}
    b = normalize(raw, "us.anthropic.claude-sonnet-4-6")
    assert "lookback-exceeded" not in {r.code for r in diff.explain(a, b)}


def test_extra_checkpoint_out_of_reach_does_not_help():
    a, _ = _lookback_shape_pair(_text(1))
    raw = {"messages": [{"role": "user", "content": [{"text": "start " * 50}]}, {"role": "assistant", "content": [{"text": "note 0"}]},
                        {"role": "user", "content": [{"text": f"note {i}"} for i in range(24)] + [CP]
                         + [{"text": f"note {i}"} for i in range(24, 30)] + [CP]}]}
    b = normalize(raw, "us.anthropic.claude-sonnet-4-6")
    reasons = {r.code: r for r in diff.explain(a, b)}
    assert reasons["lookback-exceeded"].message.startswith("25 lookback positions were added")


def _positions(messages, model="us.anthropic.claude-sonnet-4-6"):
    blocks = normalize({"messages": messages}, model).blocks
    cps = [i for i, b in enumerate(blocks) if b.checkpoint]
    return diff.lookback_positions(blocks, cps[0], cps[-1])


def test_tool_runs_split_by_text_inside_one_message():
    # tu run (1) + text (1) + tu (1) + tr run (1) + text checkpoint (1)
    assert _positions([
        {"role": "user", "content": [{"text": "x"}, CP]},
        {"role": "assistant", "content": [_render_converse("tu", 0), _render_converse("tu", 1), {"text": "more"}, _render_converse("tu", 2)]},
        {"role": "user", "content": [_render_converse("tr", 0), _render_converse("tr", 1), _render_converse("tr", 2), {"text": "end"}, CP]},
    ]) == 5


def test_tool_result_runs_do_not_merge_across_messages():
    # [tr, tr] then [tr, tr + checkpoint]: run (1) + rest of the second run (1) + checkpoint (1) = 3, not 2
    assert _positions([
        {"role": "user", "content": [{"text": "x"}, CP]},
        {"role": "user", "content": [_render_converse("tr", 0), _render_converse("tr", 1)]},
        {"role": "user", "content": [_render_converse("tr", 2), _render_converse("tr", 3), CP]},
    ]) == 3


def test_lookback_edge_cases_follow_the_rule():
    tu, tr, think = (lambda i: _render_converse("tu", i)), (lambda i: _render_converse("tr", i)), _render_converse("think", 0)
    start = {"role": "user", "content": [{"text": "x"}, CP]}
    # A checkpoint ending a tool-use run counts on its own, like one ending a tool-result run.
    assert _positions([start, {"role": "assistant", "content": [tu(0), tu(1), tu(2), CP]}]) == 2
    # A checkpoint on a thinking block still counts as its position.
    assert _positions([start, {"role": "assistant", "content": [{"text": "a"}, think, CP]}]) == 2
    # Thinking between two tool calls doesn't break their run (not measured live; thinking is skipped).
    assert _positions([start, {"role": "assistant", "content": [tu(0), think, tu(1)]},
                       {"role": "user", "content": [tr(0), tr(1), {"text": "end"}, CP]}]) == 3
    # The previous checkpoint inside a run: the rest of that run starts a new position.
    blocks = normalize({"messages": [{"role": "user", "content": [tr(0), CP, tr(1), tr(2), {"text": "e"}, CP]}]},
                       "us.anthropic.claude-sonnet-4-6").blocks
    assert diff.lookback_positions(blocks, 0, len(blocks) - 1) == 2
    # Nothing added.
    assert diff.lookback_positions(blocks, 0, 0) == 0
    # InvokeModel: cache_control inside the last tool_result's content makes it the checkpoint; a run of 2 ends there.
    inv = normalize({"anthropic_version": "bedrock-2023-05-31", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]},
        {"role": "assistant", "content": [_render_invoke("tu", 0), _render_invoke("tu", 1)]},
        {"role": "user", "content": [_render_invoke("tr", 0),
                                     {"type": "tool_result", "tool_use_id": "toolu_1",
                                      "content": [{"type": "text", "text": "ok", "cache_control": {"type": "ephemeral"}}]}]}]},
        "us.anthropic.claude-sonnet-4-6")
    assert inv.blocks[-1].checkpoint and inv.blocks[-1].kind == "tool_result"
    assert diff.lookback_positions(inv.blocks, 0, len(inv.blocks) - 1) == 3


@pytest.mark.parametrize("fixture", ["strands/agent-turn1.json strands/agent-turn2.json",
                                     "litellm/agent-turn1.json litellm/agent-turn2.json",
                                     "langchain/invoke-turn1.json langchain/invoke-turn2.json"])
def test_real_framework_wide_steps_no_longer_exceed_the_lookback(fixture):
    """Requests Strands, LiteLLM and LangChain really built for 11-12 parallel tool calls; these lost the cache
    on Oct 7 2026 and kept it on live Bedrock on Oct 30."""
    import pathlib
    root = pathlib.Path(__file__).parent / "fixtures"
    a_path, b_path = fixture.split()
    a = normalize(json.loads((root / a_path).read_text()), "us.anthropic.claude-sonnet-4-6")
    b = normalize(json.loads((root / b_path).read_text()), "us.anthropic.claude-sonnet-4-6")
    assert "lookback-exceeded" not in {r.code for r in diff.explain(a, b)}


def test_block_kinds_are_tagged_for_both_apis():
    conv = normalize({"messages": [{"role": "assistant", "content": [_render_converse(k, 0) for k in ("think", "text", "tu")]},
                                   {"role": "user", "content": [_render_converse("tr", 0)]}]})
    assert [b.kind for b in conv.blocks] == ["reasoning", "other", "tool_use", "tool_result"]
    assert [b.message for b in conv.blocks] == [0, 0, 0, 1]
    inv = normalize({"anthropic_version": "bedrock-2023-05-31", "messages": [
        {"role": "assistant", "content": [_render_invoke("think", 0), {"type": "redacted_thinking", "data": "x"}, _render_invoke("tu", 0)]},
        {"role": "user", "content": [_render_invoke("tr", 0), _render_invoke("text", 1)]}]})
    assert [b.kind for b in inv.blocks] == ["reasoning", "reasoning", "tool_use", "tool_result", "other"]
    assert normalize({"system": [{"text": "s"}]}).blocks[0].kind == "other"
