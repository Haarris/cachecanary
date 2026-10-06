"""Log-format robustness and the safety checks of scripts/live_logs_test.py (no AWS)."""

import gzip
import importlib.util
import json
from pathlib import Path

from cachecanary import logs

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(read):
    return {"schemaType": "ModelInvocationLog", "modelId": "m", "operation": "Converse", "identity": {"arn": "r"},
            "output": {"outputBodyJson": {"usage": {"inputTokens": 1, "cacheReadInputTokens": read, "cacheWriteInputTokens": 0}}}}


def test_concatenated_records_without_newlines(tmp_path):
    p = tmp_path / "batch.json.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("".join(json.dumps(_rec(i)) for i in (1, 2, 3)))
    recs = list(logs.iter_records(p))
    assert [r["output"]["outputBodyJson"]["usage"]["cacheReadInputTokens"] for r in recs] == [1, 2, 3]


def test_pretty_printed_multiline_record(tmp_path):
    p = tmp_path / "pretty.json"
    p.write_text(json.dumps(_rec(5), indent=2) + "\n" + json.dumps(_rec(6), indent=2))
    assert len(list(logs.iter_records(p))) == 2


def test_offloaded_body_objects_are_not_counted():
    body_object = {"usage": {"input_tokens": 1}, "content": [{"type": "text", "text": "x"}]}
    groups = logs.aggregate([_rec(10), body_object, {"modelId": "m", "operation": "InvokeModel", "output": {}}])
    assert groups["m"].calls == 2 and groups["m"].unparsed == 1  # body object ignored; bare record counted


def test_timestamp_prefix_with_broken_line(tmp_path):
    p = tmp_path / "cw.txt"
    p.write_text(f"2026-10-05T10:00:00Z {json.dumps(_rec(1))}\n2026-10-05T10:00:01Z {{broken\n2026-10-05T10:00:02Z {json.dumps(_rec(2))}\n")
    stats = logs.ReadStats()
    recs = list(logs.iter_records(p, stats))
    assert len(recs) == 2 and stats.bad_lines == 1


# --- live_logs_test safety checks ----------------------------------------

class FakeCW:
    def __init__(self, total):
        self.total = total

    def get_metric_statistics(self, **kw):
        assert kw["Namespace"] == "AWS/Bedrock" and kw["MetricName"] == "Invocations"
        return {"Datapoints": [{"Sum": self.total}] if self.total else []}


class FakeBedrock:
    def __init__(self, config=None):
        self.config = config

    def get_model_invocation_logging_configuration(self):
        return {"loggingConfig": self.config} if self.config else {}


def test_safety_refuses_protected_region():
    mod = _load_script("live_logs_test")
    problems = mod.check_safety("us-east-2", {"us-east-2"}, FakeCW(0), FakeBedrock())
    assert problems and "protected" in problems[0]


def test_safety_refuses_region_with_traffic_or_existing_logging():
    mod = _load_script("live_logs_test")
    assert any("invocations" in p for p in mod.check_safety("us-west-2", {"us-east-2"}, FakeCW(12), FakeBedrock()))
    assert any("already has invocation logging" in p for p in
               mod.check_safety("us-west-2", {"us-east-2"}, FakeCW(0), FakeBedrock({"s3Config": {"bucketName": "x"}})))


def test_safety_allows_idle_unconfigured_region():
    mod = _load_script("live_logs_test")
    assert mod.check_safety("us-west-2", {"us-east-2"}, FakeCW(0), FakeBedrock()) == []


def test_bucket_policy_scoped_to_account_and_region():
    mod = _load_script("live_logs_test")
    pol = json.loads(mod.bucket_policy("b", "111122223333", "us-west-2"))
    st = pol["Statement"][0]
    assert st["Principal"] == {"Service": "bedrock.amazonaws.com"} and st["Action"] == ["s3:PutObject"]
    assert st["Resource"] == ["arn:aws:s3:::b/AWSLogs/111122223333/BedrockModelInvocationLogs/*"]
    assert st["Condition"]["StringEquals"]["aws:SourceAccount"] == "111122223333"
    assert st["Condition"]["ArnLike"]["aws:SourceArn"] == "arn:aws:bedrock:us-west-2:111122223333:*"


def test_synthetic_payloads_are_valid_and_cacheable():
    from cachecanary import lint
    from cachecanary.request import normalize
    mod = _load_script("live_logs_test")
    payloads = mod.synthetic_payloads("abc")
    assert [p[0] for p in payloads] == ["converse", "converse-stream", "invoke", "invoke-stream"]
    texts = set()
    for name, payload, _ in payloads:
        req = normalize(payload, mod.SONNET)
        assert not [f for f in lint.lint(req) if f.severity == "error"], name
        texts.add(req.blocks[0].content)
    assert len(texts) == 4  # distinct prefixes so each pair's cache counts are attributable


# --- cleanup robustness ---------------------------------------------------

class _Args:
    keep = False
    region = "us-west-2"


class FlakyS3:
    """delete_bucket fails once (late log delivery), then succeeds."""
    def __init__(self):
        self.deleted, self.fail_next = False, True

    def get_paginator(self, name):
        class P:
            def paginate(self_inner, Bucket):
                return [{"Contents": [{"Key": "AWSLogs/x"}]}]
        return P()

    def delete_objects(self, Bucket, Delete):
        pass

    def delete_bucket(self, Bucket):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("BucketNotEmpty")
        self.deleted = True


class BedrockCleanup:
    def __init__(self, fail=False):
        self.fail, self.called = fail, False

    def delete_model_invocation_logging_configuration(self):
        self.called = True
        if self.fail:
            raise RuntimeError("AccessDenied")


def test_cleanup_retries_and_always_runs_both_steps(monkeypatch, capsys):
    mod = _load_script("live_logs_test")
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    s3, bed = FlakyS3(), BedrockCleanup(fail=True)
    mod.cleanup(_Args(), bed, s3, "b", True, True)
    assert bed.called and s3.deleted
    err = capsys.readouterr().err
    assert "CLEANUP INCOMPLETE" in err and "delete-model-invocation-logging-configuration" in err


def test_cleanup_noop_when_nothing_created(capsys):
    mod = _load_script("live_logs_test")
    mod.cleanup(_Args(), BedrockCleanup(), FlakyS3(), "b", False, False)
    assert capsys.readouterr().err == ""


def test_allow_invocations_tolerates_exact_known_count_only():
    mod = _load_script("live_logs_test")
    assert mod.check_safety("us-west-2", {"us-east-2"}, FakeCW(24), FakeBedrock(), allowed_invocations=24) == []
    problems = mod.check_safety("us-west-2", {"us-east-2"}, FakeCW(25), FakeBedrock(), allowed_invocations=24)
    assert problems and "25" in problems[0]
    # Protected Region can never be overridden by the allowance.
    assert mod.check_safety("us-east-2", {"us-east-2"}, FakeCW(0), FakeBedrock(), allowed_invocations=10**9)


# --- regression on REAL Bedrock log records (us-west-2, Oct 2026; redacted) -

def test_real_bedrock_log_records_parse_exactly():
    import pytest
    path = ROOT / "tests" / "fixtures" / "bedrock_invocation_logs_2026-10.json"
    if not path.exists():
        pytest.skip("real-log fixture not captured yet (run scripts/make_log_fixture.py after a live log test)")
    recs = list(logs.iter_records(path))
    assert sorted({r["operation"] for r in recs}) == ["Converse", "ConverseStream", "InvokeModel", "InvokeModelWithResponseStream"]
    groups = logs.aggregate(recs)
    total = sum((g.usage for g in groups.values()), start=type(next(iter(groups.values())).usage)())
    assert sum(g.calls for g in groups.values()) == 8
    assert sum(g.unparsed for g in groups.values()) == 0
    # Values the API returned for the same 8 calls during the live run (Oct 6 2026).
    assert (total.cache_read, total.cache_write, total.uncached_input) == (11496, 11496, 80)  # live run 789ad1af
    # Shapes observed live: ConverseStream logs a single aggregated body; the Claude stream logs events.
    by_op = {r["operation"]: r["output"]["outputBodyJson"] for r in recs}
    assert isinstance(by_op["ConverseStream"], dict) and isinstance(by_op["InvokeModelWithResponseStream"], list)
    assert all("inferenceRegion" in r for r in recs)


# --- model ID forms observed in real logs ---------------------------------

def test_logs_merge_short_id_and_invoke_arn_for_same_model():
    def rec(model):
        r = _rec(10)
        r["modelId"] = model
        return r
    arn = "arn:aws:bedrock:us-west-2:111122223333:inference-profile/us.anthropic.claude-sonnet-4-6"
    groups = logs.aggregate([rec("us.anthropic.claude-sonnet-4-6"), rec(arn)])
    assert list(groups) == ["us.anthropic.claude-sonnet-4-6"] and groups["us.anthropic.claude-sonnet-4-6"].calls == 2


def test_canonical_model_id_forms():
    from cachecanary.models import canonical_model_id, is_profile_arn
    assert canonical_model_id("arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-sonnet-4-6") == "anthropic.claude-sonnet-4-6"
    app = "arn:aws:bedrock:us-west-2:111122223333:application-inference-profile/abc123"
    assert canonical_model_id(app) == app and is_profile_arn(app)
    sysarn = "arn:aws:bedrock:us-west-2:111122223333:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert not is_profile_arn(sysarn)
    assert canonical_model_id(None) is None and canonical_model_id("us.x") == "us.x"


def test_lint_understands_system_profile_arn():
    from cachecanary import lint
    from cachecanary.request import normalize
    sysarn = "arn:aws:bedrock:us-west-2:111122223333:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    req = normalize({"system": [{"text": "short"}, {"cachePoint": {"type": "default"}}], "messages": []}, sysarn)
    found = {f.rule for f in lint.lint(req)}
    assert "profile-arn" not in found and "prefix-too-short" in found  # model recognised -> minimum checked


def test_real_logs_group_both_apis_under_one_model():
    path = ROOT / "tests" / "fixtures" / "bedrock_invocation_logs_2026-10.json"
    groups = logs.aggregate(list(logs.iter_records(path)))
    # Converse logs the short ID, InvokeModel logs the profile ARN; both must land in one group.
    assert list(groups) == ["us.anthropic.claude-sonnet-4-6"] and groups["us.anthropic.claude-sonnet-4-6"].calls == 8
