"""GitHub Actions output: annotation escaping, per-command annotations and job summary."""

import json

import pytest

from cachecanary import cli, probe
from cachecanary import github as gh
from cachecanary.usage import Usage

LONG = "Policy text that never changes. " * 400
CP = {"cachePoint": {"type": "default"}}


def req(system_text=LONG, model="us.anthropic.claude-sonnet-4-6"):
    return {"modelId": model, "system": [{"text": system_text}, CP],
            "messages": [{"role": "user", "content": [{"text": "q"}]}]}


@pytest.fixture
def summary(tmp_path, monkeypatch):
    path = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))
    return path


def write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


# --- escaping -------------------------------------------------------------

def test_annotation_escaping():
    line = gh.annotation("warning", "50% done\nnext line", file="dir/a,b:c.json", title="Rule: x, y")
    assert line == "::warning file=dir/a%2Cb%3Ac.json,title=Rule%3A x%2C y::50%25 done%0Anext line"


def test_annotation_rejects_unknown_level():
    with pytest.raises(ValueError):
        gh.annotation("fatal", "x")


def test_summary_noop_outside_actions(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert gh.append_summary("hello") is False


def test_table_escapes_pipes_and_newlines():
    out = gh.table(["a", "b"], [["x|y", "line1\nline2"]])
    assert "x\\|y" in out and "line1 line2" in out


# --- lint -----------------------------------------------------------------

def test_lint_github_annotations_and_summary(tmp_path, capsys, summary):
    path = write(tmp_path, "r.json", req(system_text="tiny prompt 2026-10-05"))
    assert cli.main(["--github", "lint", path]) == 1
    out = capsys.readouterr().out
    assert f"::error file={path},title=CacheCanary%3A prefix-too-short::" in out
    assert "::warning" in out and "dynamic-in-prefix" in out
    text = summary.read_text()
    assert "### CacheCanary lint" in text and "| error | prefix-too-short |" in text


def test_lint_clean_summary(tmp_path, capsys, summary):
    path = write(tmp_path, "r.json", req())
    assert cli.main(["--github", "lint", path]) == 0
    assert "::" not in capsys.readouterr().out.split("No caching problems found.")[1]
    assert "No caching problems found. ✅" in summary.read_text()


def test_github_flag_off_prints_no_workflow_commands(tmp_path, capsys, summary):
    path = write(tmp_path, "r.json", req(system_text="tiny"))
    cli.main(["lint", path])
    assert "::error" not in capsys.readouterr().out
    assert not summary.exists()


# --- diff -----------------------------------------------------------------

def test_diff_github(tmp_path, capsys, summary):
    a = write(tmp_path, "a.json", req(system_text=LONG + " A"))
    b = write(tmp_path, "b.json", req(system_text=LONG + " B"))
    assert cli.main(["--github", "diff", a, b]) == 1
    out = capsys.readouterr().out
    assert f"::error file={b},title=CacheCanary%3A system-changed::" in out
    assert "| system-changed |" in summary.read_text()


def test_diff_identical_is_notice(tmp_path, capsys, summary):
    a = write(tmp_path, "a.json", req())
    assert cli.main(["--github", "diff", a, a]) == 0
    assert "::notice" in capsys.readouterr().out


# --- probe ----------------------------------------------------------------

def test_probe_github_fail_and_pass(tmp_path, capsys, summary, monkeypatch):
    path = write(tmp_path, "r.json", req())
    monkeypatch.setattr(probe, "make_client", lambda region: object())
    monkeypatch.setattr(probe.time, "sleep", lambda s: None)
    calls = iter([Usage(10, 0, 0, 1), Usage(10, 0, 0, 1), Usage(10, 0, 900, 1), Usage(10, 900, 0, 1)])
    monkeypatch.setattr(probe, "_call", lambda *a, **k: next(calls))
    assert cli.main(["--github", "probe", path, "--model", "us.anthropic.claude-sonnet-4-6"]) == 1
    out = capsys.readouterr().out
    assert "::error" in out and "wrote nothing to the cache" in out
    assert cli.main(["--github", "probe", path, "--model", "us.anthropic.claude-sonnet-4-6"]) == 0
    text = summary.read_text()
    assert "FAIL ❌" in text and "PASS ✅" in text and "| 2nd call | 10 | 900 | 0 |" in text


# --- logs -----------------------------------------------------------------

def test_logs_github_threshold(tmp_path, capsys, summary):
    rec = {"schemaType": "ModelInvocationLog", "modelId": "m", "identity": {"arn": "r"},
           "output": {"outputBodyJson": {"usage": {"inputTokens": 900, "cacheReadInputTokens": 100, "cacheWriteInputTokens": 0}}}}
    path = tmp_path / "l.jsonl"
    path.write_text(json.dumps(rec))
    assert cli.main(["--github", "logs", str(path), "--min-hit", "0.5"]) == 1
    out = capsys.readouterr().out
    assert "::error title=CacheCanary%3A low cache hit rate::m: cache hit rate 10%25 is below 50%25" in out  # % is escaped per GitHub rules
    assert "| m | 1 | 10% |" in summary.read_text()


# --- errors ---------------------------------------------------------------

def test_error_annotation(tmp_path, capsys):
    assert cli.main(["--github", "lint", str(tmp_path / "missing.json")]) == 2
    out = capsys.readouterr().out
    assert "::error title=CacheCanary could not run::file not found" in out
