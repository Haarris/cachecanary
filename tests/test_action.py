"""The GitHub Action definition and its run script, executed locally with bash."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ACTION = yaml.safe_load((ROOT / "action.yml").read_text())
WORKFLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())


def run_step_script() -> str:
    return next(s for s in ACTION["runs"]["steps"] if s.get("name") == "Run CacheCanary")["run"]


def run_action(command: str, args: str, tmp_path: Path) -> subprocess.CompletedProcess:
    """Execute the action's run block exactly as the runner would (bash, env-passed inputs)."""
    bindir = Path(sys.executable).parent  # venv bin with the `cachecanary` entry point
    env = {**os.environ, "CC_COMMAND": command, "CC_ARGS": args,
           "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}", "GITHUB_STEP_SUMMARY": str(tmp_path / "s.md")}
    return subprocess.run(["bash", "-e", "-c", run_step_script()], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=60)


def test_action_metadata():
    assert ACTION["runs"]["using"] == "composite"
    assert set(ACTION["inputs"]) == {"command", "args", "python-version"}
    assert ACTION["inputs"]["command"]["required"] and ACTION["inputs"]["args"]["required"]
    for step in ACTION["runs"]["steps"]:
        if "run" in step:
            assert step["shell"] == "bash"
            assert "${{" not in step["run"], "inputs must be passed via env, never inlined into scripts"
    assert ACTION["branding"]["icon"] and ACTION["branding"]["color"]


def test_workflow_references_existing_files():
    steps = WORKFLOW["jobs"]["action-selftest"]["steps"]
    for step in steps:
        for token in (step.get("with") or {}).get("args", "").split():
            if token.endswith(".json"):
                assert (ROOT / token).exists(), token
    assert set(WORKFLOW["jobs"]["tests"]["strategy"]["matrix"]["python-version"]) >= {"3.10", "3.13"}


def test_action_script_pass_and_fail(tmp_path):
    ok = run_action("lint", "examples/request_good.json", tmp_path)
    assert ok.returncode == 0, ok.stderr
    bad = run_action("lint", "examples/request_too_short.json", tmp_path)
    assert bad.returncode == 1 and "::error file=examples/request_too_short.json" in bad.stdout
    diff = run_action("diff", "examples/request_bad.json examples/request_next.json", tmp_path)
    assert diff.returncode == 1 and "system-changed" in diff.stdout
    assert "### CacheCanary lint" in (tmp_path / "s.md").read_text()


def test_action_rejects_unknown_command(tmp_path):
    res = run_action("rm", "-rf /", tmp_path)
    assert res.returncode == 2 and "unknown command 'rm'" in res.stdout


@pytest.mark.parametrize("payload", [
    "examples/request_good.json; touch INJECTED",
    "examples/request_good.json $(touch INJECTED)",
    "examples/request_good.json `touch INJECTED`",
    "examples/request_good.json && touch INJECTED",
])
def test_action_args_are_never_executed(tmp_path, payload):
    marker = ROOT / "INJECTED"
    marker.unlink(missing_ok=True)
    res = run_action("lint", payload, tmp_path)
    assert not marker.exists(), f"shell injection via args: {payload}"
    assert res.returncode != 0  # extra tokens are rejected by the CLI as unknown arguments


# --- release workflow ------------------------------------------------------

PUBLISH = yaml.safe_load((ROOT / ".github" / "workflows" / "publish.yml").read_text())


def test_publish_workflow_is_least_privilege():
    on = PUBLISH[True] if True in PUBLISH else PUBLISH["on"]  # YAML 1.1 parses `on:` as True
    assert on["push"]["tags"] == ["v*.*.*"] and "workflow_dispatch" in on
    assert PUBLISH["permissions"] == {"contents": "read"}
    jobs = PUBLISH["jobs"]
    assert "permissions" not in jobs["build"]  # build never gets the OIDC token
    for name, env, url in (("publish-pypi", "pypi", "https://pypi.org/p/cachecanary"),
                           ("publish-testpypi", "testpypi", "https://test.pypi.org/p/cachecanary")):
        job = jobs[name]
        assert job["permissions"] == {"id-token": "write"} and job["needs"] == "build"
        assert job["environment"] == {"name": env, "url": url}
        assert any(str(s.get("uses", "")).startswith("pypa/gh-action-pypi-publish@") for s in job["steps"])
    assert jobs["publish-testpypi"]["steps"][-1]["with"]["repository-url"] == "https://test.pypi.org/legacy/"
    assert "refs/tags/v" in jobs["publish-pypi"]["if"]


def test_version_is_single_sourced():
    import cachecanary
    text = (ROOT / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in text and 'attr = "cachecanary.__version__"' in text
    assert cachecanary.__version__.count(".") == 2


def test_cli_version_flag(capsys):
    import cachecanary
    from cachecanary import cli
    with pytest.raises(SystemExit) as info:
        cli.main(["--version"])
    assert info.value.code == 0 and cachecanary.__version__ in capsys.readouterr().out


def test_license_files_and_metadata():
    lic = (ROOT / "LICENSE").read_text()
    assert "Apache License" in lic and "Version 2.0, January 2004" in lic
    assert "Copyright 2026 Haris Farooq" in (ROOT / "NOTICE").read_text()
    text = (ROOT / "pyproject.toml").read_text()
    assert 'license = "Apache-2.0"' in text and 'license-files = ["LICENSE", "NOTICE"]' in text
