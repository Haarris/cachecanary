"""Every `$ cachecanary ...` example in README.md must print exactly what the README shows."""

import contextlib
import io
import os
import re
import shlex
from pathlib import Path

import pytest

from cachecanary import cli

ROOT = Path(__file__).resolve().parents[1]


def _examples():
    text = (ROOT / "README.md").read_text()
    for block in re.findall(r"```text\n(.*?)```", text, re.S):
        lines = block.strip().split("\n")
        i = 0
        while i < len(lines):
            if lines[i].startswith("$ cachecanary "):
                cmd = lines[i][len("$ cachecanary "):]
                expected = []
                i += 1
                while i < len(lines) and not lines[i].startswith("$ "):
                    if lines[i].strip():
                        expected.append(lines[i])
                    i += 1
                yield cmd, expected
            else:
                i += 1


EXAMPLES = list(_examples())


def test_readme_has_examples():
    assert len(EXAMPLES) >= 4


@pytest.mark.parametrize("cmd,expected", EXAMPLES, ids=[c for c, _ in EXAMPLES])
def test_readme_example_output(cmd, expected, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    cli.main(shlex.split(cmd))
    got = [line for line in capsys.readouterr().out.strip().split("\n") if line.strip()]
    assert got == expected


DEMO_URL = "https://raw.githubusercontent.com/Haarris/cachecanary/main/docs/demo.png"


def test_readme_demo_image_is_in_the_repo():
    """The screenshot under the title loads on GitHub and PyPI (absolute URL) and the file it points to is committed."""
    text = (ROOT / "README.md").read_text()
    top = text.split("## Install")[0]
    match = re.search(r"^!\[([^\]]+)\]\((\S+)\)$", top, re.M)  # a standalone image line, not a badge
    assert match, "no image near the top of the README"
    alt, url = match.groups()
    assert url == DEMO_URL
    assert len(alt) > 30  # describes what the image shows, for screen readers
    png = (ROOT / "docs" / "demo.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    assert 800 <= width <= 1800 and height < width
    assert len(png) < 500_000


def test_demo_script_commands_are_real_and_shown_in_the_readme():
    """The screenshot is a real run: every command the script captures prints the same lines the README documents."""
    script = (ROOT / "scripts" / "demo_screenshot.sh").read_text()
    cmds = re.findall(r"^  run (.+)$", script, re.M)
    assert len(cmds) == 3
    documented = {line for _, expected in EXAMPLES for line in expected}
    for cmd in cmds:
        out = [line for line in _run(cmd) if line.strip()]
        assert out and all(line in documented for line in out), (cmd, out)


def _run(cmd):
    buf = io.StringIO()
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        with contextlib.redirect_stdout(buf):
            try:
                cli.main(shlex.split(cmd))
            except SystemExit:
                pass
    finally:
        os.chdir(cwd)
    return buf.getvalue().strip().split("\n")


def test_readme_points_users_to_discussions():
    assert "https://github.com/Haarris/cachecanary/discussions" in (ROOT / "README.md").read_text()
