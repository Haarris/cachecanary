"""Every `$ cachecanary ...` example in README.md must print exactly what the README shows."""

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
