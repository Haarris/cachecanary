"""GitHub Actions output: inline annotations (workflow commands) and a job summary.

Annotations appear on the pull request / run page; the summary is Markdown appended to the
file named by $GITHUB_STEP_SUMMARY. Both are no-ops outside GitHub Actions apart from printing.
"""

import os


def _escape_data(value: str) -> str:
    # Per GitHub's workflow-command escaping rules.
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(value: str) -> str:
    return _escape_data(value).replace(":", "%3A").replace(",", "%2C")


def annotation(level: str, message: str, file: str | None = None, title: str | None = None) -> str:
    """level: 'error' | 'warning' | 'notice'."""
    if level not in ("error", "warning", "notice"):
        raise ValueError(f"unknown annotation level: {level}")
    props = []
    if file:
        props.append(f"file={_escape_property(file)}")
    if title:
        props.append(f"title={_escape_property(title)}")
    head = f"::{level}" + (" " + ",".join(props) if props else "")
    return f"{head}::{_escape_data(message)}"


def append_summary(markdown: str) -> bool:
    """Append to the job summary. Returns False when not running in GitHub Actions."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return False
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(markdown.rstrip() + "\n\n")
    return True


def md_cell(value) -> str:
    """Make a value safe inside a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(md_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(out)
