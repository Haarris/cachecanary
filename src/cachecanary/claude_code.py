"""Where Claude Code spends: dollars per tool output, Bash command family, MCP server, skill and
subagent, read from the local transcripts (`cachecanary claude-code`). It reads files only: nothing
is sent anywhere and nothing is changed.

Transcripts
    $CLAUDE_CONFIG_DIR/projects (default ~/.claude/projects)/<project>/<session>.jsonl is the main chat;
    <project>/<session>/subagents/**/agent-*.jsonl are subagents, most with an agent-*.meta.json naming
    the agent type and the Agent call that started it (toolUseId). The format is internal to Claude
    Code and changes between versions (checked on 2.1.266 to 2.1.325), so only the core fields are
    required; unknown line types and fields are counted and skipped, never fatal.

Requests
    Claude Code writes one line per content block, so one API request is several assistant lines with
    the same (message.id, requestId), and a usage that grows as it streams. Each request counts once
    (also across files); for every usage field the largest value seen is kept (normally the last line;
    a later line with a smaller number is counted as odd). A line repeated with the same uuid keeps its
    first place and its last content. Entries with no prompt tokens ("<synthetic>", local errors) are
    not API calls.

Context
    What a request sent is its parent request's context plus what was added since. The parent is the
    nearest earlier request on the request's parentUuid chain. Normally that is the previous request in
    the file, and everything written between the two counts as added (parallel tool results hang off
    their own tool call, so they sit beside the chain, not on it). After a rewind or branch the context
    continues from the older request, with the lines on the chain. A compaction line starts again from
    the starting context, the summary and the messages the compaction kept.

Sizes
    P = uncached input + cache reads + cache writes. A step adds P minus the parent's P. The previous
    request's own output comes back as input at its exact output_tokens (thinking included: its size is
    output_tokens_details.thinking_tokens, as most thinking blocks are logged with no text), so it is
    charged first. The rest of the step is split across the other added items by character share.
    Items with no characters (images, documents, tool_reference results that load tool definitions)
    get what is left after the text items take their estimated size (TOKENS_PER_CHAR). The first
    request's P minus what it added is the starting context (system prompt, tools, CLAUDE.md, memory).
    A thinking_drop attachment removes earlier thinking. At a model change (another tokenizer) the old
    items are rescaled to the new prompt size. A prompt that shrinks for no known reason is counted as
    an unexplained drop and nothing is removed.

Dollars
    A prompt is a cached prefix, so each request is charged by position: its first cache_read tokens
    (the oldest items) at the cache read price, then 1-hour writes, then 5-minute writes, then
    uncached input. A new item pays its write once and a read on every later request until it leaves
    the context: its lasting cost. Output dollars go to the blocks the request produced. Web search
    fees come from Claude Code's own cost-state lines. Attributed dollars add up to the total.
    Prices are Anthropic API list prices (Bedrock prices for Bedrock model IDs): an API list price
    equivalent, not a subscription bill.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
from bisect import bisect_left, bisect_right
from functools import lru_cache
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from cachecanary import pricing

SCHEMA = "cachecanary.claude-code/1"
TESTED_VERSIONS = ((2, 1, 266), (2, 1, 325))  # Claude Code versions whose transcripts this was checked on

# Tokens per character of transcript text: used only to share one step between text items and items
# with no characters, and to size what a context's first request added. Measured on real transcripts
# (Nov 2026): 0.39 to 0.49 per rendered attachment type, median 0.44 for tool results.
TOKENS_PER_CHAR = 0.45
# Images and documents in a first request or right after a compaction, where there is no leftover to
# take: Anthropic's docs put a 1.15 megapixel image at about 1,600 tokens.
IMAGE_TOKENS = 1600
REBUILD_SHARE = 0.5        # a request that failed to read at least half of the prefix it shares with its parent
REBUILD_MIN_TOKENS = 1024
TTL_5M, TTL_1H = 300, 3600
# Token counts are whole numbers, so a leftover under half a token is float rounding from splitting a
# step into shares, never content: it creates no item and takes no branch. Float sums use math.fsum,
# which is exact and gives the same result on every Python version (3.12 changed sum() for floats).
EPS_TOKENS = 0.5
fsum = math.fsum
WEB_SEARCH_USD = 10 / 1000  # Anthropic: $10 per 1,000 searches

USED_TYPES = {"user", "assistant", "system", "attachment"}
_TYPE_PREFIX = re.compile(rb'^\{"type":"([A-Za-z0-9_-]+)"')

# Groups ("what") for things that are not tool output.
START = "starting context (system prompt, tools, CLAUDE.md, memory)"
START_SUB = "subagent starting context (system prompt, tools, task)"
SUMMARY = "compaction summary"
THINKING = "Claude's thinking"
REPLIES = "Claude's replies"
UNLOGGED = "growth not in the transcript"
WEB_FEES = "WebSearch"
TEXT_GROUPS = {
    "prompt": "your messages",
    "meta": "injected context",
    "reminder": "system reminders",
    "task": "background task notices",
    "command": "slash commands",
    "skill": "skill instructions",
    "other": "other content",
}
# Attachments that are rendered into the prompt (lines with a "rendered" field: calibrated on steps
# where one attachment type was most of what was added, 0.33 to 0.49 tokens per character; lines
# without it added about 12 to 15 tokens, so they are not sent). Types not listed here still count
# when rendered, under "other attachments", and are listed as unknown.
ATTACHMENT_GROUPS = {
    "instructions": "CLAUDE.md and instructions",
    "nested_memory": "CLAUDE.md and instructions",
    "skill_listing": "skill listing",
    "invoked_skills": "skill instructions",
    "mcp_instructions_delta": "MCP server instructions",
    "deferred_tools_delta": "deferred tool list",
    "deferred_tools_record": "deferred tool list",
    "agent_listing_delta": "subagent list",
    "environment": "environment and session info",
    "session_context": "environment and session info",
    "model": "environment and session info",
    "date": "environment and session info",
    "auto_mode": "environment and session info",
    "ultra_effort_enter": "environment and session info",
    "ultra_effort_exit": "environment and session info",
    "remote_session_change": "environment and session info",
    "credential_org": "environment and session info",
    "total_tokens_reminder": "system reminders",
    "silent_turn_reminder": "system reminders",
    "critical_system_reminder": "system reminders",
    "batching_reminder_sent": "system reminders",
    "read_truncation_notice": "system reminders",
    "task_status": "system reminders",
    "todo_reminder": "system reminders",
    "hook_success": "hook output",
    "hook_additional_context": "hook output",
    "hook_non_blocking_error": "hook output",
    "hook_blocking_error": "hook output",
    "async_hook_response": "hook output",
    "edited_text_file": "file changes and mentions",
    "file": "file changes and mentions",
    "compact_file_reference": "file changes and mentions",
    "directory": "file changes and mentions",
    "selected_lines_in_ide": "file changes and mentions",
    "queued_command": "your messages",
    "structured_output": "other attachments",
}
THINKING_REMOVED = {"thinking_drop", "thinking_stripped"}
OTHER_ATTACHMENTS = "other attachments"


# --- Bash command families ------------------------------------------------------------------------

SUBCOMMAND_TOOLS = {"git", "gh", "docker", "docker-compose", "kubectl", "helm", "npm", "pnpm", "yarn", "bun", "npx",
                    "uv", "uvx", "pip", "pip3", "poetry", "cargo", "go", "make", "aws", "gcloud", "terraform", "brew",
                    "dotnet", "mvn", "gradle", "systemctl", "conda"}
VALUE_OPTIONS = {
    "git": {"-C", "-c", "--git-dir", "--work-tree", "--namespace"},
    "docker": {"-H", "--host", "--context", "-c", "--config", "-l", "--log-level"},
    "kubectl": {"-n", "--namespace", "--context", "--kubeconfig", "--cluster", "--user", "-s", "--server"},
    "helm": {"-n", "--namespace", "--kube-context", "--kubeconfig"},
    "aws": {"--profile", "--region", "--output", "--endpoint-url", "--query", "--cli-read-timeout",
            "--cli-connect-timeout", "--color", "--ca-bundle"},
    "gcloud": {"--project", "--account", "--configuration", "--format", "--verbosity"},
    "uv": {"--directory", "--project", "--python", "-p", "--cache-dir", "--config-file", "--color"},
    "npm": {"--prefix", "-w", "--workspace"},
    "pnpm": {"-C", "--dir", "--filter", "-F"},
    "yarn": {"--cwd"},
    "make": {"-C", "--directory", "-f", "--file", "--makefile", "-j", "--jobs", "-l", "-I", "-o", "-W"},
    "cargo": {"--manifest-path", "--config", "-Z", "--color"},
    "docker-compose": {"-f", "--file", "-p", "--project-name", "--env-file", "--profile"},
}
WRAPPERS = {
    "sudo": {"-u", "-g", "-h", "-p", "-C", "-D", "-r", "-t", "-U", "-T", "--user", "--group"},
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "gtimeout": {"-s", "--signal", "-k", "--kill-after"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "-n", "-p", "--class", "--classdata"},
    "env": {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"},
    "time": {"-f", "--format", "-o", "--output"},
    "caffeinate": {"-t", "-w"},
    "nohup": set(), "command": set(), "exec": set(), "builtin": set(), "unbuffer": set(),
    "stdbuf": {"-i", "-o", "-e"}, "chronic": set(), "doas": {"-u"},
}
WRAPPER_POSITIONALS = {"timeout": 1, "gtimeout": 1}
RUNNER_VALUE_OPTIONS = {"--with", "--python", "-p", "--project", "--directory", "--env-file", "--extra", "--group",
                        "--package", "--index", "--from", "-w", "--filter", "-F", "-C", "-n", "--name"}
SETUP_COMMANDS = {"cd", "pushd", "popd", "export", "set", "unset", "source", ".", "true", "false", "alias", "ulimit",
                  "umask", "shopt", "trap", "local", "declare", "typeset", "readonly", "read", "test", "[", "[[", "]]",
                  "]", ":", "shift", "wait"}
# Commands that print little or nothing: when another command follows in a sequence (&&, ||, ;, a new
# line), the family is that command's. In a pipeline (`echo x | grep`) they stay the producer.
TRIVIAL_COMMANDS = {"date", "echo", "printf", "mkdir", "touch", "sleep", "rm", "rmdir", "pwd", "clear", "cp", "mv",
                    "ln", "chmod"}
# Shell grammar, not programs: loops and conditions are skipped to the first real command in them.
SHELL_KEYWORDS = {"for", "while", "until", "if", "elif", "else", "then", "do", "done", "fi", "case", "esac", "select",
                  "function", "in", "time", "coproc"}
# Header words whose arguments are not a command (for x in a b c; case $x in; select x in ...).
SHELL_HEADERS = {"for", "case", "select", "function"}
# Runners that run another command: the family keeps both ('uv run pytest').
RUNNERS = {("uv", "run"), ("poetry", "run"), ("pipenv", "run"), ("pdm", "run"), ("hatch", "run"), ("pnpm", "exec"),
           ("npm", "exec"), ("yarn", "exec"), ("bundle", "exec"), ("conda", "run")}
SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\+?=")
_WORD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]*$")
_OPERATOR = re.compile(r"^[;&|()]+$")


_REDIRECT_DUP = re.compile(r"\d*[<>]&(?:\d+|-)?")   # 2>&1  >&2  <&3  >&-
_REDIRECT_ALL = re.compile(r"&>>?")                   # &>/dev/null  &>>log


def _tokens(line: str) -> list[str]:
    # '&' inside a redirection is not the background operator: rewrite those first
    line = _REDIRECT_ALL.sub(" >", _REDIRECT_DUP.sub(" ", line))
    try:
        lex = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
        lex.whitespace_split = True
        lex.commenters = "#"
        return list(lex)
    except ValueError:  # unbalanced quotes: a multi-line string cut at the first line
        out = []
        for part in re.split(r"(&&|\|\||[;|&()])", line):
            part = part.strip()
            if not part:
                continue
            if _OPERATOR.match(part):
                out.append(part)
            else:
                out.extend(p.strip("'\"") for p in part.split() if p.strip("'\""))
        return out


def _segments(tokens: list[str]) -> list[tuple[list[str], bool]]:
    """Simple commands, split at && || ; | & and parentheses (also { } !), each with whether its output
    is piped into the next command."""
    segs, cur = [], []
    for tok in tokens:
        if _OPERATOR.match(tok) or tok in ("{", "}", "!"):
            if cur:
                segs.append((cur, tok in ("|", "|&")))
            elif tok in ("|", "|&") and segs:
                segs[-1] = (segs[-1][0], True)
            cur = []
        else:
            cur.append(tok)
    if cur:
        segs.append((cur, False))
    return segs


def _strip_wrappers(seg: list[str]) -> list[str]:
    i, n = 0, len(seg)
    while i < n:
        tok = seg[i]
        if _ASSIGN.match(tok):
            i += 1
            continue
        name = os.path.basename(tok)
        if name not in WRAPPERS:
            return seg[i:]
        takes, positionals = WRAPPERS[name], WRAPPER_POSITIONALS.get(name, 0)
        i += 1
        while i < n:
            t = seg[i]
            if t == "--":
                i += 1
                break
            if t.startswith("-") and len(t) > 1:
                i += 2 if (t in takes and "=" not in t) else 1
            elif name == "env" and _ASSIGN.match(t):
                i += 1
            elif positionals:
                positionals -= 1
                i += 1
            else:
                break
    return []


def bash_family(command) -> str:
    """'git status', 'python3 -m pytest', 'grep' ...: the program of the first real command, plus its
    subcommand for tools like git, docker, npm, uv and aws. Never paths or arguments."""
    if not isinstance(command, str):
        return "(empty)"
    return _bash_family(command[:4000])


FAMILY_MAX = 40


@lru_cache(maxsize=65536)
def _bash_family(command: str) -> str:
    """Cached; the name is capped and redacted, as it is shown without --details."""
    return redact(_bash_family_raw(command), FAMILY_MAX)


def _bash_family_raw(command: str) -> str:
    text = command.replace("\\\r\n", " ").replace("\\\n", " ")
    # the program is at the start of a line: long lines are cut before tokenizing
    lines = [ln[:300] for ln in (s.strip() for s in text.splitlines()) if ln and not ln.startswith("#")]
    if not lines:
        return "(empty)"
    keyword = None
    trivial = None  # first trivial command seen, kept when nothing more substantial follows
    for line in lines[:50]:
        for seg, piped in _segments(_tokens(line)):
            while seg and seg[0] in SHELL_KEYWORDS:
                keyword = keyword or seg[0]
                if seg[0] in SHELL_HEADERS:
                    seg = []
                    break
                seg = seg[1:]
            if seg and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\(\)", seg[0]):  # name() { ... }: a function definition
                continue
            seg = _strip_wrappers(seg)
            if not seg:
                continue
            prog = os.path.basename(seg[0]) or seg[0]
            if prog in SETUP_COMMANDS or prog in SHELL_KEYWORDS:
                continue
            if prog in TRIVIAL_COMMANDS and not piped:  # `date && pytest`: the cost is pytest's output
                trivial = trivial or _family_of(prog, seg[1:])
                continue
            if prog in SHELLS:
                for k in range(1, len(seg) - 1):
                    if re.fullmatch(r"-[a-z]*c[a-z]*", seg[k]):
                        inner = bash_family(seg[k + 1])
                        return inner if inner not in ("(empty)", "(setup only)") else prog
                return prog
            return _family_of(prog, seg[1:])
    if trivial:
        return trivial
    return f"shell {keyword} (no command found)" if keyword else "(setup only)"


def _family_of(prog: str, args: list[str]) -> str:
    if not _WORD.match(prog):
        return "(other)"
    if re.fullmatch(r"python(\d(\.\d+)?)?", prog):  # python, python3 and python3.12 are one family
        for k, tok in enumerate(args):
            if tok == "-m" and k + 1 < len(args):
                return f"python -m {args[k + 1]}" if _WORD.match(args[k + 1]) else "python"
            if tok.startswith("-m") and len(tok) > 2:
                return f"python -m {tok[2:]}" if _WORD.match(tok[2:]) else "python"
            if not tok.startswith("-"):
                break
        return "python"
    if prog not in SUBCOMMAND_TOOLS:
        return prog
    takes = VALUE_OPTIONS.get(prog, set())
    k = 0
    while k < len(args):
        tok = args[k]
        if tok.startswith(("-", "+")) or (prog == "make" and _ASSIGN.match(tok)):
            k += 2 if (tok in takes and "=" not in tok) else 1
            continue
        if not _WORD.match(tok):
            return prog
        if (prog, tok) in RUNNERS:
            rest = args[k + 1:]
            j = 0
            while j < len(rest) and rest[j].startswith("-"):  # runner options, e.g. uv run --with x --python 3.12
                j += 2 if rest[j] in RUNNER_VALUE_OPTIONS and "=" not in rest[j] else 1
            rest = _strip_wrappers(rest[j:])
            if rest:
                inner_prog = os.path.basename(rest[0]) or rest[0]
                return f"{prog} {tok} {_family_of(inner_prog, rest[1:])}"
        return f"{prog} {tok}"
    return prog


def mcp_parts(tool: str) -> tuple[str, str] | None:
    """mcp__<server>__<tool> -> (server, tool); the tool keeps any further '__'."""
    if not isinstance(tool, str) or not tool.startswith("mcp__"):
        return None
    server, sep, name = tool[5:].partition("__")
    return (server or "?", name if sep and name else "?")


# --- labels for --details (redacted) --------------------------------------------------------------

# Redaction for --details labels and Bash family names: specific secret shapes first, broad rules
# only where a value is clearly a credential. Normal words, paths, branch names and model IDs stay.
_HOME_PATH = re.compile(r"(?<![\w/.~-])(?:/Users|/home)/[^/\s\"']+")
_URL_CREDS = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s@]+@")
_AUTH_HEADER = re.compile(r"(?i)\b(authorization|proxy-authorization|x-api-key|api-key|cookie|x-auth-token)"
                          r"\s*:\s*[^'\"]+")
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{6,}")
_BASIC = re.compile(r"(?i)\bbasic\s+(?=[A-Za-z0-9+/]*[0-9=])[A-Za-z0-9+/]{8,}={0,2}")
_KNOWN_TOKENS = re.compile("|".join([
    r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[0-9A-Z]{16,}",           # AWS access key ids
    r"\bgh[pousr]_[A-Za-z0-9]{16,}", r"\bgithub_pat_\w{20,}",       # GitHub
    r"\bglpat-[\w-]{16,}",                                            # GitLab
    r"\bxox[abposr]-[\w-]{8,}",                                       # Slack
    r"\bnpm_[A-Za-z0-9]{8,}",                                         # npm
    r"\bsk-[A-Za-z0-9][\w-]{2,}",                                     # OpenAI / Anthropic / Stripe style
    r"\b[rp]k_(?:live|test)_[A-Za-z0-9]{8,}",                         # Stripe
    r"\bAIza[\w-]{30,}",                                              # Google API keys
    r"\beyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}",                          # JWTs
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
]))
_FLAG_SECRET = re.compile(r"(?i)--(?:password|passwd|token|secret|api-key|apikey|auth-token|access-token|"
                          r"client-secret|private-key)(?:=|\s+)(?:\"[^\"]*\"|'[^']*'|\S+)")
_MYSQL_PASS = re.compile(r"\b((?:mysql|mysqldump|mysqladmin|mariadb)\b[^|;&]*?\s-p)(?=\S)(?:\"[^\"]*\"|'[^']*'|\S+)")
_LOGIN_PASS = re.compile(r"\b((?:docker|podman|helm|nerdctl)\s+(?:registry\s+)?login\b[^|;&]*?\s-p\s+)(?:\"[^\"]*\"|'[^']*'|\S+)")
_USER_PASS = re.compile(r"(\s(?:-u|--user|--proxy-user)(?:\s+|=))([^\s:'\"]+):(?:\"[^\"]*\"|'[^']*'|\S+)")
_QUERY_SECRET = re.compile(r"(?i)([?&](?:sig|signature|sv|se|sp|token|access_token|refresh_token|key|api_key|apikey|"
                           r"code|secret|client_secret|auth)=)[^&\s'\"]+")
_KV = re.compile(r"(?<![\w-])([A-Za-z_][\w.-]*)([\"']?\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s&;|]+)")
_SECRET_NAME_ENDS = ("token", "secret", "password", "passwd", "pwd", "apikey", "api_key", "api-key", "_key", "-key",
                     ".key", "credential", "credentials", "_auth", "-auth", "_pass")
_SECRET_NAMES = {"key", "pass", "pw", "auth", "secret", "token", "password"}
_EMAIL = re.compile(r"(?<![\w.%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w:/@.-])")
_LONG_MIXED = re.compile(r"(?<![\w/.-])(?=[A-Za-z0-9+/=_-]*[A-Z])(?=[A-Za-z0-9+/=_-]*[a-z])(?=[A-Za-z0-9+/=_-]*\d)"
                         r"[A-Za-z0-9+/=_-]{32,}(?![\w/.-])")
_LONG_HEX = re.compile(r"\b(?:[0-9a-f]{32,39}|[0-9a-f]{41,})\b")  # 40 hex is a git object id: kept


def _kv(m: re.Match) -> str:
    name = m.group(1).lower()
    if name in _SECRET_NAMES or name.endswith(_SECRET_NAME_ENDS):
        return "***"
    return m.group(0)


def redact(text: str, n: int = 60) -> str:
    s = " ".join(str(text).split())
    home = str(Path.home())
    if home not in ("", "/"):
        s = s.replace(home, "~")
    s = _HOME_PATH.sub("~", s)
    s = _URL_CREDS.sub(r"\1***@", s)
    s = _AUTH_HEADER.sub(r"\1: ***", s)
    s = _BEARER.sub("***", s)
    s = _BASIC.sub("***", s)
    s = _KNOWN_TOKENS.sub("***", s)
    s = _FLAG_SECRET.sub("***", s)
    s = _MYSQL_PASS.sub(r"\1***", s)
    s = _LOGIN_PASS.sub(r"\1***", s)
    s = _USER_PASS.sub(r"\1\2:***", s)
    s = _QUERY_SECRET.sub(r"\1***", s)
    s = _KV.sub(_kv, s)
    s = _EMAIL.sub("***", s)
    s = _LONG_MIXED.sub("***", s)
    s = _LONG_HEX.sub("***", s)
    return s if len(s) <= n else s[: n - 3] + "..."


def _label(name: str, inp) -> str | None:
    """Raw material for a --details label: the command's first line or the file (kept short; it is
    redacted by label_for() only when shown)."""
    if not isinstance(inp, dict):
        return None
    if name == "Bash" and isinstance(inp.get("command"), str):
        cmd = inp["command"].lstrip()[:400]
        return cmd.split("\n", 1)[0][:200]
    for key in ("file_path", "notebook_path", "path", "pattern", "url", "query", "skill", "subagent_type"):
        if isinstance(inp.get(key), str):
            return inp[key][:200]
    return None


def label_for(item: "Item") -> str | None:
    """The redacted label shown with --details: env assignments, URL credentials, secrets, long
    tokens and home folders removed."""
    if not item.label:
        return None
    text = item.label
    if item.tool == "Bash":
        text = re.sub(r"(^|\s)[A-Za-z_][A-Za-z0-9_]*=\S*", r"\1", text).strip()
    return redact(text)


# --- reading ---------------------------------------------------------------------------------------

class Item:
    """One thing that sat in a context: a tool result or call, a message, Claude's output, an attachment."""
    __slots__ = ("kind", "group", "tool", "family", "mcp", "skill", "chars", "nochar", "image", "out_tokens",
                 "from_req", "tool_use_id", "label", "tokens", "cost_in", "cost_out", "calls", "ctx",
                 "in_window", "subagent_usd", "line", "model", "label_id")

    def __init__(self, kind: str, group: str, chars: int = 0, nochar: bool = False, image: bool = False):
        self.kind = kind            # start, summary, tool_result, tool_use, thinking, reply, text, attachment, unlogged, fee
        self.group = group          # the "what" in the main table: a tool name or a content group
        self.tool = self.family = self.mcp = self.skill = None
        self.chars, self.nochar, self.image = chars, nochar, image
        self.out_tokens = 0.0
        self.from_req = None
        self.tool_use_id = None
        self.label = None
        self.tokens = 0.0           # size when it entered the context
        self.cost_in = self.cost_out = 0.0
        self.calls = 0              # requests (in the window) it was part of
        self.ctx = None
        self.in_window = False      # entered the context in the window
        self.subagent_usd = 0.0
        self.line = None            # uuid of the line it came from
        self.model = None           # model of the request it first reached
        self.label_id = None        # a tool call's id (tool_use.id)

    @property
    def cost(self) -> float:
        return self.cost_in + self.cost_out


class Request:
    __slots__ = ("key", "model", "ts", "input", "read", "w5", "w1", "output", "thinking", "speed", "geo", "tier",
                 "skill", "blocks", "odd", "in_window", "first_line", "version", "cache_requested", "tip", "dup")

    def __init__(self, key, model, ts, first_line):
        self.key, self.model, self.ts, self.first_line = key, model, ts, first_line
        self.input = self.read = self.w5 = self.w1 = self.output = 0
        self.thinking = None
        self.speed = self.geo = self.tier = self.skill = self.version = self.cache_requested = None
        self.blocks: list[Item] = []
        self.odd = False
        self.in_window = True
        self.tip = None
        self.dup = False

    @property
    def prompt(self) -> int:
        return self.input + self.read + self.w5 + self.w1

    def merge(self, usage: dict) -> None:
        new = _usage_numbers(usage)
        old = (self.input, self.read, self.w5, self.w1, self.output)
        if any(b < a for a, b in zip(old, new)):
            self.odd = True
        self.input, self.read, self.w5, self.w1, self.output = (max(a, b) for a, b in zip(old, new))
        details = usage.get("output_tokens_details")
        if isinstance(details, dict) and "thinking_tokens" in details:
            self.thinking = max(self.thinking or 0, _int(details.get("thinking_tokens")))
        for attr, key in (("speed", "speed"), ("geo", "inference_geo"), ("tier", "service_tier")):
            if isinstance(usage.get(key), str):
                setattr(self, attr, usage[key])


def _int(value) -> int:
    if isinstance(value, str) and value.strip().isdigit():  # "123" from a writer that quotes numbers
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value < 0:
        return 0
    return int(min(value, 1e15))


def _usage_numbers(u: dict) -> tuple[int, int, int, int, int]:
    """(uncached input, cache read, 5m write, 1h write, output). The cache_creation split is trusted
    when it adds up to cache_creation_input_tokens; otherwise the total counts and anything not
    marked 1h is a 5-minute write."""
    write = _int(u.get("cache_creation_input_tokens"))
    cc = u.get("cache_creation")
    w1 = _int(cc.get("ephemeral_1h_input_tokens")) if isinstance(cc, dict) else 0
    w5 = _int(cc.get("ephemeral_5m_input_tokens")) if isinstance(cc, dict) else 0
    if w1 + w5 != write:
        w1 = min(w1, write)
        w5 = write - w1
    return _int(u.get("input_tokens")), _int(u.get("cache_read_input_tokens")), w5, w1, _int(u.get("output_tokens"))


def parse_ts(value) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def _version(value) -> tuple | None:
    if isinstance(value, str) and re.fullmatch(r"\d+(\.\d+){1,3}", value):
        return tuple(int(x) for x in value.split("."))
    return None


@dataclass
class Stats:
    files: int = 0
    files_unreadable: int = 0
    folders_unreadable: int = 0
    files_skipped_orphaned: int = 0
    lines: int = 0
    unreadable_lines: int = 0
    partial_last_lines: int = 0
    repeated_lines: int = 0
    used: Counter = field(default_factory=Counter)           # line types read
    ignored: Counter = field(default_factory=Counter)        # other line types
    odd_requests: int = 0
    not_api_calls: int = 0
    duplicate_requests: int = 0                               # same request id in another file
    no_timestamp: int = 0
    orphan_results: int = 0
    images: int = 0
    attachments_not_rendered: Counter = field(default_factory=Counter)
    unknown_attachments: Counter = field(default_factory=Counter)
    unexplained_drops: int = 0
    unexplained_drop_tokens: int = 0
    branches: int = 0
    broken_chains: int = 0
    model_switches: int = 0
    thinking_removals: int = 0
    subagents_without_meta: int = 0
    ttl_mismatch: int = 0     # asked for the 1-hour cache, wrote only 5-minute entries
    unknown_tiers: Counter = field(default_factory=Counter)
    unknown_speeds: Counter = field(default_factory=Counter)
    unknown_geos: Counter = field(default_factory=Counter)
    versions: Counter = field(default_factory=Counter)
    usage_keys: Counter = field(default_factory=Counter)
    block_types: Counter = field(default_factory=Counter)
    result_shapes: Counter = field(default_factory=Counter)
    system_subtypes: Counter = field(default_factory=Counter)
    attachment_types: Counter = field(default_factory=Counter)
    meta_keys: Counter = field(default_factory=Counter)


@dataclass
class ContextInfo:
    path: Path
    project: str
    session: str
    is_sub: bool
    agent_type: str | None = None
    parent_tool_use_id: str | None = None
    parent_agent_id: str | None = None
    workflow: str | None = None
    agent_id: str | None = None


class Line:
    """One transcript line, reduced to what the attribution needs."""
    __slots__ = ("uuid", "parent", "kind", "items", "req", "compact", "removes_thinking", "pos", "command")

    def __init__(self, uuid, parent, kind, pos):
        self.uuid, self.parent, self.kind, self.pos = uuid, parent, kind, pos
        self.items: list[Item] = []
        self.req: Request | None = None
        self.compact: dict | None = None
        self.removes_thinking = False
        self.command = None


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b["text"] for b in content if isinstance(b, dict) and isinstance(b.get("text"), str))
    return ""


def _text_group(text: str, d: dict) -> str:
    t = text.lstrip()
    if t.startswith(("<command-name>", "<command-message>", "<command-args>", "<local-command-stdout>",
                     "<local-command-stderr>", "<local-command-caveat>")):
        return "command"
    if t.startswith("<system-reminder>"):
        return "reminder"
    if t.startswith("<task-notification>"):
        return "task"
    return "meta" if d.get("isMeta") else "prompt"


_COMMAND_NAME = re.compile(r"<command-name>/?([^<\s]+)</command-name>")


def _skill_name(inp) -> str:
    if isinstance(inp, dict) and isinstance(inp.get("skill"), str) and inp["skill"]:
        return inp["skill"]
    return "(unknown skill)"


def _tool_item(kind: str, name: str, inp, chars: int, nochar: bool = False) -> Item:
    it = Item(kind, name, chars, nochar=nochar)
    it.tool = name
    if name == "Bash":
        it.family = bash_family(inp.get("command") if isinstance(inp, dict) else None)
    it.mcp = mcp_parts(name)
    if name == "Skill":
        it.skill = _skill_name(inp)
    it.label = _label(name, inp)
    return it


def _result_item(block: dict, tools: dict, stats: Stats) -> Item:
    tid = block.get("tool_use_id")
    known = tools.get(tid) if isinstance(tid, str) else None
    if known is None:
        stats.orphan_results += 1
        name, inp = "(unknown tool)", None
    else:
        name, inp = known
    content = block.get("content")
    chars, nochar, image = 0, False, False
    if isinstance(content, str):
        chars = len(content)
        stats.result_shapes["text"] += 1
    elif isinstance(content, list):
        kinds = set()
        for x in content:
            if not isinstance(x, dict):
                continue
            kinds.add(str(x.get("type")))
            if x.get("type") == "text" and isinstance(x.get("text"), str):
                chars += len(x["text"])
            else:
                nochar = True  # image, document, tool_reference: no characters in the transcript
                if x.get("type") in ("image", "document"):
                    image = True
                    stats.images += x.get("type") == "image"
        stats.result_shapes["list:" + ",".join(sorted(kinds))] += 1
    elif content is not None:
        chars = len(json.dumps(content, ensure_ascii=False))
        stats.result_shapes[type(content).__name__] += 1
    it = _tool_item("tool_result", name, inp, chars, nochar=nochar and chars == 0)
    it.image = image
    if nochar and chars:  # text plus an image: the image's tokens go with the text's share
        it.nochar = False
    it.tool_use_id = tid if isinstance(tid, str) else None
    return it


def read_file(path: Path, stats: Stats, seen_requests: set | None = None) -> tuple[list[Line], dict, list]:
    """The transcript's lines in order (repeated uuids keep their first place and last content), its
    requests by key (complete, with all their lines merged), and its web searches as (time, count).

    Web searches run outside the logged requests; only Claude Code's cost-state lines count them, as a
    running total per process run (startTime) with no timestamp of their own. Each increase is dated
    by the last timestamped line before it, so --since can leave out searches made before the window."""
    lines: list[Line] = []
    by_uuid: dict[str, Line] = {}
    requests: dict = {}
    tools: dict[str, tuple[str, object]] = {}
    last_command = None
    searches: list[tuple[float | None, int]] = []
    run_counts: dict = {}
    last_run = None
    last_ts = None
    n = -1
    with open(path, "rb") as fh:
        for raw in fh:  # streamed: transcripts can be hundreds of MB
            n += 1
            if not raw.strip():
                continue
            stats.lines += 1
            m = _TYPE_PREFIX.match(raw)
            if m and m.group(1).decode("ascii", "replace") not in USED_TYPES:
                kind = m.group(1).decode("ascii", "replace")
                stats.ignored[kind] += 1
                if kind == "cost-state":
                    found = _web_searches(raw)
                    if found is not None:
                        run, count = found
                        if run is None:  # no startTime: a lower count means a new run
                            run = last_run if count >= run_counts.get(last_run, 0) else object()
                        before = run_counts.get(run, 0)
                        if count < before:  # a restart that kept the startTime
                            before = 0
                        if count > before:
                            searches.append((last_ts, count - before))
                        run_counts[run], last_run = count, run
                continue
            try:
                d = json.loads(raw)
            except ValueError:
                if not raw.endswith(b"\n"):
                    stats.partial_last_lines += 1  # the last line, being written right now
                else:
                    stats.unreadable_lines += 1
                continue
            if not isinstance(d, dict):
                stats.unreadable_lines += 1
                continue
            line = _read_line(d, n, lines, by_uuid, requests, tools, stats, seen_requests, last_command)
            ts = parse_ts(d.get("timestamp"))
            if ts is not None:
                last_ts = ts
            if line is not None and line.command:
                last_command = line.command
    return lines, requests, searches


def _idpart(value):
    """A request id part: strings (and plain integers) only; anything else counts as missing."""
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _read_line(d: dict, n: int, lines: list, by_uuid: dict, requests: dict, tools: dict, stats: Stats,
               seen_requests: set | None, last_command: str | None) -> Line | None:
    if True:
        t = d.get("type")
        if not isinstance(t, str) or t not in USED_TYPES:
            stats.ignored[str(t)[:40]] += 1
            return
        stats.used[t] += 1
        if _version(d.get("version")):
            stats.versions[d["version"]] += 1
        uid = d.get("uuid") if isinstance(d.get("uuid"), str) else f"#{n}"
        parent = d.get("parentUuid") if isinstance(d.get("parentUuid"), str) else None
        line = Line(uid, parent, t, len(lines))
        old = by_uuid.get(uid)
        if old is not None:  # same uuid again: keep the first place, take the last content
            stats.repeated_lines += 1
            line.pos = old.pos
            lines[old.pos] = line
            if old.req is not None:
                gone = {id(it) for it in old.items}
                old.req.blocks = [b for b in old.req.blocks if id(b) not in gone]
        else:
            lines.append(line)
        by_uuid[uid] = line
        msg = d.get("message")
        if t == "assistant":
            if not isinstance(msg, dict):
                stats.ignored["assistant without message"] += 1
                return
            key = (_idpart(msg.get("id")), _idpart(d.get("requestId")))
            if key[0] is None and key[1] is None:
                key = ("uuid", uid)
            req = requests.get(key)
            if req is None:
                model = msg.get("model") if isinstance(msg.get("model"), str) else None
                req = Request(key, model, parse_ts(d.get("timestamp")), uid)
                req.version = _version(d.get("version"))
                if seen_requests is not None and key[0] != "uuid":
                    req.dup = key in seen_requests
                    seen_requests.add(key)
                requests[key] = req
            line.req = req
            usage = msg.get("usage")
            if isinstance(usage, dict):
                stats.usage_keys.update(k for k in usage)
                req.merge(usage)
            if isinstance(d.get("attributionSkill"), str) and d["attributionSkill"]:
                req.skill = d["attributionSkill"]
            if isinstance(d.get("promptCacheRequested"), str):
                req.cache_requested = d["promptCacheRequested"]
            content = msg.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type")
                stats.block_types[f"assistant:{bt}"] += 1
                if bt == "tool_use":
                    name = block.get("name") if isinstance(block.get("name"), str) else "?"
                    chars = len(json.dumps(block.get("input"), ensure_ascii=False, separators=(",", ":")))
                    it = _tool_item("tool_use", name, block.get("input"), chars)
                    if isinstance(block.get("id"), str):
                        tools[block["id"]] = (name, block.get("input"))
                        it.label_id = block["id"]
                elif bt in ("thinking", "redacted_thinking"):
                    text = block.get("thinking") if isinstance(block.get("thinking"), str) else ""
                    it = Item("thinking", THINKING, len(text))
                elif bt == "text":
                    it = Item("reply", REPLIES, len(block.get("text") or "") if isinstance(block.get("text"), str) else 0)
                else:
                    it = Item("reply", REPLIES, len(json.dumps(block, ensure_ascii=False)))
                it.from_req = req
                it.line = uid
                req.blocks.append(it)
                line.items.append(it)
        elif t == "user":
            if not isinstance(msg, dict):
                stats.ignored["user without message"] += 1
                return
            content = msg.get("content")
            if d.get("isCompactSummary"):
                it = Item("summary", SUMMARY, len(_text_of(content)))
                it.line = uid
                line.items.append(it)
                return
            head = _text_of(content)[:500]
            cmd = _COMMAND_NAME.search(head) if "<command-name>" in head else None
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type")
                stats.block_types[f"user:{bt}"] += 1
                if bt == "tool_result":
                    it = _result_item(block, tools, stats)
                elif bt == "text":
                    text = block.get("text") if isinstance(block.get("text"), str) else ""
                    it = Item("text", TEXT_GROUPS[_text_group(text, d)], len(text))
                    src = d.get("sourceToolUseID")
                    if d.get("isMeta") and isinstance(src, str) and tools.get(src, ("",))[0] == "Skill":
                        it.skill = _skill_name(tools[src][1])
                    elif d.get("isMeta") and text.lstrip().startswith("Base directory for this skill"):
                        it.skill = last_command or "(unknown skill)"
                    if it.skill:
                        it.group = TEXT_GROUPS["skill"]
                elif bt in ("image", "document"):
                    stats.images += bt == "image"
                    it = Item("text", TEXT_GROUPS["prompt"], 0, nochar=True, image=True)
                else:
                    it = Item("text", TEXT_GROUPS["other"], 0, nochar=True)
                it.line = uid
                line.items.append(it)
            if cmd:
                line.command = cmd.group(1)
        elif t == "system":
            sub = d.get("subtype")
            stats.system_subtypes[str(sub)] += 1
            meta = d.get("compactMetadata")
            if isinstance(meta, dict) or sub == "compact_boundary":
                meta = meta if isinstance(meta, dict) else {}
                kept = meta.get("preservedMessages")
                kept_ids = kept.get("allUuids") if isinstance(kept, dict) else None
                line.compact = {"trigger": meta.get("trigger") if isinstance(meta.get("trigger"), str) else None,
                                "pre_tokens": _int(meta.get("preTokens")) or None,
                                "post_tokens": _int(meta.get("postTokens")) or None,
                                "ts": parse_ts(d.get("timestamp")),
                                "kept": [u for u in kept_ids if isinstance(u, str)] if isinstance(kept_ids, list) else []}
        elif t == "attachment":
            att = d.get("attachment")
            kind = att.get("type") if isinstance(att, dict) and isinstance(att.get("type"), str) else "(none)"
            stats.attachment_types[kind] += 1
            if kind in THINKING_REMOVED:
                line.removes_thinking = True
            rendered = d.get("rendered")
            if not isinstance(rendered, list):
                stats.attachments_not_rendered[kind] += 1
                return
            chars = sum(len(x["content"]) for x in rendered if isinstance(x, dict) and isinstance(x.get("content"), str))
            if not chars:
                return
            group = ATTACHMENT_GROUPS.get(kind)
            if group is None:
                group = OTHER_ATTACHMENTS
                stats.unknown_attachments[kind] += 1
            it = Item("attachment", group, chars)
            it.line = uid
            line.items.append(it)
    return line


def _web_searches(raw: bytes) -> tuple | None:
    """(process run, running count of web searches) from one cost-state line."""
    try:
        d = json.loads(raw)
    except ValueError:
        return None
    usage = d.get("modelUsage") if isinstance(d, dict) else None
    if not isinstance(usage, dict):
        return None
    run = d.get("startTime") if isinstance(d.get("startTime"), (str, int, float)) else None
    return run, sum(_int(v.get("webSearchRequests")) for v in usage.values() if isinstance(v, dict))


# --- attribution -----------------------------------------------------------------------------------

class Node:
    """An item's place in one context path. A request's context is the path from the root to its tip;
    paths share their common start, so a branch costs nothing extra."""
    __slots__ = ("item", "size", "end", "parent", "rate", "count", "depth")

    def __init__(self, item: Item, size: float, parent: "Node | None"):
        self.item, self.size, self.parent = item, max(size, 0.0), parent
        self.end = (parent.end if parent else 0.0) + self.size
        self.depth = parent.depth + 1 if parent else 0
        self.rate = 0.0   # USD per token, added for every request whose fully-charged range ends here
        self.count = 0    # requests whose tip is here

    @property
    def start(self) -> float:
        return self.end - self.size


class Context:
    """Walks one context's requests in order, sizing items and charging dollars to them."""

    def __init__(self, info: ContextInfo, report: "Report"):
        self.info, self.report = info, report
        self.nodes: list[Node] = []
        self.path: list[Node] = []    # the current request's path, root first
        self.ends: list[float] = []   # end position of each node on it (ascending)
        self.start_item: Item | None = None
        self.model: str | None = None
        self.cost = 0.0
        self.requests = 0
        self.first_ts: float | None = None
        self.last_ts: float | None = None
        self.last_ttl: dict[int, int] = {}

    def _node(self, item: Item, size: float, parent: Node | None, in_window: bool) -> Node:
        node = Node(item, size, parent)
        if item.ctx is None:
            item.ctx = self.info
            item.model = self.model
            item.tokens = node.size
            item.in_window = in_window
            self.report.items.append(item)
        self.nodes.append(node)
        return node

    @staticmethod
    def _path(tip: Node | None) -> list[Node]:
        out = []
        while tip is not None:
            out.append(tip)
            tip = tip.parent
        out.reverse()
        return out

    def finalize(self) -> None:
        """Each node's total rate is the sum over its subtree (a range charge adds at its deepest node
        and subtracts at the node before its first one)."""
        for node in reversed(self.nodes):
            if node.parent is not None:
                node.parent.rate += node.rate
                node.parent.count += node.count
            if abs(node.rate) > 1e-15:  # what is left of +r and -r cancelling is float noise
                node.item.cost_in += node.size * node.rate
            node.item.calls += node.count
        self.nodes, self.path, self.ends = [], [], []

    def _ensure_path(self, tip: Node | None) -> None:
        """Make self.path the root-to-tip path. Normally the tip extends it, so this costs what was
        added; a branch or restart rewalks only the part that differs."""
        path, ends = self.path, self.ends
        if tip is None:
            path.clear()
            ends.clear()
            return
        chain = []
        node = tip
        while node is not None and not (node.depth < len(path) and path[node.depth] is node):
            chain.append(node)
            node = node.parent
        keep = node.depth + 1 if node is not None else 0
        del path[keep:]
        del ends[keep:]
        for x in reversed(chain):
            path.append(x)
            ends.append(x.end)

    # sizing
    @staticmethod
    def split_output(req: Request) -> None:
        """Share the request's exact output tokens across its blocks: thinking blocks get
        output_tokens_details.thinking_tokens; text and tool calls share the rest by characters."""
        blocks = req.blocks
        if not blocks:
            return
        out = float(req.output)
        thinking = [b for b in blocks if b.kind == "thinking"]
        others = [b for b in blocks if b.kind != "thinking"]
        if req.thinking is not None and thinking:
            think = min(float(req.thinking), out)
        elif thinking:
            known = sum(b.chars for b in others) * TOKENS_PER_CHAR
            think = max(out - known, sum(b.chars for b in thinking) * TOKENS_PER_CHAR) if out - known >= EPS_TOKENS else 0.0
            think = min(think, out)
        else:
            think = 0.0
        rest = out - think
        if rest < EPS_TOKENS:
            rest, think = 0.0, out
        tchars = sum(b.chars for b in thinking)
        for b in thinking:
            b.out_tokens = think * (b.chars / tchars if tchars else 1 / len(thinking))
        ochars = sum(b.chars for b in others)
        for b in others:
            b.out_tokens = rest * (b.chars / ochars if ochars else 1 / len(others))
        if not others and rest >= EPS_TOKENS and thinking:
            for b in thinking:
                b.out_tokens += rest / len(thinking)

    @staticmethod
    def _exact(it: Item) -> bool:
        return it.from_req is not None and it.from_req.prompt > 0 and not it.from_req.dup

    def _estimates(self, items: list[Item]) -> list[float]:
        out = []
        for it in items:
            if self._exact(it):
                out.append(it.out_tokens)
            elif it.nochar:
                out.append(IMAGE_TOKENS if it.image else 0.0)
            else:
                out.append(it.chars * TOKENS_PER_CHAR)
        return out

    def _grow_sizes(self, new: list[Item], growth: float) -> tuple[list[Item], list[float]]:
        """Sizes for items added in one step whose prompt grew by `growth` tokens."""
        sizes = [0.0] * len(new)
        exact = [self._exact(it) for it in new]
        a_total = fsum(it.out_tokens for it, e in zip(new, exact) if e)
        if growth >= a_total - EPS_TOKENS:
            pool = max(growth - a_total, 0.0)
            if pool < EPS_TOKENS:
                pool = 0.0
            for k, it in enumerate(new):
                if exact[k]:
                    sizes[k] = it.out_tokens
            u_idx = [k for k in range(len(new)) if not exact[k]]
            char_idx = [k for k in u_idx if not new[k].nochar]
            nochar_idx = [k for k in u_idx if new[k].nochar]
            if not u_idx:
                if pool >= EPS_TOKENS:
                    new = new + [Item("unlogged", UNLOGGED)]
                    sizes.append(pool)
            elif not nochar_idx:
                chars = sum(new[k].chars for k in char_idx)
                for k in char_idx:
                    sizes[k] = pool * (new[k].chars / chars if chars else 1 / len(char_idx))
            else:
                char_est = sum(new[k].chars for k in char_idx) * TOKENS_PER_CHAR
                take = min(char_est, pool)
                for k in char_idx:
                    sizes[k] = new[k].chars * TOKENS_PER_CHAR * (take / char_est if char_est else 0.0)
                left = pool - take
                for k in nochar_idx:
                    sizes[k] = (left if left >= EPS_TOKENS else 0.0) / len(nochar_idx)
        elif growth >= -EPS_TOKENS:
            growth = max(growth, 0.0)
            est = self._estimates(new)
            total = fsum(est)
            for k, e in enumerate(est):
                sizes[k] = growth * (e / total if total else 1 / len(new))
        else:
            sizes = self._estimates(new)
        return new, sizes

    def step(self, req: Request, parent: Request | None, new: list[Item], compaction: dict | None,
             kept: list[Item], remove_thinking: bool) -> None:
        """Size what this request added, place it on the context path, then charge the request.
        `parent` is the request this one continues (for a compaction: the last request before it)."""
        P = float(req.prompt)
        win = req.in_window
        cause = None
        rep = self.report
        self.model = req.model
        restart = compaction is not None and self.start_item is not None
        if restart or parent is None or parent.tip is None:
            if restart:
                cause = "compaction"
                start = self.start_item
                start_size = start.tokens
            else:
                cause = "first call"
                start = Item("start", START_SUB if self.info.is_sub else START)
                self.start_item = start
                start_size = None
            summaries = [it for it in new if it.kind == "summary"]
            others = [it for it in new if it.kind != "summary"]
            kept_sizes = [it.tokens if it.ctx is not None else e for it, e in zip(kept, self._estimates(kept))]
            est = self._estimates(others)
            fixed = fsum(est) + fsum(kept_sizes)
            if start_size is None:  # first call: the start takes what the added items don't
                f = min(1.0, P / fixed) if fixed else 1.0
                start_size = P - fixed * f
            else:
                start_size = min(start_size, P)
                f = min(1.0, (P - start_size) / fixed) if fixed else 1.0
            rest = P - start_size - fixed * f
            rest = rest if rest >= EPS_TOKENS else 0.0
            if restart and not summaries:
                summaries = [Item("summary", SUMMARY)]
            tip = self._node(start, start_size, None, win)
            schars = sum(s.chars for s in summaries)
            for s in summaries:
                tip = self._node(s, rest * (s.chars / schars if schars else 1 / len(summaries)), tip, win)
            if not summaries and rest >= EPS_TOKENS:
                tip = self._node(Item("unlogged", UNLOGGED), rest, tip, win)
            for it, size in zip(kept, kept_sizes):
                tip = self._node(it, size * f, tip, win)
            for it, e in zip(others, est):
                tip = self._node(it, e * f, tip, win)
            if compaction is not None:
                rep.compactions.append({**{k: v for k, v in compaction.items() if k != "kept"},
                                        "session": self.info.session, "subagent": self.info.is_sub,
                                        "prompt_before": parent.prompt if parent else None, "prompt_after": int(P),
                                        "kept_messages": len(compaction.get("kept") or []), "in_window": win})
        else:
            tip = parent.tip
            if remove_thinking:
                path = self._path(tip)
                if any(n.item.kind == "thinking" for n in path):
                    self.report.stats.thinking_removals += 1
                    cause = "thinking removed"
                    tip = None
                    for n in path:
                        if n.item.kind != "thinking":
                            tip = self._node(n.item, n.size, tip, win)
            switched = parent.model != req.model
            if switched:
                rep.stats.model_switches += 1
                cause = "model changed"
                est = self._estimates(new)
                old_total = tip.end if tip else 0.0
                est_total = fsum(est)
                f_new = min(1.0, P / est_total) if est_total else 1.0
                sizes = [e * f_new for e in est]
                f_old = max(P - fsum(sizes), 0.0) / old_total if old_total else 0.0
                path = self._path(tip)
                tip = None
                for n in path:
                    tip = self._node(n.item, n.size * f_old, tip, win)
            else:
                growth = P - float(parent.prompt)
                if cause == "thinking removed":
                    growth = P - (tip.end if tip else 0.0)
                new, sizes = self._grow_sizes(new, growth)
                if growth < -EPS_TOKENS and cause is None:
                    rep.stats.unexplained_drops += win
                    rep.stats.unexplained_drop_tokens += int(-growth) if win else 0
            for it, size in zip(new, sizes):
                tip = self._node(it, size, tip, win)
        req.tip = tip
        self.split_output(req)
        self._charge(req, parent, cause, new)

    def _charge(self, req: Request, parent: Request | None, cause: str | None, new: list[Item]) -> None:
        rep = self.report
        priced = rep.price(req)
        ttl_before = self.last_ttl.get(id(parent), TTL_5M) if parent else TTL_5M
        self.last_ttl[id(req)] = TTL_1H if req.w1 else (TTL_5M if req.w5 else ttl_before)
        if not req.in_window:
            return
        self.requests += 1
        rep.add_request(req, self.info, priced)
        if req.cache_requested == "1h" and req.w5 and not req.w1:
            rep.stats.ttl_mismatch += 1
        r = priced.rates if priced else None
        writes = req.w1 + req.w5
        # cache rebuilds: most of what could have been read was written again
        rebuild = None
        if cause == "first call":  # nothing was cached yet: not a rebuild, reported on its own
            rep.first_calls[0] += 1
            rep.first_calls[1] += writes
            if r:
                rep.first_calls[2] += (req.w1 * r.cache_write_1h + req.w5 * r.cache_write_5m) / 1e6
        elif parent is not None:
            expected = min(float(parent.prompt), float(req.prompt))
            missed = max(expected - req.read, 0.0)
            if expected >= REBUILD_MIN_TOKENS and writes and missed >= REBUILD_SHARE * expected:
                nonread = writes + req.input
                share = min(1.0, missed / nonread) if nonread else 0.0
                if cause is None:
                    gap = req.ts - parent.ts if req.ts is not None and parent.ts is not None else None
                    waited_on_agent = any(it.tool in ("Agent", "Task") and it.kind == "tool_result" for it in new)
                    if (parent.speed or "standard") != (req.speed or "standard"):
                        cause = "fast mode switched"
                    elif parent.version and req.version and parent.version != req.version:
                        cause = "Claude Code version changed"
                    elif gap is not None and gap > ttl_before and waited_on_agent:
                        cause = "waited on a subagent past the cache lifetime"
                    elif gap is not None and gap > ttl_before:
                        cause = "idle over 1h" if ttl_before == TTL_1H else "idle over 5m"
                    else:
                        cause = "unknown"
                rebuild = (cause, req.w1 * share, req.w5 * share, req.input * share)
        if rebuild:
            c, w1, w5, u = rebuild
            extra = 0.0
            if r:
                extra = (w1 * (r.cache_write_1h - r.cache_read) + w5 * (r.cache_write_5m - r.cache_read)
                         + u * (r.input - r.cache_read)) / 1e6
            rep.rebuilds[c][0] += 1
            rep.rebuilds[c][1] += extra
        tip = req.tip
        if tip is not None:
            tip.count += 1
        req_cost = 0.0
        if r:
            in_cost = (req.read * r.cache_read + req.w1 * r.cache_write_1h + req.w5 * r.cache_write_5m
                       + req.input * r.input) / 1e6
            self._charge_positions(req, r)
            out_cost = req.output * r.output / 1e6
            if out_cost:
                total = fsum(b.out_tokens for b in req.blocks)
                if req.blocks:
                    for b in req.blocks:
                        b.cost_out += out_cost * (b.out_tokens / total if total else 1 / len(req.blocks))
                else:  # output with no logged content
                    loose = Item("reply", REPLIES)
                    loose.ctx, loose.in_window, loose.cost_out = self.info, True, out_cost
                    rep.items.append(loose)
            req_cost = in_cost + out_cost
        self.cost += req_cost

    def _charge_positions(self, req: Request, r: pricing.ApiRates) -> None:
        """Charge the prompt by position: reads cover the oldest tokens, then 1h writes, 5m writes and
        uncached input. Each band costs two binary searches on the path: nodes fully inside it get the
        rate through the subtree sums of finalize(), the (at most two) nodes it cuts are charged here."""
        tip = req.tip
        P = float(req.prompt)
        if tip is None or P <= 0:
            return
        self._ensure_path(tip)
        path, ends = self.path, self.ends
        n = len(path)
        scale = tip.end / P  # path tokens per prompt token (1 unless the prompt shrank unexplained)
        bounds = [0.0, req.read, req.read + req.w1, req.read + req.w1 + req.w5, P]
        rates = [r.cache_read, r.cache_write_1h, r.cache_write_5m, r.input]
        for k in range(4):
            lo, hi = bounds[k] * scale, bounds[k + 1] * scale
            rate = rates[k] / 1e6 / scale
            if hi <= lo or rate == 0:
                continue
            i = bisect_right(ends, lo)
            if i >= n:
                continue
            j = min(bisect_left(ends, hi), n - 1)
            starts = (ends[i - 1] if i else 0.0, ends[j - 1] if j else 0.0)
            first_full = i if starts[0] >= lo and ends[i] <= hi else i + 1
            last_full = j if starts[1] >= lo and ends[j] <= hi else j - 1
            for m, start in ((i, starts[0]), (j, starts[1])):
                if not first_full <= m <= last_full:
                    overlap = min(ends[m], hi) - max(start, lo)
                    if overlap > 0:
                        path[m].item.cost_in += overlap * rate
                if i == j:
                    break
            if first_full <= last_full:
                path[last_full].rate += rate
                if first_full > 0:
                    path[first_full - 1].rate -= rate


# --- report ---------------------------------------------------------------------------------------

@dataclass
class Totals:
    requests: int = 0
    input: int = 0
    read: int = 0
    w5: int = 0
    w1: int = 0
    output: int = 0
    cost_input: float = 0.0
    cost_read: float = 0.0
    cost_w5: float = 0.0
    cost_w1: float = 0.0
    cost_output: float = 0.0

    @property
    def prompt(self) -> int:
        return self.input + self.read + self.w5 + self.w1

    @property
    def hit_ratio(self) -> float:
        return self.read / self.prompt if self.prompt else 0.0

    @property
    def cost(self) -> float:
        return self.cost_input + self.cost_read + self.cost_w5 + self.cost_w1 + self.cost_output


class Report:
    def __init__(self, custom_price: float | None = None):
        self.custom_price = custom_price
        self.stats = Stats()
        self.all, self.main, self.sub = Totals(), Totals(), Totals()
        self.unpriced: Counter = Counter()
        self.models: Counter = Counter()
        self.price_notes: Counter = Counter()
        self.rebuilds: dict = defaultdict(lambda: [0, 0.0])
        self.first_calls = [0, 0, 0.0]  # contexts' first requests: count, cache write tokens, their $
        self.compactions: list[dict] = []
        self.links: Counter = Counter()
        self.items: list[Item] = []
        self.contexts: list[Context] = []
        self.since: float | None = None
        self.until: float | None = None
        self.sessions: set = set()
        self.web_searches = 0
        self.web_searches_outside = 0
        self.web_search_usd = 0.0
        self.retention_days: int | None = None
        self.files_found = 0

    def price(self, req: Request) -> pricing.ApiPriced | None:
        return pricing.api_rates_for(req.model, req.prompt, req.speed, req.geo, self.custom_price)

    def add_request(self, req: Request, info: ContextInfo, priced) -> None:
        self.models[req.model or "?"] += 1
        self.sessions.add((info.project, info.session))
        if priced and priced.note:
            self.price_notes[priced.note] += 1
        if req.tier and req.tier != "standard":  # priority or batch have their own (contract) prices
            self.stats.unknown_tiers[req.tier] += 1
        if req.speed and req.speed not in ("standard", "fast"):
            self.stats.unknown_speeds[req.speed] += 1
        if req.geo and req.geo not in ("global", "us", "not_available"):
            self.stats.unknown_geos[req.geo] += 1
        if priced is None:
            name = req.model or "?"
            if (req.speed or "").lower() == "fast" and pricing.api_model_key(req.model):
                name += " (fast mode: no published fast price)"
            self.unpriced[name] += req.prompt + req.output
        for t in (self.all, self.sub if info.is_sub else self.main):
            t.requests += 1
            t.input += req.input
            t.read += req.read
            t.w5 += req.w5
            t.w1 += req.w1
            t.output += req.output
            if priced:
                r = priced.rates
                t.cost_input += req.input * r.input / 1e6
                t.cost_read += req.read * r.cache_read / 1e6
                t.cost_w5 += req.w5 * r.cache_write_5m / 1e6
                t.cost_w1 += req.w1 * r.cache_write_1h / 1e6
                t.cost_output += req.output * r.output / 1e6

    @property
    def total_usd(self) -> float:
        return self.all.cost + self.web_search_usd


def default_root() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR")
    return (Path(base).expanduser() if base else Path.home() / ".claude") / "projects"


def retention_days(root: Path) -> int | None:
    """cleanupPeriodDays from the user settings next to the projects folder (Claude Code deletes older
    transcripts at startup; 30 days when unset)."""
    try:
        settings = json.loads((root.parent / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = settings.get("cleanupPeriodDays") if isinstance(settings, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def find_transcripts(roots: list[Path], since: float | None, project: str | None, session: str | None,
                     stats: Stats) -> tuple[list[ContextInfo], int]:
    """(contexts to read, transcript files found under the roots before any filter)."""
    found, everything = [], 0
    project_l = project.lower() if project else None
    for root in roots:
        if not root.is_dir():
            continue
        try:
            projects = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            stats.folders_unreadable += 1
            continue
        for proj in projects:
            if not os.access(proj, os.R_OK | os.X_OK):  # glob would quietly find nothing
                stats.folders_unreadable += 1
                continue
            try:
                mains = sorted(proj.glob("*.jsonl"))
                subs = sorted(proj.glob("*/subagents/**/agent-*.jsonl"))
            except OSError:
                stats.folders_unreadable += 1
                continue
            everything += len(mains) + len(subs)
            if project_l and project_l not in proj.name.lower():
                continue
            for path in mains:
                if ".orphaned-" in path.name or ".superseded-" in path.name:
                    stats.files_skipped_orphaned += 1
                    continue
                if (session and not path.stem.startswith(session)) or _too_old(path, since):
                    continue
                found.append(ContextInfo(path, proj.name, path.stem, False))
            for path in subs:
                if ".orphaned-" in path.name:
                    stats.files_skipped_orphaned += 1
                    continue
                parts = path.relative_to(proj).parts
                sess = parts[0]
                if (session and not sess.startswith(session)) or _too_old(path, since):
                    continue
                info = ContextInfo(path, proj.name, sess, True)
                info.agent_id = path.stem[len("agent-"):] if path.stem.startswith("agent-") else path.stem
                if len(parts) >= 5 and parts[2] == "workflows":
                    info.workflow = parts[3]
                meta_path = path.with_name(path.stem + ".meta.json")
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    meta = None
                if isinstance(meta, dict):
                    stats.meta_keys.update(meta.keys())
                    if isinstance(meta.get("agentType"), str) and meta["agentType"]:
                        info.agent_type = meta["agentType"]
                    if isinstance(meta.get("toolUseId"), str):
                        info.parent_tool_use_id = meta["toolUseId"]
                    if isinstance(meta.get("parentAgentId"), str):
                        info.parent_agent_id = meta["parentAgentId"]
                else:
                    stats.subagents_without_meta += 1
                found.append(info)
    return found, everything


def _too_old(path: Path, since: float | None) -> bool:
    """A file can't hold lines newer than its last change, so older files are skipped unread."""
    if since is None:
        return False
    try:
        return path.stat().st_mtime < since
    except OSError:
        return False


def analyze(roots, since: float | None = None, project: str | None = None, session: str | None = None,
            custom_price: float | None = None, progress=None, until: float | None = None) -> Report:
    """Read and attribute every transcript in the window [since, until) (until: now when not set;
    it exists for exact comparisons on transcripts that are still being written)."""
    roots = [roots] if isinstance(roots, (str, Path)) else list(roots)
    roots = [Path(r) for r in roots]
    report = Report(custom_price)
    report.since = since
    report.until = until if until is not None else datetime.now(timezone.utc).timestamp()
    report.retention_days = retention_days(roots[0]) if roots else None
    infos, report.files_found = find_transcripts(roots, since, project, session, report.stats)
    infos.sort(key=lambda i: (i.is_sub, str(i.path)))  # main chats first
    seen_requests: set = set()
    for n, info in enumerate(infos, 1):
        if progress:
            progress(n, len(infos))
        try:
            lines, _, searches = read_file(info.path, report.stats, seen_requests)
        except (OSError, UnicodeError):
            report.stats.files_unreadable += 1
            continue
        report.stats.files += 1
        run_context(info, lines, report, since, [] if info.is_sub else searches, until)
    link_subagents(report)
    return report


def run_context(info: ContextInfo, lines: list[Line], report: Report, since: float | None,
                web_searches: list | None = None, until: float | None = None) -> Context:
    """Turn one transcript's lines into steps (parent request, added items) and attribute them."""
    ctx = Context(info, report)
    by_uuid = {ln.uuid: ln for ln in lines}
    results = {it.tool_use_id: it for ln in lines for it in ln.items if it.kind == "tool_result" and it.tool_use_id}
    prev: Request | None = None
    pending: list[Item] = []          # added since the previous request, in file order
    before_compact: list[Item] = []   # added before a compaction line (replaced by its summary)
    compact_line: Line | None = None
    remove_thinking = False
    started: set = set()
    walked: list[Request] = []
    for line in lines:
        req = line.req if line.kind == "assistant" else None
        if req is None or id(req) in started:
            if line.compact is not None:
                compact_line = line
                before_compact, pending = before_compact + pending, []
            remove_thinking = remove_thinking or line.removes_thinking
            pending.extend(line.items)
            continue
        started.add(id(req))
        if req.odd:
            report.stats.odd_requests += 1
        if req.dup or req.prompt <= 0:
            if req.dup:
                report.stats.duplicate_requests += 1
            else:
                report.stats.not_api_calls += 1
            pending.extend(line.items)  # not an API call: what it holds is plain added content
            continue
        if req.ts is None:
            report.stats.no_timestamp += 1
        req.in_window = req.ts is None or ((since is None or req.ts >= since) and (until is None or req.ts < until))
        if req.ts is not None:
            ctx.first_ts = req.ts if ctx.first_ts is None else min(ctx.first_ts, req.ts)
            ctx.last_ts = req.ts if ctx.last_ts is None else max(ctx.last_ts, req.ts)
        # the parent request: the nearest earlier request on the parentUuid chain
        parent, boundary, chain, steps = None, None, [], 0
        cur = by_uuid.get(line.parent) if line.parent else None
        while cur is not None and steps < 1_000_000:
            steps += 1
            if cur.req is not None and cur.kind == "assistant" and cur.req.tip is not None and cur.req is not req:
                parent = cur.req
                break
            if cur.compact is not None:
                boundary = cur
                break
            chain.append(cur)
            cur = by_uuid.get(cur.parent) if cur.parent else None
        if boundary is not None:
            compaction = boundary.compact
            # kept messages stay, also ones written after the last request or written again later
            kept = [it for u in compaction.get("kept") or [] if u in by_uuid for it in by_uuid[u].items]
            kept_ids = {id(it) for it in kept}
            new = pending if boundary is compact_line else [it for c in reversed(chain) for it in c.items]
            new = [it for it in new if id(it) not in kept_ids]
            ctx.step(req, prev, new, compaction, kept, False)
        elif parent is not None and prev is not None and parent is not prev:
            # a rewind or branch: continue from the older request, with what is on the chain plus the
            # parallel tool results that answer calls on it
            report.stats.branches += 1
            on_chain = [it for c in reversed(chain) for it in c.items]
            ids = {id(it) for it in on_chain}
            new = [b for b in parent.blocks if id(b) not in ids] + on_chain
            calls = [it.label_id for it in new if it.kind == "tool_use" and it.label_id]
            ids = {id(it) for it in new}
            new += [results[c] for c in calls if c in results and id(results[c]) not in ids]
            ctx.step(req, parent, new, None, [], remove_thinking)
        else:
            if parent is None and prev is not None:
                report.stats.broken_chains += 1
                parent = prev
            ctx.step(req, parent, before_compact + pending, None, [], remove_thinking)
        walked.append(req)
        pending, before_compact, compact_line, remove_thinking = list(line.items), [], None, False
        prev = req
    ctx.finalize()
    # output that never reached a later request (the last answer of a context) still cost its output
    for req in walked:
        if req.in_window:
            for b in req.blocks:
                if b.ctx is None and b.cost_out:
                    b.ctx, b.in_window = info, True
                    report.items.append(b)
    # web searches dated by the line before the cost-state line that first counted them
    in_window = sum(n for ts, n in web_searches or []
                    if (ts is None and since is None) or (ts is not None and (since is None or ts >= since)
                                                         and (until is None or ts < until)))
    report.web_searches_outside += sum(n for _, n in web_searches or []) - in_window
    if in_window:
        web_searches = in_window
        fee = Item("fee", WEB_FEES)
        fee.tool = "WebSearch"
        fee.ctx, fee.in_window = info, True
        fee.cost_out = web_searches * WEB_SEARCH_USD
        report.web_search_usd += fee.cost_out
        report.web_searches += web_searches
        report.items.append(fee)
    report.contexts.append(ctx)
    return ctx


def link_subagents(report: Report) -> None:
    """Subagent dollars: to the Agent call that started each run (meta toolUseId), else to the parent
    subagent (parentAgentId), else to its workflow run; always to its agent type."""
    calls: dict[tuple, Item] = {}
    for it in report.items:
        if it.kind == "tool_result" and it.tool in ("Agent", "Task") and it.tool_use_id and it.ctx is not None:
            calls[(it.ctx.project, it.ctx.session, it.tool_use_id)] = it
    agents = {(c.info.project, c.info.session, c.info.agent_id): c for c in report.contexts if c.info.is_sub}
    for c in report.contexts:
        info = c.info
        if not info.is_sub:
            continue
        link = "unlinked"
        if info.parent_tool_use_id and (info.project, info.session, info.parent_tool_use_id) in calls:
            calls[(info.project, info.session, info.parent_tool_use_id)].subagent_usd += c.cost
            link = "agent call"
        elif info.parent_agent_id and (info.project, info.session, info.parent_agent_id) in agents:
            link = "parent subagent"
        elif info.workflow:
            link = "workflow"
        report.links[link] += 1


# --- output ---------------------------------------------------------------------------------------

# What "costliest single items" ranks: one tool output or call, one message or attachment. The starting
# context, compaction summaries and Claude's own replies and thinking are in the main table instead.
SINGLE_KINDS = {"tool_result", "tool_use", "text", "attachment"}


def money(usd: float | None) -> str:
    if usd is None:
        return "n/a"
    if abs(usd) < 0.005:
        return "$0.00" if usd == 0 else "under $0.01"
    return f"${usd:,.2f}"


def _pct(part: float, whole: float) -> str:
    return f"{part / whole:.1%}" if whole > 0 else "0.0%"


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _counted(report: Report) -> list[Item]:
    return [it for it in report.items if it.ctx is not None and (it.cost or it.in_window)]


def _later(i: Item) -> int:
    """Requests in the window that sent the item again after the one it arrived on (cache reads, or
    writes after a rebuild). Items that arrived before the window: every request in it counts."""
    return max(i.calls - (1 if i.in_window and i.calls else 0), 0)


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def _size(i: Item) -> float:
    """An item's tokens: its size in the context, or its output tokens when no later request read it."""
    return max(i.tokens, i.out_tokens)


def _row(name: str, items: list[Item]) -> dict:
    """tokens: size of the items that cost money in the window (some entered the context before it);
    tokens_added_in_window: only those that entered in the window."""
    return {"name": name, "cost_usd": round(fsum(i.cost for i in items), 6),
            "lasting_input_usd": round(fsum(i.cost_in for i in items), 6),
            "output_usd": round(fsum(i.cost_out for i in items), 6),
            "tokens": int(fsum(_size(i) for i in items if i.cost > 0 or i.in_window)),
            "tokens_added_in_window": int(fsum(_size(i) for i in items if i.in_window)),
            "items": sum(1 for i in items if i.cost > 0 or i.in_window)}


def _grouped(items: list[Item], key) -> list[dict]:
    groups: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        k = key(it)
        if k is not None:
            groups[k].append(it)
    rows = [_row(k, v) for k, v in groups.items()]
    return sorted(rows, key=lambda r: (-r["cost_usd"], r["name"]))


def summarize(report: Report, top: int = 20, details: bool = False) -> dict:
    """The stable JSON document (schema cachecanary.claude-code/1)."""
    items = _counted(report)
    total = report.total_usd
    st = report.stats
    by_mcp = []
    servers: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        if it.mcp:
            servers[it.mcp[0]].append(it)
    for server, its in servers.items():
        row = _row(server, its)
        row["tools"] = _grouped(its, lambda i: i.mcp[1])
        by_mcp.append(row)
    by_mcp.sort(key=lambda r: (-r["cost_usd"], r["name"]))
    by_skill = _grouped(items, lambda i: i.skill)
    loads = Counter(i.skill for i in items if i.skill and i.kind == "tool_use" and i.in_window)
    for row in by_skill:
        row["loads"] = loads.get(row["name"], 0)
    agents: dict[str, list[Context]] = defaultdict(list)
    for c in report.contexts:
        if c.info.is_sub and c.requests:
            agents[c.info.agent_type or "(no type)"].append(c)
    by_agent = sorted(({"name": name, "runs": len(cs), "requests": sum(c.requests for c in cs),
                        "cost_usd": round(fsum(c.cost for c in cs), 6),
                        "starting_context_usd": round(fsum(c.start_item.cost for c in cs if c.start_item), 6),
                        "linked_to_agent_call": sum(1 for c in cs if c.info.parent_tool_use_id),
                        "workflow_runs": len({c.info.workflow for c in cs if c.info.workflow})}
                       for name, cs in agents.items()), key=lambda r: (-r["cost_usd"], r["name"]))
    ranked = sorted((i for i in items if i.kind in SINGLE_KINDS), key=lambda i: -i.cost)[:top]
    top_items = []
    for it in ranked:
        row = {"kind": it.kind, "group": it.group, "tool": it.tool, "bash_family": it.family,
               "mcp_server": it.mcp[0] if it.mcp else None, "mcp_tool": it.mcp[1] if it.mcp else None,
               "skill": it.skill, "tokens": int(_size(it)), "later_requests": _later(it),
               "cost_usd": round(it.cost, 6), "lasting_input_usd": round(it.cost_in, 6),
               "output_usd": round(it.cost_out, 6), "subagent_usd": round(it.subagent_usd, 6),
               "session": it.ctx.session[:8] if it.ctx else None, "subagent": bool(it.ctx and it.ctx.is_sub)}
        if details:
            row["label"] = label_for(it)
        top_items.append(row)
    comps = [c for c in report.compactions if c.get("in_window")]
    rebuilds = {k: {"count": v[0], "extra_usd": round(v[1], 6)} for k, v in
                sorted(report.rebuilds.items(), key=lambda kv: -kv[1][1])}
    tot = report.all

    def tokens(t: Totals) -> dict:
        return {"requests": t.requests, "uncached_input": t.input, "cache_read": t.read, "cache_write_5m": t.w5,
                "cache_write_1h": t.w1, "output": t.output, "hit_ratio": round(t.hit_ratio, 4),
                "cost_usd": round(t.cost, 6)}

    findings = findings_for(report, items)
    return {
        "schema": SCHEMA,
        "window": {"since": _iso(report.since), "until": _iso(report.until)},
        "price_basis": "custom" if report.custom_price is not None else "list",
        "price_note": "API list price equivalent, not a subscription bill",
        "total_usd": round(total, 6),
        "cost_usd": {"uncached_input": round(tot.cost_input, 6), "cache_read": round(tot.cost_read, 6),
                     "cache_write_5m": round(tot.cost_w5, 6), "cache_write_1h": round(tot.cost_w1, 6),
                     "output": round(tot.cost_output, 6), "web_search_fees": round(report.web_search_usd, 6)},
        "web_searches": report.web_searches,
        "sessions": len(report.sessions),
        "subagent_runs": sum(1 for c in report.contexts if c.info.is_sub and c.requests),
        "all": tokens(report.all), "main": tokens(report.main), "subagents": tokens(report.sub),
        "first_calls": {"count": report.first_calls[0], "cache_write_tokens": report.first_calls[1],
                        "cache_write_usd": round(report.first_calls[2], 6)},
        "rebuilds": {"count": sum(v[0] for v in report.rebuilds.values()),
                     "extra_usd": round(fsum(v[1] for v in report.rebuilds.values()), 6), "by_cause": rebuilds},
        "compactions": {"count": len(comps), "by_trigger": dict(Counter(c.get("trigger") or "unknown" for c in comps)),
                        "list": [{"session": c["session"][:8], "subagent": c["subagent"], "time": _iso(c.get("ts")),
                                  "trigger": c.get("trigger"), "pre_tokens": c.get("pre_tokens"),
                                  "post_tokens": c.get("post_tokens"), "prompt_before": c.get("prompt_before"),
                                  "prompt_after": c.get("prompt_after"), "kept_messages": c.get("kept_messages")}
                                 for c in comps]},
        "unexplained_drops": {"count": st.unexplained_drops, "tokens": st.unexplained_drop_tokens},
        "unpriced_models": dict(report.unpriced),
        "models": dict(report.models),
        "by_what": _grouped(items, lambda i: i.group),
        "by_tool": _grouped(items, lambda i: i.tool),
        "by_bash": _grouped(items, lambda i: i.family if i.tool == "Bash" else None),
        "by_mcp": by_mcp,
        "by_skill": by_skill,
        "by_agent": by_agent,
        "top_items": top_items,
        "headline": findings[0] if findings else None,
        "suggestions": findings[1:4],
        "notes": cheap_model_notes(report),
        "stats": {"files": st.files, "transcript_files_found": report.files_found, "lines": st.lines,
                  "ignored_lines_by_type": dict(st.ignored), "unreadable_lines": st.unreadable_lines,
                  "partial_last_lines": st.partial_last_lines, "repeated_lines": st.repeated_lines,
                  "odd_requests": st.odd_requests, "not_api_calls": st.not_api_calls,
                  "duplicate_requests": st.duplicate_requests, "no_timestamp": st.no_timestamp,
                  "orphan_tool_results": st.orphan_results, "branches": st.branches,
                  "broken_chains": st.broken_chains, "model_switches": st.model_switches,
                  "thinking_removals": st.thinking_removals, "ttl_mismatch_requests": st.ttl_mismatch,
                  "unknown_attachment_types": dict(st.unknown_attachments),
                  "subagents_without_meta": st.subagents_without_meta,
                  "subagent_links": dict(report.links), "skipped_orphaned_files": st.files_skipped_orphaned,
                  "unreadable_files": st.files_unreadable, "unreadable_folders": st.folders_unreadable,
                  "unknown_service_tiers": dict(st.unknown_tiers), "unknown_speeds": dict(st.unknown_speeds),
                  "unknown_inference_geos": dict(st.unknown_geos)},
    }


def findings_for(report: Report, items: list[Item] | None = None) -> list[dict]:
    """One headline, then suggestions, each with its measured cost. Nothing about savings: these are
    where the money went, and the change most likely to touch it."""
    items = _counted(report) if items is None else items
    total = report.total_usd
    if total <= 0:
        return []
    out = []
    what = [r for r in _grouped(items, lambda i: i.group) if r["name"] not in (THINKING, REPLIES)]
    if what:
        r = what[0]
        text = f"Biggest cost: {r['name']}, {money(r['cost_usd'])} ({_pct(r['cost_usd'], total)} of {money(total)})"
        if r["name"] == "Bash":
            fams = _grouped([i for i in items if i.tool == "Bash"], lambda i: i.family)[:3]
            if fams:
                text += "; mostly " + ", ".join(f"'{f['name']}' {_pct(f['cost_usd'], total)}" for f in fams)
        out.append({"kind": "headline", "cost_usd": r["cost_usd"], "text": text + "."})
    cands = []
    bash = _grouped(items, lambda i: i.family if i.tool == "Bash" else None)
    if bash:
        r = bash[0]
        cands.append({"kind": "bash", "cost_usd": r["cost_usd"],
                      "text": f"Bash '{r['name']}' cost {money(r['cost_usd'])} ({_pct(r['cost_usd'], total)}). Its output stays "
                              "in the context and is re-read on every later request: cut it down (head, tail, grep, "
                              "quieter flags) or shorten it with a PreToolUse hook."})
    mcp = _grouped([i for i in items if i.mcp], lambda i: f"{i.mcp[0]}/{i.mcp[1]}")
    if mcp:
        r = mcp[0]
        cands.append({"kind": "mcp", "cost_usd": r["cost_usd"],
                      "text": f"MCP tool {r['name']} cost {money(r['cost_usd'])} ({_pct(r['cost_usd'], total)}). Ask it for "
                              "fewer fields or results, or lower MAX_MCP_OUTPUT_TOKENS (Claude Code's default is 25,000)."})
    tools = [r for r in _grouped(items, lambda i: i.tool) if r["name"] not in ("Bash",) and not r["name"].startswith("mcp__")]
    if tools:
        r = tools[0]
        cands.append({"kind": "tool", "cost_usd": r["cost_usd"],
                      "text": f"{r['name']} cost {money(r['cost_usd'])} ({_pct(r['cost_usd'], total)}), counting its "
                              "inputs and results for as long as they stayed in the context."})
    subs = [c for c in report.contexts if c.info.is_sub and c.requests and c.start_item]
    if subs:
        usd = fsum(c.start_item.cost for c in subs)
        cands.append({"kind": "subagent_starts", "cost_usd": round(usd, 6),
                      "text": f"{len(subs):,} subagent runs each started a fresh context; their starting contexts cost "
                              f"{money(usd)} ({_pct(usd, total)}). Fewer, larger subagent tasks start fewer of them."})
    idle = [(k, v) for k, v in report.rebuilds.items() if k.startswith(("idle", "waited"))]
    if idle:
        usd = fsum(v[1] for _, v in idle)
        n = sum(v[0] for _, v in idle)
        cands.append({"kind": "idle_rebuilds", "cost_usd": round(usd, 6),
                      "text": f"{n:,} requests came after a pause longer than the cache lifetime and wrote the cache "
                              f"again: {money(usd)} more than reading it. The rewrite is the size of the whole "
                              "context, so long pauses in big chats cost the most."})
    main_start = [c.start_item for c in report.contexts if not c.info.is_sub and c.requests and c.start_item]
    if main_start:
        usd = fsum(i.cost for i in main_start)
        cands.append({"kind": "starting_context", "cost_usd": round(usd, 6),
                      "text": f"The main chats' starting context (system prompt, tools, CLAUDE.md, memory) cost "
                              f"{money(usd)} ({_pct(usd, total)}). /context shows what it is made of."})
    cands = [c for c in cands if c["cost_usd"] >= 0.03 * total]
    cands.sort(key=lambda c: -c["cost_usd"])
    return out + cands[:3]


def _table(title: str, rows: list[dict], top: int, total: float, name_width: int = 0) -> list[str]:
    if not rows:
        return []
    lines = ["", title]
    for r in rows[:top]:
        lines.append(f"  {money(r['cost_usd']):>12}  {_pct(r['cost_usd'], total):>6}  {r['tokens']:>13,} tok  {r['name']}")
    if len(rows) > top:
        rest = fsum(r["cost_usd"] for r in rows[top:])
        lines.append(f"  {money(rest):>12}  {_pct(rest, total):>6}  {'':>17}  {len(rows) - top} more")
    return lines


def render_text(report: Report, data: dict, by: str | None, top: int, details: bool) -> list[str]:
    total = data["total_usd"]
    days = (report.until - report.since) / 86400 if report.since else None
    if days is None:
        window = "all history"
    else:
        n = max(round(days), 1) if days >= 1 else None
        window = (f"since {data['window']['since'][:16].replace('T', ' ')} UTC" if n is None
                  else f"since {data['window']['since'][:10]} ({n} day{'s' if n != 1 else ''})")
    a = data["all"]
    basis = "your --price" if data["price_basis"] == "custom" else "Anthropic API list prices"
    lines = [f"Claude Code {window}: {data['sessions']:,} sessions, {data['subagent_runs']:,} subagent runs, "
             f"{a['requests']:,} requests",
             f"Total {money(total)} at {basis}. This is an API list price equivalent, not your subscription bill."]
    if data["headline"]:
        lines += ["", data["headline"]["text"]]
    if data["suggestions"]:
        lines.append("Where to look first:")
        lines += [f"  {n}. {s['text']}" for n, s in enumerate(data["suggestions"], 1)]
    c = data["cost_usd"]
    lines += ["", f"Spend: cache reads {money(c['cache_read'])}, 1h cache writes {money(c['cache_write_1h'])}, "
                  f"5m cache writes {money(c['cache_write_5m'])}, uncached input {money(c['uncached_input'])}, "
                  f"output {money(c['output'])}" + (f", web search fees {money(c['web_search_fees'])} "
                                                   f"({data['web_searches']:,} searches)" if data["web_searches"] else "")]
    m, s = data["main"], data["subagents"]
    hit = f"Cache hit rate: {a['hit_ratio']:.1%} overall"
    if m["requests"]:
        hit += f", main chats {m['hit_ratio']:.1%}"
    if s["requests"]:
        hit += f", subagents {s['hit_ratio']:.1%}"
    lines.append(hit)
    lines.append(f"Cache writes: 1h {a['cache_write_1h']:,} tokens ({money(c['cache_write_1h'])}), "
                 f"5m {a['cache_write_5m']:,} tokens ({money(c['cache_write_5m'])})")
    fc = data["first_calls"]
    if fc["count"]:
        lines.append(f"First calls: {_plural(fc['count'], 'chat or subagent run')} wrote "
                     f"{fc['cache_write_tokens']:,} tokens to the cache on their first request ({money(fc['cache_write_usd'])})")
    rb = data["rebuilds"]
    if rb["count"]:
        parts = ", ".join(f"{k} {v['count']:,} ({money(v['extra_usd'])})" for k, v in rb["by_cause"].items())
        lines.append(f"Cache rebuilds: {_plural(rb['count'], 'request')} wrote again what they could have read, "
                     f"{money(rb['extra_usd'])} more than reading it: {parts}")
    cp = data["compactions"]
    if cp["count"]:
        trig = ", ".join(f"{k} {v}" for k, v in sorted(cp["by_trigger"].items()))
        pairs = [(x["prompt_before"], x["prompt_after"]) for x in cp["list"] if x["prompt_before"] and x["prompt_after"]]
        size = ""
        if pairs:
            size = (f"; prompt {sum(p for p, _ in pairs) // len(pairs):,} tokens before, "
                    f"{sum(q for _, q in pairs) // len(pairs):,} after on average")
        lines.append(f"Compactions: {cp['count']:,} ({trig}){size}")
    if data["unexplained_drops"]["count"]:
        lines.append(f"Unexplained prompt drops: {data['unexplained_drops']['count']:,} "
                     f"({data['unexplained_drops']['tokens']:,} tokens); nothing was removed for them")
    def show(name):
        return by in (None, name)
    if by is None:
        lines += _table("Where the money went (lasting cost: written once, then re-read by every later request "
                        "until it leaves the context)", data["by_what"], top, total)
    if show("tool"):
        lines += _table("By tool (inputs and results)", data["by_tool"], top, total)
    if show("bash"):
        lines += _table("Bash command families", data["by_bash"], top, total)
    if show("mcp") and data["by_mcp"]:
        lines += ["", "MCP servers"]
        for r in data["by_mcp"][:top]:
            lines.append(f"  {money(r['cost_usd']):>12}  {_pct(r['cost_usd'], total):>6}  {r['tokens']:>13,} tok  {r['name']}")
            for t in r["tools"][:3]:
                lines.append(f"  {money(t['cost_usd']):>12}  {_pct(t['cost_usd'], total):>6}  {t['tokens']:>13,} tok    {t['name']}")
    if show("skill") and data["by_skill"]:
        lines += ["", "Skills: what their instructions cost while they stayed in the context, and how often a "
                      "Skill call loaded them"]
        for r in data["by_skill"][:top]:
            loads = f"{r['loads']:>5,} load{'s' if r['loads'] != 1 else ' '}" if r["loads"] else f"{'':>11}"
            lines.append(f"  {money(r['cost_usd']):>12}  {_pct(r['cost_usd'], total):>6}  "
                         f"{r['tokens']:>13,} tok  {loads}  {r['name']}")
    if show("agent") and data["by_agent"]:
        lines += ["", "Subagents (everything their runs spent; starting context included)"]
        for r in data["by_agent"][:top]:
            lines.append(f"  {money(r['cost_usd']):>12}  {_pct(r['cost_usd'], total):>6}  {r['runs']:>6,} runs  "
                         f"{r['name']} (starting contexts {money(r['starting_context_usd'])})")
    if show("item") and data["top_items"]:
        lines += ["", "Costliest single outputs and messages (cost, size, later requests that sent it again, what)"]
        for r in data["top_items"][:top]:
            what = r["tool"] or r["group"]
            if r["kind"] == "tool_use":
                what += " call"
            if r["bash_family"]:
                what += f" '{r['bash_family']}'"
            elif r["mcp_server"]:
                what = f"MCP {r['mcp_server']}/{r['mcp_tool']}" + (" call" if r["kind"] == "tool_use" else "")
            extra = f", plus {money(r['subagent_usd'])} inside the subagent" if r["subagent_usd"] else ""
            label = f"  {r['label']}" if details and r.get("label") else ""
            lines.append(f"  {money(r['cost_usd']):>12}  {r['tokens']:>9,} tok  {r['later_requests']:>6,} later  "
                         f"{what}{extra}{label}")
    cheap = cheap_model_notes(report)
    if cheap and show("tool"):
        lines += [""] + cheap
    notes = notes_for(report, data)
    if notes:
        lines += ["", "Notes:"] + [f"  {n}" for n in notes]
    return lines


def cheap_model_notes(report: Report, limit: int = 3) -> list[str]:
    """Tools with at least 3x more of the tokens than of the dollars, when most of their tokens went to
    one model: a fact about where they ran, nothing more."""
    items = [i for i in _counted(report) if i.tool and i.in_window]
    tok_total = fsum(i.tokens for i in items)
    usd_total = report.total_usd
    if not tok_total or not usd_total:
        return []
    by_tool: dict[str, list[Item]] = defaultdict(list)
    for i in items:
        by_tool[i.tool].append(i)
    out = []
    for tool, its in sorted(by_tool.items(), key=lambda kv: -fsum(i.tokens for i in kv[1])):
        tok = fsum(i.tokens for i in its)
        usd = fsum(i.cost for i in its)
        if tok / tok_total < 0.05 or tok / tok_total < 3 * usd / usd_total:
            continue
        models = Counter()
        for i in its:
            models[i.model or "?"] += i.tokens
        model, mtok = models.most_common(1)[0]
        line = (f"{tool}: {tok / tok_total:.1%} of the tokens added but {usd / usd_total:.1%} of the dollars"
                + (f"; {mtok / tok:.0%} of its tokens went to {model} requests" if mtok / tok >= 0.5 else "") + ".")
        out.append(line)
        if len(out) >= limit:
            break
    return out


def notes_for(report: Report, data: dict) -> list[str]:
    st = report.stats
    notes = []
    if data["price_basis"] == "list":
        notes.append("Prices: Anthropic API list (platform.claude.com pricing page, checked 2026-11-18); Bedrock "
                     "model IDs at Bedrock prices. Use --price for your own rate.")
    if data["unpriced_models"]:
        notes.append("No price for: " + ", ".join(f"{k} ({v:,} tokens)" for k, v in sorted(data["unpriced_models"].items()))
                     + "; add --price to include them.")
    if any(pricing.is_bedrock_id(m) or "@" in m for m in report.models):
        notes.append("Bedrock and Vertex model IDs are priced by rule (Bedrock: global at list, other endpoints +10%; "
                     "Vertex: at the API list price, its regional endpoints cost 10% more). CacheCanary has not "
                     "been checked on real Bedrock or Vertex transcripts yet.")
    if st.unknown_tiers or st.unknown_speeds or st.unknown_geos:
        odd = {**st.unknown_tiers, **st.unknown_speeds, **st.unknown_geos}
        notes.append("Priced at standard rates despite unknown usage values: "
                     + ", ".join(f"{k} ({v:,})" for k, v in sorted(odd.items())) + ".")
    if report.web_searches or report.web_searches_outside:
        notes.append("Web search fees ($10 per 1,000 searches) come from Claude Code's own cost-state counts, "
                     "which have no time: each search is dated by the transcript line before the count that "
                     "first includes it" + (f"; {report.web_searches_outside:,} searches outside the window were "
                                            "left out." if report.web_searches_outside else "."))
    if st.files_unreadable or st.folders_unreadable:
        notes.append(f"Skipped {st.files_unreadable:,} unreadable file(s) and {st.folders_unreadable:,} unreadable "
                     "folder(s).")
    ignored = sum(st.ignored.values())
    notes.append(f"Read {st.files:,} files, {st.lines:,} lines; {ignored:,} lines of other types were skipped"
                 + (f", {st.unreadable_lines:,} could not be read" if st.unreadable_lines else "") + ".")
    if st.odd_requests:
        notes.append(f"{st.odd_requests:,} request(s) had a later line with smaller numbers; the larger value per field was kept.")
    if st.unknown_attachments:
        notes.append(f"{sum(st.unknown_attachments.values()):,} attachments of types this version does not know are "
                     "counted under 'other attachments'.")
    if report.retention_days is not None or (report.since and report.until - report.since > 30 * 86400):
        days = report.retention_days or 30
        notes.append(f"Claude Code deletes transcripts older than {days} days (cleanupPeriodDays).")
    if st.versions:
        newest = max(_version(v) for v in st.versions)
        if newest > TESTED_VERSIONS[1]:
            notes.append(f"Transcripts from Claude Code {'.'.join(map(str, newest))}, newer than the newest checked "
                         f"({'.'.join(map(str, TESTED_VERSIONS[1]))}); run with --debug-shapes if numbers look wrong.")
    return notes


def debug_shapes(report: Report) -> dict:
    """Counts of what the transcripts contain (types and key names only, never values), for bug reports."""
    st = report.stats
    versions = sorted((_version(v) for v in st.versions), key=lambda v: v)
    newest = versions[-1] if versions else None
    return {
        "schema": SCHEMA + "/debug-shapes",
        "files": st.files, "transcript_files_found": report.files_found, "lines": st.lines,
        "line_types_read": dict(st.used), "line_types_skipped": dict(st.ignored),
        "unreadable_lines": st.unreadable_lines, "partial_last_lines": st.partial_last_lines,
        "attachment_types": dict(st.attachment_types), "attachments_not_rendered": dict(st.attachments_not_rendered),
        "unknown_attachment_types": dict(st.unknown_attachments), "system_subtypes": dict(st.system_subtypes),
        "usage_keys": dict(st.usage_keys), "content_block_types": dict(st.block_types),
        "tool_result_shapes": dict(st.result_shapes), "subagent_meta_keys": dict(st.meta_keys),
        "versions": {"min": ".".join(map(str, versions[0])) if versions else None,
                     "max": ".".join(map(str, newest)) if newest else None, "distinct": len(set(versions))},
        "newer_than_checked": bool(newest and newest > TESTED_VERSIONS[1]),
        "repeated_lines": st.repeated_lines, "duplicate_requests": st.duplicate_requests,
        "odd_requests": st.odd_requests, "not_api_calls": st.not_api_calls,
        "unexplained_drops": st.unexplained_drops, "branches": st.branches, "broken_chains": st.broken_chains,
        "orphan_tool_results": st.orphan_results, "subagents_without_meta": st.subagents_without_meta,
        "subagent_links": dict(report.links), "model_switches": st.model_switches,
        "thinking_removals": st.thinking_removals, "unknown_service_tiers": dict(st.unknown_tiers),
        "unknown_speeds": dict(st.unknown_speeds), "unknown_inference_geos": dict(st.unknown_geos),
    }
