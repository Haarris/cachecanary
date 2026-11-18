"""`cachecanary claude-code`: reading Claude Code transcripts and putting dollars on what sat in the
context. Every fixture here is synthetic: its shapes copy what real transcripts look like (checked
on Claude Code 2.1.266 to 2.1.325, keys only), never their content."""

import itertools
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from cachecanary import claude_code as cc
from cachecanary import cli, pricing

# Claude Opus 5.5 list prices, USD per token
IN, W5, W1, RD, OUT = 4e-6, 5e-6, 8e-6, 0.2e-6, 20e-6
OPUS = "claude-opus-5-5"
NOW = datetime.now(timezone.utc).replace(microsecond=0)
_IDS = itertools.count(1)  # request ids are unique across files, as in real transcripts


class T:
    """Builds one synthetic transcript, line by line, the way Claude Code writes it."""

    def __init__(self, start: datetime | None = None, model: str = OPUS, session: str = "s1"):
        self.lines: list[dict] = []
        self.last: str | None = None
        self.n = 0
        self.clock = start or (NOW - timedelta(hours=5))
        self.model = model
        self.session = session
        self.tool_n = 0

    def _id(self, prefix="u") -> str:
        self.n += 1
        return f"{prefix}{self.n:05d}"

    def _base(self, kind: str, parent="last") -> dict:
        uid = self._id()
        d = {"parentUuid": self.last if parent == "last" else parent, "isSidechain": False, "type": kind,
             "uuid": uid, "timestamp": self.clock.isoformat().replace("+00:00", "Z"), "sessionId": self.session,
             "version": "2.1.325", "cwd": "/work/demo", "userType": "external"}
        return d

    def _add(self, d: dict) -> str:
        self.lines.append(d)
        self.last = d["uuid"]
        return d["uuid"]

    def user(self, text: str, parent="last", **extra) -> str:
        d = self._base("user", parent)
        d["message"] = {"role": "user", "content": text}
        d.update(extra)
        return self._add(d)

    def blocks(self, content: list, parent="last", **extra) -> str:
        d = self._base("user", parent)
        d["message"] = {"role": "user", "content": content}
        d.update(extra)
        return self._add(d)

    def result(self, tool_use_id: str, content, parent="last", source=None) -> str:
        d = self._base("user", parent)
        d["message"] = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}]}
        if source:
            d["sourceToolAssistantUUID"] = source
        return self._add(d)

    def request(self, *, read=0, w5=0, w1=0, inp=0, out=0, blocks=None, thinking=None, model=None,
                minutes=1.0, msg_id=None, req_id=None, usages=None, parent="last", **extra) -> list[str]:
        """One API request: one assistant line per content block, the usage repeated on each line
        (or `usages`, one per line, to mimic streaming). Returns the line uuids."""
        self.clock += timedelta(minutes=minutes)
        unique = next(_IDS)
        msg_id = msg_id or f"msg_{unique:06d}"
        req_id = req_id or f"req_{unique:06d}"
        blocks = blocks if blocks is not None else [{"type": "text", "text": ""}]
        uids = []
        for k, block in enumerate(blocks):
            usage = {"input_tokens": inp, "cache_creation_input_tokens": w5 + w1, "cache_read_input_tokens": read,
                     "output_tokens": out, "cache_creation": {"ephemeral_5m_input_tokens": w5,
                                                              "ephemeral_1h_input_tokens": w1},
                     "service_tier": "standard", "inference_geo": "global", "speed": "standard",
                     "iterations": [{"type": "message"}]}
            if thinking is not None:
                usage["output_tokens_details"] = {"thinking_tokens": thinking}
            if usages:
                usage.update(usages[k])
            d = self._base("assistant", parent if k == 0 else "last")
            d["requestId"] = req_id
            d["message"] = {"id": msg_id, "type": "message", "role": "assistant", "model": model or self.model,
                            "content": [block], "usage": usage, "stop_reason": None}
            d.update(extra)
            uids.append(self._add(d))
        return uids

    def bash(self, command: str) -> dict:
        self.tool_n += 1
        return {"type": "tool_use", "id": f"toolu_{self.tool_n:04d}", "name": "Bash",
                "input": {"command": command, "description": "run"}}

    def tool(self, name: str, inp: dict) -> dict:
        self.tool_n += 1
        return {"type": "tool_use", "id": f"toolu_{self.tool_n:04d}", "name": name, "input": inp}

    def attachment(self, kind: str, rendered: str | None, **fields) -> str:
        d = self._base("attachment")
        d["attachment"] = {"type": kind, **fields}
        if rendered is not None:
            d["rendered"] = [{"content": rendered}]
            d["renderedRole"] = "system"
        return self._add(d)

    def compact(self, pre=100_000, post=10_000, kept=None, trigger="auto") -> str:
        d = self._base("system", parent=None)
        d["subtype"] = "compact_boundary"
        d["logicalParentUuid"] = self.last
        d["compactMetadata"] = {"trigger": trigger, "preTokens": pre, "postTokens": post,
                                "preservedSegment": {"headUuid": None, "anchorUuid": None, "tailUuid": None},
                                "preservedMessages": {"uuids": kept or [], "allUuids": kept or [], "anchorUuid": None}}
        return self._add(d)

    def summary(self, text: str) -> str:
        d = self._base("user")
        d["message"] = {"role": "user", "content": text}
        d["isCompactSummary"] = True
        return self._add(d)

    def raw(self, d: dict) -> None:
        self.lines.append(d)

    def cost_state(self, searches: int, total=1.0) -> None:
        self.lines.append({"type": "cost-state", "sessionId": self.session, "totalCostUSD": total,
                           "modelUsage": {OPUS: {"inputTokens": 1, "outputTokens": 1, "webSearchRequests": searches,
                                                 "costUSD": total}}})

    def write(self, path, partial_tail: str | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(json.dumps(d, separators=(",", ":")) for d in self.lines) + "\n"
        if partial_tail is not None:
            text += partial_tail
        path.write_text(text)
        return path


def run(tmp_path, *transcripts, since=None, price=None, **kw):
    root = tmp_path / "projects"
    for rel, t in transcripts:
        t.write(root / rel)
    return cc.analyze(root, since=since, custom_price=price, **kw)


def items(report, **match):
    out = [it for it in report.items if it.ctx is not None]
    for k, v in match.items():
        out = [it for it in out if getattr(it, k) == v]
    return out


def attributed(report) -> float:
    return sum(it.cost for it in report.items if it.ctx is not None)


def bash_session(after=5, command="cd /repo && FOO=1 timeout 60 git status | head", **kw) -> T:
    """A 10,000-token Bash result, then `after` more requests that only add a few tokens."""
    t = T(**kw)
    t.user("go")
    call = t.bash(command)
    t.request(w1=5000, out=50, blocks=[call])
    t.result(call["id"], "x" * 20000)
    t.request(read=5000, w5=10050, out=20, blocks=[{"type": "text", "text": "ok"}])
    p = 15050
    for _ in range(after):
        t.user("next")
        t.request(read=p, w5=50, out=20, blocks=[{"type": "text", "text": "ok"}])
        p += 50
    return t


# --- the expected-behaviour table of the spec -----------------------------------------------------

def test_one_request_written_as_three_lines_counts_once_with_the_last_output(tmp_path):
    t = T()
    t.user("hi")
    t.request(w1=1000, out=40, blocks=[{"type": "thinking", "thinking": "", "signature": "x"},
                                      {"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
              usages=[{"output_tokens": 10}, {"output_tokens": 10}, {"output_tokens": 40}])
    rep = run(tmp_path, ("p/s1.jsonl", t))
    assert rep.all.requests == 1
    assert rep.all.output == 40
    assert rep.stats.odd_requests == 0


def test_a_later_line_with_smaller_numbers_keeps_the_larger_value_and_counts_as_odd(tmp_path):
    t = T()
    t.user("hi")
    t.request(w1=1000, out=40, blocks=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
              usages=[{"output_tokens": 40, "cache_creation_input_tokens": 1000},
                      {"output_tokens": 30, "cache_creation_input_tokens": 900,
                       "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 900}}])
    rep = run(tmp_path, ("p/s1.jsonl", t))
    assert (rep.all.requests, rep.all.output, rep.all.w1) == (1, 40, 1000)
    assert rep.stats.odd_requests == 1


def test_a_big_bash_result_pays_its_write_once_then_a_read_per_later_request(tmp_path):
    rep = run(tmp_path, ("p/s1.jsonl", bash_session(after=5)))
    [result] = items(rep, kind="tool_result")
    assert result.tool == "Bash" and result.family == "git status"
    assert result.tokens == pytest.approx(10000)
    assert result.calls == 6  # the request it arrived on plus 5 more
    [row] = [r for r in cc.summarize(rep)["top_items"] if r["kind"] == "tool_result"]
    assert row["later_requests"] == 5
    assert result.cost_in == pytest.approx(10000 * W5 + 5 * 10000 * RD)
    data = cc.summarize(rep)
    [fam] = [r for r in data["by_bash"] if r["name"] == "git status"]
    [call] = items(rep, kind="tool_use")
    assert fam["cost_usd"] == pytest.approx(result.cost + call.cost, abs=1e-6)
    assert call.cost_out == pytest.approx(50 * OUT)


def test_compaction_ends_older_items_and_starts_a_summary(tmp_path):
    t = bash_session(after=1)
    t.compact(pre=15100, post=2000)
    t.summary("s" * 4000)
    t.user("continue")
    t.request(read=4999, w5=2010, out=10)
    t.user("more")
    t.request(read=7009, w5=30, out=10)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    [result] = items(rep, kind="tool_result")
    assert result.calls == 2  # not charged after the compaction
    assert result.cost_in == pytest.approx(10000 * W5 + 10000 * RD)
    [summary] = items(rep, kind="summary")
    assert summary.calls == 2
    start = items(rep, kind="start")[0]
    assert summary.tokens == pytest.approx(7009 - start.tokens - len("continue") * cc.TOKENS_PER_CHAR)
    [comp] = rep.compactions
    assert (comp["trigger"], comp["pre_tokens"], comp["post_tokens"], comp["prompt_after"]) == ("auto", 15100, 2000, 7009)
    assert attributed(rep) == pytest.approx(rep.total_usd, rel=1e-4)


def test_a_drop_with_no_compaction_line_is_counted_and_nothing_is_removed(tmp_path):
    # Revision 2 of the spec: no proportional shrink; the drop is reported as unexplained.
    t = bash_session(after=1)
    t.user("x")
    t.request(read=3000, w5=0, out=5)  # 15,100 -> 3,000 tokens, no compaction line
    rep = run(tmp_path, ("p/s1.jsonl", t))
    assert rep.stats.unexplained_drops == 1
    assert rep.stats.unexplained_drop_tokens == 15100 - 3000
    [result] = items(rep, kind="tool_result")
    assert result.calls == 3
    assert attributed(rep) == pytest.approx(rep.total_usd, rel=1e-4)


@pytest.mark.parametrize("name,server,tool", [
    ("mcp__upwork__find_jobs", "upwork", "find_jobs"),
    ("mcp__upwork__upwork__find_jobs", "upwork", "upwork__find_jobs"),
    ("mcp__claude_ai_Slack__slack_send_message", "claude_ai_Slack", "slack_send_message"),
    ("mcp__plugin_x_y__do", "plugin_x_y", "do"),
    ("mcp__lonely", "lonely", "?"),
])
def test_mcp_names(name, server, tool):
    assert cc.mcp_parts(name) == (server, tool)


def test_mcp_result_is_grouped_by_server_and_tool(tmp_path):
    t = T()
    t.user("find")
    call = t.tool("mcp__upwork__find_jobs", {"query": "python"})
    t.request(w1=3000, out=30, blocks=[call])
    t.result(call["id"], [{"type": "text", "text": "j" * 9000}])
    t.request(read=3000, w5=4030, out=10)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    data = cc.summarize(rep)
    [server] = data["by_mcp"]
    assert server["name"] == "upwork" and [x["name"] for x in server["tools"]] == ["find_jobs"]
    assert server["cost_usd"] > 0


BASH_CASES = [
    ("git status", "git status"),
    ("cd x && FOO=1 timeout 60 git status | head", "git status"),
    ("", "(empty)"),
    ("   \n  ", "(empty)"),
    (None, "(empty)"),
    ("ls -la /tmp", "ls"),
    ("/usr/bin/python3 script.py", "python"),
    ("python3 -m pytest -q tests", "python -m pytest"),
    ("python -mpytest", "python -m pytest"),
    ("python3.12 -m pip install x", "python -m pip"),
    ("sudo -u root apt-get install x", "apt-get"),
    ("nice -n 10 timeout -s KILL 900 pytest -q", "pytest"),
    ("env -i FOO=1 BAR=2 make test", "make test"),
    ("time make -C dir build", "make build"),
    ("caffeinate -i -t 100 npm run build", "npm run"),
    ("cat <<'EOF' > f.txt\nhello 'world\nEOF", "cat"),
    ("python3 - <<'EOF'\nimport os\nprint(os.getcwd())\nEOF", "python"),
    ('python3 -c "import x\nprint(1)"', "python"),
    ('grep -r "foo|bar" . | wc -l', "grep"),
    ("echo 'a && b' && ls", "ls"),
    ("date && cd x && python -m pytest -q", "python -m pytest"),
    ("mkdir -p d && cat > d/f <<EOF\nhello\nEOF", "cat"),
    ("echo; echo ---; git diff", "git diff"),
    ("date", "date"),
    ("echo hi", "echo"),
    ("echo hi | grep h", "echo"),
    ("echo --- ; grep -r x .", "grep"),
    ("printf 'a\\n' > f && wc -l f", "wc"),
    ("date; sleep 5; date", "date"),
    ("touch f && rm -f g || true", "touch"),
    ("rm -rf build && make", "make"),
    ("date\npython3 run.py", "python"),
    ("echo start && cat a | sort", "cat"),
    ("mkdir -p x", "mkdir"),
    ("cd repo 2>&1 && git status", "git status"),
    ("mkdir -p out 2>&1; pytest", "pytest"),
    ("echo start >&2; npm test", "npm test"),
    ("rm -rf build &>/dev/null; make", "make"),
    ("sleep 2 &>/dev/null && curl x", "curl"),
    ("mkdir x >/dev/null 2>&1 && git diff", "git diff"),
    ("pytest >/dev/null 2>&1 && git diff", "pytest"),
    ("touch log &>>out.log && tail log", "tail"),
    ("make & wait; ls", "make"),
    ("git " + "y" * 300, "git " + "y" * 33 + "..."),
    ("docker --context prod ps", "docker ps"),
    ("kubectl -n kube-system get pods", "kubectl get"),
    ("aws --profile x s3 ls", "aws s3"),
    ("gh pr view 12", "gh pr"),
    ("uv run pytest", "uv run pytest"),
    ("uv run --with rich python -m pytest -q", "uv run python -m pytest"),
    ("uv run python script.py", "uv run python"),
    ("poetry run black .", "poetry run black"),
    ("/x/.venv/bin/python -m pip list", "python -m pip"),
    ("for f in *.py; do\n  wc -l \"$f\"\ndone", "wc"),
    ("while true; do curl -s localhost; sleep 1; done", "curl"),
    ("while read l; do grep x $l; done < files.txt", "grep"),
    ("if [ -f x ]; then cat x; fi", "cat"),
    ("if grep -q a b; then echo y; fi", "grep"),
    ("case $x in a) docker ps;; esac", "docker ps"),
    ("for i in 1 2 3; do :; done", "shell for (no command found)"),
    ("{ ls; pwd; } > out", "ls"),
    ("export A=1; cd /x; pnpm install", "pnpm install"),
    ("# comment\nls", "ls"),
    ("bash -c 'git log -1'", "git log"),
    ("bash -lc 'cd x && npm test'", "npm test"),
    ("(cd sub && cargo build)", "cargo build"),
    ("cargo +nightly build", "cargo build"),
    ("cd /tmp", "(setup only)"),
    ("FOO=bar", "(setup only)"),
    ("git -C /some/path log --oneline", "git log"),
    ("./scripts/run.sh --fast", "run.sh"),
    ("npx prettier --check .", "npx prettier"),
    ("git status 'unterminated", "git status"),
    ("make", "make"),
    ("pip install -r req.txt", "pip install"),
    ("go test ./...", "go test"),
    ("cd a \\\n && git diff", "git diff"),
    ("source .venv/bin/activate && python -m pytest", "python -m pytest"),
    ("cd /repo\ngit add -A", "git add"),
    ("sed -n 1,10p file", "sed"),
    ("for f in *.py; do echo $f; done", "echo"),
    ("for f in *.py; do echo $f; wc -l $f; done", "wc"),
    ("git /etc/passwd", "git"),
]


@pytest.mark.parametrize("command,family", BASH_CASES, ids=[repr(c)[:40] for c, _ in BASH_CASES])
def test_bash_family(command, family):
    assert cc.bash_family(command) == family


def test_bash_family_never_returns_a_path_or_argument_values():
    for command, _ in BASH_CASES:
        fam = cc.bash_family(command)
        assert "/" not in fam and "=" not in fam


def test_shell_keywords_are_never_a_family():
    for command, _ in BASH_CASES:
        assert cc.bash_family(command).split(" ")[0] not in cc.SHELL_KEYWORDS - {"time"}


def write_subagent(root, agent_id, t: T, meta: dict | None, workflow=None, project="p", session="s1"):
    folder = root / project / session / "subagents"
    if workflow:
        folder = folder / "workflows" / workflow
    path = t.write(folder / f"agent-{agent_id}.jsonl")
    if meta is not None:
        (folder / f"agent-{agent_id}.meta.json").write_text(json.dumps(meta))
    return path


def subagent_transcript(cost_tokens=20000) -> T:
    s = T(session="s1")
    s.user("task for the agent")
    call = s.bash("rg TODO")
    s.request(w5=cost_tokens, out=10, blocks=[call])
    s.result(call["id"], "r" * 2000)
    s.request(read=cost_tokens, w5=910, out=30, blocks=[{"type": "text", "text": "found it"}])
    return s


def test_a_subagent_is_its_own_context_rolled_up_to_its_type_and_the_parent_agent_call(tmp_path):
    t = T()
    t.user("delegate")
    call = t.tool("Agent", {"subagent_type": "Explore", "description": "look", "prompt": "find TODOs"})
    t.request(w1=4000, out=40, blocks=[call])
    t.result(call["id"], "the agent says: 3 TODOs")
    t.request(read=4000, w5=50, out=10)
    root = tmp_path / "projects"
    t.write(root / "p" / "s1.jsonl")
    write_subagent(root, "a1", subagent_transcript(), {"agentType": "Explore", "toolUseId": call["id"], "spawnDepth": 1})
    rep = cc.analyze(root)
    sub = [c for c in rep.contexts if c.info.is_sub]
    assert len(sub) == 1 and sub[0].requests == 2
    expected = (20000 * W5 + 10 * OUT) + (20000 * RD + 910 * W5 + 30 * OUT)
    assert sub[0].cost == pytest.approx(expected)
    [agent_result] = [i for i in items(rep, kind="tool_result") if i.tool == "Agent"]
    assert agent_result.subagent_usd == pytest.approx(expected)
    data = cc.summarize(rep)
    [row] = data["by_agent"]
    assert (row["name"], row["runs"], row["linked_to_agent_call"]) == ("Explore", 1, 1)
    assert row["cost_usd"] == pytest.approx(expected, abs=1e-6)
    assert rep.links == {"agent call": 1}
    assert data["subagents"]["requests"] == 2 and data["main"]["requests"] == 2
    assert attributed(rep) == pytest.approx(rep.total_usd, rel=1e-4)


def test_subagent_links_fall_back_to_parent_agent_then_workflow_then_unlinked(tmp_path):
    root = tmp_path / "projects"
    T().write(root / "p" / "s1.jsonl")
    write_subagent(root, "a1", subagent_transcript(), {"agentType": "general-purpose"})
    write_subagent(root, "a2", subagent_transcript(), {"agentType": "general-purpose", "parentAgentId": "a1"})
    write_subagent(root, "w1", subagent_transcript(), {"agentType": "workflow-subagent", "workflowPhase": "x"},
                   workflow="wf_123")
    write_subagent(root, "a3", subagent_transcript(), None)
    rep = cc.analyze(root)
    assert rep.links == {"unlinked": 2, "parent subagent": 1, "workflow": 1}
    assert rep.stats.subagents_without_meta == 1
    names = {r["name"] for r in cc.summarize(rep)["by_agent"]}
    assert names == {"general-purpose", "workflow-subagent", "(no type)"}


def test_an_image_result_gets_the_leftover_of_its_step(tmp_path):
    t = T()
    t.user("look")
    shot, read = t.tool("Screenshot", {}), t.tool("Read", {"file_path": "/work/a.txt"})
    t.request(w1=3000, out=10, blocks=[shot, read])
    t.result(shot["id"], [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}])
    t.result(read["id"], "t" * 1000)
    t.request(read=3000, w5=10 + 450 + 1200, out=5)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    img = [i for i in items(rep, kind="tool_result") if i.tool == "Screenshot"][0]
    txt = [i for i in items(rep, kind="tool_result") if i.tool == "Read"][0]
    assert txt.tokens == pytest.approx(1000 * cc.TOKENS_PER_CHAR)
    assert img.tokens == pytest.approx(1200)
    assert rep.stats.images == 1


def test_a_tool_reference_result_is_sized_from_the_leftover(tmp_path):
    t = T()
    t.user("find a tool")
    call = t.tool("ToolSearch", {"query": "slack"})
    t.request(w1=3000, out=10, blocks=[call])
    t.result(call["id"], [{"type": "tool_reference", "tool_name": "mcp__slack__send"}])
    t.request(read=3000, w5=10 + 800, out=5)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    [res] = items(rep, kind="tool_result")
    assert res.tool == "ToolSearch" and res.tokens == pytest.approx(800)


def test_synthetic_and_unknown_models_count_tokens_but_no_dollars(tmp_path):
    t = T()
    t.user("x")
    t.request(model="claude-unknown-9", w1=1000, out=100)
    t.request(model="<synthetic>", out=0, blocks=[{"type": "text", "text": "API Error"}])
    rep = run(tmp_path, ("p/s1.jsonl", t))
    assert rep.unpriced == {"claude-unknown-9": 1100}
    assert rep.stats.not_api_calls == 1
    assert rep.total_usd == 0
    priced = run(tmp_path / "b", ("p/s1.jsonl", t), price=2)
    assert priced.total_usd == pytest.approx(1000 * 2 * 2 / 1e6 + 100 * 10 / 1e6)  # 1h write 2x, output 5x


def test_unknown_lines_and_fields_are_counted_never_fatal(tmp_path):
    t = bash_session(after=1)
    t.raw({"type": "brand-new-type", "payload": {"a": 1}})
    t.raw({"type": "api-request-blob", "content": "big"})
    t.raw(["not", "an", "object"])
    t.raw({"type": "assistant", "uuid": "zz1", "message": "not a dict"})
    t.raw({"type": "user", "uuid": "zz2", "message": {"content": [{"type": "mystery"}, "junk", 5]}})
    t.raw({"type": "attachment", "uuid": "zz3", "attachment": "weird", "rendered": "not a list"})
    t.raw({"type": "assistant", "uuid": "zz4", "message": {"id": "m9", "usage": {"input_tokens": "lots",
                                                                                   "output_tokens": -5,
                                                                                   "cache_creation": "x"}}})
    path = tmp_path / "projects" / "p" / "s1.jsonl"
    t.write(path)
    with path.open("a") as fh:
        fh.write("{this is not json\n\n")
    rep = cc.analyze(tmp_path / "projects")
    assert rep.stats.ignored["brand-new-type"] == 1 and rep.stats.ignored["api-request-blob"] == 1
    assert rep.stats.unreadable_lines == 2  # the list and the broken line
    assert rep.all.requests == 3
    assert attributed(rep) == pytest.approx(rep.total_usd, rel=1e-4)


def test_no_transcripts_and_bad_since_exit_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "nothing"))
    assert cli.main(["claude-code"]) == 2
    assert "no Claude Code transcripts folder" in capsys.readouterr().err
    (tmp_path / "empty" / "projects" / "p").mkdir(parents=True)
    assert cli.main(["claude-code", "--path", str(tmp_path / "empty" / "projects")]) == 2
    assert "no Claude Code transcripts (*.jsonl)" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        cli.main(["claude-code", "--since", "banana"])
    assert exc.value.code == 2
    assert "use a length like 30d" in capsys.readouterr().err
    root = tmp_path / "ok" / "projects"
    bash_session().write(root / "p" / "s1.jsonl")
    future = (NOW + timedelta(days=2)).date().isoformat()
    assert cli.main(["claude-code", "--path", str(root), "--since", future]) == 2
    assert "--since must be in the past" in capsys.readouterr().err
    for bad in ("0d", "-3d", "30x", "2026-13-45"):
        with pytest.raises(SystemExit):
            cli.main(["claude-code", "--path", str(root), "--since", bad])


def test_attributed_dollars_add_up_to_the_total(tmp_path):
    """Everything at once: parallel tools, attachments, an image, a rewind, a model switch, a
    thinking drop, a compaction, a subagent and web search fees."""
    t = T()
    t.attachment("instructions", "c" * 3000)
    t.attachment("skill_listing", "k" * 1000)
    t.attachment("prompt_snapshot", None)
    t.user("start")
    a, b = t.bash("pytest -q"), t.tool("Read", {"file_path": "/work/x.py"})
    first = t.request(w1=8000, out=120, thinking=60, blocks=[{"type": "thinking", "thinking": "", "signature": "s"}, a, b])
    t.result(a["id"], "p" * 6000, parent=first[1], source=first[1])
    t.result(b["id"], [{"type": "text", "text": "r" * 3000}, {"type": "image", "source": {}}], parent=first[2], source=first[2])
    t.request(read=8000, w5=120 + 2700 + 1350 + 500, out=40, blocks=[{"type": "text", "text": "done"}])
    keep = t.user("question two")
    t.request(read=12670, w5=60, out=30, blocks=[{"type": "text", "text": "answer"}])
    t.user("rewound question", parent=keep)
    t.request(read=12670, w5=20, out=30, blocks=[{"type": "text", "text": "other answer"}])
    t.request(model="claude-sonnet-5-5", w5=12000, out=10, minutes=2)
    t.attachment("thinking_drop", "dropped")
    t.user("go on")
    t.request(model="claude-sonnet-5-5", read=11000, w5=500, out=10, minutes=1)
    t.compact()
    t.summary("summary " * 300)
    t.request(model="claude-sonnet-5-5", read=6000, w5=1500, out=20, minutes=10)
    t.cost_state(4)
    root = tmp_path / "projects"
    t.write(root / "p" / "s1.jsonl")
    write_subagent(root, "a1", subagent_transcript(), {"agentType": "Explore"})
    rep = cc.analyze(root)
    assert rep.web_searches == 4 and rep.web_search_usd == pytest.approx(0.04)
    assert rep.stats.branches == 1 and rep.stats.model_switches == 1
    assert len(rep.compactions) == 1
    assert rep.total_usd > 0
    assert attributed(rep) == pytest.approx(rep.total_usd, rel=1e-4)
    data = cc.summarize(rep, top=1000)
    assert sum(r["cost_usd"] for r in data["by_what"]) == pytest.approx(rep.total_usd, rel=1e-4)


# --- context details ------------------------------------------------------------------------------

def test_parallel_tool_results_beside_the_chain_are_in_the_context(tmp_path):
    t = T()
    t.user("two at once")
    a, b = t.bash("ls"), t.bash("pwd")
    lines = t.request(w1=3000, out=20, blocks=[a, b])
    t.result(a["id"], "1" * 1000, parent=lines[0], source=lines[0])  # beside the chain
    t.result(b["id"], "2" * 3000, parent=lines[1], source=lines[1])
    t.request(read=3000, w5=20 + 1800, out=5)
    t.user("then")
    t.request(read=4825, w5=10, out=5)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    res = sorted(items(rep, kind="tool_result"), key=lambda i: i.chars)
    assert [r.tokens for r in res] == [pytest.approx(450), pytest.approx(1350)]
    assert all(r.calls == 2 for r in res)
    assert rep.stats.branches == 0


def test_a_rewind_continues_from_the_older_request(tmp_path):
    t = T()
    t.user("one")
    t.request(w1=3000, out=10, blocks=[{"type": "text", "text": "a"}])
    fork = t.last
    t.user("abandoned question")
    t.request(read=3010 - 10, w5=10 + 100, out=10, blocks=[{"type": "text", "text": "b"}])
    t.user("second try", parent=fork)
    t.request(read=3000, w5=10 + 40, out=10)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    assert rep.stats.branches == 1
    abandoned = [i for i in items(rep, kind="text") if i.chars == len("abandoned question")][0]
    retry = [i for i in items(rep, kind="text") if i.chars == len("second try")][0]
    assert abandoned.calls == 1 and retry.calls == 1
    assert attributed(rep) == pytest.approx(rep.total_usd, rel=1e-4)


def test_thinking_takes_its_exact_share_of_the_output(tmp_path):
    t = T()
    t.user("think")
    t.request(w1=2000, out=1000, thinking=900,
              blocks=[{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "short"}])
    t.user("ok")
    t.request(read=2000, w5=1000 + 50, out=5)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    [think] = items(rep, kind="thinking")
    [reply] = [i for i in items(rep, kind="reply") if i.chars == 5]
    assert (think.tokens, reply.tokens) == (pytest.approx(900), pytest.approx(100))
    assert think.cost_out == pytest.approx(900 * OUT)
    assert think.group == cc.THINKING


def test_a_thinking_drop_removes_earlier_thinking(tmp_path):
    t = T()
    t.user("think")
    t.request(w1=2000, out=1000, thinking=1000, blocks=[{"type": "thinking", "thinking": "", "signature": "s"}])
    t.user("ok")
    t.request(read=2000, w5=1010, out=5)
    t.attachment("thinking_drop", "earlier thinking was dropped")
    t.user("later")
    t.request(read=2000, w5=30, out=5)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    [think] = items(rep, kind="thinking")
    assert think.calls == 1
    assert rep.stats.thinking_removals == 1
    assert "thinking removed" in rep.rebuilds or rep.stats.unexplained_drops == 0


def test_a_model_switch_rescales_and_counts_as_a_rebuild(tmp_path):
    t = bash_session(after=1)
    t.user("switch")
    t.request(model="claude-haiku-4-5-20251001", w5=17000, out=5)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    assert rep.stats.model_switches == 1
    assert rep.rebuilds["model changed"][0] == 1
    assert rep.stats.unexplained_drops == 0
    assert attributed(rep) == pytest.approx(rep.total_usd, rel=1e-4)


def test_attachments_are_their_own_groups_and_unrendered_ones_are_not_sent(tmp_path):
    t = T()
    t.attachment("instructions", "c" * 2000)
    t.attachment("prompt_snapshot", None, systemPrompt="ignored")
    t.attachment("brand_new_attachment", "n" * 200)
    t.user("hi")
    t.request(w1=5000, out=5)
    t.attachment("edited_text_file", "e" * 1000, filename="x")
    t.user("again")
    t.request(read=5000, w5=5 + 450 + 3, out=5)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    groups = {i.group for i in items(rep, kind="attachment")}
    assert groups == {"CLAUDE.md and instructions", "other attachments", "file changes and mentions"}
    assert rep.stats.attachments_not_rendered == {"prompt_snapshot": 1}
    assert rep.stats.unknown_attachments == {"brand_new_attachment": 1}


def test_a_tool_result_without_its_call_is_kept_as_unknown_tool(tmp_path):
    t = T()
    t.user("x")
    t.request(w1=1000, out=5)
    t.result("toolu_missing", "orphan output")
    t.request(read=1000, w5=10, out=5)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    [res] = items(rep, kind="tool_result")
    assert res.tool == "(unknown tool)" and rep.stats.orphan_results == 1


def test_a_repeated_uuid_keeps_its_first_place_and_last_content(tmp_path):
    t = T()
    uid = t.user("a" * 100)
    t.raw({**t.lines[-1], "message": {"role": "user", "content": "b" * 400}})
    t.request(w1=1000, out=5)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    [msg] = items(rep, kind="text")
    assert msg.chars == 400 and msg.line == uid
    assert rep.stats.repeated_lines == 1


def test_a_request_seen_in_two_files_counts_once_and_orphaned_files_are_skipped(tmp_path):
    t = bash_session(after=0)
    root = tmp_path / "projects"
    t.write(root / "p" / "s1.jsonl")
    t.write(root / "p" / "s2.jsonl")
    t.write(root / "p" / "s3.orphaned-1700000000-ab12.jsonl")
    rep = cc.analyze(root)
    assert rep.all.requests == 2
    assert rep.stats.duplicate_requests == 2
    assert rep.stats.files_skipped_orphaned == 1


def test_web_search_fees_come_from_cost_state_and_restart_per_process(tmp_path):
    t = bash_session(after=0)
    for n in (1, 3, 5, 2):  # a resume in a new process restarts the count at 2
        t.cost_state(n)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    assert rep.web_searches == 7
    [fee] = items(rep, kind="fee")
    assert fee.group == "WebSearch" and fee.cost == pytest.approx(0.07)
    assert rep.total_usd == pytest.approx(rep.all.cost + 0.07)


def test_idle_rebuild_is_priced_as_write_minus_read(tmp_path):
    t = bash_session(after=0)
    t.user("back after lunch")
    t.request(read=0, w5=15050 + 30, out=5, minutes=45)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    count, extra = rep.rebuilds["idle over 5m"]
    assert count == 1
    assert extra == pytest.approx(15050 * (W5 - RD))


def test_one_hour_writes_are_charged_before_five_minute_writes(tmp_path):
    t = T()
    t.user("q" * 1000)  # 450 tokens
    t.request(w1=1000, w5=450, out=0)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    [start] = items(rep, kind="start")
    [msg] = items(rep, kind="text")
    assert start.cost_in == pytest.approx(1000 * W1)
    assert msg.cost_in == pytest.approx(450 * W5)


def test_haiku_55_long_prompt_tier_end_to_end(tmp_path):
    t = T(model="claude-haiku-5-5")
    t.user("x")
    t.request(w5=100_001, out=10)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    assert rep.total_usd == pytest.approx(100_001 * 0.625e-6 + 10 * 2.5e-6)
    t2 = T(model="claude-haiku-5-5")
    t2.user("x")
    t2.request(w5=100_000, out=10)
    assert run(tmp_path / "b", ("p/s1.jsonl", t2)).total_usd == pytest.approx(100_000 * 0.125e-6 + 10 * 0.5e-6)


def test_partial_last_line_is_skipped_quietly(tmp_path):
    path = bash_session(after=0).write(tmp_path / "projects" / "p" / "s1.jsonl", partial_tail='{"type":"user","uu')
    rep = cc.analyze(path.parents[1])
    assert rep.stats.partial_last_lines == 1 and rep.stats.unreadable_lines == 0


# --- window, filters, folders -----------------------------------------------------------------------

def test_since_filters_by_request_time_not_file_time(tmp_path):
    t = T(start=NOW - timedelta(days=40))
    t.user("old")
    t.request(w1=1000, out=5)
    t.clock = NOW - timedelta(hours=1)
    t.user("new")
    t.request(read=1000, w5=10, out=5)
    rep = run(tmp_path, ("p/s1.jsonl", t), since=(NOW - timedelta(days=30)).timestamp())
    assert rep.all.requests == 1
    assert rep.total_usd == pytest.approx(1000 * RD + 10 * W5 + 5 * OUT)
    assert attributed(rep) == pytest.approx(rep.total_usd, rel=1e-4)


def test_files_not_changed_in_the_window_are_skipped_unread(tmp_path):
    path = bash_session().write(tmp_path / "projects" / "p" / "s1.jsonl")
    old = time.time() - 40 * 86400
    os.utime(path, (old, old))
    rep = cc.analyze(tmp_path / "projects", since=time.time() - 30 * 86400)
    assert rep.stats.files == 0 and rep.files_found == 1


def test_project_and_session_filters(tmp_path, capsys):
    root = tmp_path / "projects"
    bash_session().write(root / "-work-alpha" / "aaa111.jsonl")
    bash_session(session="b").write(root / "-work-beta" / "bbb222.jsonl")
    assert cc.analyze(root, project="ALPHA").stats.files == 1
    assert cc.analyze(root, session="bbb").stats.files == 1
    assert cc.analyze(root, project="gamma").stats.files == 0
    assert cli.main(["claude-code", "--path", str(root), "--project", "gamma"]) == 0
    assert "No Claude Code requests in this window" in capsys.readouterr().out


def test_claude_config_dir_is_respected(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    bash_session().write(tmp_path / "cfg" / "projects" / "p" / "s1.jsonl")
    assert cc.default_root() == tmp_path / "cfg" / "projects"
    assert cli.main(["claude-code"]) == 0
    assert "requests" in capsys.readouterr().out
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    assert cc.default_root() == cc.Path.home() / ".claude" / "projects"


def test_path_is_repeatable(tmp_path, capsys):
    bash_session().write(tmp_path / "a" / "p" / "s1.jsonl")
    bash_session(session="x").write(tmp_path / "b" / "p" / "s9.jsonl")
    rep = cc.analyze([tmp_path / "a", tmp_path / "b"])
    assert rep.stats.files == 2


def test_retention_note_reads_cleanup_period(tmp_path, capsys):
    cfg = tmp_path / "cfg"
    bash_session().write(cfg / "projects" / "p" / "s1.jsonl")
    (cfg / "settings.json").write_text(json.dumps({"cleanupPeriodDays": 14}))
    assert cli.main(["claude-code", "--path", str(cfg / "projects")]) == 0
    assert "older than 14 days (cleanupPeriodDays)" in capsys.readouterr().out


# --- command line output --------------------------------------------------------------------------

JSON_KEYS = {"schema", "window", "price_basis", "price_note", "total_usd", "cost_usd", "web_searches", "sessions",
             "subagent_runs", "all", "main", "subagents", "first_calls", "rebuilds", "compactions", "unexplained_drops",
             "unpriced_models", "models", "by_what", "by_tool", "by_bash", "by_mcp", "by_skill", "by_agent",
             "top_items", "headline", "suggestions", "notes", "stats"}


def test_json_schema_is_stable(tmp_path, capsys):
    root = tmp_path / "projects"
    bash_session().write(root / "p" / "s1.jsonl")
    assert cli.main(["--json", "claude-code", "--path", str(root)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert cli.main(["claude-code", "--path", str(root), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data) == JSON_KEYS and set(first) == JSON_KEYS
    assert data["schema"] == "cachecanary.claude-code/1"
    assert data["price_note"] == "API list price equivalent, not a subscription bill"
    assert set(data["all"]) == {"requests", "uncached_input", "cache_read", "cache_write_5m", "cache_write_1h",
                                "output", "hit_ratio", "cost_usd"}
    assert set(data["cost_usd"]) == {"uncached_input", "cache_read", "cache_write_5m", "cache_write_1h", "output",
                                     "web_search_fees"}
    row = data["top_items"][0]
    assert set(row) == {"kind", "group", "tool", "bash_family", "mcp_server", "mcp_tool", "skill", "tokens",
                        "later_requests", "cost_usd", "lasting_input_usd", "output_usd", "subagent_usd",
                        "session", "subagent"}
    assert set(data["by_tool"][0]) == {"name", "cost_usd", "lasting_input_usd", "output_usd", "tokens",
                                        "tokens_added_in_window", "items"}
    assert data["total_usd"] == pytest.approx(sum(r["cost_usd"] for r in data["by_what"]), rel=1e-4)


def test_text_header_says_list_price_equivalent_and_no_dashes(tmp_path, capsys):
    root = tmp_path / "projects"
    bash_session().write(root / "p" / "s1.jsonl")
    assert cli.main(["claude-code", "--path", str(root)]) == 0
    out = capsys.readouterr().out
    assert "This is an API list price equivalent, not your subscription bill." in out.splitlines()[1]
    assert "Bash command families" in out and "git status" in out
    assert "\u2014" not in out and "\u2013" not in out


def test_by_shows_one_table(tmp_path, capsys):
    root = tmp_path / "projects"
    bash_session().write(root / "p" / "s1.jsonl")
    assert cli.main(["claude-code", "--path", str(root), "--by", "bash"]) == 0
    out = capsys.readouterr().out
    assert "Bash command families" in out
    assert "Where the money went" not in out and "Costliest single items" not in out


def test_min_hit_exits_1_when_missed(tmp_path, capsys):
    root = tmp_path / "projects"
    bash_session().write(root / "p" / "s1.jsonl")
    assert cli.main(["claude-code", "--path", str(root), "--min-hit", "0.99"]) == 1
    assert "below --min-hit" in capsys.readouterr().out
    assert cli.main(["claude-code", "--path", str(root), "--min-hit", "0.1"]) == 0


SECRET_COMMAND = ("API_TOKEN=sk-live-0123456789abcdef0123 curl -H 'Authorization: Bearer abcdefSECRET99' "
                  "https://bob:hunter2@api.example.com/v1?token=qwertyuiop /Users/bob/private/file")


def secret_session() -> T:
    t = T()
    t.user("my question mentions PRIVATE_PROMPT_TEXT")
    call = t.bash(SECRET_COMMAND)
    t.request(w1=3000, out=30, blocks=[call, {"type": "text", "text": "PRIVATE_REPLY"}])
    t.result(call["id"], "SECRET_OUTPUT " * 500)
    t.request(read=3000, w5=30 + 3150, out=10)
    return t


LEAKS = ("sk-live", "abcdefSECRET99", "hunter2", "qwertyuiop", "PRIVATE_PROMPT_TEXT", "PRIVATE_REPLY",
         "SECRET_OUTPUT", "/Users/bob", "example.com")


def test_default_output_has_no_transcript_content(tmp_path, capsys):
    root = tmp_path / "projects"
    secret_session().write(root / "p" / "s1.jsonl")
    for args in ([], ["--json"], ["--debug-shapes"], ["--debug-shapes", "--json"]):
        assert cli.main(["claude-code", "--path", str(root), *args]) == 0
        out = capsys.readouterr().out
        for leak in LEAKS:
            assert leak not in out, (args, leak)


def test_details_labels_are_redacted(tmp_path, capsys):
    root = tmp_path / "projects"
    secret_session().write(root / "p" / "s1.jsonl")
    assert cli.main(["claude-code", "--path", str(root), "--details", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    labels = [r["label"] for r in data["top_items"] if r.get("label")]
    assert labels and any(lab.startswith("curl") for lab in labels)
    for leak in ("sk-live", "abcdefSECRET99", "hunter2", "qwertyuiop", "/Users/bob", "SECRET_OUTPUT"):
        assert all(leak not in lab for lab in labels), leak


@pytest.mark.parametrize("text,gone", [
    ("curl -H 'Authorization: Bearer abc.def.ghi'", "abc.def"),
    ("psql postgres://user:pa55@db:5432/x", "pa55"),
    ("export KEY=AKIAABCDEFGHIJKLMNOPQRST", "AKIAABCDEFGHIJKLMNOPQRST"),
    ("deploy --password=letmein", "letmein"),
    ("cat /home/alice/.ssh/id_rsa", "alice"),
    ("echo 0123456789abcdef0123456789abcdef", "0123456789abcdef"),
    ("curl -H 'Authorization: Bearer abc'", "Bearer"),
    ("mail bob.smith@example.org", "bob.smith"),
    ("export OPENAI=sk-abc", "sk-"),
    ("login --password hunter2", "password"),
    ("DB_PASSWORD=x ./run", "PASSWORD"),
])
def test_redact(text, gone):
    assert gone not in cc.redact(text, 200)


def test_debug_shapes_counts_types_and_warns_on_new_versions(tmp_path, capsys):
    t = secret_session()
    t.lines[-1]["version"] = "2.9.0"
    root = tmp_path / "projects"
    t.write(root / "p" / "s1.jsonl")
    assert cli.main(["claude-code", "--path", str(root), "--debug-shapes", "--json"]) == 0
    shapes = json.loads(capsys.readouterr().out)
    assert shapes["newer_than_checked"] is True and shapes["versions"]["max"] == "2.9.0"
    assert shapes["line_types_read"]["assistant"] == 3
    assert "tool_use" in "".join(shapes["content_block_types"])


def test_findings_lead_with_the_biggest_cost(tmp_path):
    rep = run(tmp_path, ("p/s1.jsonl", bash_session(after=30)))
    data = cc.summarize(rep)
    assert data["headline"]["text"].startswith("Biggest cost: ")
    assert len(data["suggestions"]) <= 3
    assert any(s["kind"] == "bash" and "'git status'" in s["text"] for s in data["suggestions"])
    for s in data["suggestions"]:
        assert "save" not in s["text"].lower()


def test_existing_commands_are_still_listed():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    assert {"lint", "diff", "probe", "logs", "langfuse", "claude-code"} <= set(sub.choices)


def test_bedrock_ids_and_unknown_usage_values_are_noted(tmp_path, capsys):
    t = T(model="us.anthropic.claude-opus-5-5")
    t.user("x")
    t.request(w5=1000, out=10, usages=[{"service_tier": "priority", "speed": "turbo"}])
    root = tmp_path / "projects"
    t.write(root / "p" / "s1.jsonl")
    assert cli.main(["claude-code", "--path", str(root)]) == 0
    out = capsys.readouterr().out
    assert "not been checked on real Bedrock or Vertex transcripts" in out
    assert "priority (1)" in out and "turbo (1)" in out
    rep = cc.analyze(root)
    assert rep.total_usd == pytest.approx(1000 * 5.5e-6 + 10 * 22e-6)  # Bedrock regional 1.1x


def test_costliest_items_are_single_outputs_not_starting_contexts(tmp_path, capsys):
    root = tmp_path / "projects"
    t = bash_session(after=3)
    t.compact(pre=15100, post=2000)
    t.summary("s" * 40000)
    t.user("continue")
    t.request(read=4999, w5=18010, out=10)
    t.write(root / "p" / "s1.jsonl")
    data = cc.summarize(cc.analyze(root))
    kinds = {r["kind"] for r in data["top_items"]}
    assert kinds <= {"tool_result", "tool_use", "text", "attachment"}
    assert data["top_items"][0]["bash_family"] == "git status" and data["top_items"][0]["kind"] == "tool_result"
    assert cli.main(["claude-code", "--path", str(root), "--by", "item"]) == 0
    out = capsys.readouterr().out
    section = out.split("Costliest single outputs and messages", 1)[1].split("Notes:", 1)[0]
    assert "Bash 'git status'" in section
    assert cc.START not in section and cc.SUMMARY not in section


def test_skill_table_shows_instructions_cost_and_loads(tmp_path, capsys):
    t = T()
    t.user("use the skill")
    call = t.tool("Skill", {"skill": "anthropic-skills:pdf"})
    t.request(w1=3000, out=10, blocks=[call])
    t.result(call["id"], "Launching skill")
    t.user("Base directory for this skill: x\n" + "i" * 4000, isMeta=True, sourceToolUseID=call["id"])
    t.request(read=3000, w5=10 + 1810, out=5, attributionSkill="pdf")
    root = tmp_path / "projects"
    t.write(root / "p" / "s1.jsonl")
    data = cc.summarize(cc.analyze(root))
    [row] = data["by_skill"]
    assert (row["name"], row["loads"]) == ("anthropic-skills:pdf", 1)
    assert "spent_while_active_usd" not in row
    assert cli.main(["claude-code", "--path", str(root), "--by", "skill"]) == 0
    out = capsys.readouterr().out
    assert "1 load   anthropic-skills:pdf" in out and "while active" not in out


def test_headline_names_the_top_bash_families(tmp_path):
    t = bash_session(after=10)
    call = t.bash("sed -n 1,400p big.py")
    t.request(read=15600, w5=10, out=40, blocks=[call])
    t.result(call["id"], "s" * 8000)
    t.request(read=15650, w5=3600, out=5)
    data = cc.summarize(run(tmp_path, ("p/s1.jsonl", t)))
    text = data["headline"]["text"]
    assert text.startswith("Biggest cost: Bash, ") and "; mostly 'git status' " in text and "'sed' " in text


def test_cheap_model_note_is_factual(tmp_path):
    big = T(model="claude-haiku-5-5", session="h")
    big.user("x")
    call = big.tool("mcp__jobs__find", {"q": "a"})
    big.request(w5=3000, out=10, blocks=[call])
    big.result(call["id"], "j" * 200000)
    big.request(read=3000, w5=90010, out=5)
    small = bash_session(after=5)
    rep = run(tmp_path, ("p/h.jsonl", big), ("p/s1.jsonl", small))
    notes = cc.cheap_model_notes(rep)
    assert notes and notes[0].startswith("mcp__jobs__find: ") and "claude-haiku-5-5 requests" in notes[0]
    assert all("Bash" not in n for n in notes)


def test_every_row_with_dollars_has_tokens(tmp_path):
    """Items that entered the context before the window still show their size, and output that no
    later request read shows its output tokens."""
    t = T(start=NOW - timedelta(days=40))
    t.user("old")
    call = t.tool("Skill", {"skill": "schedule"})
    t.request(w1=3000, out=10, blocks=[call])
    t.result(call["id"], "ok")
    t.user("Base directory for this skill: x\n" + "i" * 9000, isMeta=True, sourceToolUseID=call["id"])
    t.request(read=3000, w5=10 + 4100, out=5)
    t.clock = NOW - timedelta(hours=2)
    t.user("new")
    last = t.bash("pytest -q")
    t.request(read=7110, w5=5, out=40, blocks=[last])  # its call is never read again
    root = tmp_path / "projects"
    t.write(root / "p" / "s1.jsonl")
    data = cc.summarize(cc.analyze(root, since=(NOW - timedelta(days=30)).timestamp()))
    for table in ("by_what", "by_tool", "by_bash", "by_skill"):
        for row in data[table]:
            if row["cost_usd"] > 0:
                assert row["tokens"] > 0, (table, row)
    [skill] = data["by_skill"]
    assert skill["tokens"] > 4000 and skill["tokens_added_in_window"] == 0


def test_float_rounding_never_creates_a_phantom_item(tmp_path):
    """100 output tokens shared 1:1:1 sum to 99.99999999999999 (math.fsum and Python 3.12's sum alike):
    the 1.4e-14 left over must not become a 'growth not in the transcript' item."""
    t = T()
    t.user("x")
    t.request(w1=3000, out=100, blocks=[{"type": "text", "text": c} for c in "abc"])
    t.request(read=3000, w5=100, out=5)  # grows by exactly the previous output, nothing else added
    t.request(read=3100, w5=5, out=5)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    assert not items(rep, kind="unlogged")
    assert rep.stats.unexplained_drops == 0
    assert attributed(rep) == pytest.approx(rep.total_usd, rel=1e-9)


def test_tiny_leftovers_go_nowhere():
    ctx = cc.Context(cc.ContextInfo(cc.Path("x"), "p", "s", False), cc.Report())
    req = cc.Request(("m", "r"), OPUS, None, "u")
    req.input, req.output = 10, 100
    blocks = [cc.Item("reply", cc.REPLIES, 1) for _ in range(3)]
    for b in blocks:
        b.from_req = req
    req.blocks = blocks
    ctx.split_output(req)
    image = cc.Item("tool_result", "Screenshot", nochar=True, image=True)
    growth = cc.fsum(b.out_tokens for b in blocks) + 1e-12
    new, sizes = ctx._grow_sizes(blocks + [image], growth)
    assert len(new) == 4 and sizes[3] == 0.0
    new, sizes = ctx._grow_sizes(list(blocks), growth - 2e-12)  # just under: still the exact shares
    assert sizes == [b.out_tokens for b in blocks]



# --- QA round 4 regressions ---------------------------------------------------------------------------

def test_web_search_fees_outside_the_window_are_left_out(tmp_path, capsys):
    t = T(start=NOW - timedelta(days=40))
    t.user("old")
    t.request(w1=1000, out=10)
    t.cost_state(500)
    t.clock = NOW - timedelta(hours=1)
    t.user("new")
    t.request(read=1000, w5=10, out=10)
    t.cost_state(503)
    root = tmp_path / "projects"
    t.write(root / "p" / "s1.jsonl")
    rep = cc.analyze(root, since=(NOW - timedelta(days=30)).timestamp())
    assert (rep.web_searches, rep.web_searches_outside) == (3, 500)
    assert rep.web_search_usd == pytest.approx(0.03)
    assert cli.main(["claude-code", "--path", str(root)]) == 0
    assert "500 searches outside the window were left out" in capsys.readouterr().out


def test_web_search_runs_are_told_apart_by_start_time(tmp_path):
    t = bash_session(after=0)
    for start, n in (("r1", 1), ("r1", 3), ("r2", 5), ("r2", 6)):  # run 2 starts above run 1's last count
        t.lines.append({"type": "cost-state", "sessionId": "s1", "startTime": start, "totalCostUSD": 1,
                        "modelUsage": {OPUS: {"webSearchRequests": n}}})
    rep = run(tmp_path, ("p/s1.jsonl", t))
    assert rep.web_searches == 9


@pytest.mark.parametrize("bad", [
    {"message": {"id": {"x": 1}}},
    {"requestId": ["r", 1]},
    {"type": [1]},
    {"type": {"a": "assistant"}},
    {"message": {"id": ["m"]}, "requestId": {"r": 2}},
])
def test_unhashable_ids_and_types_never_crash(tmp_path, bad):
    t = bash_session(after=1)
    line = json.loads(json.dumps(t.lines[3]))  # an assistant line
    for k, v in bad.items():
        if k == "message":
            line["message"].update(v)
            line["uuid"] = "odd-1"
        else:
            line[k] = v
    t.raw(line)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    assert rep.all.requests >= 3
    assert attributed(rep) == pytest.approx(rep.total_usd, rel=1e-9)


def test_first_calls_are_not_rebuilds(tmp_path, capsys):
    t = T()
    t.user("hi")
    t.request(read=25000, w1=3000, out=10)  # reused a shared system prompt, wrote only new tokens
    t.user("more")
    t.request(read=28010, w5=10, out=5)
    root = tmp_path / "projects"
    t.write(root / "p" / "s1.jsonl")
    rep = cc.analyze(root)
    assert dict(rep.rebuilds) == {}
    assert rep.first_calls[:2] == [1, 3000] and rep.first_calls[2] == pytest.approx(3000 * W1)
    assert cli.main(["claude-code", "--path", str(root)]) == 0
    out = capsys.readouterr().out
    assert "First calls: 1 chat or subagent run wrote 3,000 tokens" in out and "Cache rebuilds" not in out


def test_a_kept_message_written_after_the_last_request_keeps_its_tokens(tmp_path):
    t = bash_session(after=1)
    big = t.user("Q" * 20000)  # written after the last request, preserved by the compaction
    t.compact(pre=15100, post=2000, kept=[big])
    t.summary("s" * 400)
    t.request(read=4999, w5=9300, out=10)
    t.user("more")
    t.request(read=14299, w5=30, out=10)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    [kept] = [i for i in items(rep, kind="text") if i.chars == 20000]
    [summary] = items(rep, kind="summary")
    assert kept.tokens == pytest.approx(9000) and kept.calls == 2
    assert summary.tokens == pytest.approx(14299 - 4999.1 - 9000, abs=1)


def test_a_kept_message_repeated_after_the_compaction_keeps_its_tokens(tmp_path):
    t = bash_session(after=0)
    big = t.user("Q" * 20000)
    t.request(read=15050, w5=9020, out=10)
    t.compact(pre=24000, post=2000, kept=[big])
    t.raw({**[d for d in t.lines if d.get("uuid") == big][0]})  # the same line written again
    t.summary("s" * 400)
    t.request(read=4999, w5=9300, out=10)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    [kept] = [i for i in items(rep, kind="text") if i.chars == 20000]
    assert kept.calls == 2
    [summary] = items(rep, kind="summary")
    assert summary.tokens < 400


def test_requests_that_read_nothing_stay_fast(tmp_path):
    t = T(start=NOW - timedelta(days=2))
    t.user("go")
    p = 30000
    t.request(w1=p, out=10, minutes=0.01)
    for _ in range(5000):  # the old walk took about 13 s here
        t.user("x" * 40)
        p += 30
        t.request(w5=p, out=10, minutes=0.001)
    root = tmp_path / "projects"
    t.write(root / "p" / "s1.jsonl")
    start = time.time()
    rep = cc.analyze(root)
    assert time.time() - start < 4
    assert attributed(rep) == pytest.approx(rep.total_usd, rel=1e-9)


MUST_REDACT = [
    ("aws configure set aws_access_key_id AKIAABCDEFGHIJKLMNOP", "AKIAABCDEFGHIJKLMNOP"),
    ("mysql -u root -pS3cr3tPw db", "S3cr3tPw"),
    ("curl -u admin:hunter2 https://x.io", "hunter2"),
    ("docker login -p hunter2pass registry.io", "hunter2pass"),
    ("gh secret set X --body ghp_abcdefghijklmnopqrst", "ghp_abcdefghijklmnopqrst"),
    ("export GH=github_pat_11ABCDEFG0123456789abcdef", "github_pat_11ABCDEFG"),
    ("curl -d token=xoxb-123456789-abcdefgh slack.com", "xoxb-123456789"),
    ('export API_TOKEN="abc def ghi"', "def ghi"),
    ("npm config set //r/:_authToken npm_Abcdefgh12", "npm_Abcdefgh12"),
    ("curl 'https://a.blob.core.windows.net/c?sv=2020-08-04&sig=AbCdEf123%3D'", "AbCdEf123"),
    ("curl -H 'Authorization: Bearer abc.def.ghi'", "abc.def"),
    ("psql postgres://user:pa55@db:5432/x", "pa55"),
    ("deploy --password=letmein", "letmein"),
    ("cat /home/alice/.ssh/id_rsa", "alice"),
    ("mail bob.smith@example.org", "bob.smith"),
    ("export OPENAI=sk-abc", "sk-abc"),
    ("DB_PASSWORD=x ./run", "DB_PASSWORD"),
    ("echo eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2Q", "eyJhbGci"),
    ("key=AbCdEfGh1234567890IjKlMnOpQrStUv run", "AbCdEfGh1234567890"),
]
MUST_KEEP = [
    "src/pages/home/index.tsx",
    "tests/test_hubspot_webhook_auth2.py",
    "docker run claude-opus-4-5-20251101",
    "git checkout feat/quick-start-hubspot-value2",
    "max token limit",
    "basic usage",
    "author: name",
    "git clone git@github.com:org/repo.git",
    "python -m pytest -q tests/test_x.py",
    "max_tokens=4096",
    "grep -r password_reset src",
    "git show 0123456789abcdef0123456789abcdef01234567",
    "npm run build:prod",
]


@pytest.mark.parametrize("text,gone", MUST_REDACT, ids=[t[:30] for t, _ in MUST_REDACT])
def test_redact_removes_secrets(text, gone):
    assert gone not in cc.redact(text, 300)


@pytest.mark.parametrize("text", MUST_KEEP)
def test_redact_keeps_normal_labels(text):
    assert cc.redact(text, 300) == text


def test_huge_since_exits_2_and_fractions_parse(tmp_path, capsys):
    root = tmp_path / "projects"
    bash_session().write(root / "p" / "s1.jsonl")
    assert cli.main(["claude-code", "--path", str(root), "--since", "99999999d"]) == 2
    assert "too far back" in capsys.readouterr().err
    for since in ("2026-11-01T00:00:00.5Z", "2026-11-01T00:00:00.1234567+05:30"):
        assert cli._since_or_date(since).year == 2026
    with pytest.raises(SystemExit) as exc:
        cli.main(["claude-code", "--path", str(root), "--since", "999999999999d"])
    assert exc.value.code == 2


def test_usage_numbers_written_as_strings_count(tmp_path):
    t = T()
    t.user("x")
    t.request(out=5, usages=[{"input_tokens": "120", "output_tokens": "5", "cache_creation_input_tokens": "0"}])
    rep = run(tmp_path, ("p/s1.jsonl", t))
    assert (rep.all.input, rep.all.output) == (120, 5)


def test_unreadable_folders_are_skipped(tmp_path):
    root = tmp_path / "projects"
    bash_session().write(root / "ok" / "s1.jsonl")
    locked = root / "locked"
    bash_session(session="x").write(locked / "s2.jsonl")
    locked.chmod(0)
    try:
        rep = cc.analyze(root)
    finally:
        locked.chmod(0o755)
    if os.geteuid() != 0:
        assert rep.stats.folders_unreadable == 1
    assert rep.stats.files >= 1


def test_fast_mode_without_a_price_is_named(tmp_path):
    t = T(model="claude-haiku-5-5")
    t.user("x")
    t.request(w5=1000, out=10, usages=[{"speed": "fast"}])
    rep = run(tmp_path, ("p/s1.jsonl", t))
    assert list(rep.unpriced) == ["claude-haiku-5-5 (fast mode: no published fast price)"]


def test_unknown_usage_values_count_only_in_the_window(tmp_path):
    t = T(start=NOW - timedelta(days=40))
    t.user("old")
    t.request(w1=1000, out=5, usages=[{"service_tier": "priority"}])
    t.clock = NOW - timedelta(hours=1)
    t.user("new")
    t.request(read=1000, w5=10, out=5)
    rep = run(tmp_path, ("p/s1.jsonl", t), since=(NOW - timedelta(days=30)).timestamp())
    assert not rep.stats.unknown_tiers


def test_a_branch_keeps_parallel_tool_results(tmp_path):
    t = T()
    t.user("two at once")
    a, b = t.bash("ls"), t.bash("pwd")
    lines = t.request(w1=3000, out=20, blocks=[a, b])
    t.result(a["id"], "1" * 1000, parent=lines[0], source=lines[0])  # beside the chain
    r2 = t.result(b["id"], "2" * 1000, parent=lines[1], source=lines[1])
    t.request(read=3000, w5=920, out=5)
    t.user("abandoned follow-up")
    t.request(read=3920, w5=20, out=5)
    t.user("r" * 20, parent=r2)  # rewind to just after the tool results
    t.request(read=3000, w5=929, out=5)
    rep = run(tmp_path, ("p/s1.jsonl", t))
    assert rep.stats.branches == 1
    res = sorted(items(rep, kind="tool_result"), key=lambda i: i.chars)
    assert [r.calls for r in res] == [3, 3]  # both results were sent with the rewound request too
    assert attributed(rep) == pytest.approx(rep.total_usd, rel=1e-9)
