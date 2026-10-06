"""0.3 checks, each pinned to behaviour measured on live Amazon Bedrock (us-west-2, Claude Sonnet 4.6,
October 2026): misplaced and nested cache points, thinking / effort / tool choice changes, and a
conversation cache point that never moves."""

import json

import pytest

from cachecanary import cli, diff, lint
from cachecanary.request import normalize

M = "us.anthropic.claude-sonnet-4-6"
CP = {"cachePoint": {"type": "default"}}
LONG = "Rule: answer politely and cite the section. " * 200
TOOL = {"toolSpec": {"name": "lookup_order", "description": "Look up an order.",
                     "inputSchema": {"json": {"type": "object", "properties": {"id": {"type": "string"}}}}}}
TOOL2 = {"toolSpec": {"name": "refund_order", "description": "Refund an order.",
                      "inputSchema": {"json": {"type": "object", "properties": {"id": {"type": "string"}}}}}}


def rules(req):
    return {f.rule: f for f in lint.lint(req)}


def codes(a, b):
    return [r.code for r in diff.explain(normalize(a, M), normalize(b, M))]


def conv(**kw):
    req = {"messages": [{"role": "user", "content": [{"text": "Reply with OK."}]}]}
    req.update(kw)
    return req


# --- cache point with nothing before it in its own list (live: ValidationException) ----------

def test_cachepoint_first_in_a_later_message_is_an_orphan_not_a_marker_on_the_previous_message():
    req = normalize({"messages": [
        {"role": "user", "content": [{"text": LONG}]},
        {"role": "assistant", "content": [{"text": "Understood."}]},
        {"role": "user", "content": [CP, {"text": "Reply with OK."}]}]}, M)
    assert req.orphan_checkpoints == ["messages[2].content[0]"]
    assert req.checkpoint_indexes == []  # the assistant block is NOT marked
    f = rules(req)["orphan-checkpoint"]
    assert f.severity == "error" and "nothing available to cache" in f.message


@pytest.mark.parametrize("payload,loc", [
    ({"messages": [{"role": "user", "content": [{"text": LONG}]}, {"role": "assistant", "content": [{"text": "ok"}]},
                   {"role": "user", "content": [CP]}]}, "messages[2].content[0]"),
    ({"messages": [{"role": "user", "content": [{"text": LONG}]}, {"role": "assistant", "content": [CP]},
                   {"role": "user", "content": [{"text": "q"}]}]}, "messages[1].content[0]"),
    ({"messages": [{"role": "user", "content": [CP]}]}, "messages[0].content[0]"),
    (conv(system=[CP]), "system[0]"),
    (conv(system=[CP, {"text": LONG}]), "system[0]"),
    (conv(toolConfig={"tools": [CP]}), "toolConfig.tools[0]"),
    (conv(toolConfig={"tools": [CP, TOOL]}), "toolConfig.tools[0]"),
    # A list that only starts after tools: system's first item is still an orphan.
    (conv(toolConfig={"tools": [TOOL]}, system=[CP, {"text": LONG}]), "system[0]"),
], ids=["user-only", "assistant-only", "first-message-only", "system-only", "system-first",
        "tools-only", "tools-first", "system-first-after-tools"])
def test_every_live_rejected_shape_is_an_orphan(payload, loc):
    req = normalize(payload, M)
    assert loc in req.orphan_checkpoints
    assert rules(req)["orphan-checkpoint"].severity == "error"


@pytest.mark.parametrize("payload", [
    {"messages": [{"role": "user", "content": [{"text": LONG}, CP]}]},
    conv(system=[{"text": LONG}, CP]),
    conv(toolConfig={"tools": [TOOL, CP]}),
    {"messages": [{"role": "user", "content": [{"text": LONG}, {"document": {"format": "pdf", "name": "p",
                  "source": {"bytes": "JVBERi0xLjEK"}}}, CP, {"text": "q"}]}]},  # live: caches fine
], ids=["message", "system", "tools", "after-document"])
def test_valid_placements_are_not_orphans(payload):
    req = normalize(payload, M)
    assert req.orphan_checkpoints == [] and req.checkpoint_indexes
    assert "orphan-checkpoint" not in rules(req)


def test_two_cache_points_in_a_row_is_a_duplicate_not_an_orphan():
    req = normalize(conv(system=[{"text": LONG}, CP, CP]), M)  # live: accepted
    assert req.orphan_checkpoints == [] and req.duplicate_checkpoints == ["system[2]"]


def test_lint_cli_fails_on_orphan(tmp_path, capsys):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({**conv(system=[CP, {"text": LONG}]), "modelId": M}))
    assert cli.main(["lint", str(p)]) == 1
    assert "[error] orphan-checkpoint" in capsys.readouterr().out


# --- cachePoint nested in toolResult.content (live: boto3 refuses; raw HTTP caches nothing) ----

def _tool_turns(result_content, after=None):
    return {"toolConfig": {"tools": [TOOL]}, "messages": [
        {"role": "user", "content": [{"text": LONG}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "lookup_order", "input": {"id": "1"}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": result_content}}] + (after or [])}]}


def test_nested_cachepoint_is_flagged_and_not_counted():
    req = normalize(_tool_turns([{"text": "order 1 shipped"}, CP]), M)
    assert req.nested_checkpoints == ["messages[2].content[0].toolResult.content[1]"]
    assert req.checkpoint_indexes == []
    found = rules(req)
    assert found["nested-checkpoint"].severity == "error" and "no-checkpoint" in found


def test_sibling_cachepoint_after_tool_result_is_the_valid_form():
    req = normalize(_tool_turns([{"text": "order 1 shipped"}], after=[CP]), M)  # live: caches
    assert req.nested_checkpoints == [] and req.blocks[req.checkpoint_indexes[-1]].location == "messages[2].content[0]"
    assert "nested-checkpoint" not in rules(req)


def test_invoke_cache_control_inside_tool_result_counts_as_a_checkpoint():
    body = {"anthropic_version": "bedrock-2023-05-31", "messages": [
        {"role": "user", "content": LONG},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "lookup_order", "input": {"id": "1"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
            {"type": "text", "text": "order 1 shipped", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]}]}
    req = normalize(body, M)  # live: caches (InvokeModel accepts it)
    cp = req.checkpoint_indexes
    assert cp and req.blocks[cp[-1]].location == "messages[2].content[0]" and req.blocks[cp[-1]].ttl == "1h"
    assert "no-checkpoint" not in rules(req) and "nested-checkpoint" not in rules(req)


def test_moving_an_inner_invoke_marker_is_not_a_history_edit():
    def body(inner_marker: bool, extra_marker: bool):
        part = {"type": "text", "text": "order 1 shipped"}
        if inner_marker:
            part["cache_control"] = {"type": "ephemeral"}
        msgs = [{"role": "user", "content": [{"type": "text", "text": LONG, "cache_control": {"type": "ephemeral"}}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [part]}]}]
        if extra_marker:
            msgs += [{"role": "assistant", "content": [{"type": "text", "text": "done"}]},
                     {"role": "user", "content": [{"type": "text", "text": "next", "cache_control": {"type": "ephemeral"}}]}]
        return {"anthropic_version": "bedrock-2023-05-31", "messages": msgs}
    found = codes(body(True, False), body(False, True))
    assert "history-changed" not in found and found[0] == "prefix-identical"


# --- thinking (live: any change = full miss, system included; disabled == absent) ------------

def _req(extra=None, **kw):
    r = conv(system=[{"text": LONG}, CP], **kw)
    if extra is not None:
        r["additionalModelRequestFields"] = extra
    return r


def test_thinking_disabled_equals_absent():
    assert codes(_req(), _req({"thinking": {"type": "disabled"}})) == ["prefix-identical"]


@pytest.mark.parametrize("a,b,label", [
    (None, {"thinking": {"type": "enabled", "budget_tokens": 1024}}, "off -> enabled, budget 1024"),
    ({"thinking": {"type": "enabled", "budget_tokens": 1024}}, {"thinking": {"type": "enabled", "budget_tokens": 1500}},
     "enabled, budget 1024 -> enabled, budget 1500"),
    (None, {"thinking": {"type": "adaptive"}}, "off -> adaptive"),
    ({"thinking": {"type": "adaptive"}}, {"thinking": {"type": "disabled"}}, "adaptive -> off"),
])
def test_thinking_changes_are_a_full_miss(a, b, label):
    reasons = diff.explain(normalize(_req(a), M), normalize(_req(b), M))
    assert [r.code for r in reasons] == ["thinking-changed"]
    assert label in reasons[0].message and "system prompt included" in reasons[0].message


def test_thinking_change_is_reported_first_even_when_the_prompt_also_changed():
    a = _req()
    b = _req({"thinking": {"type": "adaptive"}})
    b["system"] = [{"text": LONG + " edited"}, CP]
    assert codes(a, b) == ["thinking-changed"]


def test_invoke_thinking_change():
    def body(thinking=None):
        b = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 10,
             "system": [{"type": "text", "text": LONG, "cache_control": {"type": "ephemeral"}}],
             "messages": [{"role": "user", "content": "Reply with OK."}]}
        if thinking:
            b["thinking"] = thinking
        return b
    assert codes(body(), body({"type": "enabled", "budget_tokens": 1024})) == ["thinking-changed"]
    assert codes(body(), body({"type": "disabled"})) == ["prefix-identical"]


# --- effort (live: each level its own cache; absent == high) ------------------------------

@pytest.mark.parametrize("a,b,changed", [
    (None, "high", False), ("high", None, False), (None, "low", True), (None, "medium", True),
    (None, "max", True), ("low", "medium", True), ("low", "low", False), ("LOW", "low", False),
])
def test_effort_levels(a, b, changed):
    ea = {"output_config": {"effort": a}} if a else None
    eb = {"output_config": {"effort": b}} if b else None
    found = codes(_req(ea), _req(eb))
    assert (found == ["effort-changed"]) is changed
    if not changed:
        assert found == ["prefix-identical"]


def test_invoke_effort_change():
    def body(effort=None):
        b = {"anthropic_version": "bedrock-2023-05-31",
             "system": [{"type": "text", "text": LONG, "cache_control": {"type": "ephemeral"}}],
             "messages": [{"role": "user", "content": "q"}]}
        if effort:
            b["output_config"] = {"effort": effort}
        return b
    assert codes(body(), body("low")) == ["effort-changed"]


def test_unrelated_extra_fields_and_sampling_settings_do_not_matter():
    base = _req()
    tweaked = _req({"top_k": 5})
    tweaked["inferenceConfig"] = {"maxTokens": 40, "temperature": 1.0}  # live: still a hit
    assert codes(base, tweaked) == ["prefix-identical"]


# --- tool choice (live: auto/none/absent vs any/tool = partial miss) -----------------------

def _tc(choice, tools=(TOOL, TOOL2, CP)):
    cfg = {"tools": list(tools)}
    if choice is not None:
        cfg["toolChoice"] = choice
    return {"toolConfig": cfg, "system": [{"text": LONG}, CP],
            "messages": [{"role": "user", "content": [{"text": LONG}, CP]}]}


@pytest.mark.parametrize("a,b,changed", [
    (None, {"auto": {}}, False),
    ({"auto": {}}, {"any": {}}, True),
    ({"any": {}}, {"tool": {"name": "refund_order"}}, False),
    ({"tool": {"name": "refund_order"}}, {"auto": {}}, True),
])
def test_converse_tool_choice_groups(a, b, changed):
    found = codes(_tc(a), _tc(b))
    assert ("tool-choice-changed" in found) is changed
    if changed:
        assert "prefix-identical" not in found
    else:
        assert found == ["prefix-identical"]


@pytest.mark.parametrize("a,b,changed", [
    (None, {"type": "auto"}, False), ({"type": "auto"}, {"type": "none"}, False),
    ({"type": "none"}, {"type": "tool", "name": "x"}, True), ({"type": "tool", "name": "x"}, {"type": "any"}, False),
])
def test_invoke_tool_choice_groups(a, b, changed):
    def body(choice):
        out = {"anthropic_version": "bedrock-2023-05-31",
               "tools": [{"name": "x", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}],
               "messages": [{"role": "user", "content": "q"}]}
        if choice is not None:
            out["tool_choice"] = choice
        return out
    assert ("tool-choice-changed" in codes(body(a), body(b))) is changed


def test_tool_choice_change_is_reported_alongside_a_real_prompt_change():
    a, b = _tc({"auto": {}}), _tc({"any": {}})
    b["system"] = [{"text": LONG + " edited"}, CP]
    assert codes(a, b) == ["tool-choice-changed", "system-changed"]


def test_no_tools_means_no_tool_choice_reason():
    a, b = conv(system=[{"text": LONG}, CP]), conv(system=[{"text": LONG}, CP])
    b["toolConfig"] = {"toolChoice": {"any": {}}}
    assert normalize(b, M).tool_choice is None and codes(a, b) == ["prefix-identical"]


# --- a conversation cache point that never moves (cline/cline#13738) ----------------------

def _agent(rounds: int, cp_at_end: bool):
    msgs = [{"role": "user", "content": [{"text": LONG}] + ([] if cp_at_end else [CP])}]
    for r in range(rounds):
        msgs.append({"role": "assistant", "content": [{"toolUse": {"toolUseId": f"t{r}", "name": "lookup_order", "input": {"id": str(r)}}}]})
        msgs.append({"role": "user", "content": [{"toolResult": {"toolUseId": f"t{r}", "content": [{"text": "ok"}]}}]})
    if cp_at_end:
        msgs[-1]["content"].append(CP)
    return {"toolConfig": {"tools": [TOOL]}, "messages": msgs}


def test_stale_checkpoint_in_an_agent_loop():
    f = rules(normalize(_agent(2, cp_at_end=False), M))["stale-checkpoint"]
    assert f.severity == "warn" and "4 messages back" in f.message and f.location == "messages[0].content[0]"


@pytest.mark.parametrize("payload", [
    _agent(5, cp_at_end=True),   # cache point on the newest message: fine
    _agent(1, cp_at_end=False),  # one round after it (2 messages): normal rolling pattern
], ids=["moved", "one-round"])
def test_no_stale_checkpoint(payload):
    assert "stale-checkpoint" not in rules(normalize(payload, M))


def test_plain_chat_history_after_a_checkpoint_is_not_called_stale():
    msgs = [{"role": "user", "content": [{"text": LONG}, CP]}] + [
        {"role": r, "content": [{"text": f"turn {i}"}]} for i, r in enumerate(["assistant", "user", "assistant", "user"])]
    assert "stale-checkpoint" not in rules(normalize({"messages": msgs}, M))


def test_stale_checkpoint_on_invoke_model():
    msgs = [{"role": "user", "content": [{"type": "text", "text": LONG, "cache_control": {"type": "ephemeral"}}]}]
    for r in range(2):
        msgs += [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{r}", "name": "x", "input": {}}]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{r}", "content": "ok"}]}]
    assert "stale-checkpoint" in rules(normalize({"anthropic_version": "bedrock-2023-05-31", "messages": msgs}, M))


def test_diff_checkpoint_not_moved_is_reported_but_does_not_fail(tmp_path, capsys):
    a, b = _agent(0, cp_at_end=False), _agent(1, cp_at_end=False)
    reasons = diff.explain(normalize(a, M), normalize(b, M))
    assert [r.code for r in reasons] == ["prefix-identical", "checkpoint-not-moved"]
    assert "2 new blocks" in reasons[1].message
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps({**a, "modelId": M}))
    pb.write_text(json.dumps({**b, "modelId": M}))
    assert cli.main(["diff", str(pa), str(pb)]) == 0
    assert "checkpoint-not-moved" in capsys.readouterr().out


@pytest.mark.parametrize("b_builder", [
    lambda: _agent(1, cp_at_end=True),  # the cache point moved to the end
], ids=["moved"])
def test_diff_no_not_moved_when_it_moved(b_builder):
    assert "checkpoint-not-moved" not in codes(_agent(0, cp_at_end=False), b_builder())


def test_diff_system_only_caching_is_not_called_stuck():
    a = conv(system=[{"text": LONG}, CP])
    b = conv(system=[{"text": LONG}, CP])
    b["messages"] += [{"role": "assistant", "content": [{"text": "a"}]}, {"role": "user", "content": [{"text": "b"}]}]
    assert codes(a, b) == ["prefix-identical"]


def test_diff_one_new_block_is_not_called_stuck():
    a = {"messages": [{"role": "user", "content": [{"text": LONG}, CP]}]}
    b = {"messages": [{"role": "user", "content": [{"text": LONG}, CP, {"text": "one more"}]}]}
    assert codes(a, b) == ["prefix-identical"]


# --- JSON and GitHub output carry the new codes -------------------------------------------

def test_json_and_github_output_for_new_rules(tmp_path, capsys, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    p = tmp_path / "r.json"
    p.write_text(json.dumps({**_tool_turns([{"text": "ok"}, CP]), "modelId": M}))
    assert cli.main(["--json", "lint", str(p)]) == 1
    data = json.loads(capsys.readouterr().out)
    assert any(f["rule"] == "nested-checkpoint" and f["severity"] == "error" for f in data)
    assert cli.main(["--github", "lint", str(p)]) == 1
    out = capsys.readouterr().out
    assert "::error" in out and "nested-checkpoint" in out
