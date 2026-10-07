"""The marketing site: assets exist, no inline code (strict CSP), security headers, 404 page."""

import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"
INDEX = (SITE / "index.html").read_text()
LITELLM = (SITE / "litellm" / "index.html").read_text()
STRANDS = (SITE / "strands" / "index.html").read_text()
PAGES = ("index.html", "404.html", "litellm/index.html", "strands/index.html")


def test_required_files_exist():
    for name in ("index.html", "404.html", "litellm/index.html", "strands/index.html", "style.css", "main.js", "favicon.svg", "_headers", "robots.txt"):
        assert (SITE / name).is_file(), name


def test_every_local_reference_exists():
    for page in PAGES:
        html = (SITE / page).read_text()
        for ref in re.findall(r'(?:href|src)="(/[^"#]*)"', html):
            if ref != "/":
                target = SITE / ref.lstrip("/")
                assert (target / "index.html" if ref.endswith("/") else target).is_file(), f"{page} references missing {ref}"


def test_no_inline_code_so_csp_can_stay_strict():
    for page in PAGES:
        html = (SITE / page).read_text()
        assert "<style" not in html and 'style="' not in html, page
        # Structured data (application/ld+json) is not executed, so CSP doesn't apply to it.
        assert not re.search(r"<script(?![^>]*\b(?:src=|type=\"application/ld\+json\"))[^>]*>", html), page
        assert not re.search(r"\son[a-z]+=", html), f"inline event handler in {page}"


def test_security_headers():
    headers = (SITE / "_headers").read_text()
    csp = re.search(r"Content-Security-Policy: (.+)", headers).group(1)
    assert "default-src 'none'" in csp and "style-src 'self'" in csp
    # Scripts: our own plus Cloudflare Web Analytics only (cookieless visitor counts, allowed by Haris Oct 6 2026).
    assert "script-src 'self' https://static.cloudflareinsights.com;" in csp
    assert "connect-src 'self' https://cloudflareinsights.com;" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp and "frame-ancestors 'none'" in csp
    for h in ("X-Content-Type-Options: nosniff", "Referrer-Policy:", "Strict-Transport-Security:"):
        assert h in headers


def test_page_basics():
    assert '<html lang="en">' in INDEX and 'name="viewport"' in INDEX and "<title>" in INDEX
    assert 'name="description"' in INDEX and 'rel="canonical" href="https://cachecanary.com/"' in INDEX
    assert "mailto:hello@cachecanary.com" in INDEX and "https://github.com/Haarris/cachecanary" in INDEX
    assert 'id="copy"' in INDEX and 'id="install-cmd"' in INDEX
    assert "—" not in INDEX  # writing style: no em dashes


def test_wrangler_config_serves_site_with_404_page():
    import json
    import re as _re
    raw = (SITE.parent / "wrangler.jsonc").read_text()
    cfg = json.loads(_re.sub(r"^\s*//.*$", "", raw, flags=_re.M))  # strip // comment lines
    assert cfg["name"] == "cachecanary"
    assert cfg["assets"] == {"directory": "./site", "not_found_handling": "404-page"}
    assert "main" not in cfg  # static assets only, no Worker script
    # Only the custom domains serve the site (no duplicate workers.dev copy, no public previews).
    assert cfg["routes"] == [{"pattern": "cachecanary.com", "custom_domain": True},
                             {"pattern": "www.cachecanary.com", "custom_domain": True}]
    assert cfg["workers_dev"] is False and cfg["preview_urls"] is False


def test_site_logs_example_matches_real_output(capsys, monkeypatch):
    """The logs output shown on the page is exactly what the CLI prints for the redacted real logs."""
    import html
    from cachecanary import cli
    monkeypatch.chdir(SITE.parent)
    cli.main(["logs", "tests/fixtures/bedrock_invocation_logs_2026-10.json", "--min-hit", "0.8"])
    page = html.unescape(re.sub(r"<[^>]+>", "", INDEX))
    for line in capsys.readouterr().out.strip().split("\n"):
        assert line in page, line


def test_structured_data_matches_the_package():
    import json
    from cachecanary import __version__
    m = re.search(r'<script type="application/ld\+json">(.*?)</script>', INDEX, re.S)
    data = json.loads(m.group(1))
    assert data["@type"] == "SoftwareApplication" and data["name"] == "CacheCanary"
    assert data["softwareVersion"] == __version__  # bump it with every release
    assert data["offers"]["price"] == "0" and data["url"] == "https://cachecanary.com/"


def test_sitemap_and_robots():
    import xml.etree.ElementTree as ET
    root = ET.parse(SITE / "sitemap.xml").getroot()
    locs = [e.text for e in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    assert locs == ["https://cachecanary.com/", "https://cachecanary.com/litellm/", "https://cachecanary.com/strands/"]
    assert "Sitemap: https://cachecanary.com/sitemap.xml" in (SITE / "robots.txt").read_text()


def test_site_and_readme_link_the_write_up():
    url = "https://harisfarooq.substack.com/p/five-ways-claude-prompt-caching-quietly"
    assert url in INDEX
    assert url in (SITE.parent / "README.md").read_text()


def test_llms_txt_follows_the_format_and_links_resolve_locally():
    text = (SITE / "llms.txt").read_text()
    lines = text.splitlines()
    assert lines[0] == "# CacheCanary" and any(l.startswith("> ") for l in lines[:4])
    assert "## Docs" in text and "—" not in text
    links = re.findall(r"\]\((https://[^)]+)\)", text)
    assert "https://cachecanary.com/" in links and "https://github.com/Haarris/cachecanary#readme" in links
    # Facts that must stay in step with the code.
    from cachecanary.models import MAX_BLOCKS_ADDED
    assert f"{MAX_BLOCKS_ADDED} added still hits, {MAX_BLOCKS_ADDED + 1} always misses" in text


def test_litellm_page_basics_and_links():
    assert '<html lang="en">' in LITELLM and 'name="viewport"' in LITELLM and "<title>" in LITELLM
    assert 'name="description"' in LITELLM
    assert 'rel="canonical" href="https://cachecanary.com/litellm/"' in LITELLM
    assert "—" not in LITELLM  # writing style: no em dashes
    guides = re.search(r'<h2 id="guides">.*?</ul>', INDEX, re.S).group(0)
    assert 'href="/litellm/"' in guides and 'href="/strands/"' in guides  # listed in the home page's guides section
    for page in (INDEX, LITELLM, STRANDS):  # which every top menu links to
        nav = re.search(r"<nav>(.*?)</nav>", page, re.S).group(1)
        assert 'href="#guides"' in nav or 'href="/#guides"' in nav
    assert "https://cachecanary.com/litellm/" in (SITE / "llms.txt").read_text()


def test_litellm_page_outputs_match_real_output(capsys, monkeypatch):
    """Every diff shown on the LiteLLM page is what the CLI prints for the requests LiteLLM 1.104.0 built."""
    import html
    from cachecanary import cli
    monkeypatch.chdir(SITE.parent / "tests" / "fixtures" / "litellm")
    page = html.unescape(re.sub(r"<[^>]+>", "", LITELLM))
    cmds = re.findall(r"\$ (cachecanary diff [^\n]+)", page)
    assert len(cmds) == 3
    for cmd in cmds:
        args = cmd.split()[1:]
        cli.main(args)
        out = capsys.readouterr().out.strip()
        assert out, cmd
        for line in out.split("\n"):
            assert line in page, f"{cmd}: {line}"


def _page_helper():
    import html
    code = html.unescape(re.search(r'<pre class="yaml">(def cache_points.*?)\n\n\nresponse', LITELLM, re.S).group(1))
    ns = {}
    exec(code, ns)
    return ns["cache_points"]


def _calls(n):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": f"t{i}", "type": "function", "function": {"name": "f", "arguments": "{}"}} for i in range(n)]}


def _results(n):
    return [{"role": "tool", "tool_call_id": f"t{i}", "content": "ok"} for i in range(n)]


def test_litellm_page_helper_places_points_as_described():
    cache_points = _page_helper()
    system, user = {"role": "system", "content": "s"}, {"role": "user", "content": "u"}
    indexes = lambda pts: sorted(p["index"] for p in pts if "index" in p)
    # First call: system + end; the newest user message is the end, so LiteLLM merges the two.
    pts = cache_points([system, user])
    assert {"location": "message", "role": "system"} in pts and indexes(pts) == [-1, 1]
    # A tool round: newest user message, first tool result of the round, end of conversation.
    msgs = [system, user, _calls(12)] + _results(12)
    assert indexes(cache_points(msgs)) == [-1, 1, 3]
    # A second round: the round point moves to the newest round.
    msgs2 = msgs + [_calls(3)] + _results(3)
    assert indexes(cache_points(msgs2)) == [-1, 1, 16]
    # The user speaks after the tools: no round point (the newest round is older than the user message).
    msgs3 = msgs + [{"role": "assistant", "content": "done"}, user]
    assert indexes(cache_points(msgs3)) == [-1, 16]
    # Assistant tool calls with no results yet: no point past the end.
    assert indexes(cache_points([system, user, _calls(2)])) == [-1, 1]
    # No system message, no user message: still valid, never more than 4 points.
    assert indexes(cache_points([_calls(1)] + _results(1))) == [-1, 1]
    for m in (msgs, msgs2, msgs3):
        assert len(cache_points(m)) <= 4


def test_live_script_uses_the_page_helper():
    import html
    page = html.unescape(re.search(r'<pre class="yaml">(def cache_points.*?)\n\n\nresponse', LITELLM, re.S).group(1))
    script = (SITE.parent / "scripts" / "live_litellm_checks.py").read_text()
    assert page in script


def test_social_preview_images():
    """LinkedIn, X and Slack show a blank box without og:image; each page has its own 2400x1260 card (sharp on high-resolution screens)."""
    import struct
    for html, name in ((INDEX, "og.png"), (LITELLM, "og-litellm.png"), (STRANDS, "og-strands.png")):
        # ?v=N busts LinkedIn's image cache; bump it whenever a card changes.
        assert re.search(rf'<meta property="og:image" content="https://cachecanary.com/{re.escape(name)}\?v=\d+">', html)
        assert 'content="summary_large_image"' in html
        assert re.search(r'og:title" content="[^"]{20,}"', html)  # says what it is, not just the name
        assert 'property="og:image:alt"' in html
        data = (SITE / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) < 5_000_000
        assert struct.unpack(">II", data[16:24]) == (2400, 1260)
        assert 'og:image:width" content="2400"' in html and 'og:image:height" content="1260"' in html


def test_strands_page_basics_and_links():
    assert '<html lang="en">' in STRANDS and 'name="viewport"' in STRANDS and "<title>" in STRANDS
    assert 'name="description"' in STRANDS
    assert 'rel="canonical" href="https://cachecanary.com/strands/"' in STRANDS
    assert "—" not in STRANDS  # writing style: no em dashes
    assert "https://cachecanary.com/strands/" in (SITE / "llms.txt").read_text()
    assert 'href="/strands/"' in LITELLM and 'href="/litellm/"' in STRANDS  # the guides point to each other


def test_strands_page_outputs_match_real_output(capsys, monkeypatch):
    """Every command shown on the Strands page prints exactly this for the requests Strands 1.58.1 built."""
    import html
    from cachecanary import cli
    monkeypatch.chdir(SITE.parent / "tests" / "fixtures" / "strands")
    page = html.unescape(re.sub(r"<[^>]+>", "", STRANDS))
    cmds = re.findall(r"\$ (cachecanary (?:diff|lint) [^\n]+)", page)
    assert len(cmds) == 3
    for cmd in cmds:
        cli.main(cmd.split()[1:])
        out = capsys.readouterr().out.strip()
        assert out, cmd
        for line in out.split("\n"):
            assert line in page, f"{cmd}: {line}"


def _strands_code(start, end):
    import html
    return html.unescape(re.search(rf'({re.escape(start)}.*?){re.escape(end)}', STRANDS, re.S).group(1)).rstrip()


def _with_fake_strands(code):
    """Run page code that imports strands, with tiny stand-ins (strands isn't a dependency of this repo)."""
    import sys
    import types
    hooks = types.ModuleType("strands.hooks")
    hooks.BeforeModelCallEvent, hooks.HookRegistry = object, object
    hooks.HookProvider = type("HookProvider", (), {})
    managers = types.ModuleType("strands.agent.conversation_manager")

    class SlidingWindowConversationManager:  # the parts TrimInChunks relies on
        def __init__(self, window_size=40, **kwargs):
            self.window_size = window_size

        def reduce_context(self, agent, **kwargs):
            del agent.messages[: len(agent.messages) - self.window_size]

        def restore_from_session(self, state):  # Strands checks the class name like this
            if state.get("__name__") != self.__class__.__name__:
                raise ValueError("Invalid conversation manager state.")

    managers.SlidingWindowConversationManager = SlidingWindowConversationManager
    saved = {k: sys.modules.get(k) for k in ("strands.hooks", "strands.agent.conversation_manager")}
    sys.modules.update({"strands.hooks": hooks, "strands.agent.conversation_manager": managers})
    try:
        ns = {}
        exec(code, ns)
        return ns
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_strands_page_hook_places_points_as_described():
    import types
    hook = _with_fake_strands(_strands_code("from strands.hooks import", "\n\n\nagent = Agent("))["BedrockCachePoints"]()
    CP = {"cachePoint": {"type": "default"}}

    def place(messages):
        hook.place(types.SimpleNamespace(agent=types.SimpleNamespace(messages=messages)))
        return [[i for i, b in enumerate(m["content"]) if "cachePoint" in b] for m in messages]

    typed = lambda text: {"role": "user", "content": [{"text": text}]}
    calls = lambda n: {"role": "assistant", "content": [{"toolUse": {"toolUseId": f"t{i}"}} for i in range(n)]}
    results = lambda n: {"role": "user", "content": [{"toolResult": {"toolUseId": f"t{i}"}} for i in range(n)]}
    # First call: the typed message is the end, one point.
    assert place([typed("hi")]) == [[1]]
    # A tool round: the typed message, the first result, the end (3 message points + system = 4).
    assert place([typed("hi"), calls(12), results(12)]) == [[1], [], [1, 13]]
    # Old points are removed: the next round moves them, never more than 3 in messages.
    msgs = [typed("hi"), calls(12), results(12)]
    place(msgs)
    msgs += [calls(3), results(3)]
    assert place(msgs) == [[1], [], [], [], [1, 4]]
    # One tool result: a single point at the end.
    assert place([typed("hi"), calls(1), results(1)]) == [[1], [], [1]]
    # The person speaks after the tools: only their message carries a point.
    assert place([typed("hi"), calls(2), results(2), {"role": "assistant", "content": [{"text": "done"}]}, typed("more")]) == [[], [], [], [], [1]]
    # A non-PDF document at the end: the point goes before it; a PDF keeps the point after it.
    doc = lambda fmt: {"role": "user", "content": [{"text": "read"}, {"document": {"format": fmt}}]}
    assert place([doc("txt")]) == [[1]] and place([doc("pdf")]) == [[2]]
    # A message that is only a non-PDF document gets no point (Bedrock would reject it).
    assert place([{"role": "user", "content": [{"document": {"format": "csv"}}]}]) == [[]]
    assert place([]) == []


def test_strands_page_trim_in_chunks():
    import types
    code = _strands_code("class TrimInChunks", "\n\n\nagent = Agent(")
    TrimInChunks = _with_fake_strands("from strands.agent.conversation_manager import SlidingWindowConversationManager\n" + code)["TrimInChunks"]
    manager, agent = TrimInChunks(), types.SimpleNamespace(messages=list(range(40)))
    manager.apply_management(agent)
    assert len(agent.messages) == 40  # not full yet: nothing changes
    agent.messages.append(40)
    manager.apply_management(agent)
    assert agent.messages == list(range(21, 41)) and manager.window_size == 40  # trimmed to 20 in one go
    for n in range(41, 61):
        agent.messages.append(n)
        manager.apply_management(agent)
    assert len(agent.messages) == 40 and agent.messages[0] == 21  # no trim until it is over 40 again
    # Sessions saved with the default manager load; other managers' sessions are still refused.
    TrimInChunks().restore_from_session({"__name__": "SlidingWindowConversationManager", "removed_message_count": 0})
    TrimInChunks().restore_from_session({"__name__": "TrimInChunks", "removed_message_count": 0})
    import pytest
    with pytest.raises(ValueError):
        TrimInChunks().restore_from_session({"__name__": "SummarizingConversationManager"})


def test_strands_live_script_uses_the_page_code():
    script = (SITE.parent / "scripts" / "live_strands_checks.py").read_text()
    hook = _strands_code("class BedrockCachePoints", "\n\n\nagent = Agent(")
    trim = _strands_code("class TrimInChunks", "\n\n\nagent = Agent(")
    assert hook in script and trim in script
