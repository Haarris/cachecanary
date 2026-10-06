"""The marketing site: assets exist, no inline code (strict CSP), security headers, 404 page."""

import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"
INDEX = (SITE / "index.html").read_text()


def test_required_files_exist():
    for name in ("index.html", "404.html", "style.css", "main.js", "favicon.svg", "_headers", "robots.txt"):
        assert (SITE / name).is_file(), name


def test_every_local_reference_exists():
    for page in ("index.html", "404.html"):
        html = (SITE / page).read_text()
        for ref in re.findall(r'(?:href|src)="(/[^"#]*)"', html):
            if ref != "/":
                assert (SITE / ref.lstrip("/")).is_file(), f"{page} references missing {ref}"


def test_no_inline_code_so_csp_can_stay_strict():
    for page in ("index.html", "404.html"):
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
    assert locs == ["https://cachecanary.com/"]
    assert "Sitemap: https://cachecanary.com/sitemap.xml" in (SITE / "robots.txt").read_text()
