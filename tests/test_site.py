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
        assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), page
        assert not re.search(r"\son[a-z]+=", html), f"inline event handler in {page}"


def test_security_headers():
    headers = (SITE / "_headers").read_text()
    csp = re.search(r"Content-Security-Policy: (.+)", headers).group(1)
    assert "default-src 'none'" in csp and "script-src 'self'" in csp and "style-src 'self'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp and "frame-ancestors 'none'" in csp
    for h in ("X-Content-Type-Options: nosniff", "Referrer-Policy:", "Strict-Transport-Security:"):
        assert h in headers


def test_page_basics():
    assert '<html lang="en">' in INDEX and 'name="viewport"' in INDEX and "<title>" in INDEX
    assert 'name="description"' in INDEX and 'rel="canonical" href="https://cachecanary.com/"' in INDEX
    assert "mailto:hello@cachecanary.com" in INDEX and "https://github.com/Haarris/cachecanary" in INDEX
    assert 'id="copy"' in INDEX and 'id="install-cmd"' in INDEX
    assert "—" not in INDEX  # writing style: no em dashes
