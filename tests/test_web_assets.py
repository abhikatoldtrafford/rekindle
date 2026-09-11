"""The page's own assets: no network, and no dangling references.

A browser is not available in CI and a headless one is a dependency this
project will not take, so these are the checks that can be made without one:
that the page asks nothing of the network, and that every element `app.js`
reaches for exists in `index.html`. The second catches a whole class of
silent breakage - rename an id in the markup and one control stops working
with no error anywhere a test would see.
"""

from __future__ import annotations

import re

import pytest

from rekindle.web.server import ASSETS, STATIC

#: The one external URL the page is allowed to contain. It is an XML
#: namespace in an inline SVG favicon - a name, never fetched.
SVG_NAMESPACE = "http://www.w3.org/2000/svg"

ASSET_FILES = ["index.html", *STATIC]


@pytest.mark.parametrize("name", ASSET_FILES)
def test_no_asset_reaches_the_network(name):
    """No CDN, no font service, no analytics, no image host.

    This is the promise the whole feature rests on: a page showing someone's
    family photographs must not tell anybody what is on it. The CSP enforces
    it at runtime; this catches it at the point it would be written.
    """
    text = (ASSETS / name).read_text(encoding="utf-8").replace(SVG_NAMESPACE, "")
    for scheme in ("http://", "https://", "//cdn", "ws://", "wss://"):
        assert scheme not in text, f"{name} references {scheme}"


def test_every_element_the_script_reaches_for_exists():
    html = (ASSETS / "index.html").read_text(encoding="utf-8")
    script = (ASSETS / "app.js").read_text(encoding="utf-8")
    present = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r'\$\("([^"]+)"\)', script))
    assert used, "the id-lookup pattern changed; this test is no longer checking anything"
    assert used <= present, f"app.js addresses ids that do not exist: {sorted(used - present)}"


def test_the_page_loads_its_own_stylesheet_and_script():
    html = (ASSETS / "index.html").read_text(encoding="utf-8")
    for name in STATIC:
        assert f"/assets/{name}" in html
    # And nothing else: a reference the allow-list does not serve is a 404
    # the browser reports and nothing here would.
    referenced = set(re.findall(r"/assets/([A-Za-z0-9_.-]+)", html))
    assert referenced == set(STATIC)


def test_the_script_sends_the_token_on_every_call():
    """Without this header the page's own requests are refused, so a call
    written without it is a control that silently does nothing."""
    script = (ASSETS / "app.js").read_text(encoding="utf-8")
    assert "X-Rekindle-Token" in script
    # Thumbnails go through <img>, which cannot set a header, so they use the
    # query parameter instead.
    assert "t=${encodeURIComponent(TOKEN)}" in script or "&t=" in script


def test_the_stream_is_closed_on_both_terminal_events():
    """An EventSource that is not closed RECONNECTS when the server ends the
    stream, which silently re-runs the whole build."""
    script = (ASSETS / "app.js").read_text(encoding="utf-8")
    for terminal in ("ready", "error"):
        start = script.find(f'stream.addEventListener("{terminal}"')
        assert start >= 0, f"no handler for the {terminal} event"
        # The handler's own body only: up to wherever the NEXT listener is
        # registered. Slicing a fixed number of characters instead would let
        # the sibling handler's `close()` satisfy this one.
        rest = script[start + 1 :]
        end = rest.find("stream.addEventListener(")
        body = rest[: end if end >= 0 else len(rest)]
        assert "stream.close()" in body, f"the {terminal} handler does not close the stream"
