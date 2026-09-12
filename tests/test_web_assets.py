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


def test_no_asset_carries_a_stray_control_character():
    """A byte you cannot see is a bug you cannot read.

    An editing pass wrote literal 0x08 bytes into two regular expressions in
    `app.js` - `/no/` became `/<BS>no<BS>/`, which silently matches "no"
    ANYWHERE, so "nocturne" rendered as "no.cturne". The file looked correct
    in every diff and every editor.
    """
    allowed = {0x09, 0x0A, 0x0D}
    for name in ASSET_FILES:
        text = (ASSETS / name).read_text(encoding="utf-8")
        stray = sorted({ord(c) for c in text if ord(c) < 32 and ord(c) not in allowed})
        assert not stray, f"{name} contains control bytes {[hex(c) for c in stray]}"


#: `display` values that keep an element on the page.
_SHOWING = re.compile(r"display:\s*(?!none)")


def test_a_panel_hidden_with_the_attribute_is_actually_hidden():
    """`[hidden]` is an author-defeatable UA rule, and every panel on this page
    is shown and hidden with it.

    `.sheet` sets `display: grid`, which beats the browser's own
    `[hidden] { display: none }` - so the detail overlay, a full-viewport
    95%-opaque scrim, was painted over the entire page from load. It went
    unnoticed because the stylesheet was being answered 403 and no rule
    applied at all.

    Structural rather than a search for one line: it finds the elements that
    are hidden by attribute, finds the selectors that would show them, and
    only then insists on the override.
    """
    html = (ASSETS / "index.html").read_text(encoding="utf-8")
    # Comments out first. This file explains the rule it is about in prose,
    # and a scan that reads comments finds the explanation instead of the
    # declaration - which is a test that passes on a stylesheet describing a
    # fix nobody applied.
    css = re.sub(r"/\*.*?\*/", "", (ASSETS / "app.css").read_text(encoding="utf-8"), flags=re.S)

    # Names on elements carrying the `hidden` attribute.
    hidden_names = set()
    for tag in re.findall(r"<[a-z]+[^>]*\shidden\s*/?>", html):
        for attr, pattern in (("id", r'id="([^"]+)"'), ("class", r'class="([^"]+)"')):
            found = re.search(pattern, tag)
            if found:
                prefix = "#" if attr == "id" else "."
                hidden_names.update(prefix + part for part in found.group(1).split())
    assert hidden_names, "nothing on this page is hidden by attribute; the test is stale"

    # Selectors whose block sets a display that would keep them visible.
    at_risk = set()
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if not _SHOWING.search(body):
            continue
        for name in hidden_names:
            if re.search(re.escape(name) + r"(?![\w-])", selector):
                at_risk.add(name)
    if not at_risk:
        return

    override = re.search(r"\[hidden\][^{}]*\{([^{}]*)\}", css)
    assert override, (
        f"{sorted(at_risk)} are hidden with the attribute and given a display "
        "by an author rule, so they are never actually hidden. The stylesheet "
        "needs a [hidden] rule that wins."
    )
    body = override.group(1).replace(" ", "")
    assert "display:none!important" in body, override.group(0)


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
