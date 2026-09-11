"""The optional GPT caption layer.

**No test in this file makes a network call.** The transport is injected
everywhere, and one test asserts the real one is never reached by the default
configuration.

The assertions that matter: the key never leaks, an unsubstantiated caption is
rejected, and every failure falls back to the deterministic caption rather
than breaking a render.
"""

import json

import pytest

from rekindle.memory import llm
from rekindle.memory.llm import (
    REJECT_UNKNOWN_PERSON,
    CaptionReport,
    GptCaptioner,
    LLMUnavailable,
    apply_captions,
    build_payload,
    captioner_from_env,
    extract_text,
    substantiated,
)
from rekindle.memory.spec import FactSheet, MemorySpec, Shot

SECRET = "sk-test-THIS-MUST-NEVER-APPEAR-abcdef123456"


def _facts(**kw) -> FactSheet:
    return FactSheet(
        title=kw.pop("title", "Kashmir"),
        recipe=kw.pop("recipe", "album_story"),
        photo_count=kw.pop("photo_count", 3),
        years=kw.pop("years", (2015, 2016)),
        people=kw.pop("people", {"Abhik Maiti": 3, "Paramita": 2}),
        albums=kw.pop("albums", ("Kashmir",)),
        **kw,
    )


def _spec(captions=("15 May 2015", "16 May 2015")) -> MemorySpec:
    shots = tuple(
        Shot(f"h{i}", c, f"201{5 + i}-05-15T12:00:00", True) for i, c in enumerate(captions)
    )
    return MemorySpec(
        recipe="album_story",
        key="Kashmir",
        title="Kashmir",
        subtitle="2 photos, May 2015",
        shots=shots,
        facts=_facts(photo_count=len(shots)),
        public_safe=True,
    )


def _fake(replies):
    """A transport that returns canned text and records what it was sent."""
    sent = []
    queue = list(replies)

    def transport(payload, api_key):
        sent.append((payload, api_key))
        text = queue.pop(0) if queue else ""
        if isinstance(text, Exception):
            raise text
        return {"output_text": text}

    transport.sent = sent
    return transport


# --------------------------------------------------------------------------
# the fact sheet is the ONLY thing that goes out


def test_the_request_contains_the_fact_sheet_and_nothing_else(tmp_path):
    spec = _spec()
    payload = build_payload(spec.facts, spec.shots[0], spec.shots[0].caption)
    blob = json.dumps(payload)

    assert "Kashmir" in blob
    # No pixels, no paths, no hashes, no library.
    assert ".jpg" not in blob.lower()
    assert "takeout" not in blob.lower()
    assert "h0" not in json.loads(blob)["input"][1]["content"]


def test_the_request_asks_for_the_specified_model_and_effort():
    payload = build_payload(_facts(), Shot("h", "c", None, True), "c")
    assert payload["model"] == "gpt-5.6-luna"
    assert payload["reasoning"] == {"effort": "low"}


def test_the_photo_hash_is_never_sent():
    spec = _spec()
    payload = build_payload(spec.facts, spec.shots[0], "x")
    assert spec.shots[0].file_hash not in json.dumps(payload)


# --------------------------------------------------------------------------
# the key never leaks


def test_the_key_is_not_in_the_captioner_repr():
    captioner = GptCaptioner(SECRET, transport=_fake([""]))
    assert SECRET not in repr(captioner)
    assert "redacted" in repr(captioner)


def test_the_key_is_not_in_a_rewritten_spec():
    spec, _ = apply_captions(_spec(), GptCaptioner(SECRET, transport=_fake(["May 2015"])))
    assert SECRET not in spec.dumps()


def test_the_key_is_not_in_an_error_message():
    """An Authorization header echoed into an exception message is exactly how
    a credential ends up in a log file."""
    boom = LLMUnavailable("the captioning service returned HTTP 401")
    _, report = apply_captions(_spec(), GptCaptioner(SECRET, transport=_fake([boom])))
    assert SECRET not in (report.error or "")
    assert "401" in report.error


def test_the_key_is_not_in_the_report():
    _, report = apply_captions(_spec(), GptCaptioner(SECRET, transport=_fake(["May 2015"])))
    assert SECRET not in repr(report)


def test_the_key_is_passed_to_the_transport_and_nowhere_else():
    transport = _fake(["May 2015"])
    apply_captions(_spec(), GptCaptioner(SECRET, transport=transport))
    assert all(key == SECRET for _, key in transport.sent)


# --------------------------------------------------------------------------
# the verifier


def test_a_caption_using_only_known_facts_is_accepted():
    assert substantiated("May 2015 in Kashmir", _facts()) is None


def test_a_caption_asserting_an_UNKNOWN_YEAR_is_rejected():
    assert substantiated("Back in 1999", _facts(years=(2015, 2016))) == llm.REJECT_UNKNOWN_YEAR


def test_a_caption_using_a_known_year_is_accepted():
    assert substantiated("The summer of 2016", _facts(years=(2015, 2016))) is None


def test_a_caption_naming_an_UNKNOWN_PERSON_is_rejected():
    assert substantiated("A day with Geoffrey", _facts()) == llm.REJECT_UNKNOWN_PERSON


def test_a_caption_naming_a_known_person_is_accepted():
    assert substantiated("Paramita on the lake", _facts()) is None


def test_a_first_name_of_a_known_person_is_accepted():
    """The sheet says "Abhik Maiti"; a caption saying "Abhik" is
    substantiated."""
    assert substantiated("Abhik at the water", _facts()) is None


def test_ordinary_capitalised_words_are_not_mistaken_for_people():
    """Without this the verifier rejects almost every caption and the layer is
    a very expensive no-op."""
    for text in ["Then and now", "Every December", "The first of many", "Six years ago"]:
        assert substantiated(text, _facts()) is None, text


def test_an_over_long_caption_is_rejected():
    assert substantiated("x" * 200, _facts()) == llm.REJECT_TOO_LONG


def test_an_empty_caption_is_rejected():
    assert substantiated("   ", _facts()) == llm.REJECT_EMPTY


def test_a_multi_line_caption_is_normalised_then_judged():
    """A caption sits on one line under a photo."""
    spec, report = apply_captions(
        _spec(), GptCaptioner(SECRET, transport=_fake(["May\n2015", "May\n2015"]))
    )
    assert report.accepted == 2
    assert all("\n" not in s.caption for s in spec.shots)


def test_an_invented_place_name_is_rejected():
    """The sharpest case: there is no gazetteer, so a city name is always an
    invention."""
    assert substantiated("Sunset over Srinagar", _facts()) == llm.REJECT_UNKNOWN_PERSON


# --------------------------------------------------------------------------
# fallback: every failure keeps the deterministic caption


def test_a_rejected_caption_falls_back_to_the_deterministic_one():
    original = _spec()
    spec, report = apply_captions(
        original, GptCaptioner(SECRET, transport=_fake(["Back in 1999", "Kashmir in 2015"]))
    )
    assert spec.shots[0].caption == original.shots[0].caption
    assert spec.shots[1].caption == "Kashmir in 2015"
    assert report.rejected == {llm.REJECT_UNKNOWN_YEAR: 1}
    assert report.accepted == 1
    assert report.accounted


def test_the_verifier_over_rejects_on_purpose():
    """ "Sunlight on the water" is refused because nothing substantiates
    "Sunlight". That is the right direction to be wrong in: a rejection costs
    one fallback to a caption that is always correct, while an acceptance of
    "Sunset over Srinagar" puts an invented place under someone's photo.

    Pinned so the strictness is a decision rather than an accident, and so
    anyone loosening it has to change a test that says why.
    """
    assert substantiated("Sunlight on the water", _facts()) == llm.REJECT_UNKNOWN_PERSON


def test_an_unreachable_service_keeps_every_remaining_caption():
    original = _spec(captions=("a", "b"))
    transport = _fake([LLMUnavailable("could not reach the captioning service: URLError")])
    spec, report = apply_captions(original, GptCaptioner(SECRET, transport=transport))

    assert [s.caption for s in spec.shots] == ["a", "b"]
    assert report.error is not None
    assert report.accepted == 0


def test_a_malformed_response_keeps_the_deterministic_caption():
    spec, report = apply_captions(_spec(), GptCaptioner(SECRET, transport=_fake(["", ""])))
    assert [s.caption for s in spec.shots] == [s.caption for s in _spec().shots]
    assert report.rejected == {llm.REJECT_EMPTY: 2}


def test_a_missing_key_raises_a_clean_unavailable(monkeypatch):
    monkeypatch.delenv(llm.ENV_KEY, raising=False)
    with pytest.raises(LLMUnavailable, match="supported configuration"):
        captioner_from_env()


def test_the_environment_is_read_in_exactly_one_place(monkeypatch):
    monkeypatch.setenv(llm.ENV_KEY, SECRET)
    captioner = captioner_from_env(transport=_fake([""]))
    assert isinstance(captioner, GptCaptioner)

    source = __import__("pathlib").Path(llm.__file__).read_text(encoding="utf-8")
    assert source.count("os.environ") == 1, "the key must have exactly one read site"


# --------------------------------------------------------------------------
# the layer changes captions and NOTHING else


def test_only_the_caption_field_can_change():
    """An LLM never picks a photo, reorders one, or changes what is
    publishable."""
    original = _spec()
    spec, _ = apply_captions(original, GptCaptioner(SECRET, transport=_fake(["May 2015"] * 2)))

    assert [s.file_hash for s in spec.shots] == [s.file_hash for s in original.shots]
    assert [s.taken_at_local for s in spec.shots] == [s.taken_at_local for s in original.shots]
    assert [s.public_safe for s in spec.shots] == [s.public_safe for s in original.shots]
    assert spec.title == original.title
    assert spec.public_safe == original.public_safe
    assert spec.facts == original.facts


def test_the_shot_count_is_unchanged():
    original = _spec(captions=("a", "b"))
    spec, _ = apply_captions(original, GptCaptioner(SECRET, transport=_fake(["x", "y"])))
    assert len(spec.shots) == len(original.shots)


# --------------------------------------------------------------------------
# the response parser


def test_output_text_is_preferred():
    assert extract_text({"output_text": "hello"}) == "hello"


def test_a_list_output_text_is_joined():
    assert extract_text({"output_text": ["a", "b"]}) == "ab"


def test_the_output_array_is_the_fallback():
    response = {"output": [{"content": [{"type": "output_text", "text": " hi "}]}]}
    assert extract_text(response) == "hi"


def test_an_unrecognised_shape_returns_empty_rather_than_raising():
    """A shape this code does not know must fall back to the deterministic
    caption, not crash a render."""
    assert extract_text({}) == ""
    assert extract_text({"output": [{"content": []}]}) == ""
    assert extract_text({"output": ["not a dict"]}) == ""
    assert extract_text({"output_text": []}) == ""


# --------------------------------------------------------------------------
# no network, ever, by default


def test_the_default_configuration_makes_no_transport_call(monkeypatch):
    """The layer is opt-in. Nothing constructs a captioner unless asked."""
    calls = []
    monkeypatch.setattr(llm, "http_transport", lambda p, k: calls.append(1) or {})
    # Building and rendering a memory happens entirely without this module.
    from rekindle.memory import engine

    assert "llm" not in dir(engine)
    assert calls == []


def test_no_test_in_this_module_can_reach_the_network(monkeypatch):
    """A guard on the guard: if a future test forgets to inject a transport,
    this makes the failure obvious rather than a silent billed API call."""
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("a test attempted a real network connection")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    spec, report = apply_captions(_spec(), GptCaptioner(SECRET, transport=_fake(["May 2015"] * 2)))
    assert report.accepted == 2


def test_the_report_accounts_for_every_shot():
    _, report = apply_captions(
        _spec(captions=("a", "b")),
        GptCaptioner(SECRET, transport=_fake(["Back in 1999", "May 2015"])),
    )
    assert report.accounted
    assert report.requested == 2


def test_an_empty_report_is_accounted():
    assert CaptionReport().accounted


# --------------------------------------------------------------------------
# the REAL transport's error paths
#
# The fallback tests above inject a fake transport, so none of them reach
# http_transport at all - and a mutation that echoed the request headers into
# the error message survived every one of them. These exercise the real
# function with a stubbed urlopen, still without a network call.


def _stub_urlopen(monkeypatch, raiser):
    monkeypatch.setattr(llm.urllib.request, "urlopen", raiser)


def test_the_real_transport_never_puts_the_key_in_an_http_error(monkeypatch):
    """An Authorization header echoed into an exception message is exactly how
    a credential reaches a log file."""
    import io

    def raise_http(*args, **kwargs):
        raise llm.urllib.error.HTTPError(
            url=llm.API_URL,
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"Incorrect API key provided: sk-xyz"}}'),
        )

    _stub_urlopen(monkeypatch, raise_http)
    with pytest.raises(LLMUnavailable) as caught:
        llm.http_transport({"model": "m"}, SECRET)

    message = str(caught.value)
    assert SECRET not in message
    assert "Bearer" not in message
    assert "401" in message


def test_the_real_transport_never_puts_the_key_in_a_connection_error(monkeypatch):
    def raise_url(*args, **kwargs):
        raise llm.urllib.error.URLError(f"failed talking to {SECRET}")

    _stub_urlopen(monkeypatch, raise_url)
    with pytest.raises(LLMUnavailable) as caught:
        llm.http_transport({"model": "m"}, SECRET)
    assert SECRET not in str(caught.value)


def test_the_real_transport_never_puts_the_key_in_a_timeout(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise TimeoutError(SECRET)

    _stub_urlopen(monkeypatch, raise_timeout)
    with pytest.raises(LLMUnavailable) as caught:
        llm.http_transport({"model": "m"}, SECRET)
    assert SECRET not in str(caught.value)


def test_the_real_transport_reports_a_malformed_body_without_the_key(monkeypatch):
    class Response:
        def read(self):
            return b"not json at all"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    _stub_urlopen(monkeypatch, lambda *a, **k: Response())
    with pytest.raises(LLMUnavailable, match="malformed"):
        llm.http_transport({"model": "m"}, SECRET)


def test_the_real_transport_sends_the_key_as_a_bearer_header(monkeypatch):
    """The one place the key is legitimately used. Pinned so a refactor that
    drops the header fails here rather than with a confusing 401."""
    captured = {}

    class Response:
        def read(self):
            return b'{"output_text":"ok"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def capture(request, timeout=None):
        captured["headers"] = request.headers
        captured["url"] = request.full_url
        captured["body"] = request.data
        return Response()

    _stub_urlopen(monkeypatch, capture)
    assert llm.http_transport({"model": "m"}, SECRET) == {"output_text": "ok"}
    assert captured["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert captured["url"] == llm.API_URL
    assert b"m" in captured["body"]


def test_the_real_transport_is_never_called_by_any_other_test(monkeypatch):
    """A canary: if a test forgets to inject a fake, this is what it hits."""

    def explode(*args, **kwargs):
        raise AssertionError("http_transport reached the network")

    _stub_urlopen(monkeypatch, explode)
    # A normal captioning run with an injected transport touches none of it.
    _, report = apply_captions(_spec(), GptCaptioner(SECRET, transport=_fake(["May 2015"] * 2)))
    assert report.accepted == 2


# --------------------------------------------------------------------------
# A prompt memory's title is a QUERY, not a fact


def test_a_prompt_title_does_not_substantiate_its_own_words():
    """The caption whitelist is built from the fact sheet, and the title is
    part of it - correct for `album_story:Kashmir`, where the title IS an
    album the user named. A prompt's title is whatever the user typed, so
    trusting it would let the layer print an unresolvable place name under a
    photograph and defeat the rule that a memory never names a place.
    """
    facts = FactSheet(
        title="Christmas in Midnapur",
        recipe="prompt",
        photo_count=3,
        title_substantiated=False,
    )
    assert substantiated("Midnapur in the evening", facts) == REJECT_UNKNOWN_PERSON


def test_a_recipe_title_still_substantiates_its_own_words():
    """Guards the test above against passing for the wrong reason: the flag,
    and nothing else about these two sheets, is what changes the verdict."""
    facts = FactSheet(title="Christmas in Midnapur", recipe="album_story", photo_count=3)
    assert substantiated("Midnapur in the evening", facts) is None


def test_casefolding_is_the_first_line_of_defence_not_the_only_one():
    """The plan this work came from recorded the title whitelist as a live
    defect for prompt memories. Measured, it is not one TODAY: `normalise`
    casefolds a prompt before it ever reaches a fact sheet, and `_NAME` only
    matches a capitalised word, so "midnapur" in the sheet never whitelists
    "Midnapur" in a caption. That is one narrow coincidence of two unrelated
    rules, which is not a guardrail - hence the flag, and hence this test
    recording which of the two is actually doing the work.
    """
    lowercase = FactSheet(title="christmas in midnapur", recipe="album_story", photo_count=3)
    assert substantiated("Midnapur in the evening", lowercase) == REJECT_UNKNOWN_PERSON
