"""What the bundled font can draw, and the fold that keeps captions inside it.

rekindle ships no font file. `render.frames` draws with Pillow's bundled
Aileron, and a character that face lacks is not skipped: FreeType draws
`.notdef`, a filled rectangle. The failure is silent - nothing raises, nothing
is logged, and the caption shows a box.

That shipped. The first live run of the GPT caption path returned
`9 August 2025 - Paramita` with an EN DASH, which is a box in the rendered
video; 3 of the 167 captions cached in the author's index carry one.

Two kinds of test here, and the first kind is the point:

* **Measurements of the font**, which fail if Pillow's bundled face changes in
  either direction - a key that starts drawing, or a value that stops. Without
  them `_FOLD` is a table of guesses that nothing checks.
* **Behaviour of the fold**, including the deliberate non-behaviour: a script
  with no ASCII equivalent is returned unchanged rather than transliterated.

Pillow is a core dependency, so nothing here is skipped anywhere.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw, ImageFont

from rekindle.memory import llm
from rekindle.memory.captions import _FOLD, renderable
from rekindle.memory.render.frames import caption_frame, title_card
from rekindle.memory.spec import FactSheet, MemorySpec, Shot

#: A private-use codepoint. No font assigns one, so whatever it draws IS the
#: `.notdef` glyph - which is how a missing glyph is detected here without
#: reading the font's cmap, and so without a font library rekindle does not
#: depend on.
NOTDEF_PROBE = ""

SIZE = 28


def _drawn(ch: str) -> bytes:
    """The pixels one character draws in the bundled font, at one size."""
    image = Image.new("L", (96, 52), 0)
    ImageDraw.Draw(image).text((4, 4), ch, font=ImageFont.load_default(size=SIZE), fill=255)
    return image.tobytes()


def _is_notdef(ch: str) -> bool:
    return _drawn(ch) == _drawn(NOTDEF_PROBE)


# --------------------------------------------------------------------------
# the measurement


def test_the_probe_finds_notdef_and_does_not_find_it_everywhere():
    """The detector itself, before anything is concluded from it.

    A detector that answered True for everything would make every test below
    pass while proving nothing, and one that answered False for everything
    would make them all pass too.
    """
    assert _is_notdef("\U000f0000")  # a different private-use plane
    assert _is_notdef("অ")  # Bengali A: the author's own script
    assert not _is_notdef("A")
    assert not _is_notdef("-")


@pytest.mark.parametrize("bad", sorted(_FOLD))
def test_every_character_in_the_fold_table_really_is_undrawable(bad):
    """The table is measured, not guessed. A key that starts drawing should
    leave the table - folding a character the font HAS loses typography for
    nothing.
    """
    assert _is_notdef(bad), f"U+{ord(bad):04X} draws; it does not belong in _FOLD"


@pytest.mark.parametrize("good", sorted(set(_FOLD.values())))
def test_every_replacement_is_drawable(good):
    """Folding one box into another would be worse than leaving it alone."""
    for ch in good:
        assert not _is_notdef(ch), f"U+{ord(ch):04X} is itself undrawable"


@pytest.mark.parametrize(
    "ch",
    ["…", "’", "“", "·", "°", "©"],
    ids=["ellipsis", "right-quote", "left-double-quote", "middle-dot", "degree", "copyright"],
)
def test_the_characters_deliberately_left_out_of_the_table_draw(ch):
    """These all have glyphs, so they are absent from `_FOLD` on purpose. This
    is what would catch someone adding them.
    """
    assert not _is_notdef(ch)


# --------------------------------------------------------------------------
# the fold


def test_the_caption_that_shipped():
    assert renderable("9 August 2025 – Paramita") == "9 August 2025 - Paramita"


def test_an_accented_letter_degrades_to_the_letter():
    assert renderable("José in Puri") == "Jose in Puri"


def test_a_script_with_no_ascii_equivalent_is_returned_UNCHANGED():
    """Deliberate. Transliterating would invent a name, which is the one thing
    a caption may never do; the boxes are the honest outcome and are recorded
    in `docs/known-limitations.md`.
    """
    bengali = "অভিক"
    assert renderable(bengali) == bengali


def test_ascii_is_returned_unchanged_and_identical():
    text = "8 August 2025, Paramita"
    assert renderable(text) is text


def test_the_fold_can_introduce_no_capital_and_no_digit():
    """Why the fold is allowed to run BEFORE the verifier.

    The verifier rejects an unsubstantiated YEAR (`\\b(1[89]\\d{2}|20\\d{2})\\b`)
    and an unsubstantiated NAME (`\\b[A-Z][a-z]{2,}\\b`). A fold that could
    produce a digit or a capital could turn a caption the verifier would have
    refused into one it accepts. None of these can: every replacement is
    punctuation, a space, or lowercase.
    """
    for bad, good in _FOLD.items():
        assert not bad.isalnum(), f"U+{ord(bad):04X} is a letter or digit"
        assert not any(c.isdigit() or c.isupper() for c in good), f"{good!r} can forge a claim"


# --------------------------------------------------------------------------
# the pipeline


def _transport(reply: str):
    def transport(payload, api_key):
        return {"output_text": reply}

    return transport


class _Cache:
    def __init__(self, rows=None):
        self.rows = dict(rows or {})

    def get(self, file_hash):
        return self.rows.get(file_hash)

    def put(self, file_hash, caption):
        self.rows[file_hash] = caption


def _one_shot_spec() -> MemorySpec:
    return MemorySpec(
        recipe="album_story",
        key="trip",
        title="A trip",
        subtitle="",
        shots=(Shot("hash-2025", "2025", "2025-08-09T10:00:00", True),),
        facts=FactSheet(
            title="A trip",
            recipe="album_story",
            photo_count=1,
            years=(2025,),
            people={"Paramita": 1},
        ),
        public_safe=True,
    )


def _context(shot):
    return llm.ShotFacts(year=2025, people=("Paramita",))


def test_a_model_en_dash_is_folded_before_it_is_cached():
    cache = _Cache()
    spec, report = llm.apply_captions(
        _one_shot_spec(),
        llm.GptCaptioner("sk-not-a-real-key", transport=_transport("2025 – Paramita")),
        _context,
        cache=cache,
    )
    assert report.accepted == 1
    assert spec.shots[0].caption == "2025 - Paramita"
    assert cache.rows == {"hash-2025": "2025 - Paramita"}


def test_a_cached_row_written_before_the_fold_existed_is_folded_on_the_way_out():
    """Three rows in the author's index are exactly this. A cached caption is
    never regenerated, so folding only on the way IN would leave them drawing a
    box forever.
    """
    cache = _Cache({"hash-2025": "2025 — Paramita"})
    spec, report = llm.apply_captions(
        _one_shot_spec(),
        llm.GptCaptioner("sk-not-a-real-key", transport=_transport("")),
        _context,
        cache=cache,
    )
    assert report.cached == 1
    assert spec.shots[0].caption == "2025 - Paramita"


# --------------------------------------------------------------------------
# the renderer, which is the net and not the fix


def test_caption_frame_draws_no_box_for_a_dash():
    """A frame captioned with an en dash is pixel-identical to one captioned
    with a hyphen. Before the fold it differed by a filled rectangle.
    """
    with_dash = caption_frame(Image.new("RGB", (400, 300), (0, 0, 0)), "2025 – Paramita")
    with_hyphen = caption_frame(Image.new("RGB", (400, 300), (0, 0, 0)), "2025 - Paramita")
    assert with_dash.tobytes() == with_hyphen.tobytes()


def test_the_title_card_folds_before_it_measures():
    """Not merely before it draws. The fold changes the string's WIDTH, and a
    card laid out for characters that never appear on it wraps in the wrong
    place.
    """
    dashed = title_card("A trip — 2025", "10 photos", (640, 360))
    hyphened = title_card("A trip - 2025", "10 photos", (640, 360))
    assert dashed.tobytes() == hyphened.tobytes()
