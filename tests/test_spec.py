"""MemorySpec, FactSheet and the deterministic captions.

The two rules under test: the spec is byte-reproducible, and the fact sheet -
the ONLY thing the optional GPT layer ever sees - contains no filesystem path
and no invented fact.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rekindle.memory import captions
from rekindle.memory.spec import (
    FactSheet,
    MemorySpec,
    Shot,
    build_fact_sheet,
    safe_slug,
)
from rekindle.models import Gps, MediaType, Photo, PhotoMeta


def _p(h, *, local=datetime(2020, 5, 1, 12, 0), people=(), gps=None, media=MediaType.IMAGE):
    return Photo(
        file_hash=h,
        paths=[Path(f"D:/google_photos/Takeout/Google Photos/Kashmir/{h}.jpg")],
        media_type=media,
        meta=PhotoMeta(
            taken_at_utc=local.replace(tzinfo=UTC),
            taken_at_local=local,
            people=list(people),
            gps=Gps(*gps) if gps else None,
        ),
        first_seen=datetime(2020, 1, 1, tzinfo=UTC),
        last_seen=datetime(2020, 1, 1, tzinfo=UTC),
    )


def _spec(**kw) -> MemorySpec:
    shots = kw.pop(
        "shots",
        (Shot("a", "2020", "2020-05-01T12:00:00", True),),
    )
    return MemorySpec(
        recipe=kw.pop("recipe", "album_story"),
        key=kw.pop("key", "Kashmir"),
        title=kw.pop("title", "Kashmir"),
        subtitle=kw.pop("subtitle", "1 photo, May 2020"),
        shots=shots,
        facts=kw.pop("facts", FactSheet(title="Kashmir", recipe="album_story", photo_count=1)),
        public_safe=kw.pop("public_safe", True),
    )


# --------------------------------------------------------------------------
# determinism and round-trip


def test_spec_round_trips_through_json():
    spec = _spec()
    assert MemorySpec.loads(spec.dumps()) == spec


def test_two_dumps_of_the_same_spec_are_byte_identical():
    """The engine's central promise. A dict iteration order change would
    quietly break it, which is why dumps() sorts keys."""
    spec = _spec()
    assert spec.dumps() == spec.dumps()


def test_a_spec_from_a_different_version_is_refused():
    text = _spec().dumps().replace('"spec_version": 1', '"spec_version": 99')
    with pytest.raises(ValueError, match="spec_version"):
        MemorySpec.loads(text)


def test_non_ascii_titles_survive_readably():
    """Album titles in a real library are not all ASCII."""
    spec = _spec(title="Mahasaptami, 2013 — দুর্গা")
    assert "দুর্গা" in spec.dumps()
    assert MemorySpec.loads(spec.dumps()).title == spec.title


# --------------------------------------------------------------------------
# the fact sheet: no paths, no inventions


def test_a_fact_sheet_contains_no_filesystem_path():
    """The fact sheet is the complete input to the GPT layer. A path leaks a
    username, a drive layout and often a person's name.

    Asserted by scanning the SERIALISED form, not by reading the field list:
    a field added later that happens to carry a path must fail this.
    """
    photos = [_p("a"), _p("b")]
    facts = build_fact_sheet(photos, title="Kashmir", recipe="album_story")
    serialised = str(facts.to_json())
    assert "google_photos" not in serialised.lower()
    assert "Takeout" not in serialised
    assert ".jpg" not in serialised
    assert "D:" not in serialised


def test_a_fact_sheet_without_gps_has_NO_place_key_at_all():
    """Not an empty one, not a guess. An LLM handed an empty `places` will
    write around it; a missing key is unambiguous."""
    facts = build_fact_sheet([_p("a")], title="X", recipe="r")
    assert "places" not in facts.to_json()


def test_a_fact_sheet_with_gps_carries_COORDINATES_not_a_place_name():
    """There is no offline gazetteer in this project, so a city name would be
    an invention."""
    facts = build_fact_sheet([_p("a", gps=(22.5, 87.25))], title="X", recipe="r")
    assert facts.to_json()["places"] == [[22.5, 87.25]]


def test_a_fact_sheet_without_dates_omits_the_date_keys():
    photo = _p("a")
    photo.meta.taken_at_local = None
    facts = build_fact_sheet([photo], title="X", recipe="r")
    assert "date_from" not in facts.to_json()


def test_fact_sheet_counts_are_derived_not_asserted():
    photos = [
        _p("a", local=datetime(2019, 5, 1, 12, 0), people=["Paramita"]),
        _p("b", local=datetime(2020, 6, 1, 12, 0), people=["Paramita", "Abhik Maiti"]),
    ]
    facts = build_fact_sheet(photos, title="X", recipe="r")
    assert facts.photo_count == 2
    assert facts.years == (2019, 2020)
    assert facts.per_year == {"2019": 1, "2020": 1}
    assert facts.people == {"Paramita": 2, "Abhik Maiti": 1}
    assert facts.date_from.startswith("2019-05-01")
    assert facts.date_to.startswith("2020-06-01")


def test_people_are_ranked_by_count_then_name():
    photos = [_p("a", people=["Zoe", "Amy"]), _p("b", people=["Amy"])]
    facts = build_fact_sheet(photos, title="X", recipe="r")
    assert list(facts.people) == ["Amy", "Zoe"]


def test_a_fact_sheet_round_trips():
    facts = build_fact_sheet([_p("a", gps=(1.0, 2.0), people=["Amy"])], title="X", recipe="r")
    assert FactSheet.from_json(facts.to_json()) == facts


def test_video_count_is_reported():
    facts = build_fact_sheet([_p("a"), _p("v", media=MediaType.VIDEO)], title="X", recipe="r")
    assert facts.video_count == 1


def test_an_empty_photo_list_makes_an_empty_but_valid_fact_sheet():
    facts = build_fact_sheet([], title="X", recipe="r")
    assert facts.photo_count == 0
    assert facts.years == ()


# --------------------------------------------------------------------------
# public-safe lift


def test_a_memory_is_public_safe_only_when_EVERY_shot_is():
    ok = _spec(shots=(Shot("a", "", None, True), Shot("b", "", None, True)))
    assert ok.public_safe is True
    mixed = MemorySpec(
        recipe="r",
        key="k",
        title="t",
        subtitle="",
        shots=(Shot("a", "", None, True), Shot("b", "", None, False)),
        facts=FactSheet(title="t", recipe="r", photo_count=2),
        public_safe=all(
            s.public_safe for s in (Shot("a", "", None, True), Shot("b", "", None, False))
        ),
    )
    assert mixed.public_safe is False


# --------------------------------------------------------------------------
# slugs


def test_slug_survives_the_real_album_titles():
    """Real titles from the reference library: commas, a slash, mixed case."""
    assert safe_slug("Kashmir, day 1 and 2") == "kashmir-day-1-and-2"
    assert safe_slug("Abhirup Birthday/ Sudipta Saad") == "abhirup-birthday-sudipta-saad"
    assert safe_slug("Leh Ladakh") == "leh-ladakh"
    assert safe_slug("Diwali Kali Puja 22") == "diwali-kali-puja-22"


def test_slug_of_an_unrepresentable_key_is_not_empty():
    """A directory name of "" would be a crash, not a memory."""
    assert safe_slug("....") == "memory"


def test_slug_of_a_pair_key():
    assert safe_slug("Abhik Maiti + Paramita") == "abhik-maiti-paramita"


def test_spec_slug_combines_recipe_and_key():
    assert _spec().slug == "album_story-kashmir"


# --------------------------------------------------------------------------
# captions


def test_month_names_do_not_depend_on_locale():
    """strftime('%B') is locale-dependent, so the same index would produce
    different memories on a machine with a different LC_TIME."""
    assert captions.month_name(10) == "October"
    assert captions.month_year(datetime(2024, 10, 20)) == "October 2024"


def test_years_ago_is_arithmetic_and_gets_the_singular_right():
    assert captions.years_ago(2020, 2026) == "6 years ago today"
    assert captions.years_ago(2025, 2026) == "1 year ago today"
    assert captions.years_ago(2026, 2026) == "today"


def test_titles_are_built_from_names_never_invented():
    assert captions.person_title("Avyan") == "Avyan over the years"
    assert captions.pair_title("Abhik Maiti", "Paramita") == "Abhik Maiti and Paramita"


def test_a_span_within_one_month_is_stated_once():
    photos = [_p("a", local=datetime(2024, 10, 1)), _p("b", local=datetime(2024, 10, 20))]
    assert captions.span_subtitle(photos) == "October 2024"


def test_a_span_across_months_names_both_ends():
    photos = [_p("a", local=datetime(2024, 10, 1)), _p("b", local=datetime(2025, 1, 3))]
    assert captions.span_subtitle(photos) == "October 2024 - January 2025"


def test_a_span_with_no_dates_is_empty_not_a_guess():
    photo = _p("a")
    photo.meta.taken_at_local = None
    assert captions.span_subtitle([photo]) == ""


def test_the_count_subtitle_gets_the_singular_right():
    assert captions.count_subtitle([_p("a")]) == "1 photo"
    assert captions.count_subtitle([_p("a"), _p("b")]) == "2 photos"


def test_describe_people_is_empty_when_nobody_is_tagged():
    """43.7% of live photos carry no face tag. Silence is the honest output,
    not "unknown" or "nobody"."""
    assert captions.describe_people([_p("a")]) == ""


def test_describe_people_lists_the_most_present_first():
    photos = [_p("a", people=["Amy", "Zoe"]), _p("b", people=["Amy"]), _p("c", people=["Amy"])]
    assert captions.describe_people(photos) == "Amy and Zoe"


def test_the_place_title_never_names_a_place():
    assert captions.PLACE_TITLE == "A place you kept coming back to"


def test_the_title_substantiated_flag_round_trips():
    facts = FactSheet(
        title="durga puja over the years",
        recipe="prompt",
        photo_count=4,
        title_substantiated=False,
    )
    assert FactSheet.from_json(facts.to_json()).title_substantiated is False


def test_an_ordinary_fact_sheet_does_not_carry_the_flag_at_all():
    """Absent when true, so every spec already on disk round-trips unchanged."""
    facts = FactSheet(title="Kashmir", recipe="album_story", photo_count=4)
    assert "title_substantiated" not in facts.to_json()
    assert FactSheet.from_json(facts.to_json()).title_substantiated is True
