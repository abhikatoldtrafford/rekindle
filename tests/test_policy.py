"""Guardrails: what may never surface, and the public-safe rule.

Fixture shapes come from the real index: 162 archived rows, 8,433 live photos
with NO face tags at all, photos carrying five names at once, and album titles
whose case varies (`Leh Ladakh` / `ladakh`).
"""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from rekindle.memory import policy as pol
from rekindle.memory.policy import (
    DateRange,
    ExclusionPolicy,
    PolicyError,
    append_exclusion,
    is_public_safe,
    load_policy,
)
from rekindle.models import MediaType, Photo, PhotoMeta


def _p(**kw) -> Photo:
    local = kw.pop("local", datetime(2020, 5, 1, 12, 0))
    return Photo(
        file_hash=kw.pop("h", "abc"),
        paths=kw.pop("paths", [Path("/lib/a.jpg")]),
        media_type=kw.pop("media_type", MediaType.IMAGE),
        meta=PhotoMeta(
            taken_at_utc=kw.pop("utc", datetime(2020, 5, 1, 12, 0, tzinfo=UTC)),
            taken_at_local=local,
            people=kw.pop("people", []),
            archived=kw.pop("archived", False),
            trashed=kw.pop("trashed", False),
            face_count=kw.pop("face_count", None),
            face_verdict=kw.pop("face_verdict", None),
        ),
        first_seen=datetime(2020, 1, 1, tzinfo=UTC),
        last_seen=datetime(2020, 1, 1, tzinfo=UTC),
        albums=kw.pop("albums", []),
    )


# --------------------------------------------------------------------------
# the non-configurable refusals


def test_archived_is_denied():
    assert ExclusionPolicy().deny_reason(_p(archived=True)) == pol.DENY_ARCHIVED


def test_trashed_is_denied():
    assert ExclusionPolicy().deny_reason(_p(trashed=True)) == pol.DENY_TRASHED


def test_no_configuration_can_re_admit_an_archived_photo():
    """162 rows on the reference library, flagged by the user as
    must-not-surface. There is deliberately no flag that turns this off, so
    the test is that a maximally permissive policy still refuses."""
    permissive = ExclusionPolicy(
        people=frozenset(),
        albums=frozenset(),
        paths=(),
        dates=(),
        public_safe_allow=frozenset({"anyone"}),
    )
    assert permissive.deny_reason(_p(archived=True)) == pol.DENY_ARCHIVED


def test_an_unrenderable_media_type_is_denied():
    assert ExclusionPolicy().deny_reason(_p(media_type=MediaType.UNKNOWN))


def test_a_photo_with_no_date_is_denied():
    assert ExclusionPolicy().deny_reason(_p(local=None, utc=None)) == pol.DENY_NO_DATE


# --------------------------------------------------------------------------
# person exclusions


def test_any_excluded_person_denies_the_photo_not_all():
    """A photo containing someone the user asked never to see is excluded even
    when five welcome people are also in it. The `all` reading would surface
    them constantly."""
    policy = ExclusionPolicy(people=frozenset({"Unwelcome"}))
    crowded = _p(people=["Paramita", "Abhik Maiti", "Avyan", "Maa", "Unwelcome"])
    assert policy.deny_reason(crowded) == pol.DENY_PERSON


def test_an_unrelated_person_does_not_deny():
    policy = ExclusionPolicy(people=frozenset({"Unwelcome"}))
    assert policy.deny_reason(_p(people=["Paramita"])) is None


# --------------------------------------------------------------------------
# date exclusions


def test_a_date_range_is_inclusive_at_both_ends():
    """An exclusive end silently admits the last day of a period someone asked
    never to see again."""
    policy = ExclusionPolicy(dates=(DateRange(date(2019, 3, 1), date(2019, 9, 30)),))
    assert policy.deny_reason(_p(local=datetime(2019, 3, 1, 0, 0))) == pol.DENY_DATE_RANGE
    assert policy.deny_reason(_p(local=datetime(2019, 9, 30, 23, 59))) == pol.DENY_DATE_RANGE


def test_a_day_outside_the_range_survives():
    policy = ExclusionPolicy(dates=(DateRange(date(2019, 3, 1), date(2019, 9, 30)),))
    assert policy.deny_reason(_p(local=datetime(2019, 2, 28, 23, 59))) is None
    assert policy.deny_reason(_p(local=datetime(2019, 10, 1, 0, 1))) is None


def test_the_date_test_uses_LOCAL_time():
    """A user excluding a date means the date they lived through. This photo
    is 2019-03-01 at 01:00 local but 2019-02-28 in UTC; the exclusion must
    still catch it. 13,116 rows in this library have a non-UTC local zone."""
    policy = ExclusionPolicy(dates=(DateRange(date(2019, 3, 1), date(2019, 3, 1)),))
    photo = _p(
        local=datetime(2019, 3, 1, 1, 0),
        utc=datetime(2019, 2, 28, 19, 30, tzinfo=UTC),
    )
    assert policy.deny_reason(photo) == pol.DENY_DATE_RANGE


# --------------------------------------------------------------------------
# album and path exclusions


def test_album_exclusion_is_casefolded():
    """`Leh Ladakh` and `ladakh` are the same trip under two spellings in this
    library; case must not be what decides whether an exclusion holds."""
    policy = ExclusionPolicy(albums=frozenset({"ladakh"}))
    assert policy.deny_reason(_p(albums=["Ladakh"])) == pol.DENY_ALBUM


def test_path_exclusion_matches_a_parent_directory():
    policy = ExclusionPolicy(paths=(Path("/lib/2019"),))
    assert policy.deny_reason(_p(paths=[Path("/lib/2019/may/a.jpg")])) == pol.DENY_PATH


def test_path_exclusion_does_not_match_a_string_prefix_sibling():
    """`/lib/photos2` starts with the string `/lib/photos` and is NOT inside
    it. A prefix compare here silently excludes an unrelated folder."""
    policy = ExclusionPolicy(paths=(Path("/lib/photos"),))
    assert policy.deny_reason(_p(paths=[Path("/lib/photos2/a.jpg")])) is None


def test_any_excluded_path_denies_a_photo_present_in_two_folders():
    """The same bytes in two folders is routine here. If one copy is inside an
    excluded directory, the photo is excluded."""
    policy = ExclusionPolicy(paths=(Path("/lib/private"),))
    photo = _p(paths=[Path("/lib/ok/a.jpg"), Path("/lib/private/a.jpg")])
    assert policy.deny_reason(photo) == pol.DENY_PATH


# --------------------------------------------------------------------------
# the public-safe rule


ALLOW = frozenset({"Abhik Maiti"})


def test_an_untagged_photo_is_NOT_public_safe():
    """THE most important assertion in this file.

    Face tags cover 55.9% of this library; 8,433 live photos carry no tag at
    all and any of them may contain anyone. Treating "no tags" as "no people"
    is how a stranger's face reaches a public GitHub repo. Default deny.
    """
    assert is_public_safe(_p(people=[]), ALLOW) is False


def test_a_photo_of_only_an_allowed_person_is_public_safe():
    assert is_public_safe(_p(people=["Abhik Maiti"]), ALLOW) is True


def test_a_photo_with_an_extra_person_is_not_public_safe():
    """Subset, never intersection."""
    assert is_public_safe(_p(people=["Abhik Maiti", "Paramita"]), ALLOW) is False


def test_a_photo_of_someone_else_entirely_is_not_public_safe():
    assert is_public_safe(_p(people=["Paramita"]), ALLOW) is False


def test_an_empty_allow_list_publishes_nothing():
    """Default deny, all the way down: the default policy has no allow-list,
    so nothing at all qualifies."""
    assert is_public_safe(_p(people=["Abhik Maiti"]), frozenset()) is False


def test_a_blank_face_tag_does_not_count_as_a_person():
    """An empty-string tag must not make an otherwise-untagged photo pass the
    non-empty test."""
    assert is_public_safe(_p(people=[""]), ALLOW) is False


def test_public_safe_mode_denies_at_the_policy_level():
    policy = ExclusionPolicy(public_safe_allow=ALLOW).with_public_safe(True)
    assert policy.deny_reason(_p(people=["Paramita"])) == pol.DENY_NOT_PUBLIC_SAFE
    assert policy.deny_reason(_p(people=["Abhik Maiti"])) is None


def test_public_safe_mode_is_off_by_default():
    assert ExclusionPolicy(public_safe_allow=ALLOW).deny_reason(_p(people=["Paramita"])) is None


# --------------------------------------------------------------------------
# deny order


def test_the_first_matching_reason_is_the_one_reported():
    """Order is the contract: the report groups by first match, and `archived`
    must win over a user-configured rule so that the count means what it
    says."""
    policy = ExclusionPolicy(people=frozenset({"Paramita"}))
    photo = _p(archived=True, people=["Paramita"])
    assert policy.deny_reason(photo) == pol.DENY_ARCHIVED


# --------------------------------------------------------------------------
# config loading


def test_a_missing_config_is_permissive(tmp_path):
    policy = load_policy(tmp_path / "nope.toml")
    assert policy.deny_reason(_p(people=["anyone"])) is None


def test_a_full_config_round_trips(tmp_path):
    path = tmp_path / "exclusions.toml"
    path.write_text(
        'people = ["Unwelcome"]\n'
        'albums = ["Hospital"]\n'
        'paths = ["/lib/private"]\n'
        'public_safe_allow = ["Abhik Maiti"]\n'
        "\n[[dates]]\nfrom = 2019-03-01\nto = 2019-09-30\n",
        encoding="utf-8",
    )
    policy = load_policy(path)
    assert policy.people == frozenset({"Unwelcome"})
    assert policy.albums == frozenset({"hospital"})
    assert policy.paths == (Path("/lib/private"),)
    assert policy.public_safe_allow == ALLOW
    assert policy.dates == (DateRange(date(2019, 3, 1), date(2019, 9, 30)),)


def test_a_quoted_date_is_accepted(tmp_path):
    """TOML gives a `date` for a bare YYYY-MM-DD and a `str` when quoted.
    Quoting it is the natural thing to write, so rejecting it would be a
    baffling error about a file that looks correct."""
    path = tmp_path / "e.toml"
    path.write_text('[[dates]]\nfrom = "2019-03-01"\nto = "2019-03-02"\n', encoding="utf-8")
    assert load_policy(path).dates[0].start == date(2019, 3, 1)


def test_malformed_toml_is_FATAL_not_a_warning(tmp_path):
    """Continuing with an unparsed exclusion list surfaces exactly the photos
    the user asked never to see, and a warning scrolls past."""
    path = tmp_path / "e.toml"
    path.write_text("people = [unclosed\n", encoding="utf-8")
    with pytest.raises(PolicyError):
        load_policy(path)


def test_a_date_entry_missing_a_bound_is_fatal(tmp_path):
    path = tmp_path / "e.toml"
    path.write_text("[[dates]]\nfrom = 2019-03-01\n", encoding="utf-8")
    with pytest.raises(PolicyError, match="from"):
        load_policy(path)


def test_a_backwards_date_range_is_rejected(tmp_path):
    """It would silently exclude nothing at all, which looks identical to a
    working exclusion."""
    path = tmp_path / "e.toml"
    path.write_text("[[dates]]\nfrom = 2019-09-30\nto = 2019-03-01\n", encoding="utf-8")
    with pytest.raises(PolicyError, match="backwards"):
        load_policy(path)


def test_a_wrongly_typed_key_is_fatal(tmp_path):
    path = tmp_path / "e.toml"
    path.write_text('people = "Unwelcome"\n', encoding="utf-8")
    with pytest.raises(PolicyError, match="list of strings"):
        load_policy(path)


# --------------------------------------------------------------------------
# appending


def test_append_creates_a_readable_file(tmp_path):
    path = tmp_path / "e.toml"
    append_exclusion(path, person="Unwelcome")
    assert load_policy(path).people == frozenset({"Unwelcome"})


def test_append_preserves_hand_written_keys_and_comments(tmp_path):
    path = tmp_path / "e.toml"
    path.write_text(
        '# my notes\nalbums = ["Hospital"]\npublic_safe_allow = ["Abhik Maiti"]\n',
        encoding="utf-8",
    )
    append_exclusion(path, person="Unwelcome")

    policy = load_policy(path)
    assert policy.people == frozenset({"Unwelcome"})
    assert policy.albums == frozenset({"hospital"})
    assert policy.public_safe_allow == ALLOW
    assert "# my notes" in path.read_text(encoding="utf-8")


def test_appending_a_second_person_keeps_the_first(tmp_path):
    """TOML forbids a duplicate key, so a naive append produces a file that no
    longer parses - and a file that no longer parses is, by the rule above, a
    hard failure on the user's next run."""
    path = tmp_path / "e.toml"
    append_exclusion(path, person="First")
    append_exclusion(path, person="Second")
    assert load_policy(path).people == frozenset({"First", "Second"})


def test_appending_a_date_range_then_a_person_still_parses(tmp_path):
    """In TOML every key after a table header belongs to that table, so
    writing `people = [...]` at the end of a file containing [[dates]] would
    silently make it `dates.people` and the exclusion would vanish."""
    path = tmp_path / "e.toml"
    append_exclusion(path, date_from=date(2019, 3, 1), date_to=date(2019, 9, 30))
    append_exclusion(path, person="Unwelcome")

    policy = load_policy(path)
    assert policy.people == frozenset({"Unwelcome"})
    assert len(policy.dates) == 1


def test_appending_two_date_ranges_keeps_both(tmp_path):
    path = tmp_path / "e.toml"
    append_exclusion(path, date_from=date(2019, 3, 1), date_to=date(2019, 3, 2))
    append_exclusion(path, date_from=date(2021, 1, 1), date_to=date(2021, 1, 2))
    assert len(load_policy(path).dates) == 2


def test_appending_nothing_is_an_error(tmp_path):
    with pytest.raises(PolicyError):
        append_exclusion(tmp_path / "e.toml")


def test_a_name_with_a_quote_survives_the_round_trip(tmp_path):
    path = tmp_path / "e.toml"
    append_exclusion(path, person='Someone "Nickname" Else')
    assert load_policy(path).people == frozenset({'Someone "Nickname" Else'})


def test_a_path_on_another_drive_is_simply_not_inside_the_excluded_one():
    """A library on D: and an exclusion on C: is ordinary on Windows.

    `Path.is_relative_to` is total on the supported Pythons - it returns False
    here rather than raising, which is why `_under` needs no exception
    handler. Pinning the behaviour means a future Python that DOES raise
    fails this test rather than crashing a user's render.
    """
    policy = ExclusionPolicy(paths=(Path("C:/private"),))
    assert policy.deny_reason(_p(paths=[Path("D:/photos/a.jpg")])) is None


# --------------------------------------------------------------------------
# the second public-safe test: the detector's count against the tags


ALLOW = frozenset({"Abhik Maiti"})


def test_a_tagged_photo_with_an_unnamed_face_is_refused():
    """Tags say who was RECOGNISED, not who was present.

    Google tags people the owner has named, so "tagged: Abhik" is entirely
    consistent with a stranger standing beside him - and the tag test alone
    publishes that. Measured on 60 photographs of this library tagged only
    "Abhik Maiti" and actually decoded, the detector found more faces than
    tags in 29 of them; one had thirty-six.
    """
    assert is_public_safe(_p(people=["Abhik Maiti"], face_count=1), ALLOW)
    assert not is_public_safe(_p(people=["Abhik Maiti"], face_count=2), ALLOW)
    assert not is_public_safe(_p(people=["Abhik Maiti"], face_count=36), ALLOW)


def test_fewer_faces_than_tags_is_ordinary_and_not_evidence():
    """A person facing away, a tag on a photograph of a photograph. The
    comparison is `>` and not `!=` for that reason."""
    assert is_public_safe(_p(people=["Abhik Maiti"], face_count=0), ALLOW)


def test_an_unexamined_photo_abstains_rather_than_being_trusted_or_refused():
    """`face_count is None` means the detector has never looked.

    Refusing would make `--public-safe` return nothing until
    `rekindle semantic facegate` has run, breaking every existing user;
    trusting would be a claim nobody made. It abstains, the tag test still
    applies, and `unverified_count` exists so the gap can be printed.
    """
    assert is_public_safe(_p(people=["Abhik Maiti"]), ALLOW)
    assert not pol.has_untagged_face(_p(people=["Abhik Maiti"]))


def test_a_zero_count_is_evidence_and_None_is_not():
    """The distinction the whole column turns on. A decoded photograph of an
    empty beach and a photograph nobody decoded both have "no faces found",
    and only one of them is a measurement."""
    assert pol.unverified_count([_p(people=["Abhik Maiti"], face_count=0)]) == 0
    assert pol.unverified_count([_p(people=["Abhik Maiti"])]) == 1


def test_the_count_cannot_rescue_a_photo_the_tags_refuse():
    """The two tests are an AND, and the tag test is still the first one. A
    photograph of someone not on the allow-list is refused however few faces
    the detector counts, and an untagged one stays default-deny."""
    assert not is_public_safe(_p(people=["Paramita"], face_count=1), ALLOW)
    assert not is_public_safe(_p(people=[], face_count=0), ALLOW)


def test_the_two_refusals_are_reported_apart():
    """`not_public_safe` is mostly "nobody is tagged here", a property of
    Google's coverage. `more_faces_than_tags` is the detector contradicting
    the tags. Merging the counts would hide how much work the detector does.
    """
    policy = ExclusionPolicy(public_safe_allow=ALLOW, public_safe_only=True)
    assert policy.deny_reason(_p(people=[])) == pol.DENY_NOT_PUBLIC_SAFE
    assert policy.deny_reason(_p(people=["Abhik Maiti"], face_count=4)) == pol.DENY_UNTAGGED_FACE
    assert policy.deny_reason(_p(people=["Abhik Maiti"], face_count=1)) is None
