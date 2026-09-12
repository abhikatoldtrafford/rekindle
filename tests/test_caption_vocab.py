"""The caption vocabulary and the rule that picks from it.

The header of `corpus/caption_vocab.toml` lists five things that may never be
in it. This file is what makes those rules binding rather than aspirational:
a contributor who adds "a hospital" or "a happy family" to the corpus fails
the build, and the reason is in the failure message.

Everything here runs with no model, no GPU, no embeddings and no photographs.
`vocab.choose` takes a mapping of probe to percentile, which is the whole
interface between the rules and the pixels.
"""

from __future__ import annotations

import textwrap

import pytest

from rekindle.memory import vocab
from rekindle.memory.festivals import CorpusError

# --------------------------------------------------------------------------
# the five rules the corpus header promises


#: Words that name a person, a relationship, an identity or a role. Rule 2.
_PEOPLE_WORDS = {
    "family",
    "mother",
    "father",
    "child",
    "children",
    "baby",
    "kid",
    "kids",
    "couple",
    "friend",
    "friends",
    "bride",
    "groom",
    "man",
    "woman",
    "boy",
    "girl",
    "son",
    "daughter",
    "wife",
    "husband",
    "grandmother",
    "grandfather",
}

#: Sentiment and occasion. Rule 3.
_FEELING_WORDS = {
    "happy",
    "sad",
    "joyful",
    "beautiful",
    "lovely",
    "peaceful",
    "romantic",
    "fun",
    "celebration",
    "celebrating",
    "party",
    "wedding",
    "birthday",
    "funeral",
    "holiday",
    "vacation",
    "festival",
}

#: Sensitive context. Rule 4, and the one `known-limitations.md` records a
#: real near-miss on: 270 photographs of institutional buildings labelled
#: "a hospital or a clinic", containing no hospital.
_SENSITIVE_WORDS = {
    "hospital",
    "clinic",
    "doctor",
    "nurse",
    "patient",
    "medical",
    "illness",
    "sick",
    "injury",
    "funeral",
    "grave",
    "cemetery",
    "protest",
    "police",
    "prison",
    "arrest",
    "church",
    "mosque",
    "synagogue",
    "prayer",
    "praying",
    "worship",
    "ritual",
    "priest",
    "idol",
    "god",
    "goddess",
    "alcohol",
    "drunk",
    "cigarette",
    "gun",
    "weapon",
    "naked",
    "nude",
}


def _words(text: str) -> set[str]:
    return {"".join(c for c in w if c.isalpha()).casefold() for w in text.split()}


@pytest.mark.parametrize("term", vocab.load(), ids=lambda t: t.probe[:40])
def test_no_proper_nouns(term):
    """Rule 1. A capitalised word mid-phrase is a place or a name.

    `says` may capitalise its FIRST word, because it can lead a caption. Every
    other word in either field must be lower case.
    """
    for field in (term.probe, term.says.split(" ", 1)[-1] if " " in term.says else ""):
        for word in field.split()[1:] if field is term.probe else field.split():
            assert not word[:1].isupper(), (
                f"{word!r} in {term.probe!r} is capitalised. Rule 1: no proper nouns - "
                "there is no gazetteer in this project and a capitalised word in a "
                "caption is an invented place or an invented person."
            )


@pytest.mark.parametrize("term", vocab.load(), ids=lambda t: t.probe[:40])
def test_no_people_or_relationships(term):
    """Rule 2. Who is in a photograph comes from face tags or from nowhere.

    "a large crowd of many people" is allowed and is the one exception: it
    counts bodies and names nobody. The words below name a person's role.
    """
    found = (_words(term.probe) | _words(term.says)) & _PEOPLE_WORDS
    assert not found, (
        f"{term.probe!r} contains {sorted(found)}. Rule 2: a caption may not "
        "assert who someone is, or what they are to each other."
    )


@pytest.mark.parametrize("term", vocab.load(), ids=lambda t: t.probe[:40])
def test_no_sentiment_or_occasion(term):
    found = (_words(term.probe) | _words(term.says)) & _FEELING_WORDS
    assert not found, (
        f"{term.probe!r} contains {sorted(found)}. Rule 3: a montage that tells "
        "someone how they felt is inventing the only part of the memory that was theirs."
    )


@pytest.mark.parametrize("term", vocab.load(), ids=lambda t: t.probe[:40])
def test_no_sensitive_context(term):
    found = (_words(term.probe) | _words(term.says)) & _SENSITIVE_WORDS
    assert not found, (
        f"{term.probe!r} contains {sorted(found)}. Rule 4: see known-limitations.md, "
        "where 270 photographs of institutional buildings were labelled "
        "'a hospital or a clinic' and contained no hospital."
    )


def test_every_facet_has_at_least_two_terms():
    """A facet with one term always wins its own comparison.

    `MIN_MARGIN` is measured against the runner-up, and with no runner-up the
    runner-up score is 0.0, so the margin check cannot fire. A one-term facet
    is therefore gated by the percentile alone, silently - which is a
    different rule from the one the module documents.
    """
    for facet in vocab.FACETS:
        assert len(vocab.by_facet(facet)) >= 2, facet


def test_says_fragments_compose_grammatically():
    """A subject leads, a setting and a light follow. Enforced by shape.

    Without this, adding `says = "beach"` to the setting facet produces
    "A flower beach" and nobody notices until a memory is rendered.
    """
    for term in vocab.by_facet(vocab.SUBJECT):
        assert term.says[0].isupper(), f"{term.says!r} leads a caption, so it capitalises"
    for facet in (vocab.SETTING, vocab.LIGHT):
        for term in vocab.by_facet(facet):
            assert term.says[0].islower(), f"{term.says!r} follows a subject, so it does not"
            assert term.says.split()[0] in {
                "at",
                "in",
                "on",
                "by",
                "under",
                "after",
                "indoors",
            }, f"{term.says!r} must be a prepositional phrase to follow a subject"


# --------------------------------------------------------------------------
# loading


def _write(tmp_path, body: str):
    path = tmp_path / "v.toml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    vocab.load.cache_clear()
    return path


def test_unknown_facet_is_refused(tmp_path):
    path = _write(
        tmp_path,
        """
        [[term]]
        facet = "mood"
        probe = "x"
        says = "y"
        """,
    )
    with pytest.raises(CorpusError, match="facet"):
        vocab.load(path)


def test_duplicate_probe_is_refused(tmp_path):
    path = _write(
        tmp_path,
        """
        [[term]]
        facet = "setting"
        probe = "a photograph of a beach"
        says = "at the sea"
        [[term]]
        facet = "light"
        probe = "a photograph of a beach"
        says = "at sunset"
        """,
    )
    with pytest.raises(CorpusError, match="share the probe"):
        vocab.load(path)


def test_empty_corpus_is_refused(tmp_path):
    with pytest.raises(CorpusError, match="no \\[\\[term\\]\\]"):
        vocab.load(_write(tmp_path, "version = 1\n"))


def test_missing_file_names_itself(tmp_path):
    vocab.load.cache_clear()
    with pytest.raises(CorpusError, match="could not be read"):
        vocab.load(tmp_path / "absent.toml")


@pytest.fixture(autouse=True)
def _restore_cache():
    yield
    vocab.load.cache_clear()


# --------------------------------------------------------------------------
# choose: the rule


def _scores(**overrides) -> dict[str, float]:
    """Every probe at 0, then whatever the test sets."""
    out = dict.fromkeys(vocab.probes(), 0.0)
    for term in vocab.load():
        for key, value in overrides.items():
            if key.replace("_", " ") in term.probe:
                out[term.probe] = value
    return out


def test_a_term_below_its_facet_floor_says_nothing():
    """The percentile floor is what stands between a caption and a guess."""
    scores = _scores(sandy_beach=vocab.FACET_PERCENTILE[vocab.SETTING] - 0.001)
    assert vocab.choose(scores).terms == ()
    assert vocab.REJECT_BELOW_FLOOR in vocab.choose(scores).rejected


def test_a_term_at_its_facet_floor_speaks():
    scores = _scores(sandy_beach=vocab.FACET_PERCENTILE[vocab.SETTING])
    assert [t.says for t in vocab.choose(scores).terms] == ["at the sea"]


def test_the_subject_floor_is_higher_than_the_setting_floor():
    """Measured, and the whole reason the floors are per-facet.

    An object claim asserts a particular thing is present; a setting is
    supported by the whole frame. At one floor of 0.97, three of twelve
    hand-graded captions named a vehicle that was not there - and the vehicle
    term is gone now, because at 0.995 it was still only right one time in
    six.
    """
    assert vocab.FACET_PERCENTILE[vocab.SUBJECT] > vocab.FACET_PERCENTILE[vocab.SETTING]
    at_setting_floor = _scores(tall_trees=vocab.FACET_PERCENTILE[vocab.SETTING])
    assert vocab.choose(at_setting_floor).terms == ()
    at_subject_floor = _scores(tall_trees=vocab.FACET_PERCENTILE[vocab.SUBJECT])
    assert [t.says for t in vocab.choose(at_subject_floor).terms] == ["Trees"]


def test_two_settings_too_close_together_say_nothing():
    """CLIP cannot tell a garden from open country here, so neither may we."""
    scores = _scores(garden_with_lawn=0.995)
    for term in vocab.by_facet(vocab.SETTING):
        if "open green fields" in term.probe:
            scores[term.probe] = 0.995 - vocab.MIN_MARGIN / 2
    result = vocab.choose(scores)
    assert result.terms == ()
    assert vocab.REJECT_TOO_CLOSE in result.rejected


def test_a_clear_winner_survives_the_margin():
    scores = _scores(garden_with_lawn=0.995)
    for term in vocab.by_facet(vocab.SETTING):
        if "open green fields" in term.probe:
            scores[term.probe] = 0.995 - vocab.MIN_MARGIN * 2
    assert [t.says for t in vocab.choose(scores).terms] == ["in a garden"]


def test_a_probe_absent_from_the_scores_does_not_compete():
    """A vocabulary that grew since the scores were computed must shorten a
    caption, never invent one."""
    scores = _scores(sandy_beach=0.99)
    scores.pop(vocab.by_facet(vocab.SETTING)[0].probe)
    assert vocab.choose(scores).terms == ()


def test_a_tie_is_broken_by_corpus_order_and_is_stable():
    """Two facet terms at the same percentile must give the same answer every
    run, or the engine's byte-for-byte promise dies the moment captions are on.

    Corpus order decides, because `by_facet` reads a checked-in file and the
    sort is stable - which also means a maintainer can express a preference by
    putting a phrase first. Swept with `min_margin=0`: the shipped margin
    refuses a tie outright and would make this pass whatever the sort did,
    which is a test that cannot fail.

    The pair is chosen so that corpus order and alphabetical order DISAGREE,
    so an alphabetical tiebreak sneaking back in fails here.
    """
    light = vocab.by_facet(vocab.LIGHT)
    first, second = light[1], light[3]
    assert second.probe < first.probe, "the pair must disagree with alphabetical order"
    scores = {**dict.fromkeys(vocab.probes(), 0.0), first.probe: 0.99, second.probe: 0.99}
    picked = vocab.choose(scores, min_margin=0.0).terms
    assert [t.probe for t in picked] == [first.probe]


# --------------------------------------------------------------------------
# phrase: the sentence


def _grounding(*says: str) -> vocab.Grounding:
    lookup = {t.says: t for t in vocab.load()}
    return vocab.Grounding(terms=tuple(lookup[s] for s in says))


def test_subject_and_setting_read_as_english():
    assert vocab.phrase(_grounding("A flower", "in a garden")) == "A flower in a garden"


def test_a_setting_alone_is_capitalised():
    assert vocab.phrase(_grounding("at the sea")) == "At the sea"


def test_light_alone_leads_rather_than_opening_with_a_comma():
    """Found by looking at a rendered memory: four of twenty-four captions in
    `scenery:sea` read ", at sunset", because the light fragment was only ever
    appended and nothing led it."""
    assert vocab.phrase(_grounding("at sunset")) == "At sunset"
    assert not vocab.phrase(_grounding("at sunset")).startswith(",")


def test_the_deterministic_caption_is_kept_on_the_end():
    """It is a year or a date - a fact - and worth more than the light."""
    assert vocab.phrase(_grounding("at the sea"), "2019") == "At the sea, 2019"


def test_a_tail_that_does_not_fit_is_dropped_not_truncated():
    """The light does not fit at 30 characters and the year does, so the light
    goes and the year stays - each tail is judged on its own. A caption that
    ended "at the sea, at su" would read as a bug, and the fragments are
    independently true, so dropping one loses detail and nothing else."""
    line = vocab.phrase(_grounding("A flower", "in a garden", "at sunset"), "2019", max_chars=30)
    assert line == "A flower in a garden, 2019"
    assert "at su" not in line


def test_nothing_grounded_is_a_blank_even_with_a_deterministic_caption():
    """`phrase` composes a GROUNDED caption. A caller that wants the bare
    deterministic one already has it, and returning it from here would make a
    silent CLIP look like a working one."""
    assert vocab.phrase(vocab.Grounding(terms=()), "2019") == ""


def test_every_caption_the_vocabulary_can_produce_fits():
    """The longest possible caption, from the longest term in every facet."""
    longest = [max(vocab.by_facet(f), key=lambda t: len(t.says)) for f in vocab.FACETS]
    line = vocab.phrase(vocab.Grounding(terms=tuple(longest)), "31 December 2019")
    assert 0 < len(line) <= vocab.MAX_CHARS
