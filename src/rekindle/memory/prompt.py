"""Prompt memories: the user's own words into a Selection.

The one idea this module exists for:

> **The prompt never goes to CLIP. The prompt becomes a list of visual tags -
> what such a photo would LOOK like - and CLIP matches the tags. A photo then
> ranks by how many distinct tags agree on it.**

That is not a stylistic preference, it is the fix for a measured failure.
"Durga Puja" and "Kali Puja" are nearly the same string to CLIP as event
names: sending `kalipuja diwali celebration` straight to the encoder put
**9 of 24 shots** of the resulting memory on a Durga Puja day, and only 4 on
a genuine Kali Puja or Diwali day. As *visual descriptions* the two festivals
are not close at all - a ten-armed goddess with a lion versus a black goddess
with a long red tongue - and consensus across several such descriptions is
what separates them. Measured on the reference library with the same
pipeline, the same ground truth and the same 24 slots: **24 of 24 shots on a
genuine Kali Puja or Diwali day, 0 on a Durga Puja day.** The numbers, the
ground truth and the failures are in `docs/decision-log-prompt-memories.md`.

**Agreement is computed from RANKS, never from scores.** This is the hard
constraint the whole semantic milestone is built around: an absolute cosine
means nothing across queries. The same 0.25 is a perfect hit for `durga puja`
and rank-800 junk for `food on a plate`, and `scuba diving underwater` scores
a higher z@1 against this library than `durga puja` does with zero scuba
photos in it. So this module reads a hit's POSITION in its own tag's ranking
and never its magnitude. `--min-score` exists for a user who insists; it is
unset, and nothing here reads a score except to make the ordering total.

**Unless the user already answered the question.** An album of their own that
holds at least as many photographs as a memory has slots is better evidence
than anything above, so it IS the memory: the tags are dropped, `retrieve` is
never called, and every photograph comes from that album. A SMALLER album
cannot fill a memory, so it goes into the pool and the tags still supply the
rest. `LEAD_MIN` and `match_albums` carry the measurements, including the two
rules that were tried on this library and are not here.

Nothing in here imports `rekindle.semantic`: retrieval arrives as an injected
callable, so every test in `tests/test_prompt.py` runs with no torch, no
embedding store and no network - and `rekindle --help` never loads CLIP. An
album-led memory needs no embedding store at all, which is the one case where
this whole feature works on a default install.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from rekindle.memory import albums as album_names
from rekindle.memory import strata
from rekindle.memory.index import MemoryIndex
from rekindle.memory.recipes import Selection
from rekindle.memory.recipes.base import chronological
from rekindle.memory.spec import build_fact_sheet
from rekindle.models import Photo

#: The recipe name a prompt memory records. There is NO registered recipe by
#: this name, and `tests/test_engine.py` asserts that there never is: a
#: registered recipe would make prompt memories appear in `rekindle memories`
#: and in `--auto`, which is the one thing they must never do.
RECIPE = "prompt"

#: How deep each individual tag is searched. 100 is the measured knee for the
#: single-query version of this path (K=30 starves below the 24-shot cap,
#: K=600 buys year-buckets at visible precision cost), and the relevance cliff
#: on the best query in this library sits at about rank 400, so 100 is
#: comfortably inside it. NOT swept per-tag.
TAG_K = 100

#: How many of the consensus-ranked photos are read as seeds.
SEED_K = 100

#: A capture day needs this many of the seed photos before it is treated as a
#: day the concept happened on.
#:
#: **Measured, and it turned out to be nearly inert** once `day_quorum` exists.
#: On the reference library, across the three prompts this feature was built
#: for, moving it 1 -> 2 -> 3 changes the durga memory from 23 correct shots in
#: 11 year-buckets to 24 in 10 to 24 in 7, and changes the kali memory not at
#: all. The plan this work came from expected `min_seeds = 1` to wreck month
#: purity; with one tag it would have, but the tag quorum is what actually
#: refuses a day, and it refuses the same days at every value here. Kept at 2
#: because it is the cheaper of the two gates and it costs nothing; do not
#: describe it as load-bearing.
MIN_SEEDS = 2

#: Reciprocal-rank constant. Standard RRF. Its only job here is to order
#: photos WITHIN a vote tier; see `consensus` for why 60 makes that safe.
RRF_C = 60

#: Shape phrases that mean "spread this across years". Stripped from the
#: subject, kept in the prompt text, because the text is the memory's id.
_SHAPE_PHRASES = ("over the years", "through the years", "across the years")

SHAPE_YEARS = "over_the_years"
SHAPE_SPAN = "single_span"

#: Month names, from a literal table rather than `strftime`, which is
#: locale-dependent and would make a memory id depend on the machine's locale.
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
#: Words that FRAME a request instead of naming its subject. Removed before a
#: prompt is compared with an album title, and never reported as a word that
#: narrowed nothing.
#:
#: This list is the difference between a feature that fires and one that does
#: not. All-token matching asks the album title to contain every word of the
#: subject, so on the reference library **none of the 37 named albums is
#: reachable from a prompt as ordinary as `memories of puri`** - `Puri 25`
#: does not contain "memories" or "of". Measured: 37 of 37 albums self-match
#: from their own bare title, 0 of 37 from `memories of <title>`. With this
#: list, `memories of puri` reaches `Puri 25` again.
#:
#: It is a closed, human-readable table for the same reason `_MONTHS` is: a
#: stemmer or a stop-word package would be a dependency, a locale and a
#: silently changing answer.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "in",
        "at",
        "on",
        "of",
        "and",
        "or",
        "to",
        "from",
        "for",
        "with",
        "my",
        "our",
        "me",
        "us",
        "some",
        "all",
        "any",
        "that",
        "those",
        "these",
        "this",
        "show",
        "find",
        "memories",
        "memory",
        "photo",
        "photos",
        "picture",
        "pictures",
        "pic",
        "pics",
        "image",
        "images",
        "album",
        "albums",
        "shot",
        "shots",
    }
)

#: How many photographs an album needs before it IS the memory rather than a
#: contribution to one. Equal to `engine.DEFAULT_MAX_SHOTS`, and that is the
#: whole argument: an album the user curated that can fill a memory on its own
#: needs nothing inferred to complete it, and their labelling beats anything a
#: model guesses. Asserted equal in `tests/test_prompt.py` rather than
#: imported, because `engine` imports this module.
#:
#: **The threshold is measured, not assumed.** The reference library's 37
#: named albums split cleanly around it - 25 photographs is the largest album
#: below and 48 the smallest above, so every value between 26 and 48 gives the
#: same answer on every album here. `Puri 25` holds 3 photographs and stays a
#: contribution; `Kashmir` holds 510 and becomes the memory.
#:
#: **Letting a small album lead anyway was tried and it is worse.** The
#: tempting rule is that when nothing describes the prompt visually - no
#: festival corpus entry, no cached tags, no API key - the album is the only
#: real evidence in the run and should be the memory however small it is. On
#: `memories of puri` that rule produces a pool of three photographs, two of
#: which are the same burst two seconds apart; dedup collapses them and the
#: engine refuses the memory outright with `too_few_photos`. It converts a
#: poor memory into no memory, so it is not shipped. A small album stays a
#: contribution to the pool, and the CLI says how small it was.
LEAD_MIN = 24

#: Token split that keeps Bengali (U+0980-U+09FF) whole: album titles in this
#: library are not all ASCII, and a splitter that dropped them would make the
#: album union silently never fire on them.
_TOKEN_RE = re.compile(r"[^0-9a-zঀ-৿]+")

PARSER_VERSION = 1


def normalise(text: str) -> str:
    """The canonical form of a prompt. THIS STRING IS THE MEMORY ID.

    NFKC, then casefold, then collapse internal whitespace, then strip.
    `str.lower()` is not used: it leaves a decomposed "é" different from a
    composed one, so two prompts that are the same word would mint two ids and
    a dismissal of one would not apply to the other.

    **Punctuation is preserved.** It changes meaning, and normalising it away
    would merge prompts the user meant to keep apart.
    """
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokens(text: str) -> list[str]:
    """Content tokens of a normalised string, for lexical album matching."""
    return [t for t in _TOKEN_RE.split(normalise(text)) if t]


@dataclass(frozen=True)
class PromptQuery:
    """A parsed prompt. Pure data, derived by fixed rules from the text.

    `text` is the identity and `subject` is what gets described as tags. The
    rest are narrowings the user typed explicitly, each validated against the
    index - an unmatched year or name is NOT a filter, it stays in the subject
    as an ordinary word.
    """

    text: str
    subject: str
    shape: str = SHAPE_SPAN
    years: tuple[int, ...] = ()
    months: tuple[int, ...] = ()
    people: tuple[str, ...] = ()
    parser_version: int = PARSER_VERSION

    @property
    def memory_key(self) -> str:
        return self.text

    @property
    def memory_id(self) -> str:
        return memory_key(self)


def memory_key(query: PromptQuery) -> str:
    """`prompt:<normalised prompt>`, and nothing else in it.

    A memory id comes from a memory's DEFINING FACTS, never from its contents.
    The defining fact of a prompt memory is the prompt. Putting the parse, the
    matched albums or a hash of the chosen photos in here would mint a new id
    the first time the library changed - and a dismissal that silently stops
    applying is the worst failure available to the one sensitivity control
    this project offers.
    """
    return f"{RECIPE}:{query.text}"


def parse(text: str, index: MemoryIndex) -> PromptQuery:
    """Prompt -> PromptQuery, by fixed rules. No clock, no locale, no model.

    The grammar is deliberately closed and small:

    * a trailing shape phrase ("over the years") sets the shape and is
      stripped from the subject;
    * a bare four-digit year that the library actually has photos in;
    * a month name from the literal table above;
    * `with <person>`, matched EXACTLY against the index's own people.

    Everything the index cannot confirm stays in the subject as ordinary
    words. In particular **"in" is not a place cue**: `christmas in midnapur`
    goes to the tag generator whole, because the full phrase hand-grades far
    better than its parts and because this project has no gazetteer, so
    treating "midnapur" as a place would be inventing one.
    """
    normalised = normalise(text)
    subject = normalised
    shape = SHAPE_SPAN
    for phrase in _SHAPE_PHRASES:
        if subject.endswith(" " + phrase) or subject == phrase:
            subject = subject[: -len(phrase)].strip()
            shape = SHAPE_YEARS
            break

    known_years = set(index.years())
    known_people = {name.casefold(): name for name in index.people_counts()}

    words = subject.split()
    years: list[int] = []
    months: list[int] = []
    people: list[str] = []
    kept: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        bare = word.strip(".,;:!?")
        if _YEAR_RE.match(bare) and int(bare) in known_years:
            years.append(int(bare))
            i += 1
            continue
        if bare in _MONTHS:
            months.append(_MONTHS[bare])
            kept.append(word)
            i += 1
            continue
        if bare == "with" and i + 1 < len(words):
            # Longest match first, so "with Abhik Maiti" beats "with Abhik"
            # when both are real names in this library.
            matched = None
            for span in range(min(4, len(words) - i - 1), 0, -1):
                candidate = " ".join(w.strip(".,;:!?") for w in words[i + 1 : i + 1 + span])
                if candidate in known_people:
                    matched = (known_people[candidate], span)
                    break
            if matched is not None:
                people.append(matched[0])
                i += 1 + matched[1]
                continue
        kept.append(word)
        i += 1

    return PromptQuery(
        text=normalised,
        subject=" ".join(kept),
        shape=shape,
        years=tuple(sorted(set(years))),
        months=tuple(sorted(set(months))),
        people=tuple(sorted(set(people))),
    )


# --------------------------------------------------------------------------
# consensus


#: `retrieve(text, k) -> [(file_hash, score), ...]`, best first. Injected, so
#: nothing here imports the semantic layer.
Retriever = Callable[[str, int], list[tuple[str, float]]]


@dataclass(frozen=True)
class Consensus:
    """Which photos the tags agree on, and how strongly."""

    order: tuple[str, ...]
    votes: Mapping[str, int]
    tags_of: Mapping[str, frozenset[int]]
    tag_count: int

    def top(self, k: int) -> list[str]:
        return list(self.order[:k])


def rank_hits(hits: Iterable[tuple[str, float]]) -> list[str]:
    """One tag's hits as a TOTAL order, best first.

    `top_k` breaks ties by row index, which is store insertion order, so two
    photos with the same cosine can swap places when the store is rebuilt -
    and a swap across the `MIN_SEEDS` boundary flips a whole capture day in or
    out of the memory. Rounding to six places and then breaking the tie on
    `file_hash` makes the order depend on nothing but the photos.
    """
    return [h for h, _ in sorted(hits, key=lambda hs: (-round(hs[1], 6), hs[0]))]


def consensus(tags: Sequence[str], retrieve: Retriever, *, tag_k: int = TAG_K) -> Consensus:
    """Rank photos by HOW MANY DISTINCT TAGS put them near the top.

    Each tag is searched independently and contributes at most one vote. The
    score is

        votes(p)  +  sum over tags of  1 / (60 + rank of p in that tag)

    which is an integer vote count with a reciprocal-rank tiebreak welded on.
    The tiebreak can never outweigh a vote: every tag's contribution is at
    most 1/61, so the whole tail sums to less than 1 for any tag list shorter
    than 61. A photo three tags agree on therefore always outranks a photo two
    tags agree on, however high those two ranked it - which is the entire
    point, and the reason plain rank fusion was rejected. Fusion lets one very
    confident tag carry a photo the other tags have never heard of, and that
    is precisely the independence that separates two festivals with a shared
    visual vocabulary.

    Union was rejected for the same reason from the other end: one bad tag
    poisons the whole set. Intersection returns almost nothing across 18,201
    embedded photos.

    Not a single absolute score is read. `rank_hits` uses the score only to
    put each tag's own hits in order.
    """
    votes: Counter[str] = Counter()
    reciprocal: dict[str, float] = defaultdict(float)
    tags_of: dict[str, set[int]] = defaultdict(set)
    for i, tag in enumerate(tags):
        for rank, file_hash in enumerate(rank_hits(retrieve(tag, tag_k)), start=1):
            votes[file_hash] += 1
            reciprocal[file_hash] += 1.0 / (RRF_C + rank)
            tags_of[file_hash].add(i)
    order = sorted(votes, key=lambda h: (-(votes[h] + reciprocal[h]), h))
    return Consensus(
        order=tuple(order),
        votes=dict(votes),
        tags_of={h: frozenset(v) for h, v in tags_of.items()},
        tag_count=len(tags),
    )


def day_quorum(tag_count: int) -> int:
    """How many distinct tags must reach a capture day before it is a seed day.

    Half of them, rounded up. The photo-level vote count already orders the
    seed list; this is the same question asked of a DAY, and it is what stops
    one tag's private idea of the concept from claiming a whole day - and
    through the year-stratifier, a whole year-bucket of the memory.

    Measured on `kalipuja diwali celebration` with six tags: the quorum takes
    the memory from 15 of 24 shots on a genuine Kali Puja or Diwali day to 24
    of 24, and from 7 shots on a Durga Puja day to none. It costs coverage -
    six year-buckets instead of ten - which is the honest trade and is
    reported to the user rather than hidden.
    """
    return max(1, math.ceil(tag_count / 2))


@dataclass(frozen=True)
class SeedDay:
    day: tuple[int, int, int]
    hits: int
    tags: int

    @property
    def iso(self) -> str:
        return f"{self.day[0]:04d}-{self.day[1]:02d}-{self.day[2]:02d}"


# --------------------------------------------------------------------------
# albums


def content_tokens(text: str) -> frozenset[str]:
    """The tokens of a string that could name a subject. See `STOPWORDS`."""
    return frozenset(t for t in tokens(text) if t not in STOPWORDS)


def album_matches(index: MemoryIndex, subject: str) -> list[str]:
    """Presentable albums whose title contains EVERY content token.

    All tokens, not any: on `durga puja`, any-token also matches the album
    `Diwali Kali Puja 22` on the shared word "puja" and drags Kali Puja photos
    into a Durga Puja memory - exactly the confusion this module exists to
    prevent. Because it is a SUBSET test, a shorter prompt matches MORE:
    `kashmir` reaches all three Kashmir albums and `ladakh` reaches both
    `Leh Ladakh` and `ladakh`. That is the behaviour wanted, and it is why the
    multi-album case needs no alias configuration.

    **Only presentable albums.** Google's own `Photos from YYYY` albums are on
    every photograph, and without this filter the one-word prompt `photos`
    matched all 23 of them and unioned **17,004 photographs - 88% of the
    library** - into the candidate pool. Two independent guards now stop that:
    the presentability rule here, and "photos" being a `STOPWORDS` entry so
    the prompt has no content tokens left to match on at all.
    """
    wanted = content_tokens(subject)
    if not wanted:
        return []
    return sorted(
        name
        for name in index.album_counts()
        if album_names.presentable(name) and wanted <= set(tokens(name))
    )


#: Where a set of matched albums came from.
ALBUM_NONE = "none"
ALBUM_TOKENS = "tokens"
ALBUM_MODEL = "model"

#: Why an album is the memory rather than a contribution to one. One value
#: today, and a field rather than a bare flag because the alternative was
#: tried: see `LEAD_MIN` for the measurement that rejected letting a small
#: album lead when nothing else in the run knew anything either.
LEAD_SIZE = "size"


@dataclass(frozen=True)
class AlbumMatch:
    """Which of the user's own albums this prompt names, and how much they
    weigh.

    `sizes` counts each album AFTER the narrowings the user typed, which is
    the number the decision below has to be made on: `kashmir 2015` and
    `kashmir 1999` name the same album and only one of them is a memory.
    """

    names: tuple[str, ...] = ()
    sizes: Mapping[str, int] = field(default_factory=dict)
    source: str = ALBUM_NONE
    #: `LEAD_SIZE`, or "" when the albums are a contribution to the memory
    #: rather than the memory itself.
    lead_reason: str = ""
    #: Names the model returned that this library does not have. Reported, and
    #: never searched for.
    invented: tuple[str, ...] = ()
    #: How many DISTINCT photographs the matched albums hold between them.
    #:
    #: Not the sum of `sizes`, and the difference is not academic: `ladakh`
    #: matches `Leh Ladakh` (277) and `ladakh` (237), which are one trip filed
    #: twice and share 195 photographs. Summing says 514 and the memory is
    #: built from 319. The first version of this printed both numbers two
    #: lines apart.
    total: int = 0

    @property
    def led(self) -> bool:
        return bool(self.lead_reason)

    @property
    def largest(self) -> int:
        return max(self.sizes.values(), default=0)


def user_narrowed(query: PromptQuery, photos: Iterable[Photo]) -> list[Photo]:
    """Apply only the narrowings the USER TYPED.

    Not the festival corpus's month window. An album is the user's own
    labelling and a corpus window is rekindle's guess about when a festival
    falls; letting the guess delete photographs out of the label would be the
    wrong one of the two winning. A year, a month name or a `with <person>`
    that the user typed is a different thing entirely, and an album that
    ignored them would answer a question nobody asked.
    """
    kept = list(photos)
    if query.months:
        kept = [p for p in kept if p.meta.taken_at_local.month in query.months]
    if query.years:
        kept = [p for p in kept if p.meta.taken_at_local.year in query.years]
    if query.people:
        wanted = set(query.people)
        kept = [p for p in kept if wanted & set(p.meta.people)]
    return kept


def match_albums(
    index: MemoryIndex,
    query: PromptQuery,
    *,
    shortlist: Sequence[str] = (),
    lead_min: int = LEAD_MIN,
) -> AlbumMatch:
    """The albums this prompt names, and whether one of them IS the memory.

    Token matching runs first and is authoritative. `shortlist` - names a
    language model proposed - is consulted ONLY when the tokens matched
    nothing, so the exact rule can never be overridden by a guess and the
    measured festival behaviour cannot change when a key is present. Every
    shortlisted name is checked against this index and a name the library does
    not have is discarded, not searched for.
    """
    names = album_matches(index, query.subject)
    source = ALBUM_TOKENS if names else ALBUM_NONE
    invented: tuple[str, ...] = ()
    if not names and shortlist:
        real = {
            normalise(name): name for name in index.album_counts() if album_names.presentable(name)
        }
        picked: set[str] = set()
        missing: set[str] = set()
        for candidate in shortlist:
            actual = real.get(normalise(candidate))
            if actual is None:
                missing.add(candidate)
            else:
                picked.add(actual)
        names = sorted(picked)
        invented = tuple(sorted(missing))
        source = ALBUM_MODEL if names else ALBUM_NONE

    sizes = {}
    distinct: set[str] = set()
    for name in names:
        kept = user_narrowed(query, index.by_album(name))
        if kept:
            sizes[name] = len(kept)
            distinct.update(p.file_hash for p in kept)
    return AlbumMatch(
        names=tuple(sorted(sizes)),
        sizes=sizes,
        source=source if sizes else ALBUM_NONE,
        lead_reason=LEAD_SIZE if sizes and max(sizes.values()) >= lead_min else "",
        invented=invented,
        total=len(distinct),
    )


# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptBuild:
    """Everything the CLI needs to explain what it did."""

    query: PromptQuery
    tags: tuple[str, ...]
    seed_days: tuple[SeedDay, ...]
    albums: tuple[str, ...]
    months: tuple[int, ...]
    pool: int
    resolved: int
    selection: Selection | None
    unmatched: tuple[str, ...] = ()
    festival: str | None = None
    agreement: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)
    #: The full album decision. `albums` is the same names and is kept because
    #: it is what every existing caller reads.
    album: AlbumMatch = field(default_factory=AlbumMatch)

    @property
    def album_led(self) -> bool:
        return self.album.led


def build_selection(
    index: MemoryIndex,
    query: PromptQuery,
    tags: Sequence[str],
    retrieve: Retriever,
    *,
    seed_k: int = SEED_K,
    min_seeds: int = MIN_SEEDS,
    tag_k: int = TAG_K,
    months: Sequence[int] = (),
    festival: str | None = None,
    known_words: Sequence[str] = (),
    album_match: AlbumMatch | None = None,
) -> PromptBuild:
    """Tags -> seed days -> whole capture days -> a Selection.

    **Seed and expand, not direct top-K.** Measured on this library, direct
    top-150 gives 11 year-buckets but only 4 of 24 shots carry a face tag - a
    wall of idols with no family in it, because the event albums' portraits
    sit at median rank 500-4,500 where no content query reaches them.
    Expanding a CONFIRMED seed day to its whole capture session reaches them:
    22 to 24 of 24 shots carry faces. It costs a year-bucket or two and
    improves month purity.

    The day expansion assumes a capture day is one coherent event. That
    measured well here, where these festivals are multi-day outings, but a day
    holding a pandal visit AND an unrelated lunch pulls the lunch in, and
    nobody has measured how often that happens.

    **When an album leads, none of that runs.** An album big enough to fill a
    memory is the user's own answer to their own question, so the tags are
    dropped, `retrieve` is never called and the pool is that album - which is
    what `album_story` would have built, without minting a memory id that
    changes the day the library crosses a threshold. See `match_albums`.
    """
    if album_match is None:
        album_match = match_albums(index, query)
    if album_match.led:
        tags = ()
    consensus_result = consensus(tags, retrieve, tag_k=tag_k)
    # The index is the chokepoint: a hash the policy refused simply is not in
    # this list, so it cannot create a seed day and cannot reach a memory.
    # Resolving BEFORE counting days is what makes that true - counting first
    # would let an excluded person's photos vote a day in.
    resolved = index.resolve_many(consensus_result.order)
    months_allowed = set(months) | set(query.months)
    if months_allowed:
        resolved = [p for p in resolved if p.meta.taken_at_local.month in months_allowed]
    if query.years:
        resolved = [p for p in resolved if p.meta.taken_at_local.year in query.years]
    if query.people:
        wanted = set(query.people)
        resolved = [p for p in resolved if wanted & set(p.meta.people)]
    seeds = resolved[:seed_k]

    hits: Counter[tuple[int, int, int]] = Counter()
    day_tags: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for photo in seeds:
        local = photo.meta.taken_at_local
        day = (local.year, local.month, local.day)
        hits[day] += 1
        day_tags[day] |= consensus_result.tags_of.get(photo.file_hash, frozenset())

    quorum = day_quorum(consensus_result.tag_count)
    seed_days = tuple(
        sorted(
            (
                SeedDay(day=day, hits=n, tags=len(day_tags[day]))
                for day, n in hits.items()
                if n >= min_seeds and len(day_tags[day]) >= quorum
            ),
            key=lambda s: s.day,
        )
    )

    pool: dict[str, Photo] = {}
    for seed in seed_days:
        for photo in index.by_date(*seed.day):
            if months_allowed and photo.meta.taken_at_local.month not in months_allowed:
                continue
            pool[photo.file_hash] = photo

    # The user's own labelling, added unconditionally - and narrowed by what
    # the user typed and by nothing else, so `durga puja 2013` cannot pull in
    # the whole of `Durga Puja 25`. See `user_narrowed` for why the festival
    # corpus's month window is deliberately NOT applied here.
    matched_albums = album_match.names
    for album in matched_albums:
        for photo in user_narrowed(query, index.by_album(album)):
            pool[photo.file_hash] = photo

    agreement = tag_agreement(consensus_result, seed_k)
    build = PromptBuild(
        query=query,
        tags=tuple(tags),
        seed_days=seed_days,
        albums=matched_albums,
        months=tuple(sorted(months_allowed)),
        pool=len(pool),
        resolved=len(resolved),
        selection=None,
        unmatched=(
            # A model shortlist matched on meaning rather than on characters,
            # so the words that reached it DID narrow something and saying
            # they narrowed nothing would be false.
            ()
            if album_match.source == ALBUM_MODEL
            else unmatched_words(index, query, matched_albums, known=known_words)
        ),
        festival=festival,
        agreement=agreement,
        album=album_match,
    )
    if not pool:
        return build

    photos = chronological(list(pool.values()))
    by_year = query.shape == SHAPE_YEARS
    selection = Selection(
        photos=photos,
        facts=build_fact_sheet(
            photos,
            title=query.text,
            recipe=RECIPE,
            albums=matched_albums,
            # See FactSheet.title_substantiated: a prompt's title is a query,
            # not a fact, so the caption layer may not treat its words as
            # substantiated proper nouns.
            title_substantiated=False,
        ),
        stratify=strata.BY_YEAR if by_year else strata.BY_SPAN,
        min_strata=3 if by_year else 1,
    )
    return PromptBuild(**{**build.__dict__, "selection": selection})


def unmatched_words(
    index: MemoryIndex,
    query: PromptQuery,
    albums: Sequence[str],
    known: Sequence[str] = (),
) -> tuple[str, ...]:
    """Words of the subject that matched nothing this library can narrow by.

    Used only to say so out loud. `christmas in midnapur` builds a Christmas
    memory from the whole library, and the honest line is that "midnapur"
    narrowed nothing - not silence that looks like a place filter ran.
    """
    album_tokens: set[str] = set()
    for name in (*albums, *known):
        album_tokens |= set(tokens(name))
    people_tokens: set[str] = set()
    for name in index.people_counts():
        people_tokens |= set(tokens(name))
    out = []
    for token in tokens(query.subject):
        if (
            token in STOPWORDS
            or token in album_tokens
            or token in people_tokens
            or token in _MONTHS
        ):
            continue
        out.append(token)
    return tuple(out)


# --------------------------------------------------------------------------
# the refusal signal that is measured rather than asserted


def tag_agreement(result: Consensus, seed_k: int = SEED_K) -> float:
    """How much the tags agreed, in [0, 1]. Rank-only.

    The fraction of the consensus top-`seed_k` that more than one tag voted
    for. A concept the library really contains should have its independent
    descriptions landing on the SAME photographs; a concept it does not
    contain gives each tag its own unrelated junk.

    **This is the eighth refusal signal tried on this library and it is the
    eighth to fail.** Measured over 16 concepts the library genuinely contains
    and 16 it genuinely does not (`tests/fixtures/prompt_gate_concepts.json`),
    the two distributions sit on top of each other:

        present  min 0.10  median 0.49  max 0.86
        absent   min 0.05  median 0.41  max 0.81

    At the threshold that maximises accuracy it refuses 7 of 16 absent
    concepts and wrongly refuses 2 of 16 present ones - 66% accuracy against a
    50% base rate. Worse, among the concepts described by three or more tags
    the direction REVERSES: absent median 0.68 against present median 0.49,
    because `scuba diving underwater` (0.81) and `skiing on a glacier` (0.76)
    have tags that agree beautifully with one another on the same wrong
    photographs, while `a wedding` (0.12) and `a birthday` (0.10) are
    genuinely present and genuinely photographed several unrelated ways.

    So it is computed and PRINTED, because a number beside a preview is
    information, and it is never allowed to refuse anything. There is no
    threshold here by decision, not by omission. The working verifier is the
    user's eyes, which is why a prompt memory is a preview and never an offer.
    """
    top = result.order[:seed_k]
    if not top:
        return 0.0
    return sum(1 for h in top if result.votes.get(h, 0) > 1) / len(top)
