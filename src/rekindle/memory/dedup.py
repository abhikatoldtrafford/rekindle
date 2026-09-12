"""Burst dedup: collapse near-identical frames, keep everything else.

The requirement is narrow and the failure modes are asymmetric. Leaving two
near-identical frames in a montage is mildly annoying. Collapsing two photos
that merely happened to be taken close together destroys a distinct memory and
the user never learns it happened. So the rule is precision-first, and it takes
TWO independent signals to collapse anything.

Why both signals are mandatory, measured on the reference library:

    59.2% of consecutive live images are within 30s of the one before.
    Chaining on time alone groups 14,425 of 18,201 images into multi-photo
    runs, the largest being 80 photos spanning 40 minutes.
    The MEDIAN dHash distance within a 30-second run is 22 - far above any
    plausible similarity threshold.

Photos taken seconds apart are usually genuinely different photos. Time is only
the gate that decides which pairs are worth comparing; the pixels decide.

THE PIXELS ARE NOT ENOUGH, AND THIS IS WHAT IT COST
---------------------------------------------------
A dHash is pixel-structural: a small camera move between two frames of one
moment reads as a different photograph. Measured on this library:

    PXL_20251226_091254817 / PXL_20251226_091259823
        FIVE SECONDS apart. The same mother holding the same child outside the
        same school, the pose shifted slightly and the frame pulled back.
        dHash distance 32 - indistinguishable from two unrelated photographs.
        CLIP cosine 0.927.

Both survived, inside the 30-second window, because only the hash was asked.
So when an embedding store exists, a photo may ALSO join a burst on cosine to
the anchor - the time gate is unchanged, the anchoring is unchanged, and the
precision-first doctrine is unchanged, which is what fixed the threshold at
0.92 rather than lower. Every pair below was opened and looked at:

    cos 0.900  17s  two different groups of people, one wedding, minutes apart
    cos 0.912  30s  one flower shop, two different shelves, different aspect
    cos 0.921   8s  four women at a gate: candid, then the posed version
    cos 0.925  19s  one cake being lit, wide then tight
    cos 0.927   5s  the pair above
    cos 0.931  16s  one lily pond, wide then tight

0.92 is where the eye changes its answer, and the library agrees that the
number means something: only 0.04% of RANDOM pairs reach 0.90 at all, while
within 30 seconds the median pair is already at 0.917. Time still gates;
inside the gate, now, the pixels OR the embedding decide.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from rekindle.config import CATALOGUE, active
from rekindle.models import Photo

# Frames of one burst arrive seconds apart. 30s is generous enough for a
# hand-triggered sequence and is the value the brief settled on.
DEFAULT_GAP_SECONDS = CATALOGUE["dedup.gap_seconds"].default
# Hamming distance over 64 bits. See memory.fingerprint for the measurement
# that chose 6: it touches 0.033% of unrelated pairs.
DEFAULT_THRESHOLD = CATALOGUE["dedup.phash_distance"].default
# Cosine at which two photos taken within `gap_seconds` of each other are one
# moment. See the module docstring for the six pairs this was read off.
#
# It is deliberately far above the 0.90 that a naive reading of the evidence
# suggests, because this rule DELETES a photo: at 0.900 and 0.912 the pairs
# were genuinely different pictures, and dedup's whole doctrine is that a
# wrong collapse costs a memory while a wrong keep costs a mild annoyance.
DEFAULT_COSINE = CATALOGUE["dedup.cosine"].default


def hamming(a: int, b: int) -> int:
    """Bit distance between two hashes.

    Both must be UNSIGNED. Python integers are arbitrary-precision, so a
    negative operand behaves as though it had an infinite run of leading sign
    bits and `bin(a ^ b).count("1")` silently returns a large wrong number
    rather than failing. `db._unsigned64` is what guarantees the inputs here.
    """
    return (a ^ b).bit_count()


def _sort_key(photo: Photo) -> tuple:
    # file_hash breaks ties so that two photos sharing a timestamp - 1,430
    # timestamps in this library are shared, one by 40 photos, because Takeout
    # dates are second-granular - always sort the same way.
    return (photo.meta.taken_at_utc, photo.file_hash)


def _pixels(photo: Photo) -> int:
    """Pixel count, with NULL dimensions sorting LAST.

    All 1,117 video rows in the reference library have width/height NULL, and
    so does any photo whose EXIF lacked them. `(w or 0) * (h or 0)` would rank
    those as the worst possible, which is the intent - an unknown size must
    never beat a known one - but it must also never crash, which is why this
    is a function rather than an inline expression repeated three times.
    """
    width = photo.meta.width or 0
    height = photo.meta.height or 0
    return width * height


def _better(photo: Photo) -> tuple:
    """Sort key for picking a burst's survivor. Highest wins.

    Sharpest, then largest, then the lexicographically smallest file_hash.

    The final tiebreak is the FILE CONTENT, deliberately. A tiebreak on path,
    mtime or insertion order would give different survivors on different
    machines, on a re-index, or after a folder rename - and the promise this
    engine makes is that the same library produces the same memories byte for
    byte. Negated because the smallest hash wins while the rest are maxima.
    """
    sharp = photo.meta.sharpness
    return (
        sharp if sharp is not None else -1.0,
        _pixels(photo),
        # Tuple of code points, negated element-wise, so that "smallest hash
        # wins" composes with the two maxima above in a single max() call.
        tuple(-ord(c) for c in photo.file_hash),
    )


@dataclass
class DedupReport:
    considered: int = 0
    kept: int = 0
    collapsed: int = 0
    bursts: int = 0
    # Photos that could not be compared because they carry no fingerprint.
    # They are all KEPT; this is how many of the kept ones were unjudged.
    unfingerprinted: int = 0
    sizes: dict[int, int] = field(default_factory=dict)
    # Of `collapsed`, how many joined their burst on the EMBEDDING alone -
    # frames the perceptual hash called unrelated and a viewer calls the same
    # photograph. Reported separately because it is the one number that says
    # whether the optional half is earning its place.
    semantic: int = 0

    @property
    def accounted(self) -> bool:
        return self.considered == self.kept + self.collapsed


def bursts(
    photos: list[Photo],
    *,
    gap_seconds: float | None = None,
    threshold: int | None = None,
    cosine: Callable[[Photo, Photo], float | None] | None = None,
    cosine_threshold: float | None = None,
) -> list[list[Photo]]:
    """Group photos into bursts. Every input photo appears in exactly one group.

    A photo joins the current burst only when BOTH hold:

      * it was taken within `gap_seconds` of the PREVIOUS photo, and
      * it looks like the burst's ANCHOR (its first member) - meaning its hash
        is within `threshold` of the anchor's, OR, when `cosine` is supplied,
        its embedding is within `cosine_threshold` of the anchor's.

    Time chains, appearance anchors. Chaining on time is what keeps a genuine
    12-frame burst together when it spans more than one gap. Anchoring is what
    prevents a drift chain: with a previous-photo comparison, A can match B
    and B match C while A and C look nothing alike, and all three collapse to
    one. Anchoring makes that impossible - C simply starts a new burst and
    becomes its own anchor. **The embedding test anchors too**, for exactly
    that reason: a slow pan across a room would otherwise chain into one
    group, since each frame resembles the last.

    A photo that NEITHER test can judge - no fingerprint and no vector - never
    joins a burst and never anchors one that others can join. Unknown is not
    similar; guessing here is how a video or an undecodable file would swallow
    its neighbours. A photo with only one of the two is judged on that one:
    an abstention is not a yes and not a no.
    """
    # `None` means "whatever the active configuration says NOW". Binding the
    # shipped number as a default argument would freeze it at import, so a
    # user's `rekindle.toml` would reach this function only if it happened to
    # be read before `rekindle.memory.dedup` was. `collapse` passes its own
    # arguments straight through, so this is the one place that resolves them.
    cfg = active().dedup
    gap_seconds = cfg.gap_seconds if gap_seconds is None else gap_seconds
    threshold = cfg.phash_distance if threshold is None else threshold
    cosine_threshold = cfg.cosine if cosine_threshold is None else cosine_threshold

    groups: list[list[Photo]] = []
    anchor: Photo | None = None
    previous: Photo | None = None

    for photo in sorted(photos, key=_sort_key):
        joins = False
        if previous is not None and anchor is not None:
            prev_at = previous.meta.taken_at_utc
            this_at = photo.meta.taken_at_utc
            if prev_at is not None and this_at is not None:
                within_time = abs((this_at - prev_at).total_seconds()) <= gap_seconds
                joins = within_time and _alike(anchor, photo, threshold, cosine, cosine_threshold)

        if joins:
            groups[-1].append(photo)
        else:
            groups.append([photo])
            anchor = photo
        previous = photo

    return groups


def _alike(
    anchor: Photo,
    photo: Photo,
    threshold: int,
    cosine: Callable[[Photo, Photo], float | None] | None,
    cosine_threshold: float,
) -> bool:
    """Do these two show the same moment? Either test may say yes.

    The hash is tried first and alone decides when it says yes, so a library
    with no embeddings behaves exactly as it did before this argument existed.
    """
    a_hash, p_hash = anchor.meta.phash, photo.meta.phash
    # `is not None`, never a truthiness test: dHash 0 is a real value that any
    # flat image produces.
    if a_hash is not None and p_hash is not None and hamming(a_hash, p_hash) <= threshold:
        return True
    if cosine is None:
        return False
    value = cosine(anchor, photo)
    return value is not None and value >= cosine_threshold


def _joined_semantically(group: list[Photo], threshold: int) -> int:
    """How many of `group` could only have joined on the embedding.

    A member whose hash is within `threshold` of the anchor would have been
    collapsed anyway; anything else in a group of more than one is there
    because the cosine test said yes. Derived from the grouping rather than
    recorded during it, so `bursts()` stays a pure function of its inputs.
    """
    if len(group) < 2:
        return 0
    anchor = group[0].meta.phash
    count = 0
    for photo in group[1:]:
        other = photo.meta.phash
        if anchor is None or other is None or hamming(anchor, other) > threshold:
            count += 1
    return count


def collapse(
    photos: list[Photo],
    *,
    gap_seconds: float | None = None,
    threshold: int | None = None,
    cosine: Callable[[Photo, Photo], float | None] | None = None,
    cosine_threshold: float | None = None,
) -> tuple[list[Photo], DedupReport]:
    """Keep the best photo of each burst. Returns (survivors, report).

    Survivors come back in the caller's own order, not burst order: this runs
    in the middle of a pipeline whose ordering the recipe already chose.
    """
    # Resolved HERE as well as in `bursts`, and not merely passed through:
    # the semantic accounting below compares hashes against `threshold`
    # directly, and a `None` reaching that comparison is a TypeError inside a
    # render. `tests/test_dedup.py` caught exactly that.
    cfg = active().dedup
    gap_seconds = cfg.gap_seconds if gap_seconds is None else gap_seconds
    threshold = cfg.phash_distance if threshold is None else threshold
    cosine_threshold = cfg.cosine if cosine_threshold is None else cosine_threshold

    report = DedupReport(considered=len(photos))
    groups = bursts(
        photos,
        gap_seconds=gap_seconds,
        threshold=threshold,
        cosine=cosine,
        cosine_threshold=cosine_threshold,
    )

    winners: set[str] = set()
    for group in groups:
        winners.add(max(group, key=_better).file_hash)
        report.sizes[len(group)] = report.sizes.get(len(group), 0) + 1
        if len(group) > 1:
            report.bursts += 1
            if cosine is not None:
                report.semantic += _joined_semantically(group, threshold)

    # `winners` is a set of hashes, so a filter alone would emit a photo once
    # per APPEARANCE in the input. A caller handing in the same photo twice -
    # a recipe that unions two overlapping queries, say - would then get a
    # memory showing it twice, with dedup having "run". Emit each winner once.
    seen: set[str] = set()
    kept = []
    for photo in photos:
        if photo.file_hash in winners and photo.file_hash not in seen:
            seen.add(photo.file_hash)
            kept.append(photo)
    report.kept = len(kept)
    report.collapsed = report.considered - report.kept
    report.unfingerprinted = sum(1 for p in kept if p.meta.phash is None)
    return kept, report
