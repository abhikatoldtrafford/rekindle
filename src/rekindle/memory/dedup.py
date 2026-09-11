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
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rekindle.models import Photo

# Frames of one burst arrive seconds apart. 30s is generous enough for a
# hand-triggered sequence and is the value the brief settled on.
DEFAULT_GAP_SECONDS = 30.0
# Hamming distance over 64 bits. See memory.fingerprint for the measurement
# that chose 6: it touches 0.033% of unrelated pairs.
DEFAULT_THRESHOLD = 6


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

    @property
    def accounted(self) -> bool:
        return self.considered == self.kept + self.collapsed


def bursts(
    photos: list[Photo],
    *,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
    threshold: int = DEFAULT_THRESHOLD,
) -> list[list[Photo]]:
    """Group photos into bursts. Every input photo appears in exactly one group.

    A photo joins the current burst only when BOTH hold:

      * it was taken within `gap_seconds` of the PREVIOUS photo, and
      * its hash is within `threshold` of the burst's ANCHOR (its first member).

    Time chains, pixels anchor. Chaining on time is what keeps a genuine
    12-frame burst together when it spans more than one gap. Anchoring on
    pixels is what prevents a drift chain: with a previous-photo comparison,
    A can match B and B match C while A and C look nothing alike, and all
    three collapse to one. Anchoring makes that impossible - C simply starts a
    new burst and becomes its own anchor.

    A photo with no fingerprint never joins a burst and never anchors one that
    others can join. Unknown is not similar; guessing here is how a video or
    an undecodable file would swallow its neighbours.
    """
    groups: list[list[Photo]] = []
    anchor: Photo | None = None
    previous: Photo | None = None

    for photo in sorted(photos, key=_sort_key):
        joins = False
        if previous is not None and anchor is not None:
            a_hash = anchor.meta.phash
            p_hash = photo.meta.phash
            # `is not None`, never a truthiness test: dHash 0 is a real value
            # that any flat image produces.
            if a_hash is not None and p_hash is not None:
                prev_at = previous.meta.taken_at_utc
                this_at = photo.meta.taken_at_utc
                if prev_at is not None and this_at is not None:
                    within_time = abs((this_at - prev_at).total_seconds()) <= gap_seconds
                    joins = within_time and hamming(a_hash, p_hash) <= threshold

        if joins:
            groups[-1].append(photo)
        else:
            groups.append([photo])
            anchor = photo
        previous = photo

    return groups


def collapse(
    photos: list[Photo],
    *,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
    threshold: int = DEFAULT_THRESHOLD,
) -> tuple[list[Photo], DedupReport]:
    """Keep the best photo of each burst. Returns (survivors, report).

    Survivors come back in the caller's own order, not burst order: this runs
    in the middle of a pipeline whose ordering the recipe already chose.
    """
    report = DedupReport(considered=len(photos))
    groups = bursts(photos, gap_seconds=gap_seconds, threshold=threshold)

    winners: set[str] = set()
    for group in groups:
        winners.add(max(group, key=_better).file_hash)
        report.sizes[len(group)] = report.sizes.get(len(group), 0) + 1
        if len(group) > 1:
            report.bursts += 1

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
