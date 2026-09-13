"""Contact sheets for the photographs `--public-safe` would publish.

WHY THIS EXISTS
---------------
`--public-safe` is two automated checks and neither is sufficient. Tags catch
a person the owner has named; the face detector catches a face the tags did
not account for. On a hand-checked sample of the reference library, 189
photographs passed the tags, 105 survived the detector, and **a person removed
41 more** - a child being held, a woman with her body in frame and her head
above it, a wedding frame with three faces turned away. None of those is a
face a detector can count.

So the last gate is a human, and the `PUBLIC-SAFE` marker beside each memory
says so. The problem is that saying so does not make it possible: the
survivors on this library are 520 photographs scattered across the whole
export, and nobody opens 520 files. **A gate whose final step is impractical
is a gate that does not happen.**

This turns it into three images. Every photograph the gate would admit, as
labelled thumbnails on a grid, ordered so that the ones most likely to be
wrong are in the top-left of the first sheet - because a person who looks at
one sheet and stops should still have seen the worst of it.

WHAT IT IS NOT
--------------
It never publishes, never excludes and never edits the index. It writes image
files and prints the command to exclude anything you reject. The decision
stays where it has to be.

THE ORDER IS THE DESIGN
-----------------------
`suspicion` ranks a photograph the gate has ALREADY passed, worst first:

1. **Never examined.** No detector evidence at all, so only the tags vouch
   for it. On the reference library, 48 of the 520 survivors.
2. **Tagged, and the detector found nobody.** Somebody is known to be in this
   photograph and the detector could not see them - so it is demonstrably
   missing faces in this image, and any *other* person in it would have been
   missed too. This is the case a naive ranking puts last.
3. **The rest, most faces first.** More faces is more chance one of them is a
   stranger the tags happen to cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rekindle.models import Photo

#: Thumbnail edge on the sheet, and how many fit across it. 220px is legible
#: enough to see a face at the back of a frame on a normal screen; 6 across
#: keeps a sheet under 1,400px wide so it opens without scrolling sideways.
CELL = 220
COLUMNS = 6
#: Rows per sheet. 8 x 6 = 48 photographs an image, so this library's 520
#: survivors are eleven sheets - openable, unlike 520 files.
ROWS = 8
PER_SHEET = COLUMNS * ROWS

LABEL_HEIGHT = 26
MARGIN = 8
BACKGROUND = (18, 18, 20)
TEXT = (238, 238, 240)
#: The band behind a label, by how much the photograph needs a second look.
BAD = (150, 40, 40)
WARN = (140, 100, 30)
CALM = (40, 44, 52)


@dataclass(frozen=True)
class Candidate:
    """One photograph the gate admits, with why it is ranked where it is."""

    photo: Photo
    reason: str
    rank: int

    @property
    def tags(self) -> int:
        return len({p for p in self.photo.meta.people if p})


def suspicion(photo: Photo) -> tuple[int, str]:
    """`(rank, reason)` for a photograph the gate has already passed.

    Lower ranks are shown first. See the module docstring for why the order
    is what it is - in particular why "the detector found nobody in a
    photograph somebody is tagged in" outranks a crowd.
    """
    tagged = len({p for p in photo.meta.people if p})
    count = photo.meta.face_count
    if count is None:
        return 0, "never examined - only the tags vouch for this"
    if tagged and count == 0:
        return 1, f"{tagged} tagged, detector saw NO face - it is missing faces here"
    return 2 + max(0, 40 - count), f"{count} faces, {tagged} tagged"


def candidates(photos: list[Photo]) -> list[Candidate]:
    """Every admitted photograph, worst first.

    `file_hash` breaks ties so two runs over one library produce the same
    sheets in the same order - the sheets are an artefact like any other and
    a reviewer comparing two runs should see a diff only where the library
    changed.
    """
    ranked = []
    for photo in photos:
        rank, reason = suspicion(photo)
        ranked.append(Candidate(photo=photo, reason=reason, rank=rank))
    return sorted(ranked, key=lambda c: (c.rank, c.photo.file_hash))


def _label(candidate: Candidate) -> tuple[str, tuple[int, int, int]]:
    photo = candidate.photo
    names = ", ".join(sorted({p for p in photo.meta.people if p})) or "no tags"
    count = photo.meta.face_count
    if count is None:
        return f"? faces | {names}", BAD
    if candidate.tags and count == 0:
        return f"0 faces?! | {names}", WARN
    return f"{count} faces | {names}", CALM


@dataclass
class SheetReport:
    """What was written, and what could not be.

    `admitted` is what the GATE allows, which is not the same as what would
    reach a published memory: the composition guardrails drop every video, and
    a video with no cached still cannot be drawn on a sheet or examined by the
    detector either. Reporting one number for all of that made the first run
    say "519 photos would be published, 47 have never been through the face
    detector, run facegate" - and facegate would not have touched them,
    because all 47 were videos.
    """

    admitted: int = 0
    drawn: int = 0
    unreadable: int = 0
    sheets: list[Path] = None  # type: ignore[assignment]
    #: IMAGES the detector has never examined. The number that means "run
    #: facegate", and it means nothing if videos are counted into it.
    never_examined: int = 0
    detector_saw_nothing: int = 0
    #: Videos the gate admits. They cannot be face-checked (the detector reads
    #: a cached still, and these have none) and `compose` drops them from
    #: every memory, so they are reported apart rather than as photographs.
    videos: int = 0

    def __post_init__(self) -> None:
        if self.sheets is None:
            self.sheets = []

    @property
    def accounted(self) -> bool:
        """Every admitted photograph was drawn, marked undrawable, or is a
        video that was deliberately left off the grid."""
        return self.admitted == self.drawn + self.unreadable + self.videos


def write_sheets(
    photos: list[Photo],
    out_dir: Path,
    *,
    resolve_path,
    cell: int = CELL,
    per_sheet: int = PER_SHEET,
    columns: int = COLUMNS,
) -> SheetReport:
    """Write contact sheets of `photos` into `out_dir`, worst first.

    `resolve_path` is handed in - `MemoryIndex.resolve_path` - so this module
    never turns a hash into a file itself. A reviewer must be looking at
    exactly the photographs the gate admitted, and the one thing that
    guarantees it is that both go through the same resolver.

    A photograph that will not decode is COUNTED, not skipped silently: it was
    admitted by the gate, so a reviewer who never sees it has not reviewed it.
    """
    from PIL import Image, ImageDraw

    from rekindle.memory.render.frames import _font
    from rekindle.meta.exif import open_upright
    from rekindle.models import MediaType

    report = SheetReport(admitted=len(photos))
    everything = candidates(photos)
    report.videos = sum(1 for c in everything if c.photo.media_type is MediaType.VIDEO)

    # VIDEOS ARE NOT DRAWN. They rank as "never examined" - correctly, the
    # detector has no still to read - and on the reference library that put 47
    # undrawable red boxes in the first 48 cells, so sheet 1 was entirely
    # things that cannot be published and sheet 2 was where the real risks
    # started. A reviewer who looks at one sheet must see the worst of it, and
    # `compose` drops every video from a memory anyway, so they are counted in
    # the report and left off the grid.
    ranked = [c for c in everything if c.photo.media_type is not MediaType.VIDEO]
    report.never_examined = sum(1 for c in ranked if c.rank == 0)
    report.detector_saw_nothing = sum(1 for c in ranked if c.rank == 1)
    if not ranked:
        return report

    out_dir.mkdir(parents=True, exist_ok=True)
    font = _font(13)
    step = cell + MARGIN
    label_font_pad = 5

    for sheet_no, start in enumerate(range(0, len(ranked), per_sheet), start=1):
        chunk = ranked[start : start + per_sheet]
        rows = (len(chunk) + columns - 1) // columns
        width = MARGIN + columns * step
        height = MARGIN + rows * (step + LABEL_HEIGHT)
        sheet = Image.new("RGB", (width, height), BACKGROUND)
        draw = ImageDraw.Draw(sheet)

        for i, candidate in enumerate(chunk):
            column, row = i % columns, i // columns
            x = MARGIN + column * step
            y = MARGIN + row * (step + LABEL_HEIGHT)
            source = resolve_path(candidate.photo)
            if source is None:
                report.unreadable += 1
                draw.rectangle([x, y, x + cell, y + cell], fill=(60, 20, 20))
                draw.text((x + 6, y + 6), "file missing", font=font, fill=TEXT)
            else:
                try:
                    # UPRIGHT. A reviewer looking at a sideways photograph is
                    # looking at a different photograph, and `open_upright` is
                    # what every other decode in the project goes through.
                    thumb = open_upright(source, draft=(cell * 2, cell * 2)).convert("RGB")
                    thumb.thumbnail((cell, cell), Image.Resampling.LANCZOS)
                    sheet.paste(
                        thumb, (x + (cell - thumb.width) // 2, y + (cell - thumb.height) // 2)
                    )
                    report.drawn += 1
                except (OSError, ValueError, Image.DecompressionBombError):
                    report.unreadable += 1
                    draw.rectangle([x, y, x + cell, y + cell], fill=(60, 20, 20))
                    draw.text((x + 6, y + 6), "would not decode", font=font, fill=TEXT)

            text, band = _label(candidate)
            bar_top = y + cell
            draw.rectangle([x, bar_top, x + cell, bar_top + LABEL_HEIGHT], fill=band)
            draw.text((x + label_font_pad, bar_top + 6), text[:34], font=font, fill=TEXT)

        path = out_dir / f"public-safe-review-{sheet_no:02d}.jpg"
        sheet.save(path, "JPEG", quality=88, optimize=True)
        report.sheets.append(path)

    return report
