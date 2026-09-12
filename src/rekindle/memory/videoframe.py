"""One still frame per video, extracted once and cached, so a video can be a photo.

THE HOLE
--------
1,117 of the reference library's 19,480 rows are videos, and they appear in no
memory at all. They have no stored dimensions, no perceptual hash and no
embedding, so `composition._reject` drops every one of them and nothing
downstream ever sees them.

THE DECISION, AND THE CONSTRAINT THAT SHAPES IT
-----------------------------------------------
Extract ONE representative frame and treat it as a still. Fingerprint it,
embed it, let it flow through selection like any photograph. **Not** video
segments in the rendered MP4: that would make the `MemorySpec` depend on
whether ffmpeg is installed, and a memory has to be the same photographs on
every machine.

The constraint is easy to violate by accident, so it is stated as a rule:

    Frame extraction needs ffmpeg. THE SPEC MUST NOT.

Extraction happens at fingerprint time and writes a file. Selection reads only
the index and that cached file, and never asks whether ffmpeg exists. So:

  * ffmpeg present, frames extracted -> videos participate, and their
    fingerprints are in the index like any photo's.
  * ffmpeg absent -> no frames, no fingerprints, videos stay excluded exactly
    as they are today, and `rekindle fingerprint` SAYS SO rather than
    silently producing a smaller library.
  * two machines sharing a data directory -> identical specs, because the
    frame and its fingerprint are data, not a capability.

WHICH VIDEOS: NOT ALL OF THEM. MEASURED.
----------------------------------------
The brief for this work said 1,117 videos would become 1,117 new stills. That
is wrong, and the reference library says so:

    videos                                                   1,117
      `<name>.MP` paired with `<name>.MP.jpg`                  640
      `MVIMG_*.MP4` paired with `MVIMG_*.jpg`                  235
      standalone                                              242

**875 of the 1,117 (78%) are motion-photo halves whose still is already in the
library.** Extracting a frame from those would put two near-identical images
into the same memory - the burst dedup would then have to catch a duplicate
this pass created. The first pairing rule is the one
`enrich.takeout.propagate_to_derivatives` already uses to inherit metadata
onto a `.MP`; the second is Google's older `MVIMG_` naming, which that
function does not need but this one does.

So the real gain is **242 stills, 1.24% of the library**, not 5.8%.

WHICH FRAME: THE SHARPEST OF FIVE, AND THAT IS MEASURED TOO
-----------------------------------------------------------
"First frame" is the obvious choice and it is the wrong one - it is routinely
the shutter blur or a hand over the lens. Measured on 40 random standalone
videos from the reference library, scoring each candidate with this project's
own `memory.fingerprint.sharpness`:

    strategy                      median   p10      min     ms/video
    first frame                   0.4827   0.2567   0.0720       112
    10% of duration               0.4733   -        0.1333       134
    50% of duration               0.4578   -        0.1599       132
    ffmpeg `thumbnail=100`        0.4640   -        0.1066       267
    sharpest of 6 at 1 fps        0.4955   0.3117   0.2101       290
    SHARPEST OF FIVE (chosen)     0.5183   0.3110   0.2660       915

Read the `min` column, not the median: every fixed position is the same to
within noise in the middle of the distribution, and the whole problem is the
TAIL. Best-of-five lifts the worst frame in the sample from 0.0720 to 0.2660,
a factor of 3.7, and beats the first frame on 32 of the 40 videos (on 4 of
them by more than 50%).

The decisive number is the last one measured, not in the table: of the five
fractions, **which one won was uniformly spread - 9, 7, 8, 8, 8 out of 40**.
There is no right fixed position to find. Any single offset is a coin flip on
the videos that matter.

Five of the eight blurriest first frames were opened and looked at. One was a
featureless grey blur that best-of-five replaced with an in-focus textured
surface; one was the back of someone's shoulder, replaced by the room and the
people in it; one was a dark smear replaced by a lit party with faces in it.
One of the eight was a case where the first frame was arguably the nicer
photograph - the trade is accepted, because a dull-but-sharp frame passes the
quality gates and a blur does not.

`sharpness` rather than any other measure because it is the same function
`memory.composition` gates on: the frame chosen is the frame the rest of the
pipeline judges, which is not true of, say, ffmpeg's own `thumbnail` filter.

DETERMINISM
-----------
Fixed fractions, a fixed comparison, and ties broken by the EARLIEST fraction,
so the same video yields the same frame on every machine and every run. The
extracted JPEG itself is whatever ffmpeg produces for that timestamp on that
build; the SPEC depends on the fingerprint of the cached file, which is
written once and then read, so a different ffmpeg build cannot change a
memory that has already been fingerprinted.

WHERE THE FRAMES GO
-------------------
`<data-dir>/frames/<file_hash>.jpg`. **These are personal data** - they are
frames of the user's own videos - and this is a public repo, so they must be
ignored. They are, twice over: `.gitignore` has `[Dd]ata/` and
`*.[Jj][Pp][Gg]` at any depth.

Keyed by file hash, not by path: the same video can sit in two album folders,
and one frame for one set of bytes is the whole point.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from rekindle.models import MediaType, Photo

#: Where in each video the candidate frames are taken from, as a fraction of
#: its duration. Five, evenly spread, avoiding both ends: the first frame is
#: the documented failure case and the last is often a pan away or a hand
#: reaching for the stop button. See WHICH FRAME in the module docstring.
FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)

#: A video shorter than this has nowhere to sample from - every fraction lands
#: on the same frame - so it is grabbed once at the start and not five times.
MIN_SAMPLED_SECONDS = 1.0

#: Per-invocation ffmpeg/ffprobe timeout. A corrupt file can make ffmpeg spin;
#: one bad video among 1,117 must not hang the pass. Generous, because a long
#: .MTS on a cold spinning disk is legitimately slow.
TIMEOUT_S = 60

#: Reasons written to `photos.phash_error` for a video that got no frame. Each
#: is a distinct, resumable state - see `is_video_reason`.
ERR_NO_FFMPEG = "video_no_ffmpeg"
ERR_PAIRED = "video_paired"
ERR_NO_FRAME = "video_no_frame"
#: The reason this module's predecessors wrote, still honoured on an index
#: fingerprinted before frames existed.
ERR_LEGACY = "video"

_VIDEO_REASONS = frozenset({ERR_NO_FFMPEG, ERR_PAIRED, ERR_NO_FRAME, ERR_LEGACY})

#: Reasons that a LATER run should try again, because the world may have
#: changed since. Installing ffmpeg is the whole of it: `video_paired` is a
#: fact about the library and `video_no_frame` is a fact about the file.
RETRYABLE = frozenset({ERR_NO_FFMPEG, ERR_LEGACY})


def is_video_reason(error: str | None) -> bool:
    """Is this `phash_error` "it is a video", rather than "it is broken"?

    `composition._reject` and `memory.dedup` both need to tell those apart,
    and both used to do it by comparing against the literal string "video".
    That comparison is now wrong in four ways, which is why it lives here.
    """
    return error in _VIDEO_REASONS


def frames_dir(data_dir: Path) -> Path:
    return data_dir / "frames"


def frame_path(data_dir: Path, file_hash: str) -> Path:
    return frames_dir(data_dir) / f"{file_hash}.jpg"


def cached_frame(data_dir: Path, file_hash: str) -> Path | None:
    """The extracted still for this video, if one has been cached."""
    path = frame_path(data_dir, file_hash)
    return path if path.is_file() else None


def have_ffmpeg() -> bool:
    """Both binaries, because the pass needs both and half of ffmpeg is not a
    supported configuration - `ffprobe` is how the duration is read."""
    return bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))


# ------------------------------------------------------------------ pairing


def pairs_with_a_still(video: Photo, by_dir_name: dict[tuple[Path, str], MediaType]) -> bool:
    """Does a still of this exact moment already exist in the library?

    Two naming rules, both observed on the reference export:

      * `<name>.MP` beside `<name>.MP.jpg` - Pixel motion photos, 640 rows.
        This is the rule `enrich.takeout.propagate_to_derivatives` already
        uses to inherit metadata in the other direction.
      * `MVIMG_<stamp>.MP4` beside `MVIMG_<stamp>.jpg` - Google's older
        motion-photo naming, 235 rows. The extension differs, so the STEM is
        what matches.

    Scoped to one directory, and matched case-insensitively, for exactly the
    reasons `propagate_to_derivatives` documents: a `.MP` in one album must
    never pair with a same-named still in another, and a real export mixes
    `.MP`/`.mp` and `.jpg`/`.JPG` freely.
    """
    for path in video.paths:
        if (
            path.suffix.casefold() == ".mp"
            and by_dir_name.get((path.parent, f"{path.name}.jpg".casefold())) is MediaType.IMAGE
        ):
            return True
    for path in video.paths:
        if not path.stem.upper().startswith("MVIMG"):
            continue
        for candidate in (f"{path.stem}.jpg", f"{path.name}.jpg"):
            if by_dir_name.get((path.parent, candidate.casefold())) is MediaType.IMAGE:
                return True
    return False


def index_by_dir_name(photos: Iterable[Photo]) -> dict[tuple[Path, str], MediaType]:
    """(directory, casefolded filename) -> media type, for the pairing rule."""
    out: dict[tuple[Path, str], MediaType] = {}
    for photo in photos:
        for path in photo.paths:
            out.setdefault((path.parent, path.name.casefold()), photo.media_type)
    return out


# --------------------------------------------------------------- extraction


def _run(cmd: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        # A missing binary, a permission problem or a hung file. One bad
        # video among 1,117 must not end the pass, and the caller records the
        # miss rather than raising.
        return None


def duration_s(video: Path) -> float | None:
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ]
    )
    if proc is None or proc.returncode != 0:
        return None
    try:
        value = float(proc.stdout.decode("utf-8", "replace").strip())
    except ValueError:
        return None
    return value if value > 0 else None


def _grab(video: Path, at: float, dest: Path) -> bool:
    """One frame at `at` seconds into `dest`. `-ss` before `-i` = fast seek."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-nostdin"]
    if at > 0:
        cmd += ["-ss", f"{at:.3f}"]
    cmd += ["-i", str(video), "-frames:v", "1", "-q:v", "2", str(dest)]
    proc = _run(cmd)
    if proc is None or proc.returncode != 0:
        return False
    return dest.is_file() and dest.stat().st_size > 0


@dataclass(frozen=True)
class Extracted:
    """One video's chosen frame, and enough to explain the choice."""

    path: Path
    at_fraction: float
    sharpness: float
    candidates: int


def extract_frame(video: Path, dest: Path) -> Extracted | None:
    """Write the sharpest of five sampled frames to `dest`. None if no frame.

    Never raises: a video that cannot be read is a miss the caller counts,
    not an exception that ends a pass over a thousand files.
    """
    from PIL import Image

    from rekindle.memory.fingerprint import sharpness

    dest.parent.mkdir(parents=True, exist_ok=True)
    seconds = duration_s(video)
    fractions = FRACTIONS if seconds and seconds >= MIN_SAMPLED_SECONDS else (0.0,)

    best: Extracted | None = None
    tmp = dest.with_name(f".{dest.name}.candidate.jpg")
    tried = 0
    try:
        for fraction in fractions:
            if not _grab(video, (seconds or 0.0) * fraction, tmp):
                continue
            tried += 1
            try:
                with Image.open(tmp) as im:
                    score = sharpness(im.convert("RGB"))
            except (OSError, ValueError):
                continue
            # STRICTLY greater, so a tie is won by the EARLIEST fraction and
            # the choice does not depend on iteration luck.
            if best is None or score > best.sharpness:
                if dest.exists():
                    dest.unlink()
                tmp.replace(dest)
                best = Extracted(dest, fraction, score, tried)
                # `tmp` was moved; the next grab recreates it.
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    if best is None:
        return None
    return Extracted(dest, best.at_fraction, best.sharpness, tried)


# ------------------------------------------------------------------ the pass


@dataclass
class VideoFrameReport:
    """What the extraction pass did. Buckets that add up.

    `considered == extracted + paired + failed + skipped_no_ffmpeg` is
    asserted by a test, for the reason every accounting identity in this
    project exists: a pipeline that silently drops input is a bug.
    """

    considered: int = 0
    extracted: int = 0
    #: A motion-photo half whose still is already in the library.
    paired: int = 0
    #: ffmpeg was here and still could not produce a readable frame.
    failed: int = 0
    #: ffmpeg is not installed. Videos stay excluded and this says so.
    skipped_no_ffmpeg: int = 0
    #: Already had a cached frame from an earlier run. Subset of `extracted`.
    already_cached: int = 0
    elapsed_s: float = 0.0
    #: `(name, fraction)` for every frame taken, so the choice is inspectable.
    chosen: list[tuple[str, float]] = field(default_factory=list)
    #: `{file_hash: reason}` for every video that got NO still, so the reason
    #: reaches `photos.phash_error` per row rather than only the totals above.
    #: Written by mutation testing: the four `ERR_*` constants existed and
    #: only one of them was ever stored, so `doctor` and the CLI could report
    #: "it is a video" and never "it needs ffmpeg" - which is the one a user
    #: can act on.
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def accounted(self) -> bool:
        return self.considered == (
            self.extracted + self.paired + self.failed + self.skipped_no_ffmpeg
        )

    @property
    def ffmpeg_missing(self) -> bool:
        return self.skipped_no_ffmpeg > 0


def extract_for(
    videos: list[Photo],
    data_dir: Path,
    by_dir_name: dict[tuple[Path, str], MediaType],
    *,
    ffmpeg: bool | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Path], VideoFrameReport]:
    """Cache one still per standalone video. Returns {file_hash: frame path}.

    Resumable: a video whose frame is already cached is not re-extracted, so
    an interrupted pass continues. `ffmpeg` is an override for tests; by
    default it is probed once for the whole run rather than per video.
    """
    report = VideoFrameReport()
    available = have_ffmpeg() if ffmpeg is None else ffmpeg
    started = time.perf_counter()
    frames: dict[str, Path] = {}
    total = len(videos)
    for done, video in enumerate(videos, start=1):
        report.considered += 1
        if pairs_with_a_still(video, by_dir_name):
            # Checked BEFORE ffmpeg, so the count is the same with and
            # without it and the report means the same thing on both machines.
            report.paired += 1
            report.reasons[video.file_hash] = ERR_PAIRED
        elif (existing := cached_frame(data_dir, video.file_hash)) is not None:
            frames[video.file_hash] = existing
            report.extracted += 1
            report.already_cached += 1
        elif not available:
            report.skipped_no_ffmpeg += 1
            report.reasons[video.file_hash] = ERR_NO_FFMPEG
        else:
            source = next((p for p in video.paths if p.is_file()), None)
            got = (
                extract_frame(source, frame_path(data_dir, video.file_hash))
                if source is not None
                else None
            )
            if got is None:
                report.failed += 1
                report.reasons[video.file_hash] = ERR_NO_FRAME
            else:
                frames[video.file_hash] = got.path
                report.extracted += 1
                report.chosen.append((video.paths[0].name, got.at_fraction))
        if progress is not None:
            progress(done, total)
    report.elapsed_s = time.perf_counter() - started
    return frames, report
