"""`rekindle watch` - a FOREGROUND poller.

Not a service, not a daemon, and it renders nothing. It watches the library,
re-runs index+enrich when files change, and once a day tells you what today's
anniversary is - with the exact command to render it.

**It prints a command and never runs it.** An auto-rendering watcher is the
"unwelcome memory at the wrong moment" failure mode with a scheduler attached,
and unprompted memories are where the real risk lives. The user asks; rekindle
answers.

Everything time-dependent is injected - the clock, the sleep, the number of
cycles - so the tests run instantly and none of them sleeps.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

DEFAULT_INTERVAL = 300.0

# Extensions worth waking up for. Matches the source layer's view of the
# library rather than counting every file, so a stray .txt does not trigger a
# re-index.
WATCHED_SUFFIXES = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
        ".heic",
        ".heif",
        ".dng",
        ".cr2",
        ".cr3",
        ".nef",
        ".arw",
        ".orf",
        ".rw2",
        ".raf",
        ".raw",
        ".mp4",
        ".mov",
        ".m4v",
        ".avi",
        ".mkv",
        ".wmv",
        ".webm",
        ".3gp",
        ".mts",
        ".m2ts",
        ".mpg",
        ".mpeg",
        ".mp",
    }
)


@dataclass(frozen=True)
class LibraryState:
    """A cheap fingerprint of the library: how many media files, and the
    newest mtime among them.

    Deliberately NOT a content hash. The point is to notice that something
    changed, at a cost proportional to a directory walk rather than to 61 GB
    of reads; a false negative costs one missed poll, and the next one catches
    it.
    """

    count: int = 0
    newest: float = 0.0
    total_size: int = 0


def scan_state(root: Path) -> LibraryState:
    count = 0
    newest = 0.0
    total = 0
    try:
        entries: Iterable[Path] = root.rglob("*")
    except OSError:
        return LibraryState()
    for path in entries:
        try:
            if path.suffix.lower() not in WATCHED_SUFFIXES or not path.is_file():
                continue
            stat = path.stat()
        except OSError:
            # A file that vanished mid-walk, or a permission error. Skipping
            # it is right: the next poll sees the settled state.
            continue
        count += 1
        total += stat.st_size
        newest = max(newest, stat.st_mtime)
    return LibraryState(count=count, newest=newest, total_size=total)


@dataclass
class WatchReport:
    cycles: int = 0
    rescans: int = 0
    daily_checks: int = 0
    announced: list[str] = field(default_factory=list)


def watch(
    root: Path,
    *,
    on_change: Callable[[LibraryState], None],
    announce: Callable[[date], list[str]],
    emit: Callable[[str], None],
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] | None = None,
    interval: float = DEFAULT_INTERVAL,
    cycles: int | None = None,
    state: LibraryState | None = None,
) -> WatchReport:
    """Poll until `cycles` is exhausted (None means forever).

    `announce` returns the lines to print for a given date and is what makes
    this testable without a database: the caller wires it to the engine, and a
    test wires it to a list.
    """
    report = WatchReport()
    last_state = state if state is not None else scan_state(root)
    last_day: date | None = None

    while cycles is None or report.cycles < cycles:
        report.cycles += 1
        today = now().date()

        # Once per CALENDAR DAY, not once per poll. At the default 300s
        # interval a per-poll check would print the same anniversary 288
        # times before midnight.
        if last_day != today:
            last_day = today
            report.daily_checks += 1
            for line in announce(today):
                report.announced.append(line)
                emit(line)

        current = scan_state(root)
        if current != last_state:
            report.rescans += 1
            last_state = current
            on_change(current)

        if cycles is not None and report.cycles >= cycles:
            break
        if sleep is not None:
            sleep(interval)
        else:  # pragma: no cover - only reached in a real, unbounded run
            import time

            time.sleep(interval)
    return report
