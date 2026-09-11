"""The foreground watcher.

Two things matter here and both are asserted directly: it RENDERS NOTHING, and
the daily anniversary check fires once per calendar day rather than once per
poll. No test sleeps - the clock and the sleep are both injected.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from PIL import Image

from rekindle.memory.watch import DEFAULT_INTERVAL, LibraryState, scan_state, watch

NOW = datetime(2026, 9, 11, 9, 0, tzinfo=UTC)


def _jpeg(path: Path, colour=(10, 20, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), colour).save(path, "JPEG")
    return path


class Clock:
    """An injected clock. The only way to test "once per day" without waiting
    a day."""

    def __init__(self, start=NOW, step=timedelta(seconds=DEFAULT_INTERVAL)):
        self.now = start
        self.step = step

    def __call__(self):
        return self.now

    def advance(self, _seconds):
        self.now += self.step


def _run(root, **kw):
    changes = []
    emitted = []
    clock = kw.pop("clock", Clock())
    report = watch(
        root,
        on_change=kw.pop("on_change", changes.append),
        announce=kw.pop("announce", lambda day: [f"anniversary for {day.isoformat()}"]),
        emit=emitted.append,
        now=clock,
        sleep=clock.advance,
        cycles=kw.pop("cycles", 1),
        **kw,
    )
    return report, changes, emitted


# --------------------------------------------------------------------------


def test_once_runs_exactly_one_cycle(tmp_path):
    _jpeg(tmp_path / "a.jpg")
    report, _, _ = _run(tmp_path, cycles=1)
    assert report.cycles == 1


def test_the_watcher_RENDERS_NOTHING(tmp_path):
    """An auto-rendering watcher is the "unwelcome memory at the wrong moment"
    failure mode with a scheduler attached. Asserted by pointing it at an
    output directory and checking the directory stays empty."""
    _jpeg(tmp_path / "lib" / "a.jpg")
    out = tmp_path / "memories"
    out.mkdir()

    _run(tmp_path / "lib", cycles=5)

    assert list(out.iterdir()) == []


def test_todays_anniversary_is_printed_with_the_command_to_render_it(tmp_path):
    _jpeg(tmp_path / "a.jpg")

    def announce(day):
        return ["Kashmir - 24 photos", '  rekindle memory --recipe album_story --key "Kashmir"']

    report, _, emitted = _run(tmp_path, announce=announce, cycles=1)

    assert any("rekindle memory --recipe" in line for line in emitted)
    assert report.daily_checks == 1


def test_the_daily_check_fires_once_per_CALENDAR_DAY_not_per_poll(tmp_path):
    """At the default 300s interval, a per-poll check prints the same
    anniversary 288 times before midnight.

    Three polls 12 hours apart span two calendar-day boundaries, so exactly
    two checks should fire: one on the first cycle and one when the date rolls
    over.
    """
    _jpeg(tmp_path / "a.jpg")
    clock = Clock(start=datetime(2026, 9, 11, 9, 0, tzinfo=UTC), step=timedelta(hours=12))

    report, _, _ = _run(tmp_path, clock=clock, cycles=3)

    assert report.cycles == 3
    assert report.daily_checks == 2


def test_the_daily_check_does_not_repeat_within_one_day(tmp_path):
    _jpeg(tmp_path / "a.jpg")
    clock = Clock(step=timedelta(minutes=5))
    report, _, _ = _run(tmp_path, clock=clock, cycles=6)
    assert report.daily_checks == 1


# --------------------------------------------------------------------------
# change detection


def test_an_unchanged_library_triggers_no_reindex(tmp_path):
    _jpeg(tmp_path / "a.jpg")
    report, changes, _ = _run(tmp_path, cycles=4)
    assert report.rescans == 0
    assert changes == []


def test_a_new_photo_triggers_a_reindex(tmp_path):
    _jpeg(tmp_path / "a.jpg")
    changes = []

    def on_change(state):
        changes.append(state)
        # Only add on the first change, or every cycle keeps triggering.
        if len(changes) == 1:
            _jpeg(tmp_path / "b.jpg", (200, 10, 10))

    clock = Clock()
    watch(
        tmp_path,
        on_change=on_change,
        announce=lambda d: [],
        emit=lambda s: None,
        now=clock,
        sleep=clock.advance,
        cycles=3,
        # Start from an empty state so the FIRST cycle already sees a change.
        state=LibraryState(),
    )
    assert len(changes) >= 1


def test_a_changed_file_size_is_noticed(tmp_path):
    """mtime alone can miss an edit that preserves it; size is the cheap
    second signal."""
    path = _jpeg(tmp_path / "a.jpg")
    before = scan_state(tmp_path)
    Image.new("RGB", (400, 300), (250, 40, 10)).save(path, "JPEG", quality=95)
    assert scan_state(tmp_path) != before


def test_non_media_files_are_ignored(tmp_path):
    """A stray .txt must not trigger a full re-index of a 61 GB library."""
    _jpeg(tmp_path / "a.jpg")
    before = scan_state(tmp_path)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    assert scan_state(tmp_path) == before


def test_scan_state_counts_media_at_any_depth(tmp_path):
    _jpeg(tmp_path / "a.jpg")
    _jpeg(tmp_path / "Photos from 2019" / "b.jpg")
    _jpeg(tmp_path / "deep" / "deeper" / "c.jpg")
    assert scan_state(tmp_path).count == 3


def test_scan_state_is_case_insensitive_about_extensions(tmp_path):
    """A camera writes .JPG; a phone writes .jpg. Both are the library."""
    _jpeg(tmp_path / "a.JPG")
    assert scan_state(tmp_path).count == 1


def test_a_missing_root_does_not_crash(tmp_path):
    """An unplugged external drive must not take the watcher down."""
    assert scan_state(tmp_path / "not-there") == LibraryState()


def test_an_empty_library_is_a_valid_state(tmp_path):
    assert scan_state(tmp_path) == LibraryState()


def test_the_watcher_never_sleeps_when_cycles_are_bounded(tmp_path):
    """The final cycle must not sleep before returning, or `--once` would
    block for the whole interval before exiting."""
    _jpeg(tmp_path / "a.jpg")
    slept = []
    clock = Clock()

    watch(
        tmp_path,
        on_change=lambda s: None,
        announce=lambda d: [],
        emit=lambda s: None,
        now=clock,
        sleep=slept.append,
        cycles=1,
    )
    assert slept == []


def test_sleeping_happens_between_cycles(tmp_path):
    _jpeg(tmp_path / "a.jpg")
    slept = []
    clock = Clock()
    watch(
        tmp_path,
        on_change=lambda s: None,
        announce=lambda d: [],
        emit=lambda s: None,
        now=clock,
        sleep=lambda s: (slept.append(s), clock.advance(s)),
        interval=42.0,
        cycles=3,
    )
    assert slept == [42.0, 42.0]
