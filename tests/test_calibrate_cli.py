"""`rekindle calibrate` and `rekindle config`, driven the way a person drives
them: through typer, with typed answers.

Every assertion here is about what the terminal SAYS, because that is the whole
product for anyone on a machine with no browser. A guided sequence whose
questions are unreadable, or whose safeguard prints nothing, is not a guided
sequence.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rekindle import config
from rekindle.calibrate import session as calsession
from rekindle.calibrate import state
from rekindle.cli import app
from rekindle.db import PhotoStore
from rekindle.models import MediaType, Photo, PhotoMeta

runner = CliRunner()
T0 = datetime(2025, 8, 4, 10, 0, tzinfo=UTC)


def _hist(dominant: int = 0) -> str:
    bins = [0] * 64
    bins[dominant % 64] = 255
    return "".join(f"{b:02x}" for b in bins)


def _photo(name: str, *, i: int) -> Photo:
    return Photo(
        file_hash=name,
        paths=[Path(f"/lib/{name}.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=T0 + timedelta(hours=i),
            taken_at_local=(T0 + timedelta(hours=i)).replace(tzinfo=None),
            sharpness=0.05 + i / 40.0,
            brightness=40.0 + i * 4,
            phash=i << 7,
            width=4000,
            height=3000,
            colour=_hist(i),
        ),
        first_seen=T0,
        last_seen=T0,
    )


@pytest.fixture
def library(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with PhotoStore(data_dir / "rekindle.sqlite") as store:
        store.upsert_many([_photo(f"p{i:02d}", i=i) for i in range(30)])
    yield data_dir
    # `open_index` installs the library's config as the process-wide active
    # one. Put the defaults back, or a written rekindle.toml leaks into the
    # next test in this file.
    config.activate(config.defaults())


def run(*args, stdin: str = ""):
    return runner.invoke(app, list(args), input=stdin)


# ---------------------------------------------------------------------- config


def test_config_list_shows_every_setting_and_says_nothing_is_overridden(library):
    got = run("config", "list", "--data-dir", str(library))
    assert got.exit_code == 0, got.output
    assert "min_sharpness" in got.output
    assert "Nothing overridden" in got.output


def test_config_explain_prints_the_measurement_behind_the_default(library):
    got = run("config", "explain", "composition.min_sharpness", "--data-dir", str(library))
    assert got.exit_code == 0, got.output
    assert "1,149 photos" in got.output, "the provenance is the point of the command"
    assert "Raising it" in got.output and "Lowering it" in got.output


def test_config_explain_warns_on_the_one_setting_that_can_fail_open(library):
    got = run("config", "explain", "faces.gate_threshold", "--data-dir", str(library))
    assert "weakens a safety gate" in got.output
    assert "separate confirmation" in got.output


def test_config_explain_suggests_a_name_rather_than_a_traceback(library):
    got = run("config", "explain", "sharpness", "--data-dir", str(library))
    assert got.exit_code == 2
    assert "composition.min_sharpness" in got.output


def test_config_diff_reads_the_users_own_file(library):
    (library / config.CONFIG_NAME).write_text(
        "[composition]\nmin_sharpness = 0.3\n", encoding="utf-8"
    )
    got = run("config", "diff", "--data-dir", str(library))
    assert "composition.min_sharpness: 0.12 -> 0.3" in got.output
    assert "higher" in got.output


def test_a_malformed_override_stops_the_command_rather_than_running_on_defaults(library):
    (library / config.CONFIG_NAME).write_text("[composition\n", encoding="utf-8")
    got = run("config", "list", "--data-dir", str(library))
    assert got.exit_code == 2


# ------------------------------------------------------------------- calibrate


def test_calibrate_status_on_a_fresh_library_says_never(library):
    got = run("calibrate", "--status", "--data-dir", str(library))
    assert got.exit_code == 0, got.output
    assert "Never calibrated" in got.output


def test_the_sequence_shows_its_position_and_a_question_in_plain_words(library):
    got = run("calibrate", "--data-dir", str(library), stdin="q\n")
    assert "of " in got.output
    assert "Blur" in got.output
    assert "hash" not in got.output.lower(), "no jargon reaches the first screen"


def test_quitting_saves_the_answers_and_says_so(library):
    got = run("calibrate", "--data-dir", str(library), stdin="p\ny\nq\n")
    assert "Your answers are saved" in got.output
    record = state.load(library)
    assert record.answers, "the judgement must be on disk"
    assert record.in_progress


def test_a_stopped_sitting_is_resumed_rather_than_restarted(library):
    run("calibrate", "--data-dir", str(library), stdin="p\ny\nn\nq\n")
    got = run("calibrate", "--data-dir", str(library), stdin="q\n")
    assert "part-way through" in got.output


def test_accepting_every_default_finishes_and_writes_nothing(library):
    # Every step on this library, each answered with "show me the default"
    # then "accept". `d` means the same thing on every screen, which is the
    # whole reason the modes are lettered rather than numbered.
    got = run("calibrate", "--data-dir", str(library), stdin="d\na\n" * 30)
    assert "kept every one of rekindle's defaults" in got.output
    assert not (library / config.CONFIG_NAME).exists()
    assert state.load(library).finished


def test_finishing_is_a_real_state_and_it_stops_offering(library):
    run("calibrate", "--data-dir", str(library), stdin="d\na\n" * 30)
    got = run("calibrate", "--status", "--data-dir", str(library))
    assert "Calibrated" in got.output
    assert "Never calibrated" not in got.output


def test_a_chosen_value_is_shown_with_its_consequence_before_it_is_written(library):
    got = run(
        "calibrate",
        "--data-dir",
        str(library),
        "--only",
        "composition.min_sharpness",
        stdin="d\nn\n0.4\ny\n",
    )
    assert "photographs" in got.output
    assert "would now be dropped" in got.output or "drops" in got.output
    assert config.read_override(library / config.CONFIG_NAME) == {"composition.min_sharpness": 0.4}


def test_refusing_to_write_leaves_the_file_absent(library):
    run(
        "calibrate",
        "--data-dir",
        str(library),
        "--only",
        "composition.min_sharpness",
        stdin="d\nn\n0.4\nn\n",
    )
    assert not (library / config.CONFIG_NAME).exists()
    assert (
        "Nothing written"
        in runner.invoke(
            app,
            ["calibrate", "--data-dir", str(library), "--only", "composition.min_sharpness"],
            input="d\nn\n0.4\nn\n",
        ).output
    )


def test_only_an_unknown_threshold_exits_rather_than_guessing(library):
    got = run("calibrate", "--data-dir", str(library), "--only", "nonesuch")
    assert got.exit_code == 2
    assert "No threshold called" in got.output


def test_a_threshold_this_library_cannot_answer_says_why(library):
    """No embeddings here, so the CLIP sameness step cannot run. It says so
    and names the command, rather than pretending it does not exist."""
    got = run("calibrate", "--data-dir", str(library), "--only", "dedup.cosine")
    assert got.exit_code == 3
    assert "cannot be calibrated on this library" in got.output
    assert "rekindle" in got.output


# -------------------------------------- the safeguard, from the terminal


def test_widening_the_publishing_gate_prints_the_sentence_and_refuses_a_typo(library, monkeypatch):
    """The gate step needs a detector, so the availability is forced. What is
    being tested is the terminal's half of the safeguard - that the meaning is
    printed and a wrong answer leaves the value alone."""
    from rekindle.calibrate import cli as calcli
    from rekindle.calibrate import plan

    monkeypatch.setattr(
        calcli.session,
        "detect",
        lambda photos, **kw: plan.Availability(fingerprints=True, faces=True),
    )
    got = run(
        "calibrate",
        "--data-dir",
        str(library),
        "--only",
        "faces.gate_threshold",
        stdin="d\nn\n0.4\nyes please\n",
    )
    assert "widens the publishing gate" in got.output
    assert "other people" in got.output
    assert calsession.LOOSEN_PHRASE in got.output
    assert "Left alone" in got.output
    assert not (library / config.CONFIG_NAME).exists()


def test_the_exact_sentence_widens_it_and_says_that_it_did(library, monkeypatch):
    from rekindle.calibrate import cli as calcli
    from rekindle.calibrate import plan

    monkeypatch.setattr(
        calcli.session,
        "detect",
        lambda photos, **kw: plan.Availability(fingerprints=True, faces=True),
    )
    got = run(
        "calibrate",
        "--data-dir",
        str(library),
        "--only",
        "faces.gate_threshold",
        stdin=f"d\nn\n0.4\n{calsession.LOOSEN_PHRASE}\ny\n",
    )
    assert "Widened, on your explicit confirmation" in got.output
    assert config.read_override(library / config.CONFIG_NAME) == {"faces.gate_threshold": 0.4}


def test_tightening_the_publishing_gate_asks_nothing(library, monkeypatch):
    from rekindle.calibrate import cli as calcli
    from rekindle.calibrate import plan

    monkeypatch.setattr(
        calcli.session,
        "detect",
        lambda photos, **kw: plan.Availability(fingerprints=True, faces=True),
    )
    got = run(
        "calibrate",
        "--data-dir",
        str(library),
        "--only",
        "faces.gate_threshold",
        stdin="d\nn\n0.05\ny\n",
    )
    assert "widens the publishing gate" not in got.output
    assert config.read_override(library / config.CONFIG_NAME) == {"faces.gate_threshold": 0.05}


# ---------------------------------------------------------- the exit question


def test_the_exit_question_runs_and_reports_when_there_are_memories(library, tmp_path):
    from rekindle.calibrate import impact
    from rekindle.memory.spec import FactSheet, MemorySpec, Shot

    memories = tmp_path / "memories" / "album_story-x"
    memories.mkdir(parents=True)
    spec = MemorySpec(
        recipe="album_story",
        key="x",
        title="x",
        subtitle="",
        public_safe=False,
        shots=(Shot("p00", "", None, False),),
        facts=FactSheet(title="x", recipe="album_story", photo_count=1),
    )
    (memories / impact.SPEC_NAME).write_text(spec.dumps(), encoding="utf-8")

    got = run(
        "calibrate",
        "--data-dir",
        str(library),
        "--memories",
        str(tmp_path / "memories"),
        "--only",
        "composition.min_sharpness",
        stdin="d\nn\n0.4\ny\n",
    )
    assert "Re-running selection" in got.output
    assert "Your changes affect" in got.output


def test_no_memories_yet_says_so_rather_than_offering_a_rebuild(library, tmp_path):
    got = run(
        "calibrate",
        "--data-dir",
        str(library),
        "--memories",
        str(tmp_path / "nothing"),
        "--only",
        "composition.min_sharpness",
        stdin="d\nn\n0.4\ny\n",
    )
    assert "No memories built yet" in got.output
