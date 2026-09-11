"""`rekindle music fetch`, the one command that uses the network.

**Nothing in this file touches the network.** Every test injects a fake
opener, and one test proves that the module cannot reach the network without
one by making `urllib` explode.

Fixture shapes come from the real metadata API, checked against
https://archive.org/metadata/musopen-chopin: 744 files of which 104 are
`"format": "VBR MP3"`, names carrying spaces and commas, `licenseurl` of
`http://creativecommons.org/publicdomain/zero/1.0/`.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rekindle.cli import app
from rekindle.fetch import (
    DEFAULT_ITEM,
    FetchError,
    Track,
    fetch_tracks,
    list_tracks,
)

runner = CliRunner()

AUDIO = b"ID3\x04\x00" + b"not really an mp3, but it hashes" * 4
AUDIO_SHA1 = hashlib.sha1(AUDIO).hexdigest()


def _metadata(files=None, licence="http://creativecommons.org/publicdomain/zero/1.0/"):
    return {
        "metadata": {
            "identifier": DEFAULT_ITEM,
            "title": "Musopen - Chopin",
            "licenseurl": licence,
        },
        "files": files
        if files is not None
        else [
            # Real shapes: a derived MP3, the .m4a original it came from, and
            # the scan/spectrogram clutter that shares the item.
            {
                "name": "Nocturne Op. 9, No. 2 in E Flat Major.mp3",
                "format": "VBR MP3",
                "sha1": AUDIO_SHA1,
                "size": "123",
                "title": "Nocturne",
                "length": "4:33",
            },
            {
                "name": "Allegro de Concert Op. 46 in A Major.mp3",
                "format": "VBR MP3",
                "sha1": AUDIO_SHA1,
                "size": "456",
            },
            {"name": "Nocturne Op. 9, No. 2 in E Flat Major.m4a", "format": "Apple Lossless Audio"},
            {"name": "cover.jpg", "format": "Item Tile"},
            {"name": "musopen-chopin_spectrogram.png", "format": "Spectrogram"},
        ],
    }


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _opener(routes, log=None):
    """A fake transport. `routes` maps URL -> bytes, or URL -> Exception."""

    def open_url(url):
        if log is not None:
            log.append(url)
        payload = routes.get(url)
        if payload is None:
            raise OSError(f"HTTP Error 404: {url}")
        if isinstance(payload, Exception):
            raise payload
        return _Response(payload)

    return open_url


META_URL = f"https://archive.org/metadata/{DEFAULT_ITEM}"


def _routes(metadata=None, audio=AUDIO):
    routes = {META_URL: json.dumps(metadata or _metadata()).encode()}
    for name in (
        "Nocturne%20Op.%209%2C%20No.%202%20in%20E%20Flat%20Major.mp3",
        "Allegro%20de%20Concert%20Op.%2046%20in%20A%20Major.mp3",
    ):
        routes[f"https://archive.org/download/{DEFAULT_ITEM}/{name}"] = audio
    return routes


# --------------------------------------------------------------------------
# nothing reaches the network by accident


def test_the_module_has_no_way_to_reach_the_network_without_being_asked(monkeypatch):
    """The default transport is `urllib.request.urlopen`. If any code path
    reached it without an injected opener, this test - and only this test -
    would notice, because every other test hands one in and would pass either
    way."""
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("the test suite attempted a real network call")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    with pytest.raises(AssertionError, match="attempted a real network call"):
        list_tracks()


def test_nothing_outside_the_fetch_module_imports_it_at_module_scope():
    """`music fetch` must never run implicitly - not on render, not on index,
    not on watch. The strongest cheap guard is that no module on any of those
    paths even imports the fetcher: the CLI pulls it in inside the command
    body, so an accidental call site would have to add an import first.
    """
    root = Path(__file__).resolve().parent.parent / "src" / "rekindle"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "fetch.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.startswith(("from rekindle.fetch", "import rekindle.fetch")):
                offenders.append(f"{path.relative_to(root)}:{number}")
    assert offenders == [], f"module-scope import of the fetcher: {offenders}"


# --------------------------------------------------------------------------
# resolving names through the API, never by pattern


def test_names_come_from_the_api_and_are_never_constructed():
    """A plausible-looking name is a 404: the item calls it `Nocturne Op. 9,
    No. 2 in E Flat Major.mp3`, with a comma, not `Nocturne Op. 9 No. 2.mp3`.
    Pinning or guessing names is the failure this resolves around."""
    _, tracks = list_tracks(opener=_opener(_routes()))
    assert [t.name for t in tracks] == [
        "Allegro de Concert Op. 46 in A Major.mp3",
        "Nocturne Op. 9, No. 2 in E Flat Major.mp3",
    ]


def test_no_filename_or_checksum_is_pinned_in_the_source():
    """The point of resolving at fetch time. A name or a hash written into
    this file is a promise about a third-party host forever, which is the
    whole reason the original "fetch on first run" design was declined.

    Checked over CODE, not prose: the module docstring names a filename as an
    example of what not to do, and banning the substring would forbid
    explaining the rule.
    """
    source = (Path(__file__).resolve().parent.parent / "src" / "rekindle" / "fetch.py").read_text(
        encoding="utf-8"
    )
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("#", '"""', "*"))
    )
    assert not re.search(r"""['"][^'"]*\.(mp3|m4a|ogg|flac)['"]""", code), (
        "a track filename is pinned in the source"
    )
    assert not re.search(r"\b[0-9a-f]{40}\b", code), "a checksum is pinned in the source"


def test_the_cli_default_item_matches_the_module():
    """The CLI repeats the identifier as a literal so that importing the
    fetcher is not on the `rekindle --version` path. Two copies of a constant
    drift; this is what stops them."""
    import rekindle.cli as cli_module

    assert cli_module._MUSIC_ITEM == DEFAULT_ITEM


def test_only_checksummed_mp3s_are_offered():
    """744 files in the real item, 104 of them MP3. The .m4a originals, the
    spectrograms and the scans are not music beds, and a file with no sha1 has
    nothing to verify against so it is not offered at all."""
    files = _metadata()["files"] + [
        {"name": "Unchecksummed.mp3", "format": "VBR MP3", "size": "9"},
    ]
    _, tracks = list_tracks(opener=_opener(_routes(_metadata(files))))
    assert all(t.name.endswith(".mp3") for t in tracks)
    assert "Unchecksummed.mp3" not in {t.name for t in tracks}
    assert len(tracks) == 2


def test_the_track_order_is_stable_whatever_the_api_returns():
    """`--count 5` has to mean the same five every time, or "skip what I
    already have" stops meaning anything. The API does not promise an order."""
    files = list(reversed(_metadata()["files"]))
    _, tracks = list_tracks(opener=_opener(_routes(_metadata(files))))
    assert [t.name for t in tracks] == sorted(t.name for t in tracks)


def test_an_unknown_item_is_an_error_not_an_empty_list():
    """archive.org returns `{}` with HTTP 200 for an item that does not
    exist. Treating that as "no tracks" would print a cheerful zero."""
    with pytest.raises(FetchError, match="no item called"):
        list_tracks(opener=_opener({META_URL: b"{}"}))


def test_an_unreachable_host_is_an_error_not_a_traceback():
    with pytest.raises(FetchError, match="could not reach"):
        list_tracks(opener=_opener({META_URL: OSError("Name or service not known")}))


def test_a_non_json_response_is_an_error_not_a_traceback():
    with pytest.raises(FetchError, match="did not return JSON"):
        list_tracks(opener=_opener({META_URL: b"<html>captive portal</html>"}))


# --------------------------------------------------------------------------
# the licence is read, not assumed


def test_a_cc0_item_is_recognised():
    source, _ = list_tracks(opener=_opener(_routes()))
    assert source.cc0
    assert source.page == f"https://archive.org/details/{DEFAULT_ITEM}"


@pytest.mark.parametrize(
    "licence",
    ["", "http://creativecommons.org/licenses/by-nc/4.0/", "all rights reserved"],
)
def test_an_item_that_is_not_cc0_is_not_cc0(licence):
    source, _ = list_tracks(opener=_opener(_routes(_metadata(licence=licence))))
    assert not source.cc0


# --------------------------------------------------------------------------
# downloading


def test_a_verified_download_lands_in_the_folder(tmp_path):
    _, tracks = list_tracks(opener=_opener(_routes()))
    report = fetch_tracks(tracks, tmp_path, opener=_opener(_routes()))

    assert report.downloaded == 2
    assert report.failed == 0
    assert (tmp_path / "Nocturne Op. 9, No. 2 in E Flat Major.mp3").read_bytes() == AUDIO


def test_a_corrupted_download_is_refused_and_leaves_nothing_behind(tmp_path):
    """The checksum's actual job. A truncated file left in `music/` would be
    handed to ffmpeg as a memory's soundtrack."""
    _, tracks = list_tracks(opener=_opener(_routes()))
    report = fetch_tracks(tracks, tmp_path, opener=_opener(_routes(audio=b"truncated")))

    assert report.downloaded == 0
    assert report.failed == 2
    assert "checksum mismatch" in report.errors[0]
    assert list(tmp_path.iterdir()) == []


def test_an_interrupted_download_leaves_no_partial_mp3(tmp_path):
    """Atomicity. The bytes go to a `.part` file that is renamed only after
    the checksum matches, so a dropped connection cannot leave a playable-
    looking stub where `resolve_music` will find it."""

    class _Dropping(_Response):
        def read(self, *args):
            raise OSError("connection reset by peer")

    routes = _routes()
    for url in list(routes):
        if url != META_URL:
            routes[url] = None

    def opener(url):
        if url == META_URL:
            return _Response(routes[META_URL])
        return _Dropping(b"")

    _, tracks = list_tracks(opener=_opener(_routes()))
    report = fetch_tracks(tracks, tmp_path, opener=opener)

    assert report.failed == 2
    assert list(tmp_path.iterdir()) == []


def test_a_track_already_present_and_correct_is_not_downloaded_again(tmp_path):
    """Resumability: re-running after an interruption costs the remainder,
    not the whole set."""
    _, tracks = list_tracks(opener=_opener(_routes()))
    (tmp_path / tracks[0].filename).write_bytes(AUDIO)

    log: list[str] = []
    report = fetch_tracks(tracks, tmp_path, opener=_opener(_routes(), log))

    assert report.skipped == 1
    assert report.downloaded == 1
    assert not any(tracks[0].filename.replace(" ", "%20") in url for url in log)


def test_a_track_present_but_WRONG_is_downloaded_again(tmp_path):
    """The other half of resumability. "The file exists" is not the question;
    "the file is the right file" is. A half-written MP3 from a previous run
    must not be mistaken for finished work."""
    _, tracks = list_tracks(opener=_opener(_routes()))
    (tmp_path / tracks[0].filename).write_bytes(b"half a file")

    report = fetch_tracks(tracks, tmp_path, opener=_opener(_routes()))
    assert report.skipped == 0
    assert report.downloaded == 2
    assert (tmp_path / tracks[0].filename).read_bytes() == AUDIO


def test_one_bad_track_does_not_stop_the_rest(tmp_path):
    routes = _routes()
    bad = f"https://archive.org/download/{DEFAULT_ITEM}/Allegro%20de%20Concert%20Op.%2046%20in%20A%20Major.mp3"
    routes[bad] = OSError("HTTP Error 503")

    _, tracks = list_tracks(opener=_opener(_routes()))
    report = fetch_tracks(tracks, tmp_path, opener=_opener(routes))

    assert report.downloaded == 1
    assert report.failed == 1
    assert report.accounted == 2


def test_a_track_name_cannot_escape_the_destination_folder(tmp_path):
    """An archive item is a third party's namespace. A name with a directory
    component must land in the destination or nowhere."""
    evil = Track(name="../../../escaped.mp3", sha1=AUDIO_SHA1, size=len(AUDIO))
    routes = {evil.url(DEFAULT_ITEM): AUDIO}
    dest = tmp_path / "music"
    fetch_tracks([evil], dest, opener=_opener(routes))

    assert (dest / "escaped.mp3").is_file()
    assert not (tmp_path.parent / "escaped.mp3").exists()


def test_the_download_url_escapes_spaces_and_commas():
    track = Track(name="Nocturne Op. 9, No. 2 in E Flat Major.mp3", sha1="x", size=1)
    url = track.url(DEFAULT_ITEM)
    assert " " not in url
    assert "%20" in url and "%2C" in url


# --------------------------------------------------------------------------
# the command


def _cli(monkeypatch, routes, argv):
    import rekindle.fetch as fetch_module

    monkeypatch.setattr(fetch_module, "_default_opener", _opener(routes))
    return runner.invoke(app, argv)


def test_the_command_prints_the_licence_and_the_source(monkeypatch, tmp_path):
    result = _cli(
        monkeypatch,
        _routes(),
        ["music", "fetch", "--dest", str(tmp_path), "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert "CC0 1.0" in result.output
    assert f"archive.org/details/{DEFAULT_ITEM}" in result.output


def test_the_command_refuses_an_item_that_does_not_declare_cc0(monkeypatch, tmp_path):
    """Checked before a byte is downloaded, and refused rather than warned
    about. An item can be relicensed after this code is written - which is the
    entire argument against pinning a list."""
    result = _cli(
        monkeypatch,
        _routes(_metadata(licence="http://creativecommons.org/licenses/by-nc/4.0/")),
        ["music", "fetch", "--dest", str(tmp_path), "--yes"],
    )
    assert result.exit_code == 2
    assert "does not declare CC0" in result.output
    assert list(tmp_path.iterdir()) == []


def test_the_command_asks_before_downloading(monkeypatch, tmp_path):
    """ "Deliberately, never implicitly" is the whole design. Answering no must
    download nothing."""
    import rekindle.fetch as fetch_module

    monkeypatch.setattr(fetch_module, "_default_opener", _opener(_routes()))
    result = runner.invoke(app, ["music", "fetch", "--dest", str(tmp_path)], input="n\n")

    assert result.exit_code == 0
    assert "Nothing downloaded" in result.output
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_the_command_honours_count(monkeypatch, tmp_path):
    result = _cli(
        monkeypatch,
        _routes(),
        ["music", "fetch", "--dest", str(tmp_path), "--count", "1", "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert len(list(tmp_path.iterdir())) == 1


def test_the_command_exits_nonzero_when_a_track_fails(monkeypatch, tmp_path):
    result = _cli(
        monkeypatch,
        _routes(audio=b"truncated"),
        ["music", "fetch", "--dest", str(tmp_path), "--yes"],
    )
    assert result.exit_code == 1
    assert "checksum mismatch" in result.output


def test_the_command_reports_an_unreachable_host_without_a_traceback(monkeypatch, tmp_path):
    result = _cli(
        monkeypatch,
        {META_URL: OSError("Name or service not known")},
        ["music", "fetch", "--dest", str(tmp_path), "--yes"],
    )
    assert result.exit_code == 2
    assert "could not reach" in result.output
    assert "Traceback" not in result.output
