"""Guardrails. What may never reach a memory, and the public-safe rule.

A memory system that surfaces the wrong photo at the wrong moment is worse
than no memory system. Everything in this module is a refusal.

The rules are enforced ONCE, in `MemoryIndex.open`, which is the only way a
recipe can obtain a photo at all. This module owns the predicate; `index.py`
owns the chokepoint that applies it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from rekindle.models import MediaType, Photo

CONFIG_NAME = "exclusions.toml"

# Deny reasons, in the order they are tested. The FIRST match is what gets
# reported, so the order is part of the contract.
DENY_ARCHIVED = "archived"
DENY_TRASHED = "trashed"
DENY_UNKNOWN_MEDIA = "unknown_media"
DENY_NO_DATE = "no_date"
DENY_PERSON = "excluded_person"
DENY_DATE_RANGE = "excluded_date"
DENY_ALBUM = "excluded_album"
DENY_PATH = "excluded_path"
DENY_NOT_PUBLIC_SAFE = "not_public_safe"


class PolicyError(RuntimeError):
    """The exclusions file exists but could not be read.

    Deliberately fatal. Continuing with an unparsed exclusion list would
    surface exactly the photos the user asked never to see, and a warning
    scrolls past. This is the one place in the codebase where refusing to run
    is the safe option.
    """


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def contains(self, moment: datetime) -> bool:
        """Inclusive at BOTH ends.

        A user excluding "2019-03-01 to 2019-09-30" means both those days too.
        An exclusive end silently admits the last day of a period someone
        asked never to see again.
        """
        return self.start <= moment.date() <= self.end


@dataclass(frozen=True)
class ExclusionPolicy:
    """Everything the user has asked never to see, plus the public-safe rule.

    Frozen: a policy is decided once, at startup, and a recipe that could
    mutate it mid-run would make the guarantee untestable.
    """

    people: frozenset[str] = frozenset()
    albums: frozenset[str] = frozenset()
    paths: tuple[Path, ...] = ()
    dates: tuple[DateRange, ...] = ()
    public_safe_allow: frozenset[str] = frozenset()
    public_safe_only: bool = False
    album_aliases: dict[str, str] = field(default_factory=dict)

    def with_public_safe(self, enabled: bool) -> ExclusionPolicy:
        """`--public-safe` is a CLI flag, not a config value, so it is applied
        to a loaded policy rather than parsed into one."""
        if enabled == self.public_safe_only:
            return self
        return ExclusionPolicy(
            people=self.people,
            albums=self.albums,
            paths=self.paths,
            dates=self.dates,
            public_safe_allow=self.public_safe_allow,
            public_safe_only=enabled,
            album_aliases=self.album_aliases,
        )

    def deny_reason(self, photo: Photo) -> str | None:
        """None means allowed. Otherwise the FIRST rule that refused it.

        Order is the contract: `archived` is checked before anything a user
        can configure, so no configuration can re-admit it.
        """
        meta = photo.meta

        # 1-2. Google's own flags. NOT CONFIGURABLE, deliberately. The user
        # archived these to hide them; an escape hatch here is a foot-gun, not
        # a feature. 162 rows on the reference library.
        if meta.archived:
            return DENY_ARCHIVED
        if meta.trashed:
            return DENY_TRASHED

        # 3. Nothing can render an unidentifiable file.
        if photo.media_type is MediaType.UNKNOWN:
            return DENY_UNKNOWN_MEDIA

        # 4. Every recipe is temporal. 0 rows today, but the predicate must
        # not assume the next library is as clean as this one.
        if meta.taken_at_local is None or meta.taken_at_utc is None:
            return DENY_NO_DATE

        # 5. ANY excluded person, not all. A photo containing someone the user
        # asked never to see is excluded even if five welcome people are also
        # in it - the opposite reading would surface them constantly.
        if self.people and any(p in self.people for p in meta.people):
            return DENY_PERSON

        # 6. LOCAL time. A user excluding a date means the date they lived
        # through, not a UTC instant that may fall on the day before.
        if any(r.contains(meta.taken_at_local) for r in self.dates):
            return DENY_DATE_RANGE

        if self.albums and any(a.casefold() in self.albums for a in photo.albums):
            return DENY_ALBUM

        if self.paths and any(_under(path, self.paths) for path in photo.paths):
            return DENY_PATH

        if self.public_safe_only and not is_public_safe(photo, self.public_safe_allow):
            return DENY_NOT_PUBLIC_SAFE

        return None


def _under(path: Path, roots: tuple[Path, ...]) -> bool:
    """Is `path` inside one of `roots`?

    `Path.is_relative_to`, not a string prefix compare: `/a/photos2/x.jpg`
    starts with the string `/a/photos` and is NOT inside it.

    No try/except. An earlier version wrapped this in
    `except (OSError, ValueError)` for "mismatched drives on Windows", which
    is what `Path.relative_to` does - but `is_relative_to` is total on every
    Python this project supports: measured on 3.12,
    `Path("D:/photos/a.jpg").is_relative_to(Path("C:/private"))` returns
    False rather than raising. The handler was therefore unreachable, and a
    mutation that deleted it could not fail. An untestable guard is a line
    nobody can maintain, so it is gone rather than left as decoration.
    """
    return any(path == root or path.is_relative_to(root) for root in roots)


def is_public_safe(photo: Photo, allow: frozenset[str]) -> bool:
    """May this photo be published to a public repo?

    The rule the user specified: only photos with NO OTHER PERSON in them.
    Implemented as "the face-tag set is non-empty AND a subset of the
    allow-list". Both halves matter, and default deny is the point.

    THE NON-EMPTY HALF IS THE IMPORTANT ONE. Face tags come from Google and
    cover 55.9% of this library: 8,433 live photos (43.7%) carry no tag at
    all, and any of them may contain anyone. An untagged photo is therefore
    NOT public-safe. Treating "no tags" as "no people" is the single most
    dangerous widening available in this codebase - it is how a stranger's
    face reaches a public GitHub repo - and it must not be made to get more
    memories published.

    A subset, never an intersection: a photo of Abhik AND Paramita is not
    publishable under an allow-list of {"Abhik Maiti"}.
    """
    people = {p for p in photo.meta.people if p}
    return bool(people) and people <= allow


def load_policy(path: Path) -> ExclusionPolicy:
    """Read the exclusions file. Absent is fine; malformed is fatal."""
    if not path.is_file():
        return ExclusionPolicy()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as exc:
        raise PolicyError(f"{path} could not be read: {exc}") from exc

    try:
        dates = tuple(
            DateRange(start=_as_date(d["from"]), end=_as_date(d["to"]))
            for d in raw.get("dates", [])
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise PolicyError(
            f"{path}: every [[dates]] entry needs `from` and `to` as YYYY-MM-DD ({exc})"
        ) from exc

    for range_ in dates:
        if range_.start > range_.end:
            raise PolicyError(f"{path}: date range {range_.start} to {range_.end} runs backwards")

    return ExclusionPolicy(
        people=frozenset(_as_list(raw, path, "people")),
        # Casefolded on the way in so the comparison does not have to remember.
        albums=frozenset(a.casefold() for a in _as_list(raw, path, "albums")),
        paths=tuple(Path(p) for p in _as_list(raw, path, "paths")),
        dates=dates,
        public_safe_allow=frozenset(_as_list(raw, path, "public_safe_allow")),
        album_aliases={str(k): str(v) for k, v in raw.get("album_aliases", {}).items()},
    )


def _as_list(raw: dict, path: Path, key: str) -> list[str]:
    value = raw.get(key, [])
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise PolicyError(f"{path}: `{key}` must be a list of strings")
    return value


def _as_date(value: object) -> date:
    """TOML gives a `date` for a bare YYYY-MM-DD and a `str` if it was quoted.

    Both are accepted, because quoting it is the more natural thing to write
    and rejecting it would be a baffling error about a file that looks right.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def append_exclusion(
    path: Path,
    *,
    person: str | None = None,
    album: str | None = None,
    dir_path: Path | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> None:
    """Add one exclusion, preserving everything already in the file.

    Appends TOML text rather than re-serialising a parsed document: the
    standard library can read TOML but not write it, and a hand-rolled writer
    would silently drop comments and any key it did not know about. The file
    is re-parsed afterwards so a malformed result is caught here rather than
    on the user's next run.
    """
    lines: list[str] = []
    if person:
        lines.append(f"people = [{_toml_str(person)}]")
    if album:
        lines.append(f"albums = [{_toml_str(album)}]")
    if dir_path:
        lines.append(f"paths = [{_toml_str(str(dir_path))}]")
    if date_from and date_to:
        if date_from > date_to:
            raise PolicyError(f"date range {date_from} to {date_to} runs backwards")
        lines.append(f"[[dates]]\nfrom = {date_from.isoformat()}\nto = {date_to.isoformat()}")
    if not lines:
        raise PolicyError("nothing to exclude")

    # TOML forbids a duplicate key, so appending a second `people = [...]`
    # produces a file that no longer parses. Merge scalar-list keys into what
    # is already there instead; [[dates]] is an array of tables and appends
    # cleanly, which is why it is handled separately above.
    existing = load_policy(path) if path.is_file() else ExclusionPolicy()
    merged: list[str] = []
    for key, added, current in (
        ("people", person, sorted(existing.people)),
        ("albums", album, sorted(existing.albums)),
        ("paths", str(dir_path) if dir_path else None, [str(p) for p in existing.paths]),
    ):
        if added is None:
            continue
        values = sorted({*current, added})
        merged.append(f"{key} = [{', '.join(_toml_str(v) for v in values)}]")

    body = _rewrite_keys(path, merged)
    if date_from and date_to:
        body += f"\n[[dates]]\nfrom = {date_from.isoformat()}\nto = {date_to.isoformat()}\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    load_policy(path)


def _rewrite_keys(path: Path, replacements: list[str]) -> str:
    """Replace whole-line `key = [...]` assignments, keeping everything else.

    Comments and unknown keys survive, which is the whole point of not
    re-serialising.
    """
    keys = {line.split("=", 1)[0].strip() for line in replacements}
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        if line.split("=", 1)[0].strip() in keys and "=" in line:
            continue
        kept.append(line)
    text = "\n".join(kept).rstrip()
    # A new key must go ABOVE any [[dates]] table: in TOML every key after a
    # table header belongs to that table, so appending `people = [...]` at the
    # end of a file that has a [[dates]] entry silently makes it dates.people.
    head, sep, tail = text.partition("\n[[dates]]")
    body = "\n".join([head.rstrip(), *replacements]).strip()
    return f"{body}\n{sep}{tail}\n".replace("\n\n\n", "\n\n")


def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
