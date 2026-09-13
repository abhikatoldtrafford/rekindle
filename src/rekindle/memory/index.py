"""The chokepoint. The only way a recipe can obtain a photo.

> "Exclusions must be configurable and honoured everywhere: one chokepoint
> that every recipe goes through - not a filter each recipe remembers to
> apply. If a recipe can bypass the guardrail, the guardrail does not exist."

So the guardrail is not a filter. Every `Photo` this class hands out has come
through `_StoreSource`, which applies `ExclusionPolicy.deny_reason` to every
row it turns into a `Photo` - on the opening scan and on every later fetch. A
recipe cannot ask for a photo the policy rejected because no code path exists
that would produce one.

A recipe is handed a `MemoryIndex` and nothing else - no `PhotoStore`, no
path, no connection. Bypassing the guardrail requires importing `PhotoStore`
yourself, which is exactly the reviewable act it should be. `tests/
test_memory_index.py` enumerates every public method and asserts none of them
leaks, so a query added later without filtering fails automatically.

WHY THE LIBRARY IS NOT MATERIALISED
-----------------------------------
It used to be. `MemoryIndex.open` loaded every row into a tuple of `Photo`
objects and every index pointed at those objects, which is correct and fast at
19,480 rows and untenable at 300,000: measured on a synthetic 300k library,
the photos alone are **681 MB** at 2,289 bytes apiece, while the eight derived
indexes over them are 32.5 MB. The photos are the problem, not the indexing.

So the indexes hold SQLite **rowids** - eight bytes each, in `array("q")` -
and the `Photo` objects are fetched back only when a caller actually reads
one, through a bounded LRU cache. The interface is unchanged: every query
still returns `list[Photo]`, and the single enforcement point is still single.

THE COST, STATED
----------------
Recipes scan the index several times, so this is a trade and not a free win,
and the trade is not the one it looks like from here.

A library that fits in the cache pays nothing: the first pass fills it and
every later pass is a dict lookup. Measured on the reference library, a full
`--all-recipes` pass is 7% FASTER than it was, because nothing survives
`open` for the garbage collector to trace.

A library LARGER than the cache re-reads from SQLite, and that bill lands
almost entirely on `all_offers`, which asks every recipe for every slice of
the library and then wants a count and a set of years out of it. So the index
answers both off the spine - `year_counts`, `month_day_years`, `person_years`
and their siblings, and `image_day_counts` for `recurring_event` - and six of
the nine recipes stopped loading photographs to build an offer. That took the
phase from 185.9 s to 46.5 s on a synthetic 300,000-photo library, against
4.1 s materialised. Two more followed that measurement, by the same route:
`album_span` for `album_story`'s subtitle, `image_album_years` for the album
evidence `recurring_event` titles itself from. What is left is `then_and_now`,
which must know which photographs would survive `compose()`, and
`place_cluster`'s visit split.

Net at 300k: `open` 39% faster, the whole `--all-recipes` pass 1.64x slower,
peak memory 1.6x smaller and the index itself 15.7x smaller. It is volume and
not locality: generating the library in capture order rather than at random -
the shape a real Takeout index has - buys 7%.

Every number here is in `docs/decision-log-memory-scale.md`, with the caveats
that belong to it; `tests/bench_memory_index.py` re-measures them.

`iter_all` and `iter_images` exist for the callers that only want to *count*
something across the library. Materialising 300,000 photos to count how many
lack a fingerprint would undo the whole of the above.
"""

from __future__ import annotations

import itertools
import sqlite3
import threading
from array import array
from bisect import bisect_left
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rekindle.db import PhotoStore, row_to_photo
from rekindle.memory import albums
from rekindle.memory.policy import ExclusionPolicy, is_public_safe
from rekindle.memory.videoframe import frames_dir
from rekindle.models import MediaType, Photo

# Coarse enough that one town is one cell, fine enough that two towns are not.
# The reference library's 2,318 GPS photos fall into 19 cells at this size.
GPS_CELL = 0.25

#: How many hydrated `Photo` objects the index keeps alive at once.
#:
#: 60,000 is roughly 138 MB at the 2,289 bytes a photo measured on the
#: synthetic 300k library. It is chosen so that every library up to 60,000
#: photos behaves EXACTLY as the materialised index did - one pass to fill the
#: cache, then no SQLite at all - because a change that made ordinary
#: libraries slower to make a large one possible would be a bad trade made
#: silently. Above 60,000 the cache becomes a working set and the index starts
#: re-reading; that is the regime the laziness exists for.
DEFAULT_CACHE = 60_000

#: Rows per `WHERE rowid IN (...)`. SQLITE_MAX_VARIABLE_NUMBER is 999 on
#: builds older than 3.32 and 32,766 after; 900 is under the floor and the
#: per-statement overhead at that size is already noise.
_FETCH_CHUNK = 900

#: How many photos `iter_all` holds at once. Small enough that streaming the
#: whole library costs a few megabytes, large enough that the fetch is bulk.
_STREAM_CHUNK = 2_000

#: `array` typecode for rowids. Signed 64-bit, matching SQLite's own INTEGER,
#: so no library can outgrow it. Eight bytes an entry - the same as the
#: pointer a list of `Photo` would have held, with none of the object behind
#: it.
_ROWID = "q"


@dataclass
class ExclusionReport:
    """What was withheld, counted by FIRST matching reason.

    Counts only - never the photos themselves, and never their paths. This is
    what `rekindle memories` prints, and a report that named the excluded
    photos would defeat the exclusion.
    """

    total: int = 0
    allowed: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)
    #: Of the ALLOWED photos, how many the face detector has never looked at.
    #:
    #: Only meaningful under `--public-safe`, and only there because that gate
    #: ABSTAINS on an unexamined photograph rather than refusing it. A gate
    #: that quietly abstains on most of what it approves has a measured
    #: false-pass rate that does not describe what it just approved, and the
    #: user cannot tell from the output. So it is counted and printed.
    unverified: int = 0

    @property
    def excluded(self) -> int:
        return self.total - self.allowed

    @property
    def accounted(self) -> bool:
        return self.excluded == sum(self.by_reason.values())


def _ids() -> array:
    """A fresh, empty rowid array. The `defaultdict` factory for every index."""
    return array(_ROWID)


def gps_cell(lat: float, lon: float, size: float = GPS_CELL) -> tuple[float, float]:
    """Snap a coordinate to a grid cell.

    Floor division, not `round`: rounding puts the boundary in the middle of a
    cell, so two photos 10 metres apart across a rounding boundary land in
    different cells while the cell CENTRES stay stable. Floor also behaves
    correctly for negative coordinates, where `int()` truncates towards zero
    and would fold the southern and northern hemispheres together at 0.
    """
    return (
        round((lat // size) * size, 6),
        round((lon // size) * size, 6),
    )


class _ListSource:
    """A source over photos the caller already holds.

    This is what `MemoryIndex(photos, ...)` uses. It does NOT re-apply the
    policy: the constructor has always taken an already-vetted list, and
    `tests/test_memory_index.py` depends on that to build a deliberately
    unguarded index and prove its own fixture is not vacuous.
    """

    def __init__(self, photos: Sequence[Photo]) -> None:
        self._photos = tuple(photos)
        self._by_hash = {p.file_hash: p for p in self._photos}

    def fetch(self, row_ids: Sequence[int]) -> dict[int, Photo]:
        return {i: self._photos[i] for i in row_ids}

    def lookup(self, hashes: Sequence[str]) -> dict[str, Photo]:
        found = {h: self._by_hash.get(h) for h in hashes}
        return {h: p for h, p in found.items() if p is not None}


class _StoreSource:
    """Rows on demand, guardrailed. THE enforcement point of the lazy index.

    Every `Photo` that reaches `MemoryIndex` is built here, and every `Photo`
    built here has had `ExclusionPolicy.deny_reason` applied to it - on the
    opening scan and again on every later fetch. The second application is not
    redundant decoration: with the library no longer materialised, "the index
    physically does not contain a rejected photo" is no longer what makes the
    guarantee true. What makes it true is that there is exactly one function
    that turns a row into a `Photo`, and it refuses.

    It owns its OWN read-only SQLite connection rather than borrowing the
    store's, for two reasons that are both load-bearing:

      * `web/library.py` closes the store the moment the index is built, and
        then serves every request from the index for the life of the process.
      * A `sqlite3` connection belongs to the thread that created it, and that
        same web index is read from many request threads at once. So the
        connection is thread-local, created on demand.

    `mode=ro` is belt and braces: nothing here writes, and now nothing here
    CAN write, whatever a later edit does.
    """

    def __init__(self, db_path: Path, policy: ExclusionPolicy) -> None:
        self._uri = db_path.resolve().as_uri() + "?mode=ro"
        self._policy = policy
        self._local = threading.local()

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._uri, uri=True)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Release this thread's handle on the database file.

        Reopening is transparent - the next query gets a fresh connection - so
        this is a way to let go of the FILE, not a way to end the index's life.
        Windows refuses to delete an open file, which is the whole reason it
        exists: a test that builds an index inside a `TemporaryDirectory` has
        no other way to let the directory be cleaned up.

        Only this thread's connection. `sqlite3` refuses to touch a connection
        from a thread other than the one that made it, so closing another
        thread's would raise rather than tidy.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            self._local.conn = None
            conn.close()

    def scan(self, report: ExclusionReport) -> Iterator[tuple[int, Photo]]:
        """Every allowed row, once, as `(rowid, photo)`, counting the rest.

        The rowid is what lets the index hold a 300,000-photo library as
        eight-byte integers. It is not something to rely on across writes: a
        row can be deleted and a new one allocated the same id, so an index is
        a snapshot of the library as it was when it was opened, the connection
        is read-only, and rebuilding it is how you see a changed one.

        (This used to say a rowid changes on every upsert, because `_insert`
        was INSERT OR REPLACE - which deletes the conflicting row and inserts
        a new one. It is ON CONFLICT DO UPDATE now, so an updated row KEEPS
        its rowid. The conclusion is unchanged and the reason for it was not,
        which is worth the two lines.)

        ORDER BY rowid is not decoration. A bare `SELECT * FROM photos` walks
        the table in rowid order *in practice*, and the engine's byte-for-byte
        promise must not rest on a query plan: `_build_indexes` reassembles
        merged album lists by sorting on rowid, and that reproduces the scan
        order only if the scan was ordered in the first place.
        """
        for row in self._conn.execute("SELECT rowid AS _rowid, * FROM photos ORDER BY rowid"):
            photo = row_to_photo(row)
            report.total += 1
            reason = self._policy.deny_reason(photo)
            if reason is None:
                if self._policy.public_safe_only and photo.meta.face_count is None:
                    report.unverified += 1
                yield row["_rowid"], photo
            else:
                report.by_reason[reason] = report.by_reason.get(reason, 0) + 1

    def fetch(self, row_ids: Sequence[int]) -> dict[int, Photo]:
        return self._by("rowid", row_ids, key=lambda row, photo: row["_rowid"])

    def lookup(self, hashes: Sequence[str]) -> dict[str, Photo]:
        return self._by("file_hash", hashes, key=lambda row, photo: photo.file_hash)

    def _by(self, column: str, values: Sequence, key) -> dict:
        out: dict = {}
        # Sorted and de-duplicated: sorted so a large fetch walks the table in
        # rowid order instead of jumping around it, de-duplicated so a caller
        # asking for the same photo twice costs one row.
        wanted = sorted(set(values))
        for start in range(0, len(wanted), _FETCH_CHUNK):
            chunk = wanted[start : start + _FETCH_CHUNK]
            sql = (
                f"SELECT rowid AS _rowid, * FROM photos WHERE {column} IN "
                f"({','.join('?' * len(chunk))})"
            )
            for row in self._conn.execute(sql, chunk):
                photo = row_to_photo(row)
                if self._policy.deny_reason(photo) is None:
                    out[key(row, photo)] = photo
        return out


class MemoryIndex:
    """Guardrailed view of the library, indexed by rowid and loaded lazily.

    The eight indexes hold SQLite rowids, not photos; `Photo` objects are
    fetched through `_hydrate` when a caller actually reads one, and kept in a
    bounded LRU cache. See the module docstring for the measurements that
    forced this and for what it costs.

    Doing the filtering in SQL per recipe would be faster still and is exactly
    what this class exists to prevent: it is the "filter each recipe remembers
    to apply" the guardrail was written against. The filter moved closer to the
    data; it did not multiply.
    """

    def __init__(
        self,
        photos: list[Photo],
        policy: ExclusionPolicy,
        report: ExclusionReport,
        frames: Path | None = None,
        *,
        cache: int = DEFAULT_CACHE,
    ):
        photos = list(photos)
        self._start(_ListSource(photos), policy, report, frames, cache)
        self._build_indexes(enumerate(photos))

    def _start(self, source, policy, report, frames, cache: int) -> None:
        self._source = source
        self._policy = policy
        self.report = report
        #: Where `rekindle fingerprint` cached one still frame per video.
        #: `resolve_path` hands those out INSTEAD of the video file, so the
        #: renderer, the thumbnailer and anything else that wants pixels gets
        #: a JPEG without knowing a video was involved. Optional, because a
        #: `MemoryIndex` built by hand in a test has no data directory.
        self._frames = frames
        self._cache_max = max(1, cache)
        self._cache: OrderedDict[int, Photo] = OrderedDict()
        # The index is shared across threads by `web.library`, and an
        # OrderedDict is not safe against concurrent eviction.
        self._lock = threading.Lock()

    @classmethod
    def open(cls, store: PhotoStore, policy: ExclusionPolicy | None = None) -> MemoryIndex:
        policy = policy or ExclusionPolicy()
        report = ExclusionReport()
        index = cls.__new__(cls)
        source = _StoreSource(store.db_path, policy)
        index._start(source, policy, report, frames_dir(store.db_path.parent), DEFAULT_CACHE)
        index._build_indexes(source.scan(report))
        report.allowed = index.count()
        return index

    # ---- the spine

    def _build_indexes(self, rows: Iterable[tuple[int, Photo]]) -> None:
        """One streaming pass. Nothing here keeps a `Photo`."""
        self._ids = array(_ROWID)
        # Local capture year per row, ALIGNED WITH `_ids`, so a rowid's year
        # is a bisect away. Four bytes a photo - 1.2 MB across a 300k library
        # - and it is what lets `offers()` answer "how many distinct years are
        # in this slice?" without loading the slice. See `_years_in`.
        self._years = array("i")
        # Local capture YEAR-MONTH per row as `year * 12 + (month - 1)`,
        # aligned with `_ids` exactly as `_years` is. Four more bytes a photo
        # - 1.2 MB across a 300k library - and it is what lets `album_story`
        # write its subtitle without loading an album.
        #
        # One integer rather than a (year, month) tuple because a tuple per
        # photo is a Python object per photo, which is the cost this whole
        # spine exists to avoid.
        self._yearmonths = array("i")
        self._image_ids = array(_ROWID)
        self._by_year: dict[int, array] = defaultdict(_ids)
        self._by_month: dict[int, array] = defaultdict(_ids)
        self._by_month_day: dict[tuple[int, int], array] = defaultdict(_ids)
        self._by_ymd: dict[tuple[int, int, int], array] = defaultdict(_ids)
        self._by_person: dict[str, array] = defaultdict(_ids)
        self._by_pair: dict[tuple[str, str], array] = defaultdict(_ids)
        self._by_cell: dict[tuple[float, float], array] = defaultdict(_ids)
        # Images only, so `recurring_event` can find its bursts without
        # materialising the library to count days. See `image_day_counts`.
        self._image_days: Counter[tuple[int, int, int]] = Counter()
        # Keyed on the album name AS WRITTEN. The year-suffix families can only
        # be worked out once every name is known, so merging happens below.
        raw_albums: dict[str, array] = {}

        for row_id, photo in rows:
            self._ids.append(row_id)
            local = photo.meta.taken_at_local
            # `deny_reason` already rejected a dateless photo, so this is a
            # belt-and-braces narrowing for the type checker rather than a
            # reachable branch. Year 0 keeps `_years` aligned with `_ids`; it
            # is unreachable and no index below points at it.
            if local is None:  # pragma: no cover
                self._years.append(0)
                self._yearmonths.append(0)
                continue
            self._years.append(local.year)
            self._yearmonths.append(local.year * 12 + local.month - 1)
            ymd = (local.year, local.month, local.day)
            self._by_year[local.year].append(row_id)
            self._by_month[local.month].append(row_id)
            self._by_month_day[(local.month, local.day)].append(row_id)
            self._by_ymd[ymd].append(row_id)
            if photo.media_type is MediaType.IMAGE:
                self._image_ids.append(row_id)
                self._image_days[ymd] += 1

            people = sorted({p for p in photo.meta.people if p})
            for person in people:
                self._by_person[person].append(row_id)
            # Sorted pairs, so "A and B" and "B and A" are one key. Without
            # this the pair recipe offers every pair twice.
            for pair in itertools.combinations(people, 2):
                self._by_pair[pair].append(row_id)

            for album in photo.albums:
                raw_albums.setdefault(album, array(_ROWID)).append(row_id)

            if photo.meta.gps is not None:
                self._by_cell[gps_cell(photo.meta.gps.lat, photo.meta.gps.lon)].append(row_id)

        aliases = dict(self._merged_year_suffixes(raw_albums))
        # An explicit alias always wins over the automatic rule.
        aliases.update(self._policy.album_aliases)
        self.album_merges = {
            name: target
            for name, target in aliases.items()
            if name not in self._policy.album_aliases
        }
        # Album membership under the name the PHOTOGRAPH carries, before the
        # merge and before `album_aliases`. `recurring.naming_evidence` reads
        # `photo.albums` and applies `albums.presentable` to the name as
        # written, so `image_album_years` has to answer with the same names or
        # it is not the same rule: the family of `Photos from 2019` is
        # `Photos from`, which IS presentable, and a family-keyed or merged
        # answer would title every event on this library "Photos from".
        #
        # It costs almost nothing, because the arrays are SHARED with
        # `_by_album` for every name the merge left alone - on the reference
        # library the only rowids held twice are the 19 in the two `Christmas`
        # spellings.
        self._by_album_raw = raw_albums
        spellings: dict[str, list[str]] = {}
        for name in raw_albums:
            spellings.setdefault(aliases.get(name, name), []).append(name)
        self._by_album: dict[str, array] = {}
        for target, names in spellings.items():
            if names == [target]:
                self._by_album[target] = raw_albums[target]
                continue
            # Sorted back into rowid order, which IS scan order because
            # `_StoreSource.scan` says ORDER BY rowid. Without this a merged
            # family would list one spelling's photos before the other's, and
            # `album_story` would emit a different memory than the
            # materialised index did.
            rows = sorted(row for name in names for row in raw_albums[name])
            self._by_album[target] = array(_ROWID, rows)

    @staticmethod
    def _merged_year_suffixes(raw_albums: Iterable[str]) -> dict[str, str]:
        """Album names that are the same album with a year on the end.

        `Christmas 2025` and `Christmas 15` are one recurring event that
        `album_story` otherwise publishes as two unrelated memories, one of
        eight photos and one of eleven.

        **It merges only where it actually merges.** Stripping the suffix
        everywhere is the obvious version and measured on the reference
        library it is a bad trade: it merges exactly one family and RENAMES
        seven more albums that have no partner - `Durga Puja 25` to
        `Durga Puja`, `Puri 25` to `Puri`. Each of those renames changes a
        memory id, so every dismissal of one stops applying, and it buys
        nothing. A name the user wrote is left exactly as they wrote it unless
        another name shares its family.

        Conservative in the other direction too: the rule strips a suffix and
        never matches on a prefix or a shared word, so `Diwali 25` and
        `Diwali Kali Puja 22` stay apart. `Leh Ladakh` and `ladakh` are one
        trip and are NOT merged, because no honest automatic rule separates
        that from two different places with similar names - that case is what
        `album_aliases` in `exclusions.toml` is for.
        """
        families: dict[str, set[str]] = {}
        for album in raw_albums:
            if albums.presentable(album):
                families.setdefault(albums.family(album), set()).add(album)
        return {
            name: family
            for family, names in families.items()
            if len(names) > 1
            for name in names
            if name != family
        }

    def _years_in(self, row_ids: Sequence[int]) -> set[int]:
        """The distinct local capture years of a slice, without loading it.

        `_ids` is ascending - the scan is ORDER BY rowid - so a rowid's
        position is a bisect, and `_years[position]` is its year. This is the
        one thing five recipes want from the photographs they ask for in
        `offers()`, and asking for the photographs to get it is what made a
        300,000-photo library slow: `year_in_review` alone hydrated 298,000
        of them to compute 24 integers.
        """
        ids, years = self._ids, self._years
        return {years[bisect_left(ids, row_id)] for row_id in row_ids}

    def _span_in(self, row_ids: Sequence[int]) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Earliest and latest `(year, month)` of a slice, without loading it.

        The same bisect trick as `_years_in`, against `_yearmonths`. Returns
        None for an empty slice rather than a guessed span - `span_subtitle`
        has always refused to invent a date and this must not be the place
        that starts.
        """
        if not len(row_ids):
            return None
        ids, months = self._ids, self._yearmonths
        packed = [months[bisect_left(ids, row_id)] for row_id in row_ids]
        low, high = min(packed), max(packed)
        return (low // 12, low % 12 + 1), (high // 12, high % 12 + 1)

    def _is_image(self, row_id: int) -> bool:
        """Is this rowid a still image? A bisect, not a set.

        `_image_ids` is ascending for the same reason `_ids` is. Holding the
        same rowids in a `set` as well would cost 16.8 MB at 300,000
        photographs - measured: 8.4 MB of hash table and 8.4 MB of the `int`
        objects an `array("q")` does not need - for a question
        `image_album_years` asks a few thousand times a pass.
        """
        position = bisect_left(self._image_ids, row_id)
        return position < len(self._image_ids) and self._image_ids[position] == row_id

    # ---- hydration. The ONLY route from a rowid to a Photo.

    def _hydrate(self, row_ids: Sequence[int], *, cache: bool = True) -> list[Photo]:
        """Rowids -> photos, in the order given.

        A rowid the source refuses is silently absent, exactly as a refused
        hash is absent from `resolve_many`. That cannot happen with a frozen
        policy over a read-only database - the same predicate put the rowid in
        the spine - and `test_memory_index.py` pins that the two agree.
        """
        with self._lock:
            found = {}
            for row_id in row_ids:
                photo = self._cache.get(row_id)
                if photo is not None:
                    found[row_id] = photo
                    self._cache.move_to_end(row_id)
        missing = [i for i in row_ids if i not in found]
        if missing:
            fetched = self._source.fetch(missing)
            found.update(fetched)
            # A read bigger than the cache can only evict things worth
            # keeping to make room for things it is about to drop, so it is
            # not admitted at all. `all()` over a 300k library would otherwise
            # flush every recipe's working set on the way past.
            if cache and len(row_ids) < self._cache_max:
                with self._lock:
                    for row_id, photo in fetched.items():
                        self._cache[row_id] = photo
                        self._cache.move_to_end(row_id)
                    while len(self._cache) > self._cache_max:
                        self._cache.popitem(last=False)
        return [found[i] for i in row_ids if i in found]

    def _stream(self, row_ids: Sequence[int]) -> Iterator[Photo]:
        for start in range(0, len(row_ids), _STREAM_CHUNK):
            yield from self._hydrate(row_ids[start : start + _STREAM_CHUNK], cache=False)

    def close(self) -> None:
        """Release this thread's handle on the database file, if it has one.

        A no-op for an index built from a list of photos, and transparently
        undone by the next query. See `_StoreSource.close`.
        """
        closer = getattr(self._source, "close", None)
        if closer is not None:
            closer()

    # ---- queries. Every one goes through _hydrate or reads the spine.

    def all(self) -> list[Photo]:
        return self._hydrate(self._ids)

    def iter_all(self) -> Iterator[Photo]:
        """Every photo, streamed, without ever holding the library.

        `all()` on a 300,000-photo library is 690 MB of `Photo` objects, and
        the two callers that most needed it only wanted to COUNT something -
        how many rows lack a fingerprint, how many match a search string. This
        yields in bulk-fetched chunks and does not admit them to the cache, so
        the caller's own filter decides what survives.
        """
        return self._stream(self._ids)

    def count(self) -> int:
        return len(self._ids)

    def get(self, file_hash: str) -> Photo | None:
        return self._source.lookup((file_hash,)).get(file_hash)

    def resolve_many(self, hashes: Iterable[str]) -> list[Photo]:
        """Hashes -> photos, in the order given.

        This is the seam the semantic layer comes in through. A hash the
        policy refused, or one that is not in the library at all, is silently
        ABSENT from the result - which is the guardrail doing its job, not a
        lookup failure. Measured on the reference library: all 162
        archived/trashed rows resolve to nothing here, and excluding one
        person drops 60 of 600 `durga puja` hits, with no filtering code
        anywhere in the caller.
        """
        wanted = list(hashes)
        found = self._source.lookup(wanted)
        return [found[h] for h in wanted if h in found]

    def by_date(self, year: int, month: int, day: int) -> list[Photo]:
        """Every photo taken on one LOCAL calendar day.

        Local, not UTC, for the reason `strata.bucket_key` already documents:
        13,116 rows in the reference library have a non-UTC local zone, and a
        photo taken at 00:30 would otherwise land in the previous day. Note
        that this is a different index from `by_month_day`, which folds every
        year together for anniversaries.
        """
        return self._hydrate(self._by_ymd.get((year, month, day), ()))

    def dates(self) -> list[tuple[int, int, int]]:
        """Every local calendar day the library has a photo on, in order."""
        return sorted(self._by_ymd)

    def by_year(self, year: int) -> list[Photo]:
        return self._hydrate(self._by_year.get(year, ()))

    def by_month(self, month: int) -> list[Photo]:
        return self._hydrate(self._by_month.get(month, ()))

    def by_month_day(self, month: int, day: int) -> list[Photo]:
        return self._hydrate(self._by_month_day.get((month, day), ()))

    def by_person(self, person: str) -> list[Photo]:
        return self._hydrate(self._by_person.get(person, ()))

    def by_pair(self, a: str, b: str) -> list[Photo]:
        return self._hydrate(self._by_pair.get(tuple(sorted((a, b))), ()))  # type: ignore[arg-type]

    def by_album(self, album: str) -> list[Photo]:
        return self._hydrate(self._by_album.get(album, ()))

    def by_gps_cell(self, cell: tuple[float, float]) -> list[Photo]:
        return self._hydrate(self._by_cell.get(cell, ()))

    def images(self) -> list[Photo]:
        return self._hydrate(self._image_ids)

    def iter_images(self) -> Iterator[Photo]:
        """Every image, streamed. `iter_all`'s reason, for images."""
        return self._stream(self._image_ids)

    def image_day_counts(self) -> Counter[tuple[int, int, int]]:
        """How many IMAGES fall on each local calendar day, `(y, m, d)`.

        Read off the spine, so it costs no photos at all. `recurring_event`
        needs exactly this and nothing else to find its bursts, and asking it
        for `images()` instead is what used to materialise the whole library
        inside a recipe - the one place the lazy index could not help.
        """
        return Counter(self._image_days)

    # ---- aggregate views, for building offers cheaply

    def years(self) -> list[int]:
        return sorted(self._by_year)

    def months(self) -> list[int]:
        return sorted(self._by_month)

    def month_days(self) -> list[tuple[int, int]]:
        return sorted(self._by_month_day)

    def year_counts(self) -> Counter[int]:
        return Counter({year: len(v) for year, v in self._by_year.items()})

    def month_counts(self) -> Counter[int]:
        return Counter({month: len(v) for month, v in self._by_month.items()})

    def month_day_counts(self) -> Counter[tuple[int, int]]:
        return Counter({key: len(v) for key, v in self._by_month_day.items()})

    def people_counts(self) -> Counter[str]:
        return Counter({name: len(v) for name, v in self._by_person.items()})

    def pair_counts(self) -> Counter[tuple[str, str]]:
        return Counter({pair: len(v) for pair, v in self._by_pair.items()})

    def album_span(self, album: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Earliest and latest `(year, month)` in an album, off the spine.

        `album_story.offers()` needs a count and a span to write a subtitle,
        and used to get the span by hydrating every photograph of every
        album - on a 300k library, the whole library, to produce two dates per
        album.
        """
        return self._span_in(self._by_album.get(album, ()))

    def album_counts(self) -> Counter[str]:
        return Counter({name: len(v) for name, v in self._by_album.items()})

    def gps_cells(self) -> Counter[tuple[float, float]]:
        return Counter({cell: len(v) for cell, v in self._by_cell.items()})

    def years_present(self, photos: list[Photo]) -> set[int]:
        return {p.meta.taken_at_local.year for p in photos if p.meta.taken_at_local}

    # The distinct years of a slice, off the spine. Each is exactly
    # `years_present(by_X(...))` and `test_memory_index.py` pins that against
    # every slice of a real index - but it costs no photographs, which is the
    # difference between an offers pass that reads the library and one that
    # does not. See `_years_in`.

    def month_years(self, month: int) -> set[int]:
        return self._years_in(self._by_month.get(month, ()))

    def month_day_years(self, month: int, day: int) -> set[int]:
        return self._years_in(self._by_month_day.get((month, day), ()))

    def person_years(self, person: str) -> set[int]:
        return self._years_in(self._by_person.get(person, ()))

    def pair_years(self, a: str, b: str) -> set[int]:
        return self._years_in(self._by_pair.get(tuple(sorted((a, b))), ()))  # type: ignore[arg-type]

    def album_years(self, album: str) -> set[int]:
        return self._years_in(self._by_album.get(album, ()))

    def image_album_years(self, days: Iterable[tuple[int, int, int]]) -> dict[str, set[int]]:
        """Album name -> the local capture years its IMAGES have on `days`.

        The evidence `recurring.naming_evidence` weighs, without loading a
        photograph. That recipe used to get it by hydrating every image inside
        every event's bursts - 11,422 of them on the reference library - to
        read one date and one album list off each. Both are already in the
        index, inverted, so the answer is a set intersection per album and a
        bisect per hit. Measured cold over the eleven events of the reference
        index: 0.54 s hydrating, 0.02 s here, and the recipe's whole offers
        phase 1.18 s -> 0.38 s.

        **Names as written, not families.** `presentable` has to see the name
        the user wrote: the family of `Photos from 2019` is `Photos from`,
        which passes `presentable`, so a family-keyed answer would name every
        event on this library "Photos from" - the defect `albums.py` exists to
        prevent. Grouping into families, `min_years` and the tie-break all
        stay in `recurring`, which owns the rule; this returns the evidence
        and judges nothing. It is also why the merged `_by_album` is not the
        index read here - see `_by_album_raw`.

        **Images only**, because the memory is images only and the title must
        not rest on a shot the memory cannot contain. 428 of the photographs
        inside this library's bursts are videos carrying an album.
        """
        rows: set[int] = set()
        for ymd in days:
            rows.update(row for row in self._by_ymd.get(ymd, ()) if self._is_image(row))
        if not rows:
            return {}
        out: dict[str, set[int]] = {}
        for name, members in self._by_album_raw.items():
            # A list comprehension over the album rather than over `rows`,
            # because it comes out in rowid order for `_years_in` to bisect.
            inside = [row for row in members if row in rows]
            if inside:
                out[name] = self._years_in(inside)
        return out

    # ---- public-safe

    def is_public_safe(self, photo: Photo) -> bool:
        return is_public_safe(photo, self._policy.public_safe_allow)

    @property
    def public_safe_allow(self) -> frozenset[str]:
        return self._policy.public_safe_allow

    # ---- paths

    def resolve_path(self, photo: Photo) -> Path | None:
        """The first path of this photo that exists on disk.

        A photo routinely has the same bytes in two folders, and a library can
        be partially mounted or partially extracted, so `paths[0]` is not
        reliably the one that is there. Returns None when none of them are,
        which the renderer reports as a dropped shot rather than crashing.

        FOR A VIDEO this returns its CACHED STILL FRAME and never the video
        file. That is the single substitution that lets a video flow through
        the renderer: `frames.fit_photo` opens whatever it is handed with
        Pillow, and handing it an .mp4 would raise. Returning None when no
        frame is cached is correct rather than a fallback - `composition`
        has already refused any video without one, so a shot reaching here
        without a frame is a bug worth surfacing as a dropped shot.
        """
        if photo.media_type is MediaType.VIDEO:
            if self._frames is None:
                return None
            still = self._frames / f"{photo.file_hash}.jpg"
            try:
                return still if still.is_file() else None
            except OSError:
                return None
        for path in photo.paths:
            try:
                if path.is_file():
                    return path
            except OSError:
                # A path too long for the platform, or an unreachable network
                # share. Not a reason to abort a render.
                continue
        return None

    def earliest(self, photos: list[Photo]) -> Photo | None:
        return min(photos, key=_chrono) if photos else None

    def latest(self, photos: list[Photo]) -> Photo | None:
        return max(photos, key=_chrono) if photos else None


def _chrono(photo: Photo) -> tuple[datetime, str]:
    """Chronological, with file_hash breaking ties.

    1,430 timestamps in the reference library are shared by more than one
    photo (one by 40), so a bare date key is not a total order and
    `min()`/`max()` would depend on input order.
    """
    assert photo.meta.taken_at_utc is not None  # guaranteed by deny_reason
    return (photo.meta.taken_at_utc, photo.file_hash)
