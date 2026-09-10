# Writing a source

A **source** turns some photo collection into normalised `Photo` records. Its
job is to absorb a format's quirks so nothing downstream has to know about them.

Wanted: Apple Photos, Immich, Nextcloud Memories, PhotoPrism, Synology Photos,
Amazon Photos, iCloud exports.

## The protocol

This is the whole of it, and it is the real thing — see
[`src/rekindle/sources/base.py`](../src/rekindle/sources/base.py).

```python
from pathlib import Path

from rekindle.models import Photo, SourceReport


class ImmichSource:
    name = "immich"

    def scan(self, root: Path) -> tuple[list[Photo], SourceReport]:
        """Walk a library and return normalised photos plus an honest report.

        Implementations must count every skipped file with a reason. Silently
        dropping input is a bug.
        """
```

`Source` is a `typing.Protocol`, so you do not subclass it — a class with a
`name` and a matching `scan` already satisfies it. To have a type checker (or a
test) hold you to it:

```python
from rekindle.sources.base import Source

_: Source = ImmichSource()  # static check
assert isinstance(ImmichSource(), Source)  # runtime check
```

There is **no registry yet**. `rekindle/sources/__init__.py` is empty and the
CLI constructs `FolderSource()` directly, so wiring a second source in today
means editing `src/rekindle/cli.py`. A registration decorator is planned; until
it exists, say so in your PR rather than inventing one.

`scan` returns everything at once — a list, not an iterator, and a
`SourceReport` alongside it. There is no separate `metadata()` or `report()`
call: metadata extraction happens inside `scan` (see `FolderSource._read_meta`),
and the report is built as you go.

### What a `Photo` carries

`Photo` and `PhotoMeta` are in
[`src/rekindle/models.py`](../src/rekindle/models.py). Today: `file_hash`,
`paths`, `media_type`, `first_seen`/`last_seen`, `albums`, `edited_of`,
`source`, `metadata_conflict`, and a `PhotoMeta` of `taken_at_utc`,
`taken_at_local`, `tz_source`, `gps`, `people`, `face_regions`, `keywords`,
`description`, `favorite`, `camera_make`, `camera_model`, `width`, `height`.

Set `source=self.name` explicitly. It is what lets a later source enrich rows a
different source wrote, and it cannot be added retroactively without a schema
migration that does not exist.

**Planned, not yet present** — do not write code against these:

- `PhotoMeta.sidecar_match`, the confidence tier described under "Record
  confidence" below. Until it lands, a match you are unsure of should not be
  recorded at all.
- `source_id`, for formats with stable native IDs.
- Any `RawItem` type. There isn't one; `scan` yields `Photo` records directly.

## Rules

**Report honestly.** `SourceReport` feeds `rekindle doctor`. Never silently drop
an item — an unmatched file must be counted and explained. A source that
under-reports its own failures is worse than one that fails loudly.

Concretely, the counters must **reconcile**:

```
files_seen == media_indexed + duplicates_merged + json_sidecars
              + excluded_dirs + total_skipped
```

Every `continue` in your scan loop has to increment exactly one bucket. Assert
this in your tests on a fixture library — `FolderSource` shipped for ten reviews
with three files falling through two `continue` statements into no bucket at
all, precisely because nothing asserted the sum.

Use `report.skip("<reason>")` for files you did **not** index, and a field of
its own for something that happened to a file you **did** index (as
`SourceReport.undecodable` does). Filing the latter under `skipped` breaks the
identity above. Count per photo — per hash — not per path, or the same bytes in
two folders read as two problems.

**Record confidence, not just values.** If you matched metadata to a file by
guessing, say so. This exists because aggressive metadata matching in a
well-known Takeout tool produced roughly a third of its GPS pointing at the
wrong place — silently. The `sidecar_match` tier that will carry this is not
implemented yet, so for now: if you cannot justify a match, drop the metadata
and count it.

**Never invent a value to stand in for a broken one.** A damaged EXIF rational
must reject the whole coordinate, not contribute a zero to it. A missing face
region coordinate must reject the region, not default to 0.0. Both of those
were real bugs here.

**Identity is `file_hash`** (BLAKE2b-128 of file bytes), computed by
`rekindle.identity.file_hash`. Don't invent your own, and never hash decoded
pixels — the output changes between Pillow versions and would invalidate every
cached record.

**Support incremental sync if the format can.** `scan()` may be called against a
library that is already indexed. New hashes insert; existing ones update and
refresh `last_seen`. **Never delete** — a delta cannot express deletion, and
guessing produces data loss. Merging is `models.merge_meta`; use it rather than
overwriting, or a re-index destroys whatever a previous run enriched.

**Never trust file extensions.** Sniff magic bytes with
`rekindle.identity.sniff`. Takeout ships `.heic` files that are actually JPEG,
and it won't be the only format that does. `sniff` raises `OSError` if the file
cannot be read — route that to `report.unreadable`, never to a "not media"
verdict, which is a claim about content you never saw.

**Don't assume English.** Folder and album names are localized, and so are
Google's `-edited` suffixes and its trash folder.

## Media types

Handle, or explicitly skip and report: stills (JPEG, HEIC, PNG, WebP), video,
motion photos (`.MP`, `.MV`, `MVIMG_*.jpg`), Live Photo HEIC+MOV pairs, RAW.
v1 montages stills only, but the index holds everything.

A format you do not recognise is still a file the user has. Record its
filename before you decide what it is, so its sidecar can still be matched
against it.

## Testing

Build a synthetic fixture tree in `tests/fixtures/` reproducing the format's
real quirks — especially the ugly ones. If you found a naming edge case in the
wild, it belongs in a fixture. Tests must run with no network, no GPU and no
credentials.

Write the test first and **watch it fail** before you fix anything. A test you
have not seen fail is not yet a test.
