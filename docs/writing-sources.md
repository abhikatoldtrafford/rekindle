# Writing a source

A **source** turns some photo collection into normalised `Photo` records. Its
job is to absorb a format's quirks so nothing downstream has to know about them.

Wanted: Apple Photos, Immich, Nextcloud Memories, PhotoPrism, Synology Photos,
Amazon Photos, iCloud exports.

## The protocol

```python
from rekindle.sources import Source, register


@register
class ImmichSource(Source):
    name = "immich"

    def scan(self, config) -> Iterator[RawItem]:
        """Yield every item. Cheap - no decoding, no hashing."""

    def metadata(self, item: RawItem) -> PhotoMeta:
        """Extract what this format knows. Everything is optional."""

    def report(self) -> SourceReport:
        """What you found, what you couldn't match, what looked wrong."""
```

## Rules

**Report honestly.** `SourceReport` feeds `rekindle doctor`. Never silently drop
an item — an unmatched file must be counted and explained. A source that
under-reports its own failures is worse than one that fails loudly.

**Record confidence, not just values.** If you matched metadata to a file by
guessing, say so. `PhotoMeta.sidecar_match` carries the tier, and narration is
restricted by it. This exists because aggressive metadata matching in a
well-known Takeout tool produced roughly a third of its GPS pointing at the
wrong place — silently.

**Identity is `file_hash`** (BLAKE2b of file bytes), computed by core. Don't
invent your own. If your format has stable IDs, put them in `source_id`.

**Support incremental sync if the format can.** `scan()` may be called against a
library that is already indexed. New hashes insert; existing ones update and
refresh `last_seen`. **Never delete** — a delta cannot express deletion, and
guessing produces data loss.

**Never trust file extensions.** Sniff magic bytes. Takeout ships `.heic` files
that are actually JPEG, and it won't be the only format that does.

**Don't assume English.** Folder and album names are localized.

## Media types

Handle, or explicitly skip and report: stills (JPEG, HEIC, PNG, WebP), video,
motion photos (`.MP`, `.MV`, `MVIMG_*.jpg`), Live Photo HEIC+MOV pairs, RAW.
v1 montages stills only, but the index holds everything.

## Testing

Build a synthetic fixture tree in `tests/fixtures/<yourformat>/` reproducing the
format's real quirks — especially the ugly ones. If you found a naming edge case
in the wild, it belongs in a fixture. Tests must run with no network, no GPU and
no credentials.
