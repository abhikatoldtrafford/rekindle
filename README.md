# rekindle

**Turn your photo library into memories.**

rekindle finds the photos that belong together — a trip, an anniversary, a
person over the years — and stitches them into a short montage. Everything it
picks is chosen by rules over your own metadata, so the same library always
produces the same memories.

```bash
uv run rekindle doctor ~/Pictures    # what metadata do you actually have?
uv run rekindle index ~/Pictures     # build the local index
uv run rekindle enrich ~/Pictures    # read Google Takeout JSON sidecars into the index
uv run rekindle doctor ~/Pictures --from-index   # report on the stored index
```

Point it at a folder. That's the whole setup.

Then build memories from it:

```bash
uv run rekindle fingerprint                  # one-time pass, enables dedup
uv run rekindle memories                     # what could this library produce?
uv run rekindle memory --recipe album_story --key "Kashmir"
uv run rekindle memory --auto                # today's anniversary, if any
uv run rekindle watch ~/Pictures             # foreground; prints, never renders
```

A memory is a GIF (always) plus an MP4 (when ffmpeg is on PATH), written to
`memories/` with the `MemorySpec` that produced it.

**Freeform prompts** (`rekindle memory "our trip to the coast"`) need
embeddings and are not built yet - v1 selection is deterministic and
structured. `rekindle memories` lists everything available.

Runs entirely on your machine — the default configuration makes no network
calls at all, and needs no API key.

> **Status: early development.** The memory engine is designed in
> [the M2 spec](docs/superpowers/specs/2026-09-11-memories-design.md); the
> wider architecture is in [the v1 spec](docs/superpowers/specs/2026-09-10-rekindle-design.md),
> which went through an [independent adversarial review](docs/superpowers/specs/audit-v1-resolutions.md).
> Issues and PRs welcome.

---

## Getting your photos in

**v1 reads a directory.** No accounts, no OAuth, no API keys, no vendor lock-in.
If your photos are on disk — a NAS, an external drive, a phone backup folder, an
export from anywhere — rekindle can use them now.

Metadata comes from the files themselves:

| Source | Gives you |
|---|---|
| **XMP sidecars** (Lightroom, digiKam, osxphotos) | Person names **and face regions**, keywords, descriptions |
| **EXIF** | Date taken and UTC offset, GPS, camera make and model, dimensions |
| **Filesystem** | Folder names as albums, mtime as a fallback date |

Orientation is applied, so a portrait photo is indexed as portrait.
Not read yet: IPTC and ratings.

`rekindle doctor` tells you the coverage you actually have — not what these
formats could in principle hold — before you index.

### If your photos are in Google Photos

Google removed the API that could read your library after 31 March 2025, so
there's no connector — not in rekindle, not in anything else. Export with Google
Takeout, extract it, and point rekindle at the folder. You'll get dates, GPS and
camera data from EXIF.

`rekindle enrich` then reads Google's own JSON sidecars — recovering face tags,
corrected capture dates, descriptions and album titles — into an index that
`rekindle index` already built. Run `rekindle doctor --from-index` afterwards
to see what it found.

**[→ Exporting from Google Photos](docs/connecting-google-photos.md)**

### Other libraries

Immich, Apple Photos, Nextcloud and PhotoPrism all expose their libraries
properly, and several give face regions that Google never did. Each is one
`Source` implementation: [writing-sources.md](docs/writing-sources.md).

## Quick start

```bash
git clone https://github.com/abhikatoldtrafford/rekindle
cd rekindle
uv sync

uv run rekindle doctor ~/Pictures    # what metadata do you actually have?
uv run rekindle index ~/Pictures     # build the local index
uv run rekindle enrich ~/Pictures    # read Google Takeout JSON sidecars into the index
uv run rekindle doctor ~/Pictures --from-index   # report on the stored index
```

`doctor` writes nothing at all, so it is safe to point at anything.
`enrich` requires `index` to have run first — it reads the database, not the
filesystem, so photo rows must already exist.

No `.env` needed unless you want the optional LLM narration.

## How it works

```
a folder of photos
        ↓
  read metadata          XMP → EXIF → filesystem
        ↓
  fingerprints           dHash + sharpness, one decode per photo
        ↓
  SQLite                 relational truth
        ↓
  MemoryIndex            THE chokepoint: every guardrail applied once, here
        ↓
  recipes                deterministic selection: albums, anniversaries, ...
        ↓
  engine                 dedup → rank → cap → order
        ↓
  MemorySpec (JSON)  →  GIF (always)  |  MP4 (when ffmpeg is present)
```

Photo selection is **deterministic Python** — an LLM never picks your photos,
and the whole v1 engine runs with no model, no network and no randomness. The
same library produces the same memories, byte for byte.

## Guardrails

Memories touch a nerve. rekindle tries hard not to hurt you, and is honest about
where it can't guarantee that:

- **Dismissal** — `rekindle dismiss <recipe> <key>` and that memory never
  returns. Permanent, and it survives the library growing: memories are
  identified by what they are *about*, not by which photos happen to be in
  them today. `rekindle undismiss` reverses it.
- **Exclusion list** — `rekindle exclude --person NAME`, `--album`, or
  `--from`/`--to` for a date range. Enforced at one chokepoint that every
  recipe passes through, so no memory type can bypass it.
- **Archived photos never surface.** Not configurable.
- **Junk filtering** — screenshots, wallpapers, thumbnails, panoramas,
  near-black and blown-out frames are excluded, each with a visible count and
  a reason. Nothing is dropped silently.
- **No memory renders itself.** `rekindle watch` prints the command; you run
  it. Unprompted memories are where the real risk lives.

### Known limits

We'd rather tell you than let you find out:

- **Person features need person data.** XMP sidecars give it (with face
  regions); `rekindle enrich` recovers it from a Google Takeout export too
  (names only, no regions). Without either, rekindle doesn't know who is in a
  photo, so person-based memories and person exclusions are unavailable.
  **Date-range and folder exclusions always work** — prefer them.
- **Face tags are incomplete, and that limits what can be published.** On a
  Google Takeout export they cover only part of the library. A photo with *no*
  tags is therefore never treated as safe to publish: it may still contain
  people nobody labelled. `--public-safe` admits a photo only when its tags are
  non-empty *and* every name is on your allow-list.
- **GPS is sparse** in most libraries — around 12% on the reference export —
  so place memories are thin and never name the place. There is no offline
  gazetteer, so rekindle reports coordinates rather than inventing a city.
- **Sensitive-context detection does not exist.** rekindle cannot know someone
  has died, or that a trip ended a relationship. Dismissal and the exclusion
  list are the honest mechanism, and they are what ships.
- **A photographed document still gets through.** Screenshots are detectable
  from metadata; a photo *of* a receipt or a whiteboard is not, without a model
  v1 deliberately does not have.
- **Videos do not appear in memories.** They carry no stored dimensions and no
  perceptual hash, and rendering one needs ffmpeg, which stays optional.

## Privacy

Everything runs locally by default — no API key needed, no network calls made,
no audio downloaded, and your original files are never modified. If you enable
the optional LLM captions, **only a fact sheet is sent** — dates, counts, names,
albums and coordinates drawn from your index. Never the pixels, never a file
path, never your library. See [SECURITY.md](SECURITY.md).

## Contributing

Two high-value contributions, neither requiring core changes:

- **A new memory type** — one file implementing one protocol:
  [writing-recipes.md](docs/writing-recipes.md)
- **A new photo source** — Immich, Apple Photos, Nextcloud:
  [writing-sources.md](docs/writing-sources.md)

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
