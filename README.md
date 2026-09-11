# rekindle

**Turn your photo library into memories.**

rekindle finds the photos that belong together — a trip, an anniversary, a
season in one place — and stitches them into a short narrated montage set to
music. Ask for one in plain language, or let it surface them on its own.

```bash
uv run rekindle doctor ~/Pictures    # what metadata do you actually have?
uv run rekindle index ~/Pictures     # build the local index
uv run rekindle enrich ~/Pictures    # read Google Takeout JSON sidecars into the index
uv run rekindle doctor ~/Pictures --from-index   # report on the stored index
```

Point it at a folder. That's the whole setup.

Those four commands work today. The montage itself is what's being built next:

```bash
rekindle memory "our trip to the coast, 2014"   # planned
rekindle memory --auto                          # planned
```

Runs entirely on your machine — the default configuration makes no network
calls at all, and needs no API key.

> **Status: early development.** The design is settled and written up in
> [the design spec](docs/superpowers/specs/2026-09-10-rekindle-design.md), which
> has been through an [independent adversarial review](docs/superpowers/specs/audit-v1-resolutions.md).
> Implementation is in progress. Issues and PRs welcome.

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

Not read yet: IPTC, orientation, and ratings.

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

### Semantic features (optional)

Search your library by describing a photo, group it by scene, and rank it by
predicted aesthetic quality. These need an optional extra and a one-time
model download; everything afterwards runs with no network at all.

```bash
uv sync --extra semantic-gpu        # torch + CUDA;  --extra semantic for CPU/ONNX
uv run rekindle semantic setup      # fetch and checksum the weights, once
uv run rekindle semantic doctor     # which device will actually be used?

uv run rekindle semantic embed                  # embed the indexed photos
uv run rekindle semantic find "snowy mountains" # search
uv run rekindle semantic cluster --untagged-only
uv run rekindle semantic rank --album Kashmir -k 40
uv run rekindle semantic facegate               # propose face-free photos
```

`semantic setup` is the only command in rekindle that makes a network
request. Every other command loads from the local cache and fails with a
message if something is missing, rather than downloading 1.7 GB you did not
ask for. Model revisions are pinned to exact commits and every file's sha256
is checked against `model-locks.json`; `rekindle semantic licences` prints
the licence of everything rekindle can fetch.

Without the extra, these commands print what to install and exit — nothing
else changes, and the default `uv sync` stays small.

No `.env` needed unless you want the optional LLM narration.

## How it works

```
a folder of photos
        ↓
  read metadata          XMP → EXIF → filesystem
        ↓
  local embeddings       CLIP ViT-L/14 on your GPU, or ONNX on CPU
        ↓
  SQLite + vectors       relational truth, brute-force similarity
        ↓
  recipes                deterministic selection: trips, anniversaries, ...
        ↓
  narration              template by default, LLM optional and verified
        ↓
  MemorySpec (JSON)  →  web player  |  MP4 export
```

For the built-in recipes, photo selection is **deterministic Python** — an LLM
never picks your photos. The exception is freeform mode, used when your prompt
matches no recipe: there a model curates, but only from a pool that has already
had every guardrail applied, and its choices are validated against that pool.

## Guardrails

Memories touch a nerve. rekindle tries hard not to hurt you, and is honest about
where it can't guarantee that:

- **Exclusion list** — blocklist date ranges, folders or people. Applied before
  anything else runs.
- **Sensitive contexts** — likely-painful material is held back for your
  confirmation, and every memory has a "not this period / never again" action
  that feeds back into the exclusion list.
- **Verified narration** — an independent verifier checks each claim against
  your metadata. Claims it can't substantiate are rejected, not published.
- **Junk filtering** — screenshots and receipts stay out of your memories.
- **Auto-memories are off by default.** Unprompted memories are where the real
  risk lives; you opt in.

### Known limits

We'd rather tell you than let you find out:

- **Person features need person data.** XMP sidecars give it (with face
  regions); `rekindle enrich` recovers it from a Google Takeout export too
  (names only, no regions). Without either, rekindle doesn't know who is in a
  photo, so person-based memories and person exclusions are unavailable.
  **Date-range and folder exclusions always work** — prefer them. The
  exclusion list that is meant to govern person data doesn't exist yet (see
  `docs/known-limitations.md`) — today nothing surfaces a person unprompted,
  but that must land before anything does.
- **GPS is sparse** in most libraries, so trip detection falls back to clustering
  by time alone.
- **Sensitive-context detection is weak on the cases that hurt most.** It can
  see a hospital; it cannot know someone has died, or that a trip ended a
  relationship. That's what the exclusion list and feedback action are for.

## Privacy

Everything runs locally by default — no API key needed, no network calls made,
and your original files are never modified. If you enable the OpenAI providers,
only photos that reach a memory are sent for captioning, never your whole
library. See [SECURITY.md](SECURITY.md).

## Contributing

Two high-value contributions, neither requiring core changes:

- **A new memory type** — one file implementing one protocol:
  [writing-recipes.md](docs/writing-recipes.md)
- **A new photo source** — Immich, Apple Photos, Nextcloud:
  [writing-sources.md](docs/writing-sources.md)

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
