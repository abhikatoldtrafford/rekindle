# rekindle

**Turn your photo library into memories.**

rekindle finds the photos that belong together — a trip, an anniversary, a
season in one place — and stitches them into a short narrated montage set to
music. Ask for one in plain language, or let it surface them on its own.

```
rekindle index ~/Pictures
rekindle memory "our trip to the coast, 2014"
rekindle memory --auto          # anniversaries, "N years ago today"
```

Point it at a folder. That's the whole setup.

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
| **XMP sidecars** (Lightroom, digiKam, osxphotos) | Person names **and face regions**, keywords, ratings |
| **EXIF / IPTC** | Date taken, GPS, camera, orientation, keywords |
| **Filesystem** | Folder names as albums, mtime as a fallback date |

`rekindle doctor` tells you what coverage you actually have before you index.

### If your photos are in Google Photos

Google removed the API that could read your library after 31 March 2025, so
there's no connector — not in rekindle, not in anything else. Export with Google
Takeout, extract it, and point rekindle at the folder. You'll get dates, GPS and
camera data from EXIF.

A dedicated Takeout parser that also reads Google's JSON sidecars — recovering
face tags and descriptions — is the next source planned.

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

rekindle doctor ~/Pictures    # what metadata do you actually have?
rekindle index ~/Pictures     # build the local index
rekindle serve                # open the player
```

No `.env` needed unless you want the optional LLM narration.

## How it works

```
a folder of photos
        ↓
  read metadata          XMP → EXIF/IPTC → filesystem
        ↓
  local embeddings       SigLIP on your GPU, or CPU
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

- **Person features need person data.** Without XMP sidecars, rekindle doesn't
  know who is in a photo, so person-based memories and person exclusions are
  unavailable. **Date-range and folder exclusions always work** — prefer them.
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
