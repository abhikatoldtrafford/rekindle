# rekindle

**Turn your photo library into memories.**

rekindle finds the photos that belong together — a trip, an anniversary, a
person across a decade — and stitches them into a short narrated montage set to
music. Ask for one in plain language, or let it surface them on its own.

```
rekindle memory "Goa trip 2014"
rekindle memory "me and Mom over the years"
rekindle memory --auto          # anniversaries, "N years ago today"
```

Local-first: your photos never leave your machine.

> **Status: early development.** The design is settled and written up in
> [the design spec](docs/superpowers/specs/2026-09-10-rekindle-design.md);
> the implementation is in progress. Issues and PRs welcome.

---

## Getting your photos in

**Google removed the API that could read your library** on 1 April 2025, so
there is no "connect to Google Photos" button any more — not in rekindle, not
in anything else. Google Takeout is the only route that still carries your face
tags, GPS and descriptions.

The export takes hours, so **start it before anything else**:

**[→ Connecting your Google Photos library](docs/connecting-google-photos.md)**

rekindle also reads plain folders, so it works fine with any photo collection.

## Quick start

```bash
git clone https://github.com/abhikatoldtrafford/rekindle
cd rekindle
uv sync

cp .env.example .env      # optional: add an OpenAI key, or stay fully offline

rekindle doctor           # validate your Takeout export
rekindle index            # build the local index
rekindle serve            # open the player
```

## How it works

```
Takeout / Picker / folders
        ↓
  junk filtering          screenshots, receipts, blur, burst duplicates
        ↓
  local embeddings        SigLIP on your GPU, or CPU
        ↓
  LanceDB index           disk-backed, scales to hundreds of thousands
        ↓
  recipes                 deterministic selection: trips, anniversaries, ...
        ↓
  narration               grounded strictly in your photos' metadata
        ↓
  MemorySpec (JSON)  →  web player  |  MP4 export
```

Photo selection is **deterministic Python**, not an LLM guess. The model only
routes your prompt to a recipe and writes prose over facts it has been handed.
That is what makes the guardrails real rather than hopeful.

## Guardrails

Memories touch a nerve. rekindle is built so it cannot casually hurt you:

- **Exclusion list** — blocklist people, albums or date ranges. They are removed
  from the candidate pool *before* anything else runs.
- **Sensitive contexts** — likely-painful settings are held back for your
  confirmation, never set to upbeat music by surprise.
- **Strict grounding** — narration may only assert what your metadata supports.
  Untraceable claims are rejected, not published.
- **Junk filtering** — screenshots and receipts stay out of your memories.

## Privacy

Everything runs locally. If you enable the OpenAI providers, only photos that
reach a memory are sent for captioning — never your whole library. Set
`REKINDLE_CAPTION_PROVIDER=local` to keep everything on your machine.

## Contributing

The easiest and most valuable contribution is **a new memory type**. A recipe is
one file implementing one protocol, with no core changes:
[writing-recipes.md](docs/writing-recipes.md).

New photo sources (Apple Photos, Immich, Nextcloud) and model backends are
equally welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
