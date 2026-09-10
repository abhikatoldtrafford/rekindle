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

Runs entirely on your machine. The default configuration makes no network calls
at all.

> **Status: early development.** The design is settled and written up in
> [the design spec](docs/superpowers/specs/2026-09-10-rekindle-design.md), which
> has been through an [independent adversarial review](docs/superpowers/specs/audit-v1-resolutions.md).
> Implementation is in progress. Issues and PRs welcome.

---

## Getting your photos in

**Google removed the API that could read your library** after 31 March 2025, so
there is no "connect to Google Photos" button any more — not in rekindle, not in
anything else. Google Takeout is the only route that still carries face tags,
GPS and descriptions.

The export takes hours, so **start it before anything else**:

**[→ Connecting your Google Photos library](docs/connecting-google-photos.md)**

Since June 2026 scheduled Takeout exports are *incremental*, which makes them a
genuine recurring feed. rekindle merges each delta into your existing index.

rekindle also reads plain folders, so it works with any photo collection.

## Quick start

```bash
git clone https://github.com/abhikatoldtrafford/rekindle
cd rekindle
uv sync

cp .env.example .env      # optional — the defaults are fully offline

rekindle doctor           # validate your Takeout export
rekindle index            # build the local index
rekindle serve            # open the player
```

## How it works

```
Takeout / folders
        ↓
  parse + match          sidecar matching, with confidence tiers
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

- **Exclusion list** — blocklist date ranges, albums or people. Applied before
  anything else runs.
- **Sensitive contexts** — likely-painful material is held back for your
  confirmation, and every memory has a "not this person / not this period /
  never again" action that feeds back into the exclusion list.
- **Verified narration** — an independent verifier checks each claim against
  your metadata. Claims it can't substantiate are rejected, not published.
- **Junk filtering** — screenshots and receipts stay out of your memories.
- **Auto-memories are off by default.** Unprompted memories are where the real
  risk lives; you opt in.

### Known limits

We'd rather tell you than let you find out:

- **Blocking a *person* only removes photos tagged with them.** Untagged photos
  of that person will still appear. Face tags are opt-in, and unavailable in
  Illinois and Texas. **Date-range and album exclusions are reliable; person
  exclusions are best-effort.**
- **Sensitive-context detection is weak on the cases that hurt most.** It can
  see a hospital; it cannot know someone has died, or that a trip ended a
  relationship. That's what the exclusion list and feedback action are for.
- **Takeout doesn't include photos other people added to shared albums.** Group
  trips and weddings may be thinner than you remember.

## Privacy

Everything runs locally by default — no API key needed, no network calls made.
If you enable the OpenAI providers, only photos that reach a memory are sent for
captioning, never your whole library. See [SECURITY.md](SECURITY.md).

## Contributing

The easiest and most valuable contribution is **a new memory type**. A recipe is
one file implementing one protocol, registered via entry points, with no core
changes: [writing-recipes.md](docs/writing-recipes.md).

New photo sources (Apple Photos, Immich, Nextcloud) and model backends are
equally welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
