# rekindle

**Your photos already remember. This helps them say it out loud.**

[![CI](https://github.com/abhikatoldtrafford/rekindle/actions/workflows/ci.yml/badge.svg)](https://github.com/abhikatoldtrafford/rekindle/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![No cloud](https://img.shields.io/badge/runs-100%25%20local-brightgreen.svg)](#privacy)

Twenty thousand photos sitting in a folder is not a memory. It's a filing
cabinet. Somewhere in there is the week in Kashmir, your kid going from
crawling to running, every December you've ever photographed — and you will
never scroll far enough to find them.

**rekindle finds them for you and turns them into something you'd actually
watch.** A trip. An anniversary. One person, aged across fifteen years in
twenty-four frames. It picks every photo by rules over your own metadata, so
the same library always produces the same memories — no model deciding what
mattered, nothing uploaded, no API key, no account.

Point it at a folder. That's the whole setup.

---

## 🎬 What comes out

<!--
  GALLERY SLOTS - drop a GIF in docs/assets/ and swap the line beneath it.
  Every memory below is real output; see docs/gallery.md.
-->

|  |  |
|---|---|
| **Kashmir** · an album becomes a trip<br>_24 photos · May 2015_<br>`--recipe album_story --key "Kashmir"` | **Maa: then and now**<br>_2 photos · Dec 2010 → Sep 2026_<br>`--recipe then_and_now --key "Maa"` |
| **Avyan over the years**<br>_24 photos · one child, every year_<br>`--recipe person_years --key "Avyan"` | **Every October**<br>_24 photos · 2013 → 2022_<br>`--recipe on_this_month --key "10"` |

> 🚧 **Gallery GIFs land here.** The samples above are generated from a real
> 19,480-photo library; the images are being selected for publication. Run the
> commands on your own library and you'll get the equivalent in about a minute.

---

## 🧠 The nine kinds of memory

Each one is a **recipe** — a small, self-contained rule for which photos belong
together and in what order. Writing a new one is the best first contribution:
one file, one protocol, one registry entry.

| Recipe | What it finds | Example |
|---|---|---|
| 🗺️ **album_story** | A named album, told in order | `"Kashmir"`, `"Leh Ladakh"`, `"Diwali Kali Puja 22"` |
| 📅 **on_this_day** | The same date, across every year you own | `"25 December"` — 2012 → 2021 |
| 🗓️ **on_this_month** | The same month, when a single day is too thin | `"Every May"` — 2015 → 2026 |
| 🌱 **person_years** | One person, aged across time | `"Avyan over the years"` |
| 👥 **pair_years** | Two people, only where they appear *together* | `"Bapi and Maa"` |
| ⏳ **then_and_now** | Earliest and latest, side by side | `"Maa: then and now"` |
| 🎞️ **year_in_review** | One year, spread across its months | `"2016"` |
| 📍 **place_cluster** | Trips, from GPS | `"A place you kept coming back to"` |
| 🎊 **recurring_event** | A burst of photos that comes back every year, even when the date moves — found from timestamps alone, so a lunar festival is caught where `on_this_day` structurally cannot | `"Mid October, most years"` — 12 years |

---

## ✨ Why it's built this way

**Deterministic.** Same library in, same memories out — byte for byte. No
randomness, no model deciding what mattered. You can re-run it in a year and
get the same film.

**It never hurts you.** Archived photos never surface. Dismiss any memory and
it's gone for good, and it *stays* gone as your library grows. Exclude a
person, a date range or an album and every recipe honours it — enforced at one
chokepoint no recipe can route around.

**It shows its working.** Every photo a guardrail rejects is counted with a
reason. A memory that silently drops your favourite photo is indistinguishable
from a bug, so it doesn't do that.

**It's honest about what it can't do.** [Known limits](#known-limits) is a real
section, not a disclaimer. Videos don't appear in memories yet. A photographed
document can slip through. We write those down.

**Nothing leaves your machine.** The default configuration makes no network
calls at all.

---

## ⚡ Quick start

```bash
uv sync
uv run rekindle doctor ~/Pictures    # what metadata do you actually have?
uv run rekindle index ~/Pictures     # build the local index
uv run rekindle enrich ~/Pictures    # read Google Takeout sidecars, if you have them
uv run rekindle fingerprint          # one-time pass; enables dedup
```

Then make something:

```bash
uv run rekindle memories                     # what could this library produce?
uv run rekindle memory --recipe album_story --key "Kashmir"
uv run rekindle memory --auto                # today's anniversary, if any
uv run rekindle ui                           # edit a memory in your browser
uv run rekindle watch ~/Pictures             # foreground; prints, never renders
```

A memory is a **GIF** (always) plus an **MP4** (when ffmpeg is on PATH),
written to `memories/` alongside the `MemorySpec` that produced it — so you can
see exactly which photos were chosen and why.

### 🎵 Music

```bash
uv run rekindle music fetch     # the one command that touches the network
```

Fetches a small set of CC0 tracks. Or drop your own `.mp3` into `music/` —
each memory picks a track deterministically, so the same memory always sounds
the same. Silence is a perfectly good default.

**Freeform prompts** (`rekindle memory "our trip to the coast"`) need embeddings
and aren't built yet — v1 selection is deterministic and structured.
`rekindle memories` lists everything available.

> **Status: early development.** The memory engine is designed in
> [the M2 spec](docs/superpowers/specs/2026-09-11-memories-design.md); the
> wider architecture is in [the v1 spec](docs/superpowers/specs/2026-09-10-rekindle-design.md),
> which went through an [independent adversarial review](docs/superpowers/specs/audit-v1-resolutions.md).
> How it was actually built, including the bugs that shaped it, is in the
> [decision log](docs/decision-log-memory-engine.md). Issues and PRs welcome.

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


`doctor` writes nothing at all, so it is safe to point at anything. `enrich`
requires `index` to have run first — it reads the database, not the filesystem,
so photo rows must already exist.

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

### Memories from your own words (optional)

With the semantic extra installed and the library embedded:

```bash
uv run rekindle memory "durga puja over the years"
```

**Your words never go to the model that looks at your photos.** A prompt is
turned into *visual descriptions* — what such a photograph would look like —
and each of those is searched separately. A photo then ranks by how many of
those descriptions agree on it. That is not a flourish: "Durga Puja" and "Kali
Puja" are nearly the same string to an image-text model, and searching the
names put nine of twenty-four shots of a Kali Puja memory on a Durga Puja day.
A ten-armed goddess with a lion and a black goddess with a red tongue are not
close at all. Measured again with the descriptions: none.

The descriptions come from a checked-in
[festival corpus](src/rekindle/memory/corpus/festivals.toml) and a checked-in
[tag cache](src/rekindle/memory/corpus/prompt_tags.json) — both editable data
files, both working with no API key. Set `OPENAI_API_KEY` and anything they do
not cover is described by a language model once and cached, so the same prompt
gives the same memory forever.

**A prompt memory is a preview, never an offer.** It is built only when you
ask for it by name. It never appears in `rekindle memories` and `--auto` will
never show you one, because **nothing can check that the photos match your
words** — eight statistics have now been tested as a refusal signal on the
reference library and all eight failed. So the command prints what it searched
for, which days it found, the month and year histogram of what it built, and
then says: *look at the memory before you keep it.* The whole measurement is
in [the decision log](docs/decision-log-prompt-memories.md).

`semantic setup` is the only command in rekindle that makes a network
request. Every other command loads from the local cache and fails with a
message if something is missing, rather than downloading 1.7 GB you did not
ask for. Model revisions are pinned to exact commits and every file's sha256
is checked against `model-locks.json`; `rekindle semantic licences` prints
the licence of everything rekindle can fetch.

Without the extra, these commands print what to install and exit — nothing
else changes, and the default `uv sync` stays small.

### 🖼️ Editing a memory by hand

```bash
uv run rekindle ui
```

Opens a page on `127.0.0.1` where you can see what a memory is made of and
change it: **drop a shot**, **see what the guardrails rejected and overrule
it**, **drag to reorder**, **pick a different frame from a burst**, **pull in
the rest of that afternoon**, set the pace and the music, and render.

**The CLI stays fully capable and the page is a lens onto it.** It selects
nothing of its own — candidates come from the same recipes, guardrail counts
from the same `compose` and `collapse`, chosen shots from the same
`engine.build` — and **it cannot show you a photo the CLI would refuse**,
because every lookup goes through the same `MemoryIndex` chokepoint. An
archived photo has no thumbnail, cannot be searched for and cannot be added.

When you render, it hands you the command that rebuilds exactly what you made:

```bash
rekindle render memories/2026-09-12-album_story-kashmir/memory.json --frame-ms 1100
```

That is `rekindle render`, a first-class command: no browser, no server, no
selection re-run. An afternoon of editing becomes something repeatable. And
because the spec names photos by hash and the renderer resolves them through
the index, excluding someone tomorrow removes them from every memory you have
already saved — without editing a single file.

**No new dependency.** The server is `http.server` from the standard library
and the page is one HTML file with no build step, so `uv sync` stays at four
packages and CI runs the tests rather than skipping them. Prompt search inside
the page needs the `semantic` extra; without it the page still opens and
everything else works, and the prompt box says what to install.

Nothing is fetched from the network — no CDN, no font, no analytics, enforced
by a `default-src 'self'` policy on every response. The URL carries a token
that changes every run, `Host` and `Origin` are checked, and request logging is
off by default, because a request log is a record of which of your photographs
you looked at. How it was built, and the three controls that were cut, is in
[the decision log](docs/decision-log-memory-builder.md).

## How it works

```
a folder of photos
        ↓
  read metadata          XMP → EXIF → filesystem
        ↓
  local embeddings       CLIP ViT-L/14 on your GPU, or ONNX on CPU (optional)
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
                     ↑
  rekindle ui        |  a local page that edits the spec, and hands back
                        `rekindle render <spec>` to rebuild it
```

Photo selection is **deterministic Python** — an LLM never picks your photos,
and the whole v1 engine runs with no model, no network and no randomness. The
same library produces the same memories, byte for byte.

Each recipe also declares the dimension its memory is *about* — years for "on
this day", months for a year in review, the album's own span for an album
story — and slots are spread across it before quality ranking chooses within
each period. Without that, selection collapses onto whichever week happened to
photograph best.

### Optional GPT captions

`--captions gpt` rewrites the caption strings only. It is **off by default**,
needs `OPENAI_API_KEY`, and sees **only a fact sheet** — dates, counts, names,
albums and coordinates already derived from your index. Never the pixels, never
a path, never your library. Every caption it returns is checked back against
that fact sheet and rejected if it asserts a year or a name the facts do not
contain.

**It currently adds little.** Measured across five recipes, the model returned
the existing deterministic caption verbatim in four of five cases. That is the
prompt working as intended rather than a fault, but it means enabling this
costs an API call for a change you will usually not see. Leave it off unless
you are experimenting.

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
- **Nothing verifies that a prompt memory matches your prompt.** `rekindle
  memory "scuba diving underwater"` builds a perfectly confident 24-shot
  memory from a library with no scuba photographs in it. Eight statistics have
  been tested as a refusal signal and none separates a concept the library
  holds from one it does not, so no threshold ships. A prompt memory is a
  preview you are expected to look at.
- **A prompt cannot filter by place.** There is no gazetteer, so a place name
  in a prompt is searched for as a *picture* and nothing else. `christmas in
  midnapur` builds a good Christmas memory across seven Decembers, none of
  which is filtered by where it was taken, and the command says so.

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
