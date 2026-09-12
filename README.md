# rekindle

**Your photos already remember. This helps them say it out loud.**

[![CI](https://github.com/abhikatoldtrafford/rekindle/actions/workflows/ci.yml/badge.svg)](https://github.com/abhikatoldtrafford/rekindle/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/abhikatoldtrafford/rekindle/blob/main/LICENSE)
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

|  |  |
|---|---|
| <img src="https://raw.githubusercontent.com/abhikatoldtrafford/rekindle/main/docs/assets/2017.webp" width="100%"><br>**2017** · a year in fifteen frames<br>_Himalayas, May → December_<br>`rekindle memory --recipe year_in_review --key 2017` | <img src="https://raw.githubusercontent.com/abhikatoldtrafford/rekindle/main/docs/assets/gopalpur.webp" width="100%"><br>**Gopalpur** · an album becomes a trip<br>_24 photos · August 2025_<br>`rekindle memory --recipe album_story --key Gopalpur` |
| <img src="https://raw.githubusercontent.com/abhikatoldtrafford/rekindle/main/docs/assets/11-june.webp" width="100%"><br>**11 June** · the same date, every year<br>_2023 → 2026_<br>`rekindle memory --recipe on_this_day --key 06-11` | <img src="https://raw.githubusercontent.com/abhikatoldtrafford/rekindle/main/docs/assets/24-december-kolkata.webp" width="100%"><br>**24 December in Kolkata**<br>_10 photos_<br>`rekindle memory --recipe album_story --key "24dec in Kolkata"` |

*Real output from a 19,318-photo library, rendered by the commands beneath them.
Each is the WebP preview; the MP4 carries the full resolution, the music and the
Ken Burns motion. See
[docs/gallery.md](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/gallery.md).*


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
section, not a disclaimer. A video appears only as one extracted still, and
only if ffmpeg is installed. A photographed document can slip through. We
write those down.

**Nothing leaves your machine.** With no `OPENAI_API_KEY` in your environment,
rekindle makes no network calls at all — and it says so out loud on the one
path where a key in your environment is enough to change that. See
[Privacy](#privacy).

---

## 📦 Install

**If you just want to use it**, and have Python 3.12 or newer:

```bash
pipx install rekindle          # or: pip install rekindle
rekindle --version
```

`pipx` is the right tool here — it puts `rekindle` on your PATH in its own
environment, so it cannot collide with anything else you have installed.
Four dependencies, no compiler, no CUDA, no account. Everything below then
works as plain `rekindle ...` with no `uv run` in front of it.

**If you want to work on it**, clone and use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/abhikatoldtrafford/rekindle && cd rekindle
uv sync
uv run rekindle --version
```

Optional extras, in either world:

| Extra | What it adds | Cost |
|---|---|---|
| `heic` | iPhone `.HEIC` photos | ~10 MB |
| `semantic` | Search by description, scene clusters, aesthetics, the face gate, grounded captions — all on CPU | ~120 MB |
| `semantic-gpu` | The same, on an NVIDIA GPU | ~2.5 GB |

```bash
pipx install 'rekindle[semantic]'      # installed copy
uv sync --extra semantic               # source checkout
```

Every command tells you which of these it needs, **phrased for how you
installed it** — a `pipx` user is never told to run `uv sync`. Nothing else
degrades: without any extra you still get the index, all nine recipes, GIF,
WebP and MP4.

> **A CUDA caveat that packaging cannot fix.** PyPI's Windows and Linux
> `torch` wheels are CPU-only; the CUDA builds live on PyTorch's own index.
> This repository points `uv` at it (see `[tool.uv.sources]`), and that
> instruction is a *uv* setting — it cannot be expressed in wheel metadata, so
> `pipx install 'rekindle[semantic-gpu]'` gives you a CPU torch. `rekindle
> semantic doctor` says so in as many words rather than letting you discover
> it as unexplained slowness. To get CUDA from an installed copy, follow
> [pytorch.org](https://pytorch.org/get-started/locally/) and install torch
> yourself into the same environment.

**ffmpeg** is optional and unbundled. Without it you get GIF and WebP but no
MP4, and no still frames from your videos. With it on PATH, both appear.

---

## ⚡ Quick start

```bash
rekindle doctor ~/Pictures    # what metadata do you actually have?
rekindle index ~/Pictures     # build the local index
rekindle enrich ~/Pictures    # read Google Takeout sidecars, if you have them
rekindle fingerprint          # one-time pass; enables dedup and video stills
```

Then make something:

```bash
rekindle memories                     # what could this library produce?
rekindle memory --recipe album_story --key "Kashmir"
rekindle memory --all-recipes         # one of each, from every recipe
rekindle memory --auto                # today's anniversary, if any
rekindle ui                           # edit a memory in your browser
rekindle watch ~/Pictures             # foreground; prints, never renders
```

*(In a source checkout, put `uv run` in front of each.)*

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
> [the M2 spec](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/superpowers/specs/2026-09-11-memories-design.md); the
> wider architecture is in [the v1 spec](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/superpowers/specs/2026-09-10-rekindle-design.md),
> which went through an [independent adversarial review](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/superpowers/specs/audit-v1-resolutions.md).
> How it was actually built, including the bugs that shaped it, is in the
> [decision log](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/decision-log-memory-engine.md). Issues and PRs welcome.

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

**[→ Exporting from Google Photos](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/connecting-google-photos.md)**

### Other libraries

Immich, Apple Photos, Nextcloud and PhotoPrism all expose their libraries
properly, and several give face regions that Google never did. Each is one
`Source` implementation: [writing-sources.md](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/writing-sources.md).


`doctor` writes nothing at all, so it is safe to point at anything. `enrich`
requires `index` to have run first — it reads the database, not the filesystem,
so photo rows must already exist.

### Semantic features (optional)

Search your library by describing a photo, group it by scene, and rank it by
predicted aesthetic quality. These need an optional extra and a one-time
model download; everything afterwards runs with no network at all.

```bash
uv sync --extra semantic-gpu        # torch + CUDA;  --extra semantic for CPU/ONNX
                                    # installed copy: pipx install 'rekindle[semantic]'
uv run rekindle semantic setup      # fetch and checksum the weights, once
uv run rekindle semantic doctor     # which device will actually be used?

uv run rekindle semantic embed                  # embed the indexed photos
uv run rekindle semantic find "snowy mountains" # search
uv run rekindle semantic cluster --untagged-only
uv run rekindle semantic rank --album Kashmir -k 40
uv run rekindle semantic facegate               # propose face-free photos
```

### Memories from your own words (optional)

With the semantic extra installed and the library embedded — or, when the
words name one of your own albums, without either:

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
[festival corpus](https://github.com/abhikatoldtrafford/rekindle/blob/main/src/rekindle/memory/corpus/festivals.toml) and a checked-in
[tag cache](https://github.com/abhikatoldtrafford/rekindle/blob/main/src/rekindle/memory/corpus/prompt_tags.json) — both editable data
files, both working with no API key. Set `OPENAI_API_KEY` and anything they do
not cover is described by a language model once and cached, so the same prompt
gives the same memory forever.

**If one of your own albums answers the question, that is the answer.**
`rekindle memory "memories of kashmir"` finds the `Kashmir`, `Kashmir day 3`
and `Kashmir, day 1 and 2` albums, notices they hold 534 photographs between
them — more than a memory has slots — and builds from those and nothing else.
No descriptions are generated, no search runs, and **no embeddings are needed
at all**, so that command works on a plain `uv sync`. Several albums of one
trip are one memory; the two spellings of `Leh Ladakh` need no configuration,
because matching is on the words and a shorter prompt matches more.

A *small* album is a contribution rather than an answer. `Puri 25` holds three
photographs, which is not a memory however well curated it is, so those three
go into the candidate pool and the search supplies the rest — and the command
says so, with the count, rather than quietly returning three shots. The
threshold, the measurements behind it and two rules that were tried and
rejected are in [the decision log](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/decision-log-prompt-memories.md).
With `OPENAI_API_KEY` set, and only where the words match no album exactly, the
model is shown your album *titles* and asked which are about the same thing —
that is how a misspelling or another transliteration finds the right trip. It
is cached, and it can only ever point at an album you have.

**A prompt memory is a preview, never an offer.** It is built only when you
ask for it by name. It never appears in `rekindle memories` and `--auto` will
never show you one, because **nothing can check that the photos match your
words** — eight statistics have now been tested as a refusal signal on the
reference library and all eight failed. So the command prints what it searched
for, which days it found, the month and year histogram of what it built, and
then says: *look at the memory before you keep it.* The whole measurement is
in [the decision log](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/decision-log-prompt-memories.md).

`semantic setup` is the only command in rekindle that makes a network
request. Every other command loads from the local cache and fails with a
message if something is missing, rather than downloading 1.7 GB you did not
ask for. Model revisions are pinned to exact commits and every file's sha256
is checked against `model-locks.json`; `rekindle semantic licences` prints
the licence of everything rekindle can fetch.

Without the extra, these commands print what to install and exit — nothing
else changes, and the default `uv sync` stays small.

### 🏞️ Memories about a place rather than a person (optional)

```bash
uv run rekindle scenery --list          # what it knows about
uv run rekindle scenery sea mountains   # build two
uv run rekindle scenery --all
```

Every recipe above keys off a person, an album, a date or GPS. **On the
reference library 6,419 photographs — 33.2% — have none of the first three**:
no face tag, no album anyone named, no coordinates. The beaches, the hills,
the flowers, the food, the rain. They can appear inside `year_in_review` and
`on_this_day`, where the subject is the calendar, and they can never be what a
memory is *about*.

A scenery memory makes them the subject. It works the same way a prompt memory
does — a concept becomes visual descriptions, each is searched separately, and
a photograph ranks by how many of them agree — except that the concepts come
from a checked-in, editable
[scenery corpus](https://github.com/abhikatoldtrafford/rekindle/blob/main/src/rekindle/memory/corpus/scenery.toml) rather than from
whatever you typed.

**Every entry in that file was graded by looking at what it returned**, and
the count is written next to it: sea 22 of 24, mountains 24 of 24, flowers 24
of 24, temples 23 of 24, food 22 of 24. The two weak ones are still there,
with what is wrong with them recorded — `rain` at about 13 of 24, and `night`,
which retrieves night photographs correctly and mostly finds that this
library's night photographs did not come out.

The scene clusters `rekindle semantic cluster` already produces are **not**
used for this, and cannot be:
[known-limitations.md](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/known-limitations.md) records cluster #16, 270
photographs of institutional buildings labelled *"a hospital or a clinic"*,
containing no hospital. A curated file is something you can read, disagree
with, and fix.

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
[the decision log](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/decision-log-memory-builder.md).

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
  Timeline               when each shot appears, and for how long
        ↓
  MemorySpec (JSON)  →  GIF + WebP (always, hard cuts)
                     →  MP4 (when ffmpeg is present: dissolves, Ken Burns)
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

### Captions: three layers, each able to veto

A wrong caption on a photograph of your family is worse than no caption, so
`--captions` builds up in strict steps and every step can refuse.

```bash
uv run rekindle memory  --recipe album_story --key Kashmir --captions clip
uv run rekindle scenery sea --captions gpt
```

| mode | what it adds | needs |
|---|---|---|
| `deterministic` *(default)* | the year, the date, or nothing | nothing |
| `clip` | what an image model recognised, from a **closed vocabulary** | the semantic extra + embeddings |
| `gpt` | a language model phrasing those terms and the fact sheet | the above + `OPENAI_API_KEY` |

**1. CLIP grounds it.** Every word that can appear comes from
[a checked-in vocabulary](https://github.com/abhikatoldtrafford/rekindle/blob/main/src/rekindle/memory/corpus/caption_vocab.toml) with
five rules about what may never be in it — no proper nouns, no people or
relationships, no sentiment or occasion, no sensitive context, nothing not
physically in the frame — and the test suite fails the build if you add one.
A photograph is scored against each phrase as a **percentile of that phrase's
own distribution over your whole library**, because comparing "a plate of
food" with "a sandy beach" on one image compares two different queries on two
different scales.

Hand-graded on 72 random photographs in two samples of 36: **25% get a
caption and all of them were right or defensible.** The other 75% are blanks,
which is the correct answer for an indoor portrait. Naming an object needs a
much higher bar than naming a place, and that is measured rather than
assumed — at one threshold for everything, every clear error was an object
("A vehicle by the water", on an empty lake shore).

**2. GPT phrases it.** Off by default. It receives the CLIP terms and the fact
sheet — dates, counts, names, albums, coordinates. **Never the pixels, never a
path, never your library.**

**3. The index vetoes it, per photograph.** A caption may not name a year that
photo does not have or a person not tagged in *that frame* — not merely
someone in the memory. Anything else falls back to the deterministic caption.

Each caption is generated **once per photograph** and cached in your index, so
the same memory rebuilds identically forever and a second run costs nothing.
The cache is personal data and is gitignored with the index.

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

## 🎚️ Tuning it to your photographs

Every threshold rekindle ships was derived by hand from **one** library — a
Bengali family archive spanning 2000–2026. Sharpness 0.12, burst 30 s, dedup
cosine 0.92. Each was measured carefully; none of them knows anything about
your photographs.

So they are visible, and you can re-derive them the same way they were derived:

```bash
rekindle config list                              # every value, and what you changed
rekindle config explain composition.min_sharpness # what it does, and where 0.12 came from
rekindle calibrate                                # derive them from YOUR photographs
```

`rekindle calibrate` is a guided sequence, not a settings page. It shows you
photographs from your own library, chosen near the current cut, and asks a
plain question about each — *"too blurry to use?"*, *"same moment, or two
different photographs?"* You never see a number. You can also pick a number
directly and watch the effect, or just look at what rekindle's default does
and accept it. It runs once; finishing is a real state and it does not ask
again. Stop any time — your answers are saved.

Three things it will always do:

- **Show the consequence before writing.** "This collapses 3,264 MORE
  photographs away as duplicates." Then it writes.
- **Say what a number rests on.** "Rests on 7 judgements, but only ONE of them
  fell on the other side of the line, so the whole threshold is resting on a
  single photograph."
- **Tell you which memories would actually change.** On the way out it re-runs
  *selection* for every memory you have built — no frames, no encoding — and
  diffs the shot lists: *"Your changes affect 14 of 189 memories. 11 gain
  photos, 3 lose one."* Rebuild those, rebuild everything, or nothing.

Your choices land in `rekindle.toml` beside the index. Only what you changed
is in it, so an upgrade improves the values you did *not* choose. It contains
numbers and nothing else — no paths, no names — so it is safe to commit.

Your *answers* land somewhere else, in `labels.jsonl` beside the index, and
that file is append-only: a later sitting adds to it rather than replacing it,
so what you thought of your photographs in September survives a recalibration
in March. **It is personal data and is never committed** — it lives in the
data directory, which `.gitignore` excludes. Nothing reads it. It is not a
model and there is no learned component in rekindle: fitted to the seventeen
labels one real sitting produces, a learned scorer scores exactly at the "say
keep to everything" baseline while replacing 85% of the shots across 25 of 27
memories, and [the measurement is written down](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/decision-log-calibration-labels.md)
so the idea can be reconsidered on evidence rather than re-argued.

**The face gate is different.** It decides what may reach a public repository
and it is calibratable, but tightening it is free and widening it needs a
separate confirmation in which you type out, in plain words, that more
photographs of other people will become publishable. A drag of a slider can
never widen it.

The same flow is in `rekindle ui`, where you see the photographs in the page
rather than opening the paths it prints.

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
no audio downloaded, and your original files are never modified.

Exactly two things send anything to a language model, and **both need
`OPENAI_API_KEY` set in your environment**. Unset it and neither can run.

| Path | Trigger | What is sent |
| --- | --- | --- |
| Captions | `--captions gpt` | A fact sheet for one memory — dates, counts, names, albums, coordinates. |
| Prompt memories | a prompt like `rekindle memory "photos of puri"` — **no flag** | Your phrase, your album titles, the first and last year, and two counts. |

Never the pixels, never a file path. The prompt path is reached by the
environment variable alone, so it prints a line saying what it is about to
send before the first request leaves. Model weights are downloaded once by
`rekindle semantic setup`, and `rekindle fetch` downloads music only when you
ask it to. See [SECURITY.md](https://github.com/abhikatoldtrafford/rekindle/blob/main/SECURITY.md).

## Contributing

Two high-value contributions, neither requiring core changes:

- **A new memory type** — one file implementing one protocol:
  [writing-recipes.md](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/writing-recipes.md)
- **A new photo source** — Immich, Apple Photos, Nextcloud:
  [writing-sources.md](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/writing-sources.md)

See [CONTRIBUTING.md](https://github.com/abhikatoldtrafford/rekindle/blob/main/CONTRIBUTING.md).

## License

MIT
