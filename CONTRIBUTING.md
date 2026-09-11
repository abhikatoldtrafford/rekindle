# Contributing to rekindle

Thanks for considering it. This project is meant to be extended.

## The three extension points

Almost every contribution fits one of these, and none of them require touching
the core engine.

### 1. A new memory type (best first contribution)

A `Recipe` decides which photos belong in a memory and how they are ordered.
One file, one protocol, one registry entry:

```python
class Recipe(Protocol):
    name: str
    params_model: type[BaseModel]

    def candidates(self, params, index) -> list[Photo]: ...
    def order(self, photos, params) -> list[PhotoRef]: ...
    def fact_sheet(self, ordered, params) -> FactSheet: ...
```

Recipes choose photos and their order. They do **not** own timing — a separate
`Timeline` stage does, so beat-synced music doesn't break every recipe. Full
guide: [writing-recipes.md](docs/writing-recipes.md).

Ideas nobody has built yet: *Kids Growing Up*, *Every Sunset*, *This Café Over
The Years*, *Seasons In One Place*, *Everyone Who Came To Dinner*.

### 2. A new photo source

`Source` normalises any library into `Photo` records. Apple Photos, Immich,
Nextcloud, Synology Photos and PhotoPrism are all wanted. One protocol, two
members. Full guide: [writing-sources.md](docs/writing-sources.md).

```python
class Source(Protocol):
    name: str

    def scan(self, root: Path) -> tuple[list[Photo], SourceReport]: ...
```

### 3. A new model backend

`Embedder`, `Captioner` and `Narrator` are swappable. Ollama, llama.cpp,
Gemini and local VLMs all fit.

## Ground rules

**Never commit personal data.** No photos, no Takeout exports, no `.env`. The
`.gitignore` is deliberately aggressive — please keep it that way. It ignores
the usual export folders (`Takeout/`, `Data/`, `Photos/`, `Thumbnails/`) **and**
every common photo and video extension at any depth — `.jpg`, `.JPG`, `.heic`,
`.dng`, `.CR2`, `.mp4`, `.MOV`, `.MP` and friends — each spelled with
`[Jj][Pp][Gg]`-style brackets so it still holds on a case-sensitive filesystem,
where an export extracted outside those folders would otherwise have no
protection at all. This repo is public and that mistake cannot be undone. If you
genuinely need to commit a media test asset, `git add -f` it and add an explicit
`!` negation beside the rule.

**Tests must run without photos, without a GPU, and without API keys.** CI has
none of those. Use the synthetic fixture generator in `tests/fixtures/` and the
`Fake*` providers. If your change needs a real model to be tested, it needs a
fake too.

**Cross-platform.** Windows, macOS and Linux are all supported. Use `pathlib`,
never hardcode separators or drive letters, and don't assume ffmpeg's location.

**Found a new Takeout quirk?** That's a genuinely valuable bug report. Please
include the filename shape and what `rekindle doctor` said — and if you can,
add a fixture case reproducing it.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
```

### Testing against a real Takeout export

Two of this project's three worst bugs were invisible to a green test suite and
obvious within one run against real data. If you have a Google Takeout export:

```bash
REKINDLE_TAKEOUT_DIR="/path/to/Takeout/Google Photos" uv run pytest \
    tests/test_takeout_conformance.py -v
```

Those tests skip without the variable, so CI never needs your photos. Never
commit an export, or any file from one.

Every genuine bug in the Takeout enrichment milestone was found this way and
none by reading. If you want to know what that means in practice before writing
a patch, [docs/decision-log-takeout-enrichment.md](docs/decision-log-takeout-enrichment.md)
records the defects, the judgement calls and the testing discipline they led to.

## Pull requests

Keep them focused, explain the why, and add tests. If you're planning something
large, open an issue first so we can agree on the shape before you build it.
