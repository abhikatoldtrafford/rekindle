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
`Timeline` stage does, so beat-synced music doesn't break every recipe. That
stage now exists (`memory/render/timeline.py`) and nothing above it had to
change to get crossfades, Ken Burns or beat-snapped cuts. Full guide:
[writing-recipes.md](docs/writing-recipes.md).

A recipe also cannot reach the semantic layer, which is why **scenery memories
are not a recipe**: selection needs a retriever, `Recipe.select` has nowhere to
receive one, and CI has no embeddings at all. They hand the engine a
`Selection` directly, exactly as prompt memories do, and every guardrail below
`engine.build` still applies. See `memory/scenery.py`.

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

### 4. A corpus entry (the smallest useful contribution)

Three checked-in data files decide what rekindle can find and what it may say,
and none of them needs a line of code:

- `corpus/festivals.toml` — a festival, and what a photograph of it looks like.
- `corpus/scenery.toml` — a scene that can be the subject of a memory.
- `corpus/caption_vocab.toml` — a thing a caption is allowed to mention.

**Grade what you add.** Every entry in the first two carries the count someone
got by looking at a contact sheet of what it returned, and the weak entries
say what is wrong with them rather than being quietly dropped. A description
that "seems right" is how nine of twenty-four shots of a Kali Puja memory
ended up on a Durga Puja day. The caption vocabulary is stricter still: five
rules about what may never be in it, all enforced by
`tests/test_caption_vocab.py`, which fails the build rather than trusting a
comment.

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

### Packaging

```bash
uv build                      # wheel + sdist into dist/
uv run pytest tests/test_packaging.py
```

`tests/test_packaging.py` checks the wheel's contents only when `dist/` holds
one, so build first or the interesting half silently skips.

**Publishing is not automated and no agent does it.** See
[docs/publishing.md](https://github.com/abhikatoldtrafford/rekindle/blob/main/docs/publishing.md)
for the single command, what it irreversibly does, and the checklist before it.

### The semantic extras

`rekindle semantic ...` (embeddings, search, scene clusters, aesthetic
ranking, the face gate) lives behind two optional extras, and **CI installs
neither**:

```bash
uv sync --extra semantic       # numpy + onnxruntime, CPU, ~120 MB
uv sync --extra semantic-gpu   # adds torch + transformers, CUDA, ~2.5 GB
```

You do not need either to work on rekindle, or to run the tests. The semantic
tests use `tests/fixtures/semantic.py`, whose `ToyEncoder` is a real (tiny)
joint image/text embedding function over a colour grid — not a mock. If you
add a test there, assert the *answer*, not that the encoder was called: a test
that mocks the model and checks the mock was called proves nothing.

Three rules for anything under `src/rekindle/semantic/`:

- **No heavy import at module level.** `rekindle --help` must not import
  numpy, torch, onnxruntime or transformers.
  `tests/test_semantic_imports.py` checks this in a clean subprocess.
- **Every feature degrades to a sentence**, never an ImportError, and exits 3.
- **Model weights are pinned to a commit and checksummed.** Adding a model
  means adding a `RepoPin` with a 40-hex revision, a licence with a URL, and
  an entry in `model-locks.json` (`rekindle semantic setup --write-lock`).

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
none by reading. The memory engine repeated the pattern: two of its bugs were
invisible to 700 passing tests and obvious within one run against real data.

If you have an index, the memory engine has its own conformance suite:

```bash
REKINDLE_MEMORY_DB=data/rekindle.sqlite uv run pytest \
    tests/test_memory_conformance.py -v
```

Both decision logs record the defects, the judgement calls and the testing
discipline they led to:
[Takeout enrichment](docs/decision-log-takeout-enrichment.md) ·
[the memory engine](docs/decision-log-memory-engine.md).

## Pull requests

Keep them focused, explain the why, and add tests. If you're planning something
large, open an issue first so we can agree on the shape before you build it.
