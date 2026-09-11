# Writing a recipe

A **recipe** decides which photos belong in a memory and what order they go in.
It is the main way to extend rekindle, and it needs no changes to core code.

> **This document was rewritten in M2 to describe the protocol that actually
> exists.** The previous version specified `candidates()`/`order()`/
> `fact_sheet()` over an `index.search(text=...)` with `pydantic` parameter
> models. None of that shipped: pydantic is not a dependency, and text search
> needs embeddings that land in M3. A contributor guide describing a fiction is
> a bug this project has shipped once already.

## What a recipe is not

A recipe does **not** decide timing, transitions, music, the cap, dedup, or the
guardrails. The engine owns all of those and every recipe passes through it, so
a recipe *cannot* forget to dedup, *cannot* exceed the shot cap, and *cannot*
surface a photo the user excluded.

A recipe answers exactly two questions: *which memories could this library
produce*, and *which photos go in one of them*.

## The protocol

```python
from rekindle.memory.recipes import Offer, Selection, register
from rekindle.memory.recipes.base import CHRONOLOGICAL, chronological
from rekindle.memory.spec import build_fact_sheet


@register
class EverySunset:
    name = "every_sunset"      # stable: it is half of every memory id
    title = "Every sunset"     # human label for `rekindle memories`

    def offers(self, index) -> list[Offer]:
        """Every memory this recipe could build from this library.

        Cheap and total: metadata only, no pixels. `rekindle memories` and
        `--auto` both call this, so it must be fast and it must be
        deterministically ordered.
        """
        out = []
        for year in index.years():
            photos = index.by_year(year)
            if len(photos) < 20:
                continue
            out.append(
                Offer(
                    recipe=self.name,
                    key=str(year),          # STABLE - see "Memory identity"
                    title=f"Sunsets of {year}",
                    subtitle=f"{len(photos)} photos",
                    size=len(photos),
                )
            )
        return sorted(out, key=lambda o: (-o.size, o.key))

    def select(self, index, offer) -> Selection | None:
        """The photos for ONE offer. Generous and ordered.

        Return None when the offer no longer yields enough - it may have been
        built against a larger library.
        """
        photos = index.by_year(int(offer.key))
        if len(photos) < 3:
            return None
        ordered = chronological(photos)
        return Selection(
            photos=ordered,
            facts=build_fact_sheet(ordered, title=offer.title, recipe=self.name),
            ordering=CHRONOLOGICAL,
            captions={p.file_hash: str(p.meta.taken_at_local.year) for p in ordered},
        )
```

Ship a recipe in your own package with an entry point:

```toml
[project.entry-points."rekindle.recipes"]
every_sunset = "my_package:EverySunset"
```

## What the index gives you

`MemoryIndex` is the **only** way to obtain a photo, and everything it returns
has already passed every guardrail. It has no `search(text=...)`: selection in
v1 is structured queries over real metadata.

| Method | Returns |
|---|---|
| `all()`, `images()`, `count()` | every allowed photo |
| `by_year(y)`, `by_month(m)`, `by_month_day(m, d)` | temporal slices |
| `by_person(name)`, `by_pair(a, b)` | face tags (order-independent pairs) |
| `by_album(title)` | album membership, after any configured aliases |
| `by_gps_cell(cell)` | a 0.25° GPS cell |
| `years()`, `months()`, `month_days()` | what exists, for building offers |
| `people_counts()`, `pair_counts()`, `album_counts()`, `gps_cells()` | counts |
| `earliest(photos)`, `latest(photos)` | chronological ends, tie-broken stably |
| `resolve_path(photo)` | the first path that exists on disk, or None |
| `is_public_safe(photo)` | the publish rule |

## Rules

**Never reach past the index.** It is handed to you already filtered:
archived and trashed photos, excluded people, excluded date ranges, albums and
paths, and (in public-safe mode) anything not publishable are all gone before
you see them. Importing `PhotoStore` yourself bypasses every one of those, and
it is the one thing a recipe must not do.

**Make `key` stable.** It becomes half of the memory id (`recipe:key`), which
is what dismissal and the resurfacing cooldown are keyed on. Derive it from the
memory's *defining facts* - an album title, a person's name, a month-day - and
never from the set of photos in it. A key derived from contents means one new
photo produces a "new" memory and a user's dismissal silently stops working.

**Be generous in `select`.** The engine applies composition guardrails, burst
dedup, stratification, content diversity, ranking and the cap afterwards.
Returning 400 photos is normal; the engine keeps the best 24 distinct ones.

**Declare what your memory is ABOUT.** `Selection.stratify` names the dimension
across which shots are spread — `BY_YEAR` for anything whose premise is
spanning time, `BY_MONTH` for a single year, `BY_SPAN` (adaptive) for an album
or a trip, or `None` for a recipe that wants the extremes rather than a spread.
Without it, selection collapses onto whichever period happens to photograph
best: before this existed, 16 of 37 rendered memories were confined to a single
year. If spanning time is your recipe's whole point, also set `min_strata=2` so
it is refused rather than silently narrowed when the gates leave one period
standing.

**Order deliberately.** `CHRONOLOGICAL` is the safe default and the engine
restores it after the cap. Use `AS_GIVEN` only when the sequence *is* the
content - `then_and_now` juxtaposes two photos and would be destroyed by a
re-sort. A recipe whose form is a fixed, smaller number of shots sets
`Selection.min_shots`.

**Put only substantiable facts in the `FactSheet`.** It is the complete input
to the optional GPT caption layer, so a speculative field is a licence to
hallucinate. It carries no filesystem paths, and it carries coordinates rather
than place names - there is no offline gazetteer in this project, so a city
name would be invented.

**Never invent a title.** Every string must be copied from metadata or computed
arithmetically from it. If a photo has no GPS, the memory says nothing about
where it was.

**Return None rather than raising.** `rekindle memory --recipe X --key Y`
builds an `Offer` by hand from the command line, so `select` can be called with
a key that no longer exists.

## Testing

```bash
uv run pytest tests/test_recipes.py -k every_sunset
```

Write these, at minimum - they are the ones `tests/test_recipes.py` applies to
every built-in recipe automatically:

- An **empty index** yields no offers.
- Offers are **deterministically ordered** across two calls.
- `select` on a **stale offer** returns None rather than raising.
- Every offer has a **non-empty title** and a stable `memory_id`.
- Your thresholds are respected **at the boundary**: n-1 excluded, n included.

Then mutation-test them: break the line each test protects, watch it fail,
restore it. A test you have not seen fail is not a test. Two real examples from
M2 - a fixture whose "distinct" hashes were 2 bits apart, and an assertion on
the word `fingerprint` that was satisfied by pytest's own temp directory name -
both passed while protecting nothing.

**Fixtures must reproduce shapes you have actually observed.** Inventing a
convention and pinning it with a test is the most expensive mistake available
in this codebase; it has happened three times. If you have a Takeout export,
run the conformance suite (see [CONTRIBUTING.md](../CONTRIBUTING.md)).

## Adding a similarity signal

Selection avoids showing the same picture twice by penalising candidates that
look like something already chosen. Today that judgement comes from pixel
statistics — a perceptual hash and a colour histogram — and those cannot see
*semantic* redundancy: six restaurant-table photos from six different days are
different pixels and the same idea.

That is the highest-value contribution available here, and it needs no changes
to selection. Implement the protocol in `rekindle.memory.diversity`:

```python
class DissimilaritySignal(Protocol):
    name: str
    weight: float

    def between(self, a: Photo, b: Photo) -> float | None:
        """0.0 indistinguishable, 1.0 unrelated, None when you cannot judge."""
```

Three rules, all load-bearing:

- **Return `None`, never a guess,** when the data you need is missing. None is
  neither 0 nor 1: your signal abstains and the others decide. Returning 0
  would let one photo without an embedding suppress its neighbours.
- **Be deterministic.** The same library must produce the same memory, byte for
  byte, and there is a test that asserts it.
- **Never raise.** A signal that throws takes down a render.

Then add it to `CompositeSignal.signals` with a weight. `test_diversity.py`
substitutes a custom signal to prove the seam works; copy that test.

## Ideas nobody has built

*Kids Growing Up* · *Every Sunset* · *This Café Over The Years* · *Seasons In One
Place* · *Everyone Who Came To Dinner* · *The Year You Moved* · *Same Place,
Different Weather* · *First And Last Photo Of Every Trip*
