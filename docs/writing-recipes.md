# Writing a recipe

A **recipe** decides which photos belong in a memory and what order they go in.
It is the main way to extend rekindle, and it needs no changes to core code.

## What a recipe is not

A recipe does **not** decide timing, transitions, music or prose. The `Timeline`
stage owns timing (so beat-synced music doesn't break every recipe), and the
narrator owns words. A recipe answers exactly two questions: *which photos*, and
*in what order*.

## The protocol

```python
from rekindle.memory.recipes import Recipe, register
from pydantic import BaseModel


class SunsetParams(BaseModel):
    year: int | None = None
    place: str | None = None


@register
class EverySunset(Recipe):
    name = "every_sunset"
    params_model = SunsetParams

    def candidates(self, params: SunsetParams, index) -> list[Photo]:
        """Everything that could plausibly belong. Be generous."""
        return index.search(
            text="a sunset over the horizon",
            year=params.year,
            limit=400,
        )

    def order(self, photos: list[Photo], params: SunsetParams) -> list[PhotoRef]:
        """Pick and sequence. Return only what should appear."""
        best = sorted(photos, key=lambda p: p.quality_score, reverse=True)[:30]
        return [PhotoRef(p.file_hash) for p in sorted(best, key=lambda p: p.taken_at_utc)]

    def fact_sheet(self, ordered, params) -> FactSheet:
        """Facts the narrator may use. Nothing else reaches it."""
        return FactSheet.from_photos(ordered, title_hint="Every sunset")
```

Register via entry points to ship a recipe in your own package:

```toml
[project.entry-points."rekindle.recipes"]
every_sunset = "my_package:EverySunset"
```

## Rules

**Never bypass the index for filtering.** Exclusions and sensitive gating are
applied by `index.search()`. If you read photos some other way, you lose them,
and your recipe can surface material the user explicitly blocked.

**Return fewer photos than you think.** Thirty is a lot for a montage. Sixty is
unwatchable. Quality beats completeness.

**Order deliberately.** Chronological is the safe default, but not always right —
`BeforeAndNow` deliberately juxtaposes across years. Say why in a comment.

**Put only substantiable facts in the `FactSheet`.** The narrator can assert
nothing that isn't there, and an independent verifier rejects claims it can't
check. Adding a speculative field doesn't produce better prose; it produces
rejected prose.

**Respect match tiers.** `Photo.sidecar_match` records how confidently a photo
was paired with its metadata. If your recipe depends on place or date, prefer
high-tier photos — narration is forbidden from asserting place or date derived
from a heuristic match.

## Testing

```bash
uv run pytest tests/recipes/test_every_sunset.py
```

Write two kinds of test:

**Plumbing** — against synthetic fixtures. Does it return a well-formed
`MemorySpec`? Does it handle an empty candidate set, a single photo, photos with
no GPS, photos with no people tags?

**Selection quality** — against the CC0 corpus (`tests/corpus/`, fetched on
demand). Does it actually find sunsets? Declare expectations as
`"prompt X should return at least 8 of these 12 IDs"`.

Synthetic images have no semantics, so plumbing tests alone will pass while your
recipe returns nonsense. Both kinds matter.

> **Note:** the CC0 corpus and eval harness land in M3. Until then, selection
> quality is reviewed by hand — say in your PR what you tested against.

## Ideas nobody has built

*Kids Growing Up* · *Every Sunset* · *This Café Over The Years* · *Seasons In One
Place* · *Everyone Who Came To Dinner* · *The Year You Moved* · *Same Place,
Different Weather* · *First And Last Photo Of Every Trip*
