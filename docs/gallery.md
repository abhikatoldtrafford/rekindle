# Making your own gallery

The GIFs in the README are real output. Here is how to produce the equivalent
from your own library, and how the ones in this repo were chosen.

## Make some memories

```bash
uv run rekindle index ~/Pictures
uv run rekindle enrich ~/Pictures     # only if you have a Google Takeout export
uv run rekindle fingerprint           # one-time; enables dedup
uv run rekindle memories              # see what your library could produce
uv run rekindle memory --recipe album_story --key "Kashmir"
```

Every memory lands in `memories/<id>/` as three files:

| File | What it is |
|---|---|
| `memory.gif` | Silent preview, sized to embed in a README or a chat |
| `memory.mp4` | The real thing — higher resolution, with music if you have any |
| `memory.json` | The `MemorySpec`: every photo chosen, and why |

That third file is the point. If a memory surprises you, open it — it records
which recipe ran, which photos were selected, which were rejected and by which
guardrail. A memory you cannot audit is indistinguishable from a bug.

## Publishing one

`memories/` is gitignored on purpose. Photos of your family should not land in
a public repository by accident, so nothing in it is ever staged.

To put one in a README, copy the GIF into `docs/assets/` deliberately:

```bash
cp memories/album_story-kashmir/memory.gif docs/assets/kashmir.gif
```

Then reference it:

```markdown
![Kashmir — 24 photos, May 2015](docs/assets/kashmir.gif)
```

### The public-safe filter

If you are publishing from a library containing other people, build with the
filter rather than trusting yourself to check afterwards:

```bash
uv run rekindle memory --recipe album_story --key "Kashmir" --public-safe
```

A photo qualifies only when its face-tag set is **non-empty and a subset of the
allow-list**. Everything else is refused, including photos with no face tags at
all — because an untagged photo may still contain someone. Default deny.

Two things to understand before relying on it:

- **A memory needs every shot to qualify.** Building normally and filtering
  afterwards will usually leave you nothing; `--public-safe` restricts the pool
  *before* selection, which is why it produces shorter but publishable memories.
- **Face tags come from your source, not from rekindle.** On a Google Takeout
  export they cover roughly half a library. The filter is only as complete as
  those tags, so it is a strong assistant and not a guarantee. Look at what you
  publish.

## Keeping GIFs small

GIF is capped at 256 colours per frame, so photographs band no matter the
resolution, and raising the resolution grows the file fast. For a README:

- fewer frames beats more pixels — a 12-frame teaser reads fine
- keep the preview width modest and let the MP4 carry the real resolution
- if your host renders them, animated WebP or APNG look dramatically better
  than GIF at a fraction of the size

## What is in this repo's gallery

Nothing yet. The samples in the README were generated from a real 19,480-photo
library and are being selected for publication under the rule above. When they
land they will live in `docs/assets/`, and this file will say which recipe and
which command produced each one — so every image in the README can be traced
back to a command you can run yourself.
