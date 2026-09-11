# Decision log — the interactive memory builder (M5)

`rekindle ui`: a local web page where you type a prompt, watch the photographs
arrive, edit the selection by hand, and render a memory — and `rekindle
render`, the command that rebuilds by hand what you just made.

This is the record of how it was actually built: the architecture and the rule
that shapes it, the numbers, the controls that were cut and why, the two
defects running it against the real library turned up, the fifty mutations run
against its own tests, and the places where the brief this work came from
turned out to be wrong.

Every measurement below was taken against the live library — 19,480 indexed
rows, 19,318 after the guardrails, 18,201 CLIP ViT-L/14 vectors — on the
reference machine, with the ONNX CPU runtime. Nothing is carried forward from
an earlier document without re-measurement.

---

## The rule everything else follows from

> **The UI selects nothing. It is a lens onto the engine, and the engine is
> reached through the same chokepoint the CLI goes through.**

Concretely, and each of these is pinned by a test rather than by convention:

* Every candidate list comes from `recipe.select` or `prompt.build_selection`.
* Every guardrail count comes from `composition.compose` and `dedup.collapse`.
* Every chosen shot comes from `engine.build`.
* Every photo the page can display was resolved through `MemoryIndex`.
* Every edit ends as a `MemorySpec` on disk plus one command that rebuilds it.

The failure this rule exists to prevent is specific. A UI that re-implements
"which photos are good" drifts from the CLI in a month, and the first symptom
is a photograph someone asked never to see appearing in a browser tab. So
there is no selection code in `rekindle/web/` at all — no scoring, no date
filtering, no dedup rule, no second renderer.

### The chokepoint, restated for HTTP

`MemoryIndex.open` loads every row, applies the policy once, and keeps only
the survivors. Nothing rejected is ever stored on the object. The web package
inherits that by never opening `PhotoStore` to find a photo:

| Route | Lookup | What a withheld photo gets |
|---|---|---|
| `GET /api/thumb/<hash>` | `MemoryIndex.get` | 404, identical to a hash that never existed |
| `POST /api/edit` `add` | `MemoryIndex.get` | refused, with the reason |
| `GET /api/search` (either mode) | `MemoryIndex.resolve_many` | absent from the results |
| `GET /api/similar` | `MemoryIndex.resolve_many` | absent |
| `GET /api/day` | `MemoryIndex.by_date` | absent |
| `rekindle render` | `MemoryIndex.get` via `build_frames` | dropped from the render, and counted |

The last row is the one worth dwelling on. A `memory.json` is a durable
artefact that outlives the session that made it. Excluding a person tomorrow
must remove them from every spec already written, **without editing a single
file** — and it does, because the spec names hashes and the renderer resolves
them through the index every time. `test_a_spec_naming_an_excluded_photo_
renders_without_it` writes a spec, excludes the person, re-renders, and
asserts the frame count fell by exactly the number of shots that person is in
while the spec on disk is unchanged.

---

## Architecture

```
  browser (one static page, no build step)
        |  fetch + EventSource, same origin, token on every request
  server.py      stdlib http.server: routing, token, CSP, SSE, byte ranges
        |
  api.py         route -> plain function -> plain data. No HTTP knowledge.
        |
  draft.py       the editable memory: order, removed, added, pace
        |             |
  library.py     renderer.py
   MemoryIndex    build_frames -> write_webp / write_gif / write_mp4
        |
  ================ everything below is M1-M4, unchanged ================
  engine.build  composition.compose  dedup.collapse  strata.stratify
  prompt.build_selection  tags.resolve  MemoryIndex  PhotoStore
```

`api.py` is the seam that makes the whole thing testable. Every route is a
function of `(Workshop, arguments)` returning data, and `server.py` does
nothing but parse a request into those arguments and serialise the answer. So
`tests/test_web_api.py` drives the entire prompt path — parse, tag resolution,
consensus, seed days, engine build — with no socket, no browser and no event
loop, and `tests/test_web_server.py` separately pins the HTTP behaviour over a
real socket.

### Why the build is a generator

`api.build_events` yields `{"event": ..., "data": ...}` and `server.py` turns
those into `text/event-stream` frames. That shape was chosen because it is the
shape the streaming requirement actually has, and because a generator is
consumed identically by an SSE writer and by a test's `list()`.

Measured over the wire on the real library, reading the response with
`read1` — `HTTPResponse.read(n)` blocks until *n* bytes exist and collapses a
stream into one blob, which is how a "first event" timing becomes a lie:

| Stage of `durga puja over the years` | Arrives at |
|---|---|
| parse the prompt | 0.03 s |
| four visual descriptions, from the festival corpus | 0.03 s |
| load the CLIP text encoder — once per process | 3.82 s |
| the four searches, reported one at a time | 3.97 s |
| 16 capture days agreed, expanded to 799 candidates | 3.97 s |
| the 24 chosen shots | 4.46 s |

Without streaming that is four and a half seconds of blank page, and **3.8 s
of it is loading a model** — a stage a user cannot be told about at all if the
answer only arrives at the end. With the encoder already warm the whole build
is **0.67 s**, so the streaming matters most on exactly the request where a
progress bar is worth having.

The injected-retriever design in `memory/prompt.py` is what makes per-tag
progress possible: `build_selection` takes retrieval as a callable, so the UI
passes a wrapper that records each search. The consensus arithmetic never sees
the wrapper.

### Why the index loads once, in a thread

`MemoryIndex.open` over 19,480 rows takes **1.86 s** (median of three; min
1.58 s). Per request that is unusable; at startup it is invisible, because the
page itself is static and is served while the load is still running.
`/api/status` returns `{"state": "loading"}` and the page polls until it is
ready — the first HTTP request in the measured run came back `loading` and the
index was ready 1.8 s later.

SQLite connections belong to the thread that made them, so nothing holds one
open: the index is built in the loader thread and the store closed
immediately, and the two operations that write — recording that a memory was
surfaced, and dismissing one — open a fresh `PhotoStore` inside the calling
thread. The index itself is immutable in-memory data, so reader threads share
it without a lock.

---

## Dependencies: none

This is the largest departure from the brief, which asked for "a new optional
extra, like `semantic`" whose absence prints an install command. There is no
such extra, because there is nothing to install. Measured with this project's
own resolver on 2026-09-11:

| Option | Packages added |
|---|---|
| `fastapi` + `uvicorn` | 12 |
| `flask` | 7 |
| `starlette` (+ an ASGI server to run it) | 4 (+3) |
| `bottle` | 1 |
| **`http.server`** | **0** |

`rekindle` ships four runtime dependencies. What this page needs is routing,
JSON, static files, byte ranges and server-sent events — 188 lines of
`server.py` — and none of what a framework is for: there is no deployment
story, no untrusted input from the internet, no templating (the page is one
static HTML file), and the concurrency model is "a thread per connection,
because there is one user".

The argument that settled it is not the package count. **A framework behind an
optional extra means the server tests skip on every machine that does not
install it**, and this project's own `pyproject.toml` already states the
position, about numpy:

> A skipped test is not a passing test, and those three modules are where the
> arithmetic that can be silently wrong lives.

The guardrail tests in `tests/test_web_server.py` are exactly that kind of
code. They run on a plain `uv sync`, on all six CI legs, with no extra.

The frontend is the same decision: 560 lines of ES2020 in one file, no build
step, no bundler, no framework, no CDN. A page whose whole promise is that it
makes no network request cannot load a font from Google, and a project whose
install is four packages should not acquire a `node_modules`.

**The degradation requirement did not disappear** — it moved to where the
dependency actually is. Prompt search needs the `semantic` extra and an
embedded library, and without either the page still opens, still lists every
recipe memory, still edits and renders. The prompt box reports:

> Prompt search needs the semantic extra: `uv sync --extra semantic`, then
> `rekindle semantic embed`. Everything else on this page works without it.

`tests/test_web_api.py::test_a_prompt_without_the_extra_says_what_to_install`
and its HTTP twin pin that by making `availability.probe` report nothing
installed and asserting the install command comes back.

---

## Security: four rules, because these are someone's family photographs

Loopback is not a permission boundary. Any process and any other user on the
machine can open a socket to 127.0.0.1, and a page on the public internet can
make a browser send requests there.

1. **The socket binds `127.0.0.1` and there is no option to change it.** Not a
   default — `build_app` has no `host` parameter. A flag that exposes a photo
   library on a LAN is a foot-gun with no good use, and
   `test_there_is_no_option_to_bind_anywhere_else` asserts on the signature so
   it cannot be added by accident.
2. **Every route needs a per-run secret**, printed in the URL, compared with
   `secrets.compare_digest`. `<img>` cannot set a header, so thumbnails accept
   it as `?t=`; everything else sends `X-Rekindle-Token`.
3. **`Host` must name the loopback** — the DNS-rebinding defence. A page at
   `photos.example.com` that resolves its own name to 127.0.0.1 gets 403.
4. **`Origin` must be absent or loopback on anything that changes state**,
   with `Sec-Fetch-Site` as the backstop for a request carrying neither.

Plus, on every response: `Content-Security-Policy: default-src 'self'` — so a
bug in `app.js` still cannot fetch a script, a font or a tracking pixel from
anywhere — `Referrer-Policy: no-referrer`, `X-Content-Type-Options: nosniff`,
`frame-ancestors 'none'`.

Two smaller ones worth naming:

* **Request logging is off by default.** A request log of this server is a
  record of which of someone's photographs they looked at, printed to a
  terminal they may be sharing. `--verbose` turns it on.
* **Static assets and rendered outputs are allow-lists, not path joins.** The
  page may fetch `app.css` and `app.js` and nothing else; a session's outputs
  may be `memory.{json,webp,gif,mp4}` and nothing else. The traversal test
  aims at a file that genuinely exists — the index database, two directories
  up from a memory folder — because a traversal to a path that happens not to
  exist proves nothing.

---

## The four interaction modes, and the one that was cut

### 1. Review and remove — the core loop

Every shot is shown with **why it is there**: its caption, its date, whether
it was chosen by the engine or added by you, and whether it is public-safe.
Removing it moves it to the rejected pile under `removed_by_you`, where it can
be put back.

`Draft.order` is the single source of truth, and `to_spec()` reads it
verbatim — no re-sorting, no re-ranking. That is the whole mechanism behind "a
dropped photo stays dropped", and mutating `to_spec` to re-impose chronological
order kills two tests.

### 2. Add photos — see what was rejected, and overrule it

The rejected pile is every photo the recipe considered that something removed,
grouped by reason. On the real `album_story:Kashmir` (510 candidates):

| Reason | Photos | Example the page shows |
|---|---|---|
| `not_enough_slots` — survived everything, lost the 24-shot cap | 475 | |
| `out_of_focus` | 4 | `Jammu and Kashmir 21st May 2015 052.JPG` |
| `near_duplicate` — collapsed into another frame of the same burst | 3 | |
| `video` — never in a memory in v1 | 2 | `…21st May 2015 224.MOV` |
| `too_dark` | 1 | `Jammu and Kashmir 21st May 2015 068.JPG` |
| `minority_orientation` | 1 | `Jammu and Kashmir 21st May 2015 335.JPG` |

The shape of that is worth reading rather than skipping: **the guardrails
reject almost nothing here — 8 photos of 510 — and the cap rejects 475.** The
panel a user spends their time in is therefore the one full of perfectly good
photographs that simply did not fit, not the one full of rejects. Six of the
510 are members of a burst with alternates to swap between.

The per-photo reason is a problem the engine does not solve: `compose` reports
counts, not verdicts. Rather than copy its rule ladder — which would drift the
first time a threshold moved — `draft.composition_reason` asks `compose` itself
about a one-photo list with `enforce_orientation=False`. That is valid because
orientation is a property of the *set* and a set of one is always its own
majority, so the report from that call contains exactly the per-photo verdict;
a photo that survives it but was dropped by the real call can only be the
minority orientation.

`test_catalogue_reasons_reconcile_with_the_engine` then compares the histogram
of displayed reasons against `CompositionReport.dropped`, which
`engine.build` produced independently. Two derivations of the same fact; if
either drifts, the test fails.

Beyond the rejected pile there are three ways to reach the rest of the
library: metadata search (album, person, description, filename, ISO date —
**no model, no extra, works everywhere**), semantic search when the extra is
installed, and the two cluster controls below.

### 3. Reorder and pace — and why per-shot holds were cut

Drag to reorder, using native HTML5 drag-and-drop rather than a sortable
library. The browser sends the whole new order and the server refuses anything
that is not a permutation of the current one — a reorder carrying an extra
hash is an `add` in disguise, and `test_reorder_does_not_admit_a_new_photo`
pins that.

**Per-shot hold times are deliberately absent.** The brief asks for "adjust how
long each photo holds", and the honest implementation is one hold for the
memory, not one per photo. The reasoning is a chain:

1. The reproduce command has to rebuild what was made. The mechanism is the
   `MemorySpec`, which is the project's durable artefact.
2. `Shot` has four fields and `SPEC_VERSION` is 1. A per-shot duration has to
   live in the spec or it cannot be reproduced.
3. Bumping `SPEC_VERSION` invalidates every `memory.json` already on disk —
   `MemorySpec.loads` raises on a version it does not know.

A control whose result cannot be handed back as a command is worse than no
control. So the pace is `--frame-ms` and `--title-ms`, which all three writers
already accept and which the reproduce command carries. `test_the_pace_reaches_
the_file` reads the frame durations back out of the GIF rather than asserting
the parameter was passed.

Music is chosen by **name** from the music folder and resolved against its
listing, never joined as a path — so `../../data/rekindle.sqlite` is not a
track.

### 4. Dedup and cluster controls

`dedup.bursts(usable)` is called with the same defaults the engine uses, so
the burst grouping displayed *is* the burst grouping applied. Each collapsed
frame says which frame was kept instead, and "use this frame instead" swaps in
place — implemented as remove-then-add at the same index, so it cannot bypass
the guardrail in `add`.

"Pull in a whole scene cluster" became two controls rather than one, because
the obvious implementation does not exist:

* **`rekindle semantic cluster` does not persist anything.** It runs spherical
  k-means and prints; there is no cluster id in the store to look up. Measured
  on this library: `spherical_kmeans(18,201 vectors, k=95)` takes **4.3 s** and
  41 iterations to converge. That is not a click.
* **"More like this"** — a nearest-neighbour query over the same vectors —
  answers the same question for the photo in front of the user in **2 ms**,
  two thousand times cheaper, and needs nothing stored. It is one cosine pass
  through `SemanticSearch.search_vector`; `similar_to` is deliberately not
  used, for the thread-safety reason written up below.
* **"The whole capture day"** (`MemoryIndex.by_date`) needs no model and no
  extra at all, and is what actually matters after a burst — "give me the rest
  of that afternoon". It is the same call the prompt path uses to expand a
  seed day, so the UI and the engine mean the same thing by "that day".

---

## The reproduce command

The UI hands back two lines. The provenance line says where the memory came
from:

```
rekindle memory --recipe album_story --key Kashmir
```

The reproduce line rebuilds what was made by hand:

```
rekindle render memories/2026-09-12-album_story-kashmir/memory.json --frame-ms 1100
```

It is deliberately **not** `rekindle memory --recipe … --key …`. That re-runs
selection, and selection is precisely what the user has just overruled. The
spec *is* the edit, so the command that reproduces the edit is the one that
renders the spec.

Three properties make that a real promise rather than a printed string:

* **`rekindle render` is a first-class CLI verb.** It imports nothing from
  `server.py`, opens no socket, needs no browser.
* **It defaults `--out` to the spec's own folder**, so re-running it rewrites
  the memory instead of minting a new dated folder beside it. A reproduce
  command that reproduces somewhere else is not one.
* **It does not rewrite the spec it was given.** An input a command overwrites
  is not an input — and a user who hand-edits a caption would otherwise find
  it silently normalised away.

`test_the_reproduce_command_rebuilds_the_same_files` edits a memory the way a
person would (drop two shots, reverse the rest, change the pace), renders it
through the API, copies the output, runs the printed argv through the real
CLI, and compares `memory.json`, `memory.webp` and `memory.gif` byte for byte.

Measured on the real `album_story:Kashmir` memory: **identical**, all three
files, across the UI render and the CLI render.

---

## Thumbnails

46 GB of originals is not a page load. Measured over the 18,201 indexed
images: median short edge 2,976 px, mean file 2.53 MB, median 1.29 MB, 46.1 GB
in total. A 200-thumbnail grid decoded from originals on every visit is half a
gigabyte of reads and, at the rate below, twenty seconds of CPU.

The cache is keyed by `file_hash` and width — **not by path**, because the same
bytes appear under several paths in a Takeout export and a path-keyed cache
would duplicate every one of them and miss on a folder rename. Two widths
only (320 for the grid, 900 for the detail sheet), so a hostile or buggy query
string cannot fill the disk with four thousand sizes of one photo; the route
clamps and the cache refuses independently.

Measured over HTTP, 24 real shots from `album_story:Kashmir`, cache emptied
first:

| | Total | Per thumbnail |
|---|---|---|
| cold (decode + encode + write) | 1.58 s | 66 ms |
| warm (read from disk, over the socket) | 0.32 s | 13 ms |

Mean cached thumbnail: **21 KB**. A 200-photo grid is therefore about 4.2 MB
warm, against roughly 507 MB of the originals it stands for — a factor of 120.

Decoding goes through `meta.exif.open_upright`, which is the one place in this
project that turns pixels the right way up. That is not defensive
boilerplate — a missing transpose at two `semantic` decode sites cost the face
gate 14.5% of its verdicts on rotated photos, and the symptom in a thumbnail
grid would be a wall of phone photographs lying on their sides.
`test_web_thumbs.py` asserts the *pixels* for all eight EXIF orientations,
using the quadrant-marker fixture, because a size assertion cannot see four of
the eight tag values at all.

---

## What was measured

Against the live library — 19,480 indexed rows, 19,318 after the guardrails,
18,201 CLIP ViT-L/14 vectors — on the reference machine. The text encoder ran
on **ONNX CPU**, not the RTX A4000: the `semantic` extra rather than
`semantic-gpu`, which is the cheaper install and the honest floor for these
numbers. Medians of three or five runs; the script is in the branch history.

| | |
|---|---|
| Photos after the guardrails | 19,318 |
| Withheld | 162, all `archived` |
| `MemoryIndex.open`, 19,480 rows | 1.86 s (min 1.58) |
| `/api/offers`, every recipe | 1.68 s, **490 offers** |
| `album_story:Kashmir`, offer to editable draft | 1.79 s (min 1.18) |
| …of which candidates / shots | 510 -> 24 |
| One edit, including the whole session payload back | **4 ms** |
| First SSE event, recipe build | < 1 ms |
| First SSE event, prompt build | 16 ms |
| Load the CLIP text encoder and the matrix, once per process | 2.78 s |
| One text query over 18,201 vectors | 23 ms |
| Nearest neighbours of one photo | 2 ms |
| Metadata search `kashmir` over 19,318 rows | 42 ms |
| `durga puja over the years`, cold (includes the encoder load) | 4.5 s |
| `durga puja over the years`, encoder warm | 0.67 s |
| …of which seed days / candidate pool / shots | 16 / 799 / 24 across 10 years |
| 24 thumbnails, cold / warm | 1.58 s / 0.32 s |
| Mean cached thumbnail | 21 KB |
| Render, 23 shots, WebP + GIF + MP4 at 2560 px | 56 s |
| `spherical_kmeans(18,201, k=95)`, for comparison | 4.3 s, 41 iterations |

Two of these are worth a sentence.

**490 offers, not a handful.** `on_this_day` alone makes 192 and `pair_years`
137, so the offers panel needed a filter box rather than a list — a scrolling
wall of 490 rows is not a chooser.

**An edit costs 4 ms.** That is what makes "the browser never computes what
the memory contains; it asks" affordable: every edit round-trips to the server
and gets the complete recomputed state back, including the 510-row catalogue,
and it is still imperceptible.

---

## Mutations run

Every genuine bug in this project was found by running against real data, and
the recurring defect is a test that cannot fail. So each test here was checked
by breaking the line it protects, watching it fail, and restoring it.

**Fifty mutations run against 120 tests. Forty-nine killed; one turned out
not to be a mutation at all** — and six of the forty-nine only after the
test was rewritten, which is the part worth reading.

The invalid one was `prompt-resolves-hits-outside-the-index`: it replaced
`MemoryIndex.resolve_many` with a lookup that still went through
`MemoryIndex.get`, so nothing was bypassed and the test correctly did not
fail. It was replaced by one that genuinely removes the guardrail —
`Library.load` forgetting `MemoryState.apply_to(policy)`, which is a plausible
one-line regression — and that kills three tests.

<details>
<summary>The fifty</summary>

| Mutation | Killed by |
|---|---|
| thumbnail route looks the photo up in `PhotoStore` instead of the index | archived + excluded thumbnail tests |
| `Draft.add` skips the index lookup | 3 tests |
| `Draft.remove` does not remove | 2 tests |
| `to_spec` re-imposes chronological order | 2 tests |
| `reorder` accepts any list | 2 tests |
| `composition_reason` returns a constant | reconciliation test |
| `swap` appends instead of replacing in place | swap test |
| burst survivor is not tracked | burst test |
| token check removed | 2 tests |
| `Host` check removed | rebinding test |
| `Origin` check removed | cross-origin test |
| `Sec-Fetch-Site` check removed | cross-site test |
| assets served by path join | traversal test |
| output names not checked | output allow-list test |
| `Range` header ignored | range test |
| renderer ignores the pace | GIF duration test |
| `render` overwrites the spec it was given | untouched-spec test |
| renderer resolves shots outside the index | excluded-render test |
| metadata search reads the store, not the index | search guardrail test |
| thumbnails never cached | cache test |
| cache accepts any width | cache width test |
| server does not clamp an unknown width | width clamp test |
| thumbnails ignore the EXIF rotation | 8 orientation tests |
| catalogue hides the rejected pile | 2 tests |
| `Library.load` forgets to apply dismissals to the policy | 3 tests |
| render does not record the memory as surfaced | cooldown parity test |
| dismiss only notes it locally | dismissal parity test |
| music joined as a path instead of looked up | music test |
| a prompt's title is treated as a fact | prompt fact-sheet test |
| session state hands back the engine order, not the user's | 2 tests |
| `public_safe` asserted rather than computed | 2 tests |
| SSE frames lose their event name | stream test |
| neighbours go back to `SemanticSearch.similar_to` (the real bug) | threading test |
| a photo becomes its own nearest neighbour | 2 tests |
| the matrix row lookup is off by one | ranking test |
| an unembedded photo raises instead of returning nothing | no-embedding test |
| the page loads a font from a CDN | 3 tests |
| an element id is renamed in the markup | id cross-reference test |
| the EventSource is never closed | stream-close test |
| the script stops sending the token | token test |
| `render` registered as a bare `def render`, shadowing `doctor.render` | shadowing test + the whole doctor suite |
| `ui` does not check for an index | missing-index test |
| the printed URL omits the token | ui serve test |
| no semantic hint when the extra is missing | degradation test |
| `serve_forever=False` returns a socket nobody is serving | ui serve test |
| a search hit does not say WHICH name matched | namesake test |

</details>

### The six tests that could not fail, and what was wrong with them

**1. The output allow-list test aimed at files that do not exist.** It asked
for `../../rekindle.sqlite` from a memory folder — which resolves to a path
where no database lives, so it 404s whether the check runs or not. Rewritten
to aim at `../../data/rekindle.sqlite`, which genuinely exists, with an
assertion that the target is a real file before the request is made, plus a
real `notes.txt` written into the folder to test the suffix check against a
file that is actually there.

**2. The "render does not rewrite the spec" test wrote the spec in exactly the
format `MemorySpec.dumps` produces.** `json.dumps(raw, indent=2,
sort_keys=True) + "\n"` is byte-identical to `dumps()`, so rewriting the file
was invisible. Rewritten to hand-edit with a different indent *and* an extra
top-level key the loader ignores — neither of which survives a rewrite.

**3. The thumbnail width test asserted a directory does not exist.** The route
clamps `?w=4096` to 900 before the cache ever sees it, so the check inside
`ThumbnailCache` was unreachable from HTTP and `cache/4096/` was never going to
appear. Split into two: a route test that junk widths always return one of the
two legitimate images and that the cache directory holds exactly two sizes,
and a direct unit test on `ThumbnailCache.get`.

**4. The prompt guardrail test asserted a day was refused that could never
have been accepted.** The fixture had one photo of the excluded person, and
`prompt.MIN_SEEDS` is 2, so that capture day could never become a seed day
whatever the policy said. Fixed in the fixture: two photos of that person on
one day, so the day *is* a seed day the moment the exclusion stops being
applied — plus a mirror test showing those same two photos do build a memory
when nothing excludes them. The paired mutation
(`Library.load` forgetting `MemoryState.apply_to`) now kills three tests.

**5. The stream-close test read past the end of the handler it was checking.**
It looked for `stream.close()` within 200 characters of each terminal
handler — and the `error` handler sits within 200 characters of the `ready`
one, so removing `close()` from `ready` was satisfied by its neighbour's.
Rewritten to slice each handler's body up to the next
`stream.addEventListener(` instead of a fixed window.

**6. The thread-safety test asserted against its own copy of the code.** The
fixture built `retrieve` and `neighbours` by hand, because
`Library._load_semantic` needs a model registry and an ONNX runtime CI does
not install — so the regression it exists for could have come back in
`library.py` with every test still green. The pair is now built by
`library.semantic_callables`, a free function the loader itself calls, and the
fixture substitutes only where the store and the encoder come from. A
companion test proves `EmbeddingStore`'s connection really is thread-bound, so
the regression test cannot pass vacuously either.

That is six tests of 120 — **5%** — that could not fail, found in my own work
before anyone else looked at it. The previous milestones found seven, fourteen
and three.

---

## What running it found

Both defects below were found by pointing the real UI at the real library.
Neither was visible by reading, and neither would have been caught by the test
suite as it stood.

### "More like this" worked once, then killed the connection

`SemanticSearch.similar_to` begins with `store.get(file_hash)`, which reads the
manifest over `EmbeddingStore`'s own `sqlite3` connection — and a `sqlite3`
connection belongs to the thread that created it. The semantic layer is loaded
lazily by whichever request thread asks first, so the *second* request for a
photo's neighbours landed on a different thread of the pool, raised
`sqlite3.ProgrammingError`, and the socket closed with no response at all:
`RemoteDisconnected` in the client, a traceback nowhere the user could see.

Text search hid it completely. `search_vector` reads only the in-memory
matrix, so the whole prompt path — the headline feature — worked perfectly
while the neighbour lookup beside it was broken.

The fix takes the vector from that same in-memory matrix, whose rows are the
store's rows by construction. It is also two thousand times faster than the
k-means it replaced conceptually: **2 ms**.

### A 24-shot memory reported itself as "16 of 16 shots"

The render result carried the *preview's* frame count as the memory's. The
WebP and GIF stop at `--preview-frames` (16 by default); the MP4 carries every
shot. So a perfectly correct 24-shot memory printed a number that was wrong
twice over — wrong denominator, wrong numerator — and a user checking whether
their edit took would have read it as eight shots lost. `RenderResult` now
carries `shots`, `preview_rendered` and `video_rendered` separately.

---

## What the brief got wrong

**"A new optional extra, like `semantic`."** There is nothing to install; see
*Dependencies* above. The degradation requirement was real and is honoured,
but it belongs to the `semantic` extra, which the prompt panel genuinely
needs.

**"Pull in a whole scene cluster at once."** There are no stored clusters to
pull in — `rekindle semantic cluster` prints a partition and keeps none of
it. The two controls that replaced it are per-photo neighbourhoods and whole
capture days, and the second needs no model at all.

**"Adjust how long each photo holds."** Not per photo; see above. The spec
format is the constraint, and the reproduce promise outranks the control.

---

## Known limits

* **Sessions die with the process.** The durable artefact is `memory.json`,
  which `rekindle render` rebuilds from, so nothing that matters is lost — but
  a half-finished edit is. That is deliberate: a second private database of
  in-progress edits is a thing to migrate, to leak, and to get out of step
  with the spec format.
* **The plausibility judge is not run in the UI.** It exists to avoid spending
  a render on an incoherent query, and in this window the user sees the
  photographs before anything is rendered — which is the working verifier the
  judge was standing in for. `rekindle memory` keeps it.
* **Nothing verifies that a prompt memory matches your prompt.** That limit is
  unchanged and is the reason this UI exists: the page shows the tags that were
  searched, the days that agreed, the year and month histogram, and the
  measured-useless agreement figure, and then says to look at the photographs.
* **No undo stack.** Edits are individually reversible (a removed photo can be
  put back, a swap can be swapped again) but there is no history to step
  through.
* **The page is not accessible enough.** Drag-and-drop reordering has no
  keyboard equivalent. That is a real gap, not a deferred nicety.
* **One user.** There is no locking between two browser tabs editing the same
  session; the last edit wins.
* **Rendering the MP4 takes a minute and the page only says "rendering…".**
  56 s for 23 shots at 2,560 px, and `write_mp4` reports nothing until ffmpeg
  exits, so there is no honest progress to show. Untick *skip the MP4* and
  the WebP is back in a few seconds.
* **The preview is the first 16 shots, not all of them.** That is
  `--preview-frames`, inherited from `rekindle memory`. The MP4 carries every
  shot and the page now says which is which, but the animation you look at
  while editing is not the whole memory.
* **What the page shows is not verified by a browser.** There is no headless
  browser in the test suite and none is going to be added for this. What is
  checked without one: the JavaScript parses, every element id it addresses
  exists in the markup, no asset reaches the network, the token is sent, and
  both terminal stream handlers close the EventSource — and every endpoint the
  page calls is driven end to end over a real socket against the real library.
  **What is NOT verified is the rendering itself**: layout at any width, the
  drag-and-drop reorder, whether a thumbnail grid of 500 stays responsive.
  Nobody has looked at this page in a browser. Say so rather than let the test
  count imply otherwise.
