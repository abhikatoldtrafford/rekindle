# Decision log — prompt memories (M4)

`rekindle memory "durga puja over the years"`.

This is the record of how the feature was actually built: the one idea it
rests on, the numbers that idea was measured against, the eighth refusal
signal and why it is shipped switched off, and the places where the plan this
work came from turned out to be wrong.

Every number below was measured this session against the live index
(19,318 photos after the guardrails, from 19,480 rows) and the live CLIP
ViT-L/14 store (18,201 vectors, dim 768) on an RTX A4000. Nothing is carried
forward from an earlier document without re-measurement.

---

## The one idea

> **The prompt does not go to CLIP. The prompt becomes visual tags — what such
> a photograph would *look* like — and CLIP matches the tags. A photo then
> ranks by how many distinct tags agree on it.**

It exists because of a measured failure. `kalipuja diwali celebration` sent
straight to the encoder, through the unmodified `compose → collapse →
stratify → cap` pipeline, builds a 24-shot memory of which **9 shots are from
a Durga Puja day and 4 from a genuine Kali Puja or Diwali day**. The other 11
are neither.

"Durga Puja" and "Kali Puja" are nearly the same string to an image-text model
as *event names*. As *pictures* they are not close at all: a ten-armed golden
goddess with a lion, against a black goddess with a long red tongue. Six
visual descriptions of the second festival, searched independently and
combined by agreement, give:

| `kalipuja diwali celebration` | Kali/Diwali | **Durga (bleed)** | neither | years |
|---|---|---|---|---|
| the prompt straight to CLIP | 4 / 24 | **9 / 24** | 11 / 24 | 10 |
| + the corpus month window | 4 / 24 | **9 / 24** | 11 / 24 | 10 |
| tag consensus, no window | 15 / 24 | **7 / 24** | 2 / 24 | 10 |
| tag consensus + day quorum | 24 / 24 | **0 / 24** | 0 / 24 | 5 |
| **shipped** (quorum, window, MIN_SEEDS=2) | **24 / 24** | **0 / 24** | 0 / 24 | 6 |

The month window on its own does nothing, because October holds both
festivals. The tags are what separate them.

I looked at all 24 shots. They are oil lamps on steps, strings of fairy
lights, a rangoli of diyas, and family in festive clothes at night. There is
not one idol of Durga in it.

### The ground truth

Festival bleed is measured against dates, not against overlap with the naive
query — measuring "did it stop agreeing with the bad answer" would be
circular. Vijaya Dashami and Kali Puja dates for 2010–2025 come from human
recall and were then corroborated twice before use:

* every Dashami date sits at the end of, or one day after, that year's dense
  September–October capture cluster in this library (`2011-10-03..05` →
  Dashami 10-06, `2014-09-30..10-05` → 10-03, `2019-10-04..07` → 10-08, and so
  on for ten years);
* four festival-years are pinned independently by the user's own album titles:
  `Durga Puja 25` (2025-09-28…30), `Mahasaptami, 2013` (2013-10-10),
  `Diwali Kali Puja 22` (2022-10-22…25), `Diwali 25` (2025-10-19/20). All four
  agree with the recalled dates.

The table lives in `tests/test_prompt_real.py`, not in the shipped corpus.
**rekindle asserts no festival date anywhere.**

---

## The consensus function

```
score(p) = votes(p) + Σ over tags  1 / (60 + rank of p in that tag)
```

`votes(p)` is how many distinct tags put `p` in their top 100. The second term
is standard reciprocal-rank fusion and its only job is to order photos *within*
a vote tier: each tag contributes at most 1/61, so the whole tail sums to less
than 1 for any tag list shorter than 61 and **an integer vote can never be
outweighed**.

That property is the design. Three alternatives were rejected:

* **Union** — one bad tag poisons the whole set.
* **Intersection** — returns almost nothing across 18,201 photos.
* **Plain rank fusion** — lets one very confident tag carry a photo the other
  tags have never heard of, which dilutes exactly the independence that
  separates the two festivals.

Not one absolute cosine is read anywhere in this path. The only use of a score
is `rank_hits`, which sorts each tag's own hits and breaks ties on
`file_hash` — `top_k` otherwise breaks ties by store insertion order, and a
swap across the `MIN_SEEDS` boundary flips a whole capture day in or out.

**The mutation that survived first time was this line.** Replacing the
ordering key with pure reciprocal-rank fusion — which deletes the entire idea
of the module — passed the whole suite. The test used ranks 50 and 60, where
two votes win under fusion anyway. It now pins ranks 99 and 100, where they do
not, and asserts the fixture really is the interesting case before testing it.

### The day quorum

A capture day becomes a seed day when at least `ceil(n_tags / 2)` distinct
tags reached it. This is the same question the photo-level vote count asks,
asked of a day.

It matters because of the stratifier. Seed-day *precision* was already good —
17 of 19 Durga seed days were genuine — but the two wrong ones were
`2015-05-18` (144 photos) and `2020-11-23`, each the only day in its year, so
each claimed an entire year-bucket of a 24-shot memory. Two bad days out of
nineteen cost seven shots.

Measured effect on `kalipuja diwali celebration`: 15/24 correct → **24/24**,
7/24 bleed → **0/24**, at a cost of four year-buckets (10 → 6). That trade is
printed to the user rather than hidden.

---

## Tag quality is the whole game, and generic tags are poison

The clearest single result of the session. Two tag sets for Durga Puja,
everything else identical:

| tags | correct | neither | note |
|---|---|---|---|
| 6 tags, four of them generic | 12 / 24 | 12 / 24 | pulled in a 271-photo wedding |
| 4 tags, all distinctive | 17 / 24 | 7 / 24 | |

The generic ones were *"a crowd walking under an illuminated arch of lights in
a street"*, *"women in white saris with red borders smearing red powder"* and
*"drummers playing large dhaak drums beside an idol"*. Every one of them is
also a wedding, also another puja, also a street at Christmas.

So the instruction to the model, and the rule in `festivals.toml`, is explicit:
**a tag must be distinctive enough that a photograph matching it is unlikely to
be of anything else.** Four sharp tags beat six blunt ones.

There is a second-order finding here worth stating plainly, because it cuts
against the feature's own headline:

> **Tag consensus helps where the event name is ambiguous to CLIP, and HURTS
> where it is not.**

`durga puja` sent straight to CLIP already gives 23 of 24 shots on a real
Durga Puja day — CLIP genuinely sees this festival. The tags alone, with the
best tag set, give 17. It is the corpus month window that recovers it to
**24 of 24 across 10 year-buckets**. For a prompt with no corpus entry, the
tag path is what runs, and on a concept CLIP already resolves well it is a
small step backwards.

---

## Seed and expand

Direct top-K gives a wall of idols: the event albums' portraits sit at median
rank 500–4,500, where no content query reaches them. Expanding a *confirmed*
seed day to its whole capture session does reach them. With visual tags the
direct path is starker still than the plan measured for the bare query: the
tags describe idols and decorations, not an occasion people are photographed
at.

| `durga puja over the years` | shots with face tags |
|---|---|
| the top 24 of the consensus ranking | **0 / 24** |
| seed-and-expand (shipped) | **21 / 24** |

The album union is all-tokens, not any-token, and the difference is measured:
`durga puja` all-tokens matches only `Durga Puja 25`, which is what gets 2025
into the memory at all — those six photos are portraits and rank about 800 on
any content query. Any-token also matches `Diwali Kali Puja 22` on the shared
word "puja" and drags Kali Puja photos into a Durga Puja memory: precisely the
confusion the feature exists to prevent.

The known cost: **day expansion assumes a capture day is one coherent event.**
It measured well here because these festivals are multi-day outings, but the
shipped Durga memory contains two shots of a college lawn that happen to share
a day with a pandal visit. Nobody has measured how often that happens.

---

## The refusal gate — the eighth signal, and the eighth failure

Seven statistics had been tested as refusal signals on this library before
this milestone and all seven failed. Tag disagreement was the eighth and the
one with a real reason to hope: it is built from how much *independent*
descriptions of a concept converge, which does not depend on absolute scores
the way the other seven did.

**It failed.** Measured over 16 concepts this library genuinely contains and
16 it genuinely does not (`tests/fixtures/prompt_gate_concepts.json`, written
before running anything, including two deliberately hard cases):

| `tag_agreement` | min | median | max |
|---|---|---|---|
| present (16) | 0.10 | 0.49 | 0.86 |
| absent (16) | 0.05 | 0.41 | 0.81 |

At the threshold that maximises accuracy it **correctly refuses 7 of 16 absent
concepts and wrongly refuses 2 of 16 present ones** — 66% accuracy against a
50% base rate. Among the concepts described by three or more tags the
direction *reverses*: absent median 0.68, present median 0.49.

The reason is legible once you see it. `scuba diving underwater` (0.81) and
`skiing on a glacier` (0.76) have tags that agree beautifully with one another
— on the same wrong photographs, because all three descriptions of a scuba
photo push CLIP into the same blue-and-bubbly corner of the library. Meanwhile
`a wedding` (0.12) and `a birthday` (0.10) are genuinely present and genuinely
photographed several unrelated ways.

**So it ships disabled.** The number is computed and printed, with the
sentence saying it refuses nothing, because a number beside a preview is
information. There is no threshold, by decision and not by omission.

### One that looked good and is not shipped

Candidate pool size separates the two sets at 88% accuracy: refuse above 1,347
photos and you refuse 12 of 16 absent concepts and 0 of 16 present ones. It is
not shipped, for two reasons that are not close calls:

1. The threshold sits **exactly** on the largest present value (`a wedding`,
   1,347). Zero margin, fitted to one point of 32.
2. A raw pool size is a library-scale quantity. 1,347 means nothing on a
   library of 200,000 photos, which is the absolute-magnitude trap that killed
   the earlier seven signals wearing a different hat.

Recorded here so nobody rediscovers it and ships it.

### The other gate

LLM plausibility — give the model the library's *vocabulary* (album titles,
year range, tagged-people count, distinct GPS cells) and ask whether the
*query* is coherent. It is implemented, it is biased hard towards letting
things through, an unparseable reply is never a refusal, and `--no-judge`
turns it off.

**It is unmeasured on this library**, because measuring it needs an API key
and this session had none. Do not describe it as validated.

---

## What a prompt memory is

> **A preview, never an offer.**

It is built only when asked for by name. It never enters `all_offers()`, never
appears in `rekindle memories`, and `--auto` will never produce one.
`tests/test_engine.py` asserts no registered recipe is named `prompt`, and
`tests/test_memory_cli.py` asserts it from the other side.

This is the honest response to a limit that cannot be engineered away: with no
working refusal signal, the only verifier is the user's eyes, so looking is a
required step rather than one a threshold pretends to replace. The CLI says so
after every build, in those words.

The memory id is `prompt:<normalised prompt>` and nothing else — not the
parse, not the matched albums, not a hash of the photos. An id derived from
the contents mints a new memory the first time the library changes, and every
dismissal of the old one silently stops applying.

---

## Determinism

The same prompt, the same index and the same embedding store on the same
machine produce a byte-identical spec. The tag cache is committed so the same
prompt always yields the same tags; the parse is pure (NFKC + casefold, a
literal month table, exact person matching, no clock and no locale); hit
ordering is made total on `file_hash`.

Across devices it is **not** bit-identical — CUDA and CPU query embeddings
differ by about 1.8e-4 — and re-embedding the library is a new store and a new
answer. Do not claim more than that.

---

## Where the plan was wrong

The 812-line plan this work came from was unusually good, and four things in
it did not survive contact with the library.

**1. The `llm.py` title leak is not a live defect.** The plan named
`substantiated()`'s `allowed_names |= set(facts.title.split())` as a bug that
would let `christmas in midnapur` whitelist "Midnapur". Measured, it does not:
`normalise` casefolds a prompt before it reaches a fact sheet, and `_NAME`
only matches capitalised words. Two unrelated rules coinciding is not a
guardrail, so `FactSheet.title_substantiated` exists anyway and a test records
which of the two is actually doing the work.

**2. `MIN_SEEDS` is nearly inert.** The plan expected `min_seeds = 1` to wreck
month purity. With one tag it would have. With the day quorum it does not: the
quorum refuses the same days at every value. Measured across the three
acceptance prompts, 1 → 2 → 3 takes the Durga memory from 23 correct shots in
11 year-buckets, to 24 in 10, to 24 in 7, and changes the Kali memory not at
all. It is kept at 2 because it is free, not because it is load-bearing.

**3. The month window cannot separate the two festivals, and the plan implied
a date window could.** It cannot: Durga Puja and Kali Puja both fall in
October in most years in this library. A window narrows a *year's* worth of
noise, not a festival's. The tags do the separating.

**4. The measured baseline is 47%, not 51%.** Re-running the plan's own
seed-weight measurement gives 38 of 81 seed hits on a Durga Puja seed day, not
41 of 81. The final-shot figure it also quotes — 9 of 24 — reproduced exactly.
The difference is a slightly different seed-day set; the point stands either
way, and this is exactly why the plan told its reader to re-measure.

One thing in the brief also did not hold: **`christmas in midnapur` does not
resolve to eight photos from one day in 2019.** That is what the *place*
filter would have yielded, and there is no place filter. What it actually
builds is 24 shots across seven Decembers — santa hats, a church, a nativity
scene, a mall Christmas tree, "MERRY CHRISTMAS" banners — with **0 of 24
carrying GPS**, and the CLI says in as many words that "midnapur" narrowed
nothing and was searched for as a picture.

---

## What was deliberately left out

* **GeoNames and place names.** Decided against for this milestone and it is a
  real cut, not an oversight. The payoff for the query that motivates it is
  eight photos from one morning of 2019, about two after burst dedup; it needs
  a fetch command, a checksum, a CC BY 4.0 attribution obligation, a reverse
  geocode at index time and a `places` table; and bolting a gazetteer onto a
  prompt parser is how "never name a place" gets quietly broken. The honest
  line in the CLI is what ships instead.
* **Any cosine threshold.** `--min-score` exists for someone who insists and
  is unset. A 0.25 floor returns 938 photos for `durga puja` and 0 for
  `food on a plate`.
* **Video.** All 1,117 videos are unembedded, so no prompt memory can contain
  one. The CLI says so rather than leaving it to a document.
* **An LLM parser.** The rules parser covers all three acceptance prompts and
  a model cannot help where the difficulty is.
* **Relative time, seasons, booleans, negation, `in <place>` as a cue.** Each
  either makes the memory id time-dependent, invents a fact about a West
  Bengal library, or measured worse than the whole phrase.
