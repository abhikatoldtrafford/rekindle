# Decision log — prompt memories (M4)

`rekindle memory "durga puja over the years"`.

This is the record of how the feature was actually built: the one idea it
rests on, the numbers that idea was measured against, the eighth refusal
signal and why it is shipped switched off, and the places where the plan this
work came from turned out to be wrong.

Every number below was measured against the live index (19,318 photos after
the guardrails, from 19,480 rows) and the live CLIP ViT-L/14 store (18,201
vectors, dim 768) on an RTX A4000. Nothing is carried forward from an earlier
document without re-measurement.

## Re-measured after the orientation fix (`81d2565`)

Every figure in the first version of this document was measured against an
embedding store in which `semantic/embed.py:_decode` had never applied the
EXIF orientation tag, so roughly one image in seven was embedded on its side.
The library was re-embedded upright and **every number below was re-measured
on both stores**, old beside new. Comparing them vector by vector: 2,534 of
18,201 vectors changed (13.9%), the changed ones have a median self-cosine of
0.9352 against their upright replacement, and two unrelated photos of this
library sit at 0.549 — so in the worst case (0.595) a photograph was further
from itself than an unrelated pair is from each other.

The short version:

> **Every conclusion survived, and the shipped outputs did not move at all.**
> `durga puja over the years` and `kalipuja diwali celebration` build *the same
> 24 photographs in the same order* from the corrupted store and the corrected
> one. `christmas in midnapur` shares 22 of its 24.

What did move is smaller than that and mostly went the right way. What moved
*most* was not caused by the defect at all: three rows of the bleed table below
do not reproduce against the very store they were measured on. They are
corrected here and the discrepancy is written up at the end, under
"What the re-measurement found".

---

## The one idea

> **The prompt does not go to CLIP. The prompt becomes visual tags — what such
> a photograph would *look* like — and CLIP matches the tags. A photo then
> ranks by how many distinct tags agree on it.**

It exists because of a measured failure. `kalipuja diwali celebration` sent
straight to the encoder, through the unmodified `compose → collapse →
stratify → cap` pipeline, builds a 24-shot memory of which **9 shots are from
a Durga Puja day**. That is the number the whole feature exists to kill, and
it is the most robust figure in this document: 9 of 24, on both stores, at
every value of `MIN_SEEDS`.

"Durga Puja" and "Kali Puja" are nearly the same string to an image-text model
as *event names*. As *pictures* they are not close at all: a ten-armed golden
goddess with a lion, against a black goddess with a long red tongue. Six
visual descriptions of the second festival, searched independently and
combined by agreement, give:

| `kalipuja diwali celebration` | Kali/Diwali | **Durga (bleed)** | neither | years |
|---|---|---|---|---|
| the prompt straight to CLIP | 6 → **7** / 24 | **9 / 24** | 9 → **8** / 24 | 10 |
| + the corpus month window | 6 → **7** / 24 | **8 / 24** | 10 → **9** / 24 | 10 |
| tag consensus, no quorum, no window | **17 / 24** | **4 / 24** | 3 / 24 | 10 |
| tag consensus + day quorum, no window | **23 / 24** | **0 / 24** | 1 / 24 | 7 |
| **shipped** (quorum, window, MIN_SEEDS=2) | **24 / 24** | **0 / 24** | 0 / 24 | 6 |

`a → b` is the sideways store's number before the arrow and the upright
store's after it; a single number means the two stores agree exactly. Every
row is at the shipped `MIN_SEEDS = 2`, `TAG_K = 100` and `SEED_K = 100`, with
only the gate named in the row changed — which the first version of this table
was not, and that is where its middle rows came from. See the end of this
document.

The month window on its own moves one shot, because October holds both
festivals. The tags are what separate them, and the day quorum is what
finishes the job.

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
17 of 19 Durga seed days genuine on the sideways store, 16 of 18 on the
corrected one — and **the two wrong ones are the same two on both stores**:
`2015-05-18` (144 photos) and `2020-11-23`, each the only day in its year, so
each claimed an entire year-bucket of a 24-shot memory. Two bad days cost
seven shots, on both stores: the Durga memory goes from 17 of 24 correct
without the quorum to 24 of 24 with it.

Measured effect on `kalipuja diwali celebration`: 17/24 correct → **24/24**,
4/24 bleed → **0/24**, at a cost of four year-buckets (10 → 6). Identical on
both stores. That trade is printed to the user rather than hidden.

---

## Tag quality is the whole game, and generic tags are poison

The clearest single result of the session. Two tag sets for Durga Puja,
everything else identical:

| tags | correct | neither | note |
|---|---|---|---|
| 6 tags, four of them generic | 12 / 24 | 12 / 24 | pulled in a 271-photo wedding |
| 4 tags, all distinctive | **17 / 24** | **7 / 24** | reproduces exactly, both stores |

The second row re-measures exactly, on both stores, with the day quorum off —
which is what "everything else identical" has to mean here, because the quorum
is `ceil(n_tags / 2)` and therefore moves when the tag count does.

**The first row is not reproducible, because this document only ever wrote
down three of its four generic tags.** Reconstructions from the three that are
named, added to the corpus set with the quorum off, give 10 of 24 correct on
the sideways store and 11 on the upright one: the same direction, worse than
logged, but not the same measurement. The claim itself is confirmed twice
over. The 271-photo day is real — it is `2012-11-24`, the largest capture day
in that month of the library — and *"women in white saris with red borders
smearing red powder"* puts **22 of its own top 100 on that single day**, on
both stores, and makes it a seed day. A tag list is a measurement input, and
writing down three of four is how a row stops being checkable.

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

All three of those numbers — 23, 17, 24-across-10 — reproduce exactly on both
stores. This paragraph is the part of the document that most deserved to be
wrong and is not.

---

## Seed and expand

Direct top-K gives a wall of idols: the event albums' portraits sit at median
rank **1,800–4,800** across the four Durga tags (best single rank 572, worst
14,431; on the bare query `durga puja` the median is 2,650), where no content
query reaches them. The "500–4,500" first written here was the range of best
ranks and medians mixed together; the medians are what the sentence claims and
they are higher. Both stores agree to within about 2%. Expanding a *confirmed*
seed day to its whole capture session does reach them. With visual tags the
direct path is starker still than the plan measured for the bare query: the
tags describe idols and decorations, not an occasion people are photographed
at.

| `durga puja over the years` | shots with face tags |
|---|---|
| the top 24 of the consensus ranking | **0 / 24** |
| seed-and-expand (shipped) | **21 / 24** |

Both rows reproduce exactly on both stores. One number in the same family does
not: `prompt.build_selection`'s own docstring says direct top-150 "gives 11
year-buckets but only 4 of 24 shots carry a face tag". Measured now, on
*both* stores, direct top-150 gives **14 year-buckets and 0 of 24 shots with a
face tag**. The docstring's conclusion is if anything understated; its numbers
are wrong and were wrong before the re-embed.

The album union is all-tokens, not any-token, and the difference is measured
(and see *Albums: ground truth, a contribution, or noise* below, which found
that this comparison had never once been run against a prompt with an
ordinary framing word in it):
`durga puja` all-tokens matches only `Durga Puja 25` — re-confirmed on the
live index — which is what gets 2025 into the memory at all. Those six photos
are portraits: their *best* rank on any of the four tags is 572 and on the
bare query `durga puja` it is 841, with medians between 1,800 and 4,800. None
of the six is anywhere in the consensus order that the tags actually build.
 Any-token also matches `Diwali Kali Puja 22` on the shared
word "puja" and drags Kali Puja photos into a Durga Puja memory: precisely the
confusion the feature exists to prevent.

The known cost: **day expansion assumes a capture day is one coherent event.**
It measured well here because these festivals are multi-day outings, but the
shipped Durga memory contains two shots of a college lawn that happen to share
a day with a pandal visit. Nobody has measured how often that happens.


---

## Albums: ground truth, a contribution, or noise

The owner's observation was that the pipeline did not take advantage of
albums, and that `memories of puri` ought to look for a Puri album first. They
were right that albums were underused, and the reason turned out to be sharper
and more embarrassing than "albums are only a supplementary signal".

### The album union never fired on a sentence

All-token matching asks the album title to contain *every* content token of
the subject, and the subject is the whole prompt. `Puri 25` does not contain
the words "memories" or "of", so `memories of puri` matched **nothing**.
Measured on the reference library over its 37 named albums:

| prompt shape | albums reached |
|---|---|
| the album's own bare title (`puri 25`) | 37 / 37 |
| the title with its year stripped (`puri`) | 37 / 37 |
| `memories of <title>` | **0 / 37** |

Any framing word at all - "memories of", "our", "show me", "photos of" -
turned the feature off completely. Every album measurement previously recorded
here was taken with a bare keyword prompt, which is why this never showed up.

The fix is a closed, human-readable list of words that frame a request instead
of naming a subject (`prompt.STOPWORDS`), removed before the comparison. It is
a table for the same reason `_MONTHS` is a table: a stemmer or a stop-word
package would be a dependency, a locale, and a silently changing answer. After
it, `memories of <title>` reaches **37 of 37**.

All-token matching itself is kept, and the reason it was chosen still holds:
`durga puja` matches only `Durga Puja 25`, where any-token also matches
`Diwali Kali Puja 22` on the shared word "puja". Because it is a *subset*
test, a shorter prompt matches more, and that is what makes the multi-album
case work without any alias configuration: `kashmir` reaches all three Kashmir
albums and `ladakh` reaches both `Leh Ladakh` and `ladakh`.

### A live defect the same measurement found

`album_matches` did not apply the presentability rule. Google writes a
`Photos from YYYY` album for every year and every photograph is in one, so the
one-word prompt `photos` matched **all 23 of them and unioned 17,004
photographs - 88% of the library - into the candidate pool**. Two independent
guards now stop it: the presentability filter, and "photos" being a stop word,
so the prompt has no content token left to match on.

### When an album is the memory, and when it is a contribution

Album-first is *not* simply better, and the album-size distribution is why.
The reference library's 37 named albums run from 1,134 photographs (`Avyan`)
and 510 (`Kashmir`) down to 6 (`Durga Puja 25`), 3 (`Puri 25`) and 1
(`vanu biye`).

An album that holds at least as many photographs as a memory has slots
(`LEAD_MIN = 24`, which is `engine.DEFAULT_MAX_SHOTS`) can answer the question
on its own, and the user's own curation beats anything a model infers. Below
that it cannot, and the tags are still needed. The threshold is measured, not
assumed: 25 photographs is the largest album below it and 48 the smallest
above, so **every value between 26 and 48 gives the same answer on every album
in this library** - the decision does not sit on a cliff.

When an album leads, the tags are dropped, the plausibility judge is skipped,
`retrieve` is never called and the pool is the album. That is what
`album_story` would have built. The prompt is deliberately **not** routed to
that recipe: `album_story` can name only one album where `kashmir` needs
three, and the memory id would become `album_story:Kashmir`, so it would
change the day an album crossed the threshold and every dismissal of it would
stop applying. The id stays `prompt:<text>` and the several albums are one
cluster.

One arithmetic trap, found by reading the output rather than the code: the
album sizes must not be summed. `Leh Ladakh` (277) and `ladakh` (237) are one
trip filed twice and share 195 photographs, so the sum says 514 where the
memory is built from 319. The CLI printed both numbers two lines apart until
`AlbumMatch.total` was changed to count distinct photographs.

Measured, before and after, with no API key on the machine:

| prompt | before | after |
|---|---|---|
| `memories of puri` | 0 albums matched; 24 shots over 9 years and 7 months, **none from the Puri trip** | `Puri 25` (3) joins the pool; still 24 search-led shots, and the CLI now says the album was too small to be the memory |
| `durga puja` | `Durga Puja 25`; 24 shots, 10 years, 19 in October | **identical, shot for shot** |
| `durga puja over the years` | `Durga Puja 25`; 24 shots, 10 years, 21 with faces | **identical, shot for shot** |
| `kashmir` | pool 870 from tags + albums; 22 of 24 shots in a Kashmir album, one stray from 2017 | album-led, pool 534; **24 of 24** in a Kashmir album, all 2015-05, 24 with faces |
| `ladakh` | pool 334; 24 shots, all 2018 | album-led, pool 319; the same 24 shots, and no CLIP load at all |
| `memories of kashmir` | 0 albums matched; pool 790 from tags alone; 23 of 24 in the album | album-led, pool 534; **24 of 24**, and it works with no `semantic` extra installed |

The festival results are byte-identical, which is the point: `Durga Puja 25`
holds six photographs, so it never leads, and the union that every number in
this document depends on is untouched.

### Two things that were tried and are not shipped

**Expanding a small album to its capture days.** `Puri 25`'s three
photographs sit on two days that hold 57 photographs between them, which looks
like exactly the trip the owner wanted. It is not defensible. The tag path
expands a day only after *quorum* evidence that the day is the concept; an
album says nothing about the other photographs that share its dates. Measured
across all 37 named albums, the amplification is unbounded and sometimes
absurd: `Archive` (3 photographs) expands to **331**, because its "day" is a
bulk-import timestamp holding photographs filed under 2012, 2019, 2020 and
2021; `Rumpa Di marriage` (1) expands to **115**; `Abhirup Birthday/ Sudipta
Saad` (4) to **242**. Requiring two album photographs on a day helps some
cases (`Mamabari iburo bhat` 351 -> 21) and not others (`Archive` stays 331).
And the Puri days themselves are 13 photographs in a ten-minute burst and 44
in a half-hour burst, every one of them also filed under `Avyan` - a child,
not a beach.

**Letting a small album lead when nothing else knows anything.** With no
festival-corpus entry, no cached tags and no API key, the tag path falls back
to searching for the prompt's own words, which is the measured-bad path; the
album is then the only real evidence in the run. Implemented, measured,
reverted. On `memories of puri` it produces a pool of three photographs, two
of which are the same burst two seconds apart, dedup collapses them, and the
engine refuses the memory outright with `too_few_photos`. **It converts a poor
memory into no memory.**

That second result is also the honest correction to the framing this work
started from, which said the search "finds a couple of dozen from that trip"
where the album finds three. It does not. On this library, with no key, the
24 shots `memories of puri` returns contain **not one photograph from the Puri
album or its days** - nine different years and seven different months. Both
answers are bad. The library holds three photographs of Puri, two of them near
identical, and no design turns that into a memory.

### The language model shortlist

Token matching is exact and brittle: it cannot connect `ladhak` to
`Leh Ladakh`, and it cannot connect a transliteration to a Bengali album name.
So the optional layer gets a job it is genuinely good at - the model is handed
the prompt and the album *titles*, and asked which are about the same subject.

Four constraints, each of them enforced rather than requested:

1. **It is consulted only where token matching found nothing.** The exact rule
   stays authoritative, so a model cannot answer `Diwali Kali Puja 22` to
   `durga puja` and reach the pool, and the presence of an API key cannot move
   any number in this document.
2. **It cannot invent an album.** Every returned line is looked up in the list
   the model was given, matched on the normalised title so a change of case is
   not a rejection, and the library's own spelling comes back. Anything else
   is dropped, counted, and reported to the user.
3. **It is cached**, in the data directory beside the tag cache, with a digest
   of the album titles it was chosen from. A cached answer to "which of *these*
   albums" is only an answer while the album list is the same one. An empty
   answer is cached too, so a prompt that names no album does not pay for a
   call on every run.
4. **There is no committed starter cache**, unlike `prompt_tags.json`. A
   shortlist maps somebody's prompt to *their own* album titles, and a
   checked-in file would ship one person's private labelling to every user.

It opens no new channel out of the library: the plausibility judge is already
given the album titles, and this request carries the same titles and nothing
else - no photograph, no path, no date, no name, no count.

**It is unmeasured against a real model**, because this session had no API
key. Every assertion about it is about what the layer does with a reply, on an
injected transport. Do not describe it as validated.
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
| present (16), sideways | 0.10 | 0.49 | 0.86 |
| absent (16), sideways | 0.05 | 0.41 | 0.81 |
| **present (16), upright** | **0.09** | **0.45** | **0.87** |
| **absent (16), upright** | **0.06** | **0.40** | **0.81** |

At the threshold that maximises accuracy it **correctly refuses 7 of 16 absent
concepts and wrongly refuses 2 of 16 present ones** — 66% accuracy against a
50% base rate. **On the corrected store the best threshold is the same 0.23
and the accuracy is the same 66%.** Among the concepts described by three or
more tags the direction *reverses* on both stores: absent median 0.68 against
present 0.49 sideways, absent 0.69 against present 0.45 upright.

This was the measurement most worth hoping about, because it is the one an
upright store could plausibly have rescued: a signal built from whether
independent descriptions converge is exactly the kind of thing a sideways
image would scramble. It did not rescue it. Both distributions shifted down by
about 0.04 and stayed on top of each other, and every concept named below kept
its place in the ordering. The gate stays off.

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
photos and you refuse 12 of 16 absent concepts and 0 of 16 present ones.
**Re-measured upright it is 84%** — refuse above 1,348 and you refuse 11 of 16
absent concepts and 0 of 16 present ones. It got worse, and the one property
that made it unshippable did not change at all: the threshold still sits
exactly on `a wedding`, which moved from 1,347 to 1,348 photos and stayed the
largest present value. It is not shipped, for two reasons that are not close
calls:

1. The threshold sits **exactly** on the largest present value (`a wedding`,
   1,347 sideways and 1,348 upright). Zero margin, fitted to one point of 32,
   and a re-embed moved the fit by one photo while costing it four points of
   accuracy — which is what fitting to one point of 32 looks like from the
   outside.
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
differ by up to 2.5e-4 elementwise, re-measured over three queries — and
re-embedding the library is a new store and a new answer. Do not claim more
than that.

Re-embedding the library is, however, a *smaller* new answer than that
sentence implies, and this document now has evidence for how much smaller.
Replacing 13.9% of the store's vectors with meaningfully different ones (see
the header) left the shipped Durga and Kali memories byte-identical. The
seed-day quorum is the reason: it asks a question about *days* that half a
dozen independently-ranked tag lists have to agree on, and a day survives
losing several of its photographs from several of those lists.

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
all. Upright the same sweep reads 22 in 11, 24 in 10, 24 in 6 — the two ends
each lost one, the shipped middle did not move, and the Kali memory is still
untouched at all three values. It is kept at 2 because it is free, not because
it is load-bearing.

**3. The month window cannot separate the two festivals, and the plan implied
a date window could.** It cannot: Durga Puja and Kali Puja both fall in
October in most years in this library. A window narrows a *year's* worth of
noise, not a festival's. The tags do the separating.

**4. The measured baseline is 47%, not 51%.** Re-running the plan's own
seed-weight measurement gives 38 of 81 seed hits on a Durga Puja seed day, not
41 of 81. The final-shot figure it also quotes — 9 of 24 — reproduced exactly,
and reproduces again now, on both stores.

The 38-of-81 half **could not be re-measured**, because neither the plan nor
this document records what produced an 81-seed list; the shipped `SEED_K` is
100 and the obvious reconstruction of "the naive seeds for `durga puja`" gives
93 of 100 on the sideways store and 94 of 100 on the upright one, which is
plainly measuring something else. The number is left as written, flagged as
unreproducible rather than quietly corrected to a figure from a different
procedure. A percentage whose denominator nobody wrote down is not a
measurement anyone can check.

One thing in the brief also did not hold: **`christmas in midnapur` does not
resolve to eight photos from one day in 2019.** That is what the *place*
filter would have yielded, and there is no place filter. What it actually
builds is 24 shots across seven Decembers — santa hats, a church, a nativity
scene, a mall Christmas tree, "MERRY CHRISTMAS" banners — with **0 of 24
carrying GPS**, and the CLI says in as many words that "midnapur" narrowed
nothing and was searched for as a picture.

Re-measured: 24 shots, all in December, across the same seven Decembers
(2013, 2014, 2016, 2017, 2018, 2020, 2025), 0 of 24 with GPS, "midnapur" still
unmatched. 22 of the 24 photographs are the same ones the sideways store
chose. The eight-photos-in-2019 figure the brief was arguing against is also
still exactly right as a description of what a place filter *would* have done:
this library holds 8 photographs with coordinates in the Medinipur cell in the
week of Christmas, and all eight are from the morning of 2019-12-22.

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
  is unset. A 0.25 floor returns 938 photos for `durga puja` sideways and 948
  upright, and 0 for `food on a plate` on both — which is the point: the same
  floor is a useful filter for one query and a total refusal for another, and
  re-embedding the library moved one of the two by 1%.
* **Video.** All 1,117 videos are unembedded, so no prompt memory can contain
  one. The CLI says so rather than leaving it to a document.
* **An LLM parser.** The rules parser covers all three acceptance prompts and
  a model cannot help where the difficulty is.
* **Relative time, seasons, booleans, negation, `in <place>` as a cue.** Each
  either makes the memory id time-dependent, invents a fact about a West
  Bengal library, or measured worse than the whole phrase.

---

## What the re-measurement found

Re-running this whole document against the corrected store, with the sideways
store kept beside it. Ordered by how much it should change anyone's mind.

### The architecture conclusion survived, and now has a stronger argument

The claim was always structural: tag consensus separates two festivals that
share a name-shaped embedding because it asks several *independent* visual
questions and keeps only the photographs several of them agree on. That
argument predicts robustness to noise in any one tag's ranking, and the
re-embed turned out to be an unusually good test of it, with 13.9% of the
store's vectors replaced by meaningfully different ones. The prediction held
exactly: **the shipped Durga and Kali memories are the same 24 photographs in
the same order from both stores.**

I looked at all 72 shots of the three memories on the corrected store, rather
than at their scores. The Kali memory is oil lamps on steps, a woman lighting
diyas, a house outlined in fairy lights, a rangoli of diyas, and family in
festive clothes at night — **not one idol of Durga in it**, which is the
sentence the first version of this document earned and the corrected store
does not take away. The Durga memory is idols, pandals, and families in front
of pandals, with the one college-lawn shot the day-expansion caveat above
predicts. The Christmas memory is santa hats, a nativity tableau, a church
with a MERRY CHRISTMAS banner, and a mall tree.

### Three rows of the bleed table were wrong when they were written

This is the finding worth acting on, and it has nothing to do with
orientation. Rows 1-3 of the original table **do not reproduce against the
sideways store they were measured on**. A sweep of 288 configurations
(`tag_k` x `seed_k` x `MIN_SEEDS` x tags x window x quorum), plus twelve
direct-top-K variants, found nothing that produces them:

| original row | as logged | reproducible? |
|---|---|---|
| the prompt straight to CLIP | 4 / 9 / 11, 10 years | counts only at `MIN_SEEDS=1`, where the year count is 13 |
| + the corpus month window | 4 / 9 / 11, 10 years | same, and the window is *not* a no-op at `MIN_SEEDS=2` |
| tag consensus, no window | 15 / 7 / 2, 10 years | **no configuration produces it** |
| tag consensus + day quorum | 24 / 0 / 0, 5 years | only at `SEED_K=50`, which is not what ships |
| **shipped** | **24 / 0 / 0, 6 years** | **exactly, at the shipped constants, on both stores** |

The pattern says what happened: the rows were measured one at a time while
other knobs moved, and the `years` column looks copied down. The two rows that
were measured at shipped settings — the headline 9/24 bleed and the shipped
24/24 — are both exactly right, on both stores. The corrected table above
holds everything but the named gate fixed, which is what a table like this has
to mean if its rows are to be read against each other.

That makes seven figures this project has carried forward without
re-measurement. Every one was caught by re-measuring, none by anyone doubting
it first.

### What moved, in full

| figure | sideways | upright | verdict |
|---|---|---|---|
| `kalipuja` bleed, prompt straight to CLIP | 9 / 24 | 9 / 24 | unchanged |
| `kalipuja` shipped: correct / bleed / years | 24 / 0 / 6 | 24 / 0 / 6 | unchanged, same photos |
| `durga` shipped: correct / years / faces | 24 / 10 / 21 | 24 / 10 / 21 | unchanged, same photos |
| `durga` straight to CLIP | 23 / 24 | 23 / 24 | unchanged |
| `durga` tags alone, no window, no quorum | 17 / 24 | 17 / 24 | unchanged |
| `christmas`: Decembers / shots with GPS | 7 / 0 | 7 / 0 | unchanged, 22 of 24 same photos |
| faces: consensus top-24 / shipped | 0 / 21 | 0 / 21 | unchanged |
| `tag_agreement` best accuracy | 66% | 66% | unchanged |
| `tag_agreement` present min / median / max | 0.10 / 0.49 / 0.86 | 0.09 / 0.45 / 0.87 | median -0.04 |
| `tag_agreement` absent min / median / max | 0.05 / 0.41 / 0.81 | 0.06 / 0.40 / 0.81 | median -0.01 |
| three-or-more-tag reversal (absent vs present) | 0.68 vs 0.49 | 0.69 vs 0.45 | unchanged, still reversed |
| pool-size gate accuracy | 88% | 84% | **worse**, still fitted to one point |
| Durga seed days genuine, no quorum | 17 / 19 | 16 / 18 | same two wrong days |
| `MIN_SEEDS` 1 / 2 / 3, Durga correct shots | 23, 24, 24 | 22, 24, 24 | shipped value unchanged |
| `MIN_SEEDS` 1 / 2 / 3, Durga year-buckets | 11, 10, 7 | 11, 10, 6 | shipped value unchanged |
| photos over a 0.25 cosine for `durga puja` | 938 | 948 | +1% |
| photos over a 0.25 cosine for `food on a plate` | 0 | 0 | unchanged |
| CUDA vs CPU query embedding drift | ~1.8e-4 | up to 2.5e-4 | same order |

Library-level figures re-confirmed unchanged: 19,480 rows, 19,318 photos after
the guardrails, 18,201 embedded images, 1,117 unembedded videos, `durga puja`
matching exactly one album (`Durga Puja 25`) under all-tokens matching, and 8
GPS-bearing photographs in the Medinipur cell in the week of Christmas.

### Three numbers were wrong independently of the store

1. **The bleed table's middle rows**, above.
2. **`prompt.build_selection`'s docstring** says direct top-150 "gives 11
   year-buckets but only 4 of 24 shots carry a face tag". Measured on both
   stores it is **14 year-buckets and 0 of 24 with a face tag**. The same
   docstring's "22 to 24 of 24 shots carry faces" for the shipped path is 21
   on both stores, which the table above already says. Its conclusion is if
   anything understated by its own numbers. Left uncorrected in source by the
   scope of this pass; recorded here so nobody re-derives it.
3. **"median rank 500-4,500"** for the album portraits, corrected above to
   1,800-4,800.

One number could not be re-measured at all: the plan's 38-of-81 seed weight,
whose procedure nobody wrote down. It is flagged in place rather than replaced
by a figure from a different procedure.

### A limitation the eye check found that no number would have

Three of the 24 shots in the shipped Kali Puja memory are turned on their side
by the orientation fix, not by its absence. Their EXIF orientation tag is
stale: the pixels on disk are already upright, and applying the tag rotates
them 90 degrees. `open_upright` is correct and the files are lying. Written up
in `docs/known-limitations.md`, because it is a decode fault affecting a small
population of this library and not a prompt-memory fault at all. It was
invisible to every number in this document, and obvious the moment anyone
opened the photographs.
