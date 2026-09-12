# Decision log: the independent audit

An agent was handed this repository with no context — not told what it does,
what is good about it, or what was suspected to be wrong — and asked to work
that out, run the thing against the real 19,480-row library, and report
defects with evidence. It returned fourteen findings it had confirmed, eight
its own sub-agents had found that it had not re-verified, and three explicit
suspicions.

Every one was reproduced before anything was changed. Two did not hold as
reported. This log records what was found, what was done, and the two places
the report was wrong, because a finding that does not reproduce costs more
than it is worth and the record should say which was which.

## Why an audit rather than more tests

The suite was 2,553 tests and green on three operating systems. It did not
catch a single thing in this document. That is the point worth keeping: the
tests encode what the authors already understood. Nine of the twenty-two
defects are a claim in a docstring or a README that the code does not honour,
which no test can find unless someone thinks to compare the two, and the worst
one is a column list that nothing cross-checked against the schema it belongs
to.

## The one that mattered

**`rekindle index` destroyed every orientation verdict.**

`db.py::_insert` was `INSERT OR REPLACE` over a hand-typed list of 34 columns.
The table has 36: schema v6 added `orient_ignore_exif` and `orient_evidence`
and did not add them here. `INSERT OR REPLACE` deletes the conflicting row and
inserts a new one, so a column missing from the list is not left alone — it is
reset to its default, on every write, silently.

Measured on a copy of the real index:

```
before  18,363 orientation verdicts, 212 corrections
re-index 19,480 UNCHANGED photographs            3.9 s
after   0 verdicts, 0 corrections
```

The 212 corrected photographs then render, embed and fingerprint sideways
again, and the orientation pass — two image decodes per photograph over the
whole library — has to re-examine 18,363 rows to find them.

**Three previous versions caught this pattern and the fourth did not.** v3, v4
and v5 each have a preservation test written after the fact. That is the
signal that a convention is not enough, so the fix is structural rather than
another entry on the list:

* the statement is **built from one tuple**, so the column names, the
  placeholders and the SET clause cannot disagree;
* `ON CONFLICT DO UPDATE` names what it overwrites, so a forgotten column now
  **preserves** a stale value instead of destroying a real one;
* `_NOT_INSERTED` declares the other side of the line, and a test asserts the
  two tuples account for every column of `photos` — adding a column to the
  schema fails the suite until someone says, in writing, which side it is on.

## The two that did not hold

**"A complete Takeout export triggers INCOMPLETE EXPORT."** The mechanism is
real: Google writes `shared_album_comments.json` and
`user-generated-memory-titles.json` at the root of `Google Photos/`, they name
no photograph, and they were counted as orphan sidecars. On an export with no
real orphans those two alone fire the loudest warning the tool has.

But the report said it was reproduced on this library, and it was not. Running
`rekindle doctor` on the real export gives **2,444 orphans, not 2**, and
enumerating them shows 2,442 are sidecars for photographs that genuinely are
not on disk. The warning is telling the truth here. Fixed anyway, because the
mechanism is real for other users — and it surfaced something nobody was
looking for: this export is missing about 2,442 photographs.

**The `/api/output` path escape.** `D:evil.gif` has no separator, no `..` and
an allowed suffix, and joining it onto a folder discards the folder. True —
but only when the drive letter DIFFERS from the folder's. A drive-relative
name on the same drive joins inside it:

```
Path("C:/Users/Public/session") / "C:evil.gif"  ->  C:\Users\Public\session\evil.gif
Path("C:/Users/Public/session") / "D:evil.gif"  ->  D:evil.gif
```

Fixed with containment after resolution, which does not care which of the two
it is looking at.

## Things that were true and had been written down as false

Nine findings are a statement in the code contradicted by the code beside it.
Collected here because the pattern is more useful than any one of them.

| Where | Said | Did |
| --- | --- | --- |
| `memory/llm.py` | "the ONLY place in rekindle that touches a network" | `memory/tags.py` posts to the same endpoint whenever `OPENAI_API_KEY` is set |
| `README.md` | "Nothing leaves your machine" | true only with no key in the environment |
| `README.md` | "a drag of a slider can never widen it" | `loosens()` compared against the shipped default, not the user's value |
| `caption_vocab.toml` | rule 4: "not a place of worship named as such" | shipped `"at a temple"`, eighteen lines below the rule |
| `captioning.py` | a cache "read back forever after" | nothing read the CLIP rows back, ever |
| `GroundingReport.cached` | "so a dead cache shows up as a zero" | the counter itself was the dead thing |
| `frames.py` | the third guard "makes the promise UNCONDITIONAL" | the title had none of the three |
| `vectors.top_k` | "ties broken by index" | argpartition order |
| `cluster.Cluster.separation` | "Mean cosine" | a max |
| `web/server.py` | "the path is never joined from user input" | it was |
| `sidecars.py` | "the `met[a-z]*` wildcard covers truncation" | covers it only after sixteen characters |

The two caption-vocabulary entries are the ones worth dwelling on. Rule 4 was
prose in a TOML comment and `_SENSITIVE_WORDS` was the list that enforced it,
and **nothing connected them** — so the rule could name a category the list did
not cover and every test still passed. It named "a place of worship" and the
list had church, mosque and synagogue. 429 photographs got `at a temple`.
There is now a test asserting the list covers the words the rule uses, which
is the only thing that makes prose and code stay in step.

## Reporting that looked like success

Three findings share a shape: the tool said something reassuring while the
thing it described had not happened.

* `GPT captions: 24/24 accepted` printed while the model cache was a dangling
  symlink, so every caption was written from the fact sheet alone. A high
  acceptance rate with no grounding is the worst-looking failure this layer
  has, because it is indistinguishable from success.
* `3 shots in this spec are no longer admitted by the guardrails` when five
  were, because the count came off the preview's frame report and the preview
  stops at sixteen shots.
* `Dates corrected: 1` for a photograph whose real EXIF date had just been
  nulled by a dateless donor — after which `deny_reason` refuses it as
  `no_date` and it vanishes from every memory.

`CaptionReport.accounted` belongs here too: it was false whenever the service
died mid-memory, because `requested` counted iterations and the loop breaks.
A report that stops reconciling exactly when it is being read for a diagnosis
is worse than no report.

## Performance

`decode_colour` was the hottest function in a real build — it parses a
128-character hex histogram, and `between` is called on PAIRS, so the same
string is decoded once per photograph it is compared against. 108,824 calls,
2.25 s of a 12.88 s run under cProfile.

Memoised, and returning a tuple rather than rebuilding a list on every hit:

```
year_in_review 2025, 19,480-row index, alternating runs
before   7.80s  7.87s
after    4.55s  4.58s      memory.json identical, sha256 847173bae5dfdcaa
```

`_ranked` counted a sorted list linearly inside a sort key. `bisect_left`
gives the identical answer — the index of the first element not less than the
value IS how many are strictly below it — for 494 ms against 0.7 ms at
n=4,000. It did not show up in the profile because `decode_colour` dominated
it; it would have, at a larger library.

## What the audit says about the codebase

Worth recording, because it is evidence and not flattery. The determinism
claim holds: two runs produce identical `memory.json`, `.webp` and `.gif`, and
still do after twenty-two changes. `--public-safe` was checked shot by shot
against the raw tags with zero violations. All three accounting identities
hold on real data. Exclusions really do apply to specs already written.

The single biggest risk it named was the hand-maintained column list, and that
is now the one thing in this document that cannot recur the same way.
