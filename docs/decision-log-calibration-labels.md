# Decision log — the calibration labels

> *"There's no learned component anywhere — `score()` is a hand-written
> function. That's deliberate and defensible, but the calibrate flow is
> already collecting labeled judgments from users. That's a training set
> sitting there unused."*

The observation is correct. This is the measurement of what the training set
is worth, and the answer is **not yet, and not for that**.

Everything below was measured on the reference library (19,480 rows, 19,318
usable) with the real `calibration.json` from a real sitting, by
`tests/experiment_calibration_labels.py`. Re-run it with:

```
python tests/experiment_calibration_labels.py --data-dir data --blast-radius
```

---

## 1. There are fewer labels than the question assumes

The brief that raised this put the number at "13 calibration steps at roughly
ten judgements each is **~130 labels**". Measured, that is wrong, and it is
wrong in the direction that matters.

| | |
|---|---|
| Steps in the sequence | 13 |
| Steps that can produce a judgement at all (`blind` mode) | **6** |
| ...of those, that need no extra install | **4** |
| Blind judgements per step (`--rounds`) | 10 |
| **Ceiling on a machine with every extra** | **60** |
| **Ceiling on a plain `uv sync`** | **40** |
| Labels actually on disk after the reference sitting | **17** |

The other seven steps are `slider` or `default` only. A slider step records a
number the user typed; a default step records "I looked, and yours is right".
Neither is a labelled example of anything. `plan.STEPS` is the authority and
`census()` in the experiment counts it rather than assuming.

The 17 real labels are 10 about blur and 7 about darkness, 6 of them
rejections. The sitting was never finished.

One more correction while the census is open: **the labels are not thrown
away today.** `calibration.json` holds every answer with the file hash it was
about, and `Session._replay` re-derives every threshold from them on resume,
so a sitting can be stopped and picked up without re-asking. What is thrown
away is narrower and is the thing §6 fixes: an answer is *replaced* when the
same photograph is judged again, and *deleted* when a threshold is redone.

## 2. The labels do work — for the one thing they are already used for

Leave-one-out, per setting: fit the decision stump `judge.derive` already
fits on n−1 labels, predict the held-out one, and compare against the shipped
constant and against always answering the commoner label.

| setting | labels | learned (LOO) | shipped constant | majority class |
|---|---|---|---|---|
| `composition.min_sharpness` | 10 | **0.90** | 0.70 | 0.60 |
| `composition.min_brightness` | 7 | 0.71 | 0.43 | **0.71** |

Sharpness is a real result: ten judgements beat the shipped 0.12 by twenty
points and beat the do-nothing baseline by thirty. Brightness is not — at
seven labels the stump lands exactly on the majority-class baseline, which is
another way of saying it learned "usually keep".

The derived cut is stable under leave-one-out, which is worth recording
because it is the one part of this that could have gone either way:

| setting | cut range across folds | as a fraction of the shipped value |
|---|---|---|
| `composition.min_sharpness` | 0.1510 – 0.1558 | 4.0% |
| `composition.min_brightness` | 7.76 – 8.42 | 3.3% |

**None of this is new capability.** It is `judge.derive`, which already runs,
already uses every label, and already reports its own confidence. The labels
are not "unused"; they are used, exhaustively, by the thing they were
collected for.

## 3. A learned scorer fitted to them is not better than `score()`

The generous reading: pool every per-photo label into one "would you show
this?" set — 17 rows — and fit a seven-feature ridge-penalised logistic
regression (sharpness, brightness, megapixels, has-people, favourite,
exact-sidecar, has-GPS). Leave-one-out against the hand-written function,
whose own cut is also fitted on each training fold so the comparison is not
rigged.

The ridge strength is **swept and the best reported**, which is generous to
the learned model on purpose: a negative result resting on one badly chosen
hyper-parameter would be worthless. The sweep is also where the story is:

| ridge L2 | leave-one-out accuracy |
|---|---|
| 0.0 (no penalty) | 0.471 |
| 0.001 | 0.529 |
| 0.01 | 0.471 |
| 0.1 | 0.471 |
| 1.0 | **0.647** |

**Unpenalised, the model is worse than a coin.** It only reaches 0.647 when
the penalty is strong enough that the weights collapse and it predicts the
base rate for everything — which is to say, it reaches its best score by
ceasing to be a model. Against the baselines:

| | leave-one-out accuracy |
|---|---|
| Fitted seven-feature scorer, best over the sweep | **0.647** |
| Hand-written `score()`, cut fitted per fold | **0.765** |
| Majority class (answer "keep" every time) | **0.647** |
| 95% Wilson half-width at n=17 | **±0.207** |

The best fitted model lands exactly on the do-nothing baseline. The
hand-written function is twelve points ahead of it. **Neither gap is
significant** — at n=17 the interval on 0.647 runs from 0.44 to 0.85 — so the
honest sentence is not "the learned scorer is worse", it is **"there is no
measurable signal here either way"**, and a change made on no signal is a
change made on nothing.

## 4. …and it would rewrite almost every memory

An accuracy comparison says whether a model is better. This says what is at
stake when it is not. Substituting the fitted scorer for `engine.score` —
fitted at the *weakest* penalty in the sweep, so the version that moves most —
and rebuilding every memory on the reference library:

| | |
|---|---|
| Memories before / after | 27 / 27 |
| **Memories whose shot list changed** | **25** |
| Shots before | 582 |
| **Shots surviving in the same memory** | **86** |
| **Fraction of shots replaced** | **85.2%** |

A model with no demonstrable advantage that replaces six shots in every seven
is not a neutral experiment to ship. This is the number that turns "probably
not worth it" into "no".

## 5. The structural reason, which more sittings do not fix

Small *n* is the obvious objection and it is not the deepest one.

**`score()` ranks; the labels gate.** Every question the sequence asks is
"is this photograph unusable?" — too blurry, too dark, too washed out, the
same moment as that one. `score()` answers a different question: *of the
photographs that are usable, which belong in the twenty-four slots?* Measured,
the two are nearly unrelated: the AUC of `score()` against these labels is
**0.591**, where 0.5 is "knows nothing about them".

There is no step in `plan.STEPS` that produces a preference between two
acceptable photographs, and none of the six blind steps could be reworded into
one — a pairwise "which of these belongs in the memory?" is a different kind
of question with a different sampling strategy and a different derivation. So
finishing the sitting, or doing three more, produces more gate labels. It
does not produce a single ranking label.

**What would change the answer**, stated so that it can be checked later:

1. A pairwise preference step, which is a design change and not a fit over
   existing data.
2. Labels in the hundreds rather than the tens. Section 1 puts the per-sitting
   ceiling at 40 on a plain install, so this only becomes reachable if labels
   accumulate across sittings — which is what section 6 is for.
3. A re-run of this experiment showing the fitted scorer beating both the
   hand-written function *and* the majority baseline by more than the interval,
   with the blast radius reported beside it.

## 6. What was built instead: `labels.jsonl`

The honest deliverable for a negative result is to stop destroying the
evidence, so that the question can be asked again later with more of it.

`calibration.json` is **working state**. `Calibration.record` deliberately
*replaces* an earlier answer about the same photograph — someone walking back
through a step is changing their mind, and a threshold must not average both
answers. `reset_setting` deliberately *deletes* a threshold's answers, because
`rekindle calibrate --redo` has to start clean. Both are correct for working
state. Both destroy evidence: recalibrate twice and the first sitting is gone.

So `src/rekindle/calibrate/labels.py` writes a second file with the other
lifetime — **append-only, one JSON object per line, never rewritten, never
truncated by a reset**:

```json
{"v": 1, "at": "2026-09-12T11:24:43+00:00", "setting": "composition.min_sharpness",
 "subject": "<file hash>", "value": 0.0886, "rejected": true,
 "question": "Too blurry to put in a memory?", "library_size": 19318,
 "rekindle": "0.1.0"}
```

`question` is stored **verbatim** rather than as a step id, and that is the
field that makes the log worth keeping. A reworded question is a different
question — "too blurry to use?" and "too blurry to put in a memory?" will not
get the same answers — and a log keyed only on `composition.min_sharpness`
would pool them silently. `library_size` and `rekindle` date a label against
the library and the code that produced it.

It is written at `Session.answer`, the single place a judgement is recorded,
so both front ends get it and neither can forget. It satisfies every
constraint the brief set for a learned component, trivially, by not being one:

* **Determinism.** Nothing in the memory engine, the recipes or the config
  loader imports `calibrate.labels`. The log cannot change a byte of output.
* **Explainability.** No score changed, so no rejection lost its reason.
* **It degrades.** No file, no labels, no behaviour change. That is the
  default and it stays the default.
* **Personal data.** File hashes of a specific person's photographs and what
  they thought of them. It lives in the data directory, which `.gitignore`
  excludes as `[Dd]ata/`, alongside `calibration.json` and the caption cache.
  `summary()` is the only function whose output is safe to print, and it
  returns counts.
* **No new runtime dependency.** Stdlib `json`. The experiment that produced
  every number above is stdlib too — the logistic fit is forty lines of plain
  Python — deliberately, because a fit that needed numpy would have to justify
  itself before it had been shown to help.

`rekindle calibrate --status` names the file, says it is append-only and says
it is never committed, once there is anything in it. A file the user did not
ask for, holding judgements about their own photographs, has to be
discoverable; it counts rather than listing, because naming a photograph there
would defeat the point of keeping it private.

## 7. Mutations run

Every line broken, tests run, line restored. All seven caught.

| # | mutation | caught by |
|---|---|---|
| 1 | the log is opened for writing instead of appending | five tests, including the reset one |
| 2 | `Session.answer` stops writing a label | `test_the_session_writes_a_label_for_every_answer` |
| 3 | the question text is dropped from the line | `test_a_label_round_trips` |
| 4 | a malformed line aborts the read instead of being skipped | `test_a_malformed_line_is_skipped_and_the_rest_survive` |
| 5 | the version check on a line is dropped | `test_a_line_from_a_future_version_is_skipped_rather_than_guessed_at` |
| 6 | `summary` leaks the subject it counted | `test_the_summary_counts_and_names_nothing_else` |
| 7 | the log is written outside the data directory | seven tests |

Mutation 5 **survived the first run**, and the reason is the shape this
project keeps finding: the bad-version line in the test was
`{"v": 999, "setting": "b"}`, which is skipped whether or not the version is
checked, because the missing fields raise anyway. The test could not fail. It
was replaced with a line that is complete and valid in every way except its
version, and split out so the two properties are pinned separately.

## 8. The precedent this follows

This project already ships one signal deliberately disabled with a test that
fails if it ever starts working. The reasoning there and here is the same: a
measurement that does not support a change is a reason not to make it, and
recording the measurement is what stops the same idea arriving again next
milestone with no memory of why it was declined.
