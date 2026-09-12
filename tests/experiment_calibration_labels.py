"""Are the calibration labels a training set? Measured, not argued.

NOT a test - the filename is outside pytest's `test_*.py` pattern, like
`bench_memory_index.py`, because it needs a real library and a real
`calibration.json` and neither exists in CI. It lives in `tests/` so it ships
and so the numbers in `docs/decision-log-calibration-labels.md` can be
re-measured rather than believed.

    python tests/experiment_calibration_labels.py --data-dir data

It reads only what is already on disk and writes nothing. **The labels are
judgements about a specific person's photographs and never leave the machine;
this prints counts, accuracies and threshold values, never a file hash and
never a path.**

Stdlib only, deliberately - no numpy. A fit that needed numpy would belong
behind the `semantic` extra, and the first question is whether a fit is worth
having at all.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from rekindle import config
from rekindle.calibrate import plan, state
from rekindle.calibrate.judge import MIN_JUDGEMENTS, Judgement, derive
from rekindle.db import PhotoStore
from rekindle.memory.engine import score
from rekindle.memory.index import MemoryIndex
from rekindle.models import Photo

# ---------------------------------------------------------------- the census


def census(record: state.Calibration) -> dict:
    """What labels exist, and what a FINISHED sitting could ever produce.

    The distinction matters more than it looks. Only the `blind` steps produce
    a judgement at all: a slider step records a number the user typed and a
    default step records nothing but "I looked". So the ceiling on labels is
    not 13 steps x 10 rounds.
    """
    blind = [s for s in plan.STEPS if plan.MODE_BLIND in s.modes]
    no_extras = [s for s in blind if s.needs in (plan.NEEDS_NOTHING, plan.NEEDS_FINGERPRINTS)]
    by_setting: dict[str, list[state.Answer]] = {}
    for answer in record.answers:
        by_setting.setdefault(answer.setting, []).append(answer)
    return {
        "steps_total": len(plan.STEPS),
        "steps_that_can_label": len(blind),
        "steps_that_can_label_without_extras": len(no_extras),
        "rounds_per_step": 10,
        "ceiling_with_extras": len(blind) * 10,
        "ceiling_without_extras": len(no_extras) * 10,
        "labels_on_disk": len(record.answers),
        "labelled_settings": {k: len(v) for k, v in sorted(by_setting.items())},
        "rejected": sum(1 for a in record.answers if a.rejected),
        "finished": record.finished,
    }


# --------------------------------------------------- one-dimensional stumps


def loo_stump(answers: list[state.Answer], *, setting: str) -> dict:
    """Leave-one-out: does a cut learned from n-1 labels beat the shipped one?

    The comparison the whole of part 2 rests on, at its narrowest and most
    favourable to the labels: the model is the one the labels were collected
    FOR, on the feature they were collected ABOUT.
    """
    step = plan.BY_SETTING[setting]
    shipped = config.CATALOGUE[setting].default
    rows = [Judgement(value=a.value, rejected=a.rejected, subject=a.subject) for a in answers]

    learned_right = shipped_right = 0
    cuts: list[float] = []
    refused = 0
    for i in range(len(rows)):
        held, rest = rows[i], rows[:i] + rows[i + 1 :]
        found = derive(rest, direction=step.direction, safer=step.safer)
        if not found.usable:
            refused += 1
            continue
        cuts.append(found.value)
        learned_right += _predicts(found.value, held, step.direction) == held.rejected
        shipped_right += _predicts(shipped, held, step.direction) == held.rejected
    scored = len(cuts)
    full = derive(rows, direction=step.direction, safer=step.safer)
    return {
        "setting": setting,
        "labels": len(rows),
        "rejected": sum(1 for r in rows if r.rejected),
        "folds_scored": scored,
        "folds_refused": refused,
        "learned_accuracy": learned_right / scored if scored else float("nan"),
        "shipped_accuracy": shipped_right / scored if scored else float("nan"),
        "majority_accuracy": _majority(rows),
        "shipped_value": shipped,
        "full_fit_value": full.value if full.usable else float("nan"),
        "cut_min": min(cuts) if cuts else float("nan"),
        "cut_max": max(cuts) if cuts else float("nan"),
        "cut_spread_vs_shipped": ((max(cuts) - min(cuts)) / shipped) if cuts and shipped else 0.0,
    }


def _predicts(cut: float, row: Judgement, direction: str) -> bool:
    return row.value < cut if direction == plan.REJECT_BELOW else row.value > cut


def _majority(rows: list[Judgement]) -> float:
    """The accuracy of answering the commoner label every time. Any model that
    does not beat THIS has learned nothing at all."""
    if not rows:
        return float("nan")
    yes = sum(1 for r in rows if r.rejected)
    return max(yes, len(rows) - yes) / len(rows)


# ------------------------------------------- the pooled "would you show it?"


FEATURES = (
    "sharpness",
    "brightness",
    "megapixels",
    "has_people",
    "favourite",
    "sidecar_exact",
    "has_gps",
)


@dataclass
class Row:
    x: tuple[float, ...]
    y: int
    hand: float  # what the hand-written `score()` says


def _features(photo: Photo) -> tuple[float, ...]:
    m = photo.meta
    width, height = m.width or 0, m.height or 0
    return (
        m.sharpness if m.sharpness is not None else 0.0,
        (m.brightness if m.brightness is not None else 0.0) / 255.0,
        (width * height) / 1e7,
        1.0 if any(m.people) else 0.0,
        1.0 if m.favorite else 0.0,
        1.0 if photo.sidecar_match == "exact" else 0.0,
        1.0 if m.gps is not None else 0.0,
    )


def build_rows(record: state.Calibration, index: MemoryIndex) -> list[Row]:
    """Every per-photo label, pooled into one "would you show this?" set.

    Pooled deliberately. Ten labels about blur and seven about darkness are
    two questions, but a SCORER has to answer one - "how well does this photo
    present?" - so pooling them is the most generous reading available, and it
    is the reading under which a learned scorer has the most data.

    Pair labels (`dedup.phash_distance`, `dedup.cosine`) are excluded: they
    are judgements about a RELATION between two photographs, and no per-photo
    scorer can consume one.
    """
    per_photo = {
        s.setting
        for s in plan.STEPS
        if plan.MODE_BLIND in s.modes and s.attribute and s.setting in plan.BY_SETTING
    }
    rows: list[Row] = []
    ranked = _sharp_ranks(record, index)
    for answer in record.answers:
        if answer.setting not in per_photo:
            continue
        photo = index.get(answer.subject)
        if photo is None:
            continue
        rows.append(
            Row(
                x=_features(photo),
                y=1 if answer.rejected else 0,
                hand=score(photo, ranked.get(photo.file_hash, 0.5)),
            )
        )
    return rows


def _sharp_ranks(record: state.Calibration, index: MemoryIndex) -> dict[str, float]:
    """`score()` takes a sharpness PERCENTILE within the set being ranked, so
    the percentile is computed within the labelled set - the same way the
    engine computes it within one memory's candidates."""
    photos = [p for p in (index.get(a.subject) for a in record.answers) if p is not None]
    measured = sorted(p.meta.sharpness for p in photos if p.meta.sharpness is not None)
    out = {}
    for photo in photos:
        value = photo.meta.sharpness
        if value is None or not measured:
            out[photo.file_hash] = 0.5
        else:
            out[photo.file_hash] = sum(1 for m in measured if m < value) / len(measured)
    return out


def _fit(rows: list[Row], *, l2: float = 1.0, steps: int = 4000, lr: float = 0.2) -> list[float]:
    """Ridge-penalised logistic regression, plain Python, batch gradient.

    No numpy: the base install is four packages and a fit that needed a fifth
    would have to justify itself before it had even been shown to help. With
    at most a few dozen rows and seven features, 4,000 batch steps is a few
    milliseconds.
    """
    n_features = len(rows[0].x) if rows else 0
    w = [0.0] * (n_features + 1)  # last is the bias
    if not rows:
        return w
    for _ in range(steps):
        grad = [0.0] * len(w)
        for row in rows:
            z = sum(wi * xi for wi, xi in zip(w[:-1], row.x, strict=True)) + w[-1]
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            err = p - row.y
            for j, xj in enumerate(row.x):
                grad[j] += err * xj
            grad[-1] += err
        for j in range(len(w) - 1):
            grad[j] = grad[j] / len(rows) + l2 * w[j]
        grad[-1] /= len(rows)
        for j in range(len(w)):
            w[j] -= lr * grad[j]
    return w


def _predict(w: list[float], x: tuple[float, ...]) -> float:
    z = sum(wi * xi for wi, xi in zip(w[:-1], x, strict=True)) + w[-1]
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


#: Ridge strengths swept for the learned model. The BEST is reported, which
#: is generous to it and deliberately so: a negative result that rested on one
#: badly chosen hyper-parameter would be worthless. At L2=1.0 with seventeen
#: rows the penalty dominates the likelihood and the fit collapses to the base
#: rate, which would have produced the answer this experiment expects for the
#: wrong reason.
L2_GRID = (0.0, 0.001, 0.01, 0.1, 1.0)


def loo_scorer(rows: list[Row]) -> dict:
    """Leave-one-out: a fitted multi-feature scorer against the hand-written
    one, on the same labels, with the do-nothing baseline beside them."""
    if len(rows) < MIN_JUDGEMENTS + 1:
        return {"rows": len(rows), "refusal": "too few labels to hold any out"}
    by_l2: dict[float, float] = {}
    for l2 in L2_GRID:
        right = 0
        for i in range(len(rows)):
            held, rest = rows[i], rows[:i] + rows[i + 1 :]
            w = _fit(rest, l2=l2)
            right += (1 if _predict(w, held.x) >= 0.5 else 0) == held.y
        by_l2[l2] = right / len(rows)
    hand = 0
    for i in range(len(rows)):
        held, rest = rows[i], rows[:i] + rows[i + 1 :]
        # The hand-written baseline gets the same courtesy: its cut is fitted
        # on the training fold too, so this is not a rigged comparison.
        cut = _best_cut([(r.hand, r.y) for r in rest])
        hand += (1 if held.hand < cut else 0) == held.y
    best = max(by_l2.values())
    yes = sum(r.y for r in rows)
    return {
        "rows": len(rows),
        "rejected": yes,
        "learned_accuracy_by_l2": {str(k): v for k, v in by_l2.items()},
        "learned_accuracy_best": best,
        "hand_written_accuracy": hand / len(rows),
        "majority_accuracy": max(yes, len(rows) - yes) / len(rows),
        "hand_written_auc": _auc([(r.hand, r.y) for r in rows]),
        "wilson_95_halfwidth": _wilson(best, len(rows)),
    }


def _best_cut(pairs: list[tuple[float, int]]) -> float:
    """The `score()` cut that best separates rejects on the training fold.
    A LOW score is the reject side, because `score()` is "presents well"."""
    values = sorted({v for v, _ in pairs})
    if not values:
        return 0.0
    cuts = [values[0] - 1.0] + [(a + b) / 2 for a, b in zip(values, values[1:], strict=False)]
    cuts.append(values[-1] + 1.0)
    return min(cuts, key=lambda c: sum(1 for v, y in pairs if (1 if v < c else 0) != y))


def _auc(pairs: list[tuple[float, int]]) -> float:
    """Probability that a randomly chosen KEPT photo scores above a randomly
    chosen REJECTED one. 0.5 is "this score knows nothing about these
    labels" - which is a finding, not a failure."""
    pos = [v for v, y in pairs if y == 0]  # kept: score() should be higher
    neg = [v for v, y in pairs if y == 1]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def _wilson(p: float, n: int) -> float:
    """Half-width of a 95% Wilson interval. Printed beside every accuracy,
    because at n=17 the interval is most of the number line and a report that
    hid that would be the sort of confident wrongness this project avoids."""
    z = 1.96
    denom = 1 + z * z / n
    return (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom


# ------------------------------------------- what a fitted scorer would DO


def blast_radius(rows: list[Row], index: MemoryIndex) -> dict:
    """How much would swapping `score()` for the fitted model actually change?

    An accuracy comparison says whether the model is better. This says what is
    at stake if it is not: it rebuilds every memory in the library with the
    fitted scorer substituted for the hand-written one and counts the shots
    that move. A model with no measurable advantage that rewrites a third of
    every montage is not a neutral experiment to ship.
    """
    from rekindle.memory import engine

    before = _shots(index)
    # The weakest penalty in the grid, i.e. the model that fits these labels
    # hardest. The point of this measurement is how much a scorer would MOVE,
    # so the version that moves most is the honest one to measure.
    weights = _fit(rows, l2=min(L2_GRID))

    def learned(photo: Photo, sharp_rank: float) -> float:
        # Higher is better, as `score()` is - the model predicts REJECTED, so
        # the score is one minus it. `sharp_rank` is accepted and ignored: the
        # model already has sharpness as a feature.
        return 1.0 - _predict(weights, _features(photo))

    original = engine.score
    engine.score = learned  # type: ignore[assignment]
    try:
        after = _shots(index)
    finally:
        engine.score = original  # type: ignore[assignment]

    ids = sorted(set(before) | set(after))
    changed = [k for k in ids if before.get(k) != after.get(k)]
    kept = sum(len(set(before.get(k, ())) & set(after.get(k, ()))) for k in ids)
    total = sum(len(v) for v in before.values())
    return {
        "memories_before": len(before),
        "memories_after": len(after),
        "memories_changed": len(changed),
        "shots_before": total,
        "shots_after": sum(len(v) for v in after.values()),
        "shots_surviving_in_the_same_memory": kept,
        "fraction_of_shots_replaced": 1 - kept / total if total else 0.0,
    }


def _shots(index: MemoryIndex) -> dict[str, tuple[str, ...]]:
    from rekindle.memory.engine import all_offers, build_all

    built, _ = build_all(index, all_offers(index), per_recipe_limit=3)
    return {f"{m.recipe}:{m.key}": tuple(s.file_hash for s in m.shots) for m in built}


# ------------------------------------------------------------------- report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--json", type=Path, help="also write the whole report here")
    ap.add_argument(
        "--blast-radius",
        action="store_true",
        help="also rebuild every memory with the fitted scorer and diff (slow)",
    )
    args = ap.parse_args()

    record = state.load(args.data_dir)
    out: dict = {"census": census(record)}

    out["per_setting"] = []
    for setting, answers in sorted(_by_setting(record).items()):
        if setting not in plan.BY_SETTING or len(answers) <= MIN_JUDGEMENTS:
            out["per_setting"].append(
                {"setting": setting, "labels": len(answers), "refusal": "too few to hold one out"}
            )
            continue
        out["per_setting"].append(loo_stump(answers, setting=setting))

    db = args.data_dir / "rekindle.sqlite"
    if db.is_file():
        store = PhotoStore(db)
        index = MemoryIndex.open(store)
        try:
            rows = build_rows(record, index)
            out["pooled_scorer"] = loo_scorer(rows)
            if args.blast_radius and rows:
                out["blast_radius"] = blast_radius(rows, index)
        finally:
            index.close()
            store.close()
    else:
        out["pooled_scorer"] = {"refusal": f"no library at {db}"}

    print(json.dumps(out, indent=2, default=float))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")


def _by_setting(record: state.Calibration) -> dict[str, list[state.Answer]]:
    out: dict[str, list[state.Answer]] = {}
    for answer in record.answers:
        out.setdefault(answer.setting, []).append(answer)
    return out


if __name__ == "__main__":
    main()
