"""Synthetic-library benchmark for `MemoryIndex`.

NOT a test - the filename is deliberately outside pytest's `test_*.py` pattern,
because this takes minutes and needs a multi-hundred-megabyte scratch database.
It lives in `tests/` so it ships with the sdist and anyone can re-measure the
numbers in `docs/decision-log-memory-scale.md` rather than trusting them.

    python tests/bench_memory_index.py --rows 300000 --db <scratch>/big.sqlite

It generates rows only - no image files - which is all a *selection* benchmark
needs: nothing between `MemoryIndex.open` and `build_all` opens a pixel.

Peak memory is `tracemalloc`, i.e. the Python heap, not RSS. That is the right
number here: the thing being measured is how many Python objects the index
holds, and RSS on Windows is dominated by the interpreter and by an allocator
that does not return freed arenas to the OS.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
import tracemalloc
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rekindle.db import PhotoStore
from rekindle.models import FaceRegion, Gps, MediaType, Photo, PhotoMeta, TzSource

# Shape taken from the reference library (19,480 rows), scaled where a bigger
# library would plainly have more of something. Measured there:
#   paths 137 chars mean, albums 23, people 13, colour 113, 12.0% with GPS,
#   94.3% images, 66 album names, 40 people, 24 years.
YEARS = tuple(range(2003, 2027))
GPS_FRACTION = 0.12
IMAGE_FRACTION = 0.943
FAVOURITE_FRACTION = 0.0002


def _albums(n: int) -> list[str]:
    """`Photos from <year>` for every year, plus named albums.

    The year folders matter: they are the biggest albums in a real Takeout and
    they are the ones `albums.presentable` filters out, so a benchmark without
    them measures the wrong `album_story` offer count.
    """
    named = [f"Trip {i:03d}" for i in range(n)]
    named += [f"Christmas {y}" for y in YEARS[-6:]]
    return [f"Photos from {y}" for y in YEARS] + named


def generate(
    db_path: Path, rows: int, *, seed: int = 20260912, chronological: bool = False
) -> None:
    """Build a synthetic library.

    `chronological` decides how ROWID ORDER relates to capture time, and it is
    the single most consequential knob in this file for the lazy index.

    Measured on the reference library, 93.9% of adjacent rows are already in
    capture order, because `rekindle index` walks Takeout's per-year folders.
    Every date-keyed query - `by_year`, `by_month_day`, `by_date` - therefore
    asks for a nearly contiguous run of rowids, and the LRU cache serves it
    well. The default here is the OPPOSITE: dates are uniformly random per
    row, so a date-keyed query scatters across the whole table and the cache
    is as useless as it can be. That is deliberate for a headline number - a
    benchmark should not flatter the design - but it means the timings from
    the default are a worst case and a real library will do better.
    """
    rng = random.Random(seed)
    album_names = _albums(max(8, rows // 700))
    people_names = [f"Person {i:03d}" for i in range(max(8, rows // 2500))]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    start = datetime(YEARS[0], 1, 1, tzinfo=UTC)
    span = int((datetime(YEARS[-1], 12, 31, tzinfo=UTC) - start).total_seconds())
    if chronological:
        moments = sorted(start + timedelta(seconds=rng.randrange(span)) for _ in range(rows))
    batch: list[Photo] = []
    with PhotoStore(db_path) as store:
        for i in range(rows):
            taken = moments[i] if chronological else start + timedelta(seconds=rng.randrange(span))
            local = taken.replace(tzinfo=None) + timedelta(hours=5, minutes=30)
            people = rng.sample(people_names, rng.choices((0, 1, 2, 3), (44, 30, 18, 8))[0])
            mine = rng.sample(album_names, rng.choices((1, 2), (85, 15))[0])
            gps = None
            if rng.random() < GPS_FRACTION:
                gps = Gps(lat=rng.uniform(8.0, 35.0), lon=rng.uniform(68.0, 97.0))
            image = rng.random() < IMAGE_FRACTION
            batch.append(
                Photo(
                    file_hash=f"{i:064x}",
                    paths=[
                        Path(
                            f"D:/Takeout/Google Photos/{mine[0]}/"
                            f"IMG_{i:08d}_{rng.randrange(10**9):09d}.jpg"
                        )
                    ],
                    media_type=MediaType.IMAGE if image else MediaType.VIDEO,
                    meta=PhotoMeta(
                        taken_at_utc=taken,
                        taken_at_local=local,
                        tz_source=TzSource.TAKEOUT,
                        gps=gps,
                        people=people,
                        face_regions=[
                            FaceRegion(name=p, x=0.5, y=0.4, w=0.1, h=0.1) for p in people
                        ],
                        keywords=[],
                        description=None,
                        favorite=rng.random() < FAVOURITE_FRACTION,
                        camera_make="Synthetic",
                        camera_model="Bench",
                        width=4032,
                        height=3024,
                        archived=rng.random() < 0.008,
                        trashed=False,
                        phash=rng.getrandbits(64) if image else None,
                        sharpness=rng.uniform(1.0, 400.0) if image else None,
                        phash_error=None if image else "video",
                        brightness=rng.uniform(20.0, 220.0) if image else None,
                        colour=f"{rng.getrandbits(128):032x}" * 4 if image else None,
                    ),
                    first_seen=taken,
                    last_seen=taken,
                    albums=mine,
                    sidecar_match="exact" if rng.random() < 0.8 else "none",
                )
            )
            if len(batch) >= 5000:
                store.upsert_many(batch)
                batch.clear()
        if batch:
            store.upsert_many(batch)


def measure(
    db_path: Path,
    *,
    recipes: bool = True,
    per_recipe: int = 1,
    trace: bool = True,
    recipe: str | None = None,
) -> dict[str, float]:
    """One pass. `trace` costs roughly 2x wall-clock, so time and memory are
    measured in SEPARATE runs rather than reported from the same one.

    `recipe` narrows to one, which is the case a person actually runs:
    `rekindle memory --recipe album_story --key X` asks one recipe for its
    offers and builds one memory. `--all-recipes` is the worst case for a lazy
    index and the best case for a materialised one, so measuring only that
    would answer a question nobody asks.
    """
    from rekindle.memory.engine import all_offers, build_all
    from rekindle.memory.index import MemoryIndex
    from rekindle.memory.recipes import registered

    gc.collect()
    if trace:
        tracemalloc.start()
        tracemalloc.reset_peak()
    t0 = time.perf_counter()
    store = PhotoStore(db_path)
    index = MemoryIndex.open(store)
    open_s = time.perf_counter() - t0

    result: dict[str, float] = {
        "rows": index.report.total,
        "allowed": index.count(),
        "open_s": open_s,
    }
    if trace:
        open_current, open_peak = tracemalloc.get_traced_memory()
        result["open_peak_mb"] = open_peak / 1e6
        result["after_open_mb"] = open_current / 1e6
    if recipes:
        t1 = time.perf_counter()
        if recipe:
            one = next(r for r in registered() if r.name == recipe)
            offers = one.offers(index)
            result["recipe"] = recipe
        else:
            offers = all_offers(index)
        offers_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        built, report = build_all(index, offers, per_recipe_limit=per_recipe)
        build_s = time.perf_counter() - t2
        result.update(
            {
                "offers": len(offers),
                "offers_s": offers_s,
                "built": len(built),
                "build_s": build_s,
                "total_s": open_s + offers_s + build_s,
                "deduped": report.deduped,
            }
        )
        if trace:
            current, peak = tracemalloc.get_traced_memory()
            result["peak_mb"] = peak / 1e6
            result["resident_mb"] = current / 1e6
    if trace:
        tracemalloc.stop()
    store.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=300_000)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--generate", action="store_true", help="(re)build the synthetic database")
    ap.add_argument("--open-only", action="store_true")
    ap.add_argument("--per-recipe", type=int, default=1)
    ap.add_argument("--no-trace", action="store_true", help="wall-clock only, no tracemalloc")
    ap.add_argument(
        "--chronological",
        action="store_true",
        help="generate rows in capture order, as a real Takeout index is (see generate)",
    )
    ap.add_argument("--recipe", help="measure ONE recipe rather than --all-recipes")
    args = ap.parse_args()

    if args.generate or not args.db.exists():
        t = time.perf_counter()
        generate(args.db, args.rows, chronological=args.chronological)
        print(f"generated {args.rows} rows in {time.perf_counter() - t:.1f}s -> {args.db}")

    out = measure(
        args.db,
        recipes=not args.open_only,
        per_recipe=args.per_recipe,
        trace=not args.no_trace,
        recipe=args.recipe,
    )
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
