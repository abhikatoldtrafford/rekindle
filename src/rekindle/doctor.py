"""Coverage reporting. Tells the user what metadata they actually have
before they spend time indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from rekindle.db import PhotoStore
from rekindle.enrich.takeout import EnrichReport
from rekindle.extras import install_command_markup
from rekindle.meta.exif import HEIF_AVAILABLE
from rekindle.models import SourceReport, TzSource

_LOW_DATE_PCT = 50.0


@dataclass(frozen=True)
class Diagnosis:
    report: SourceReport
    warnings: list[str] = field(default_factory=list)

    def pct(self, n: int) -> float:
        total = self.report.media_indexed
        return round(100.0 * n / total, 1) if total else 0.0


def diagnose(report: SourceReport) -> Diagnosis:
    warnings: list[str] = []
    total = report.media_indexed

    if total:
        if 100.0 * report.with_date / total < _LOW_DATE_PCT:
            warnings.append(
                "Low date coverage - most photos have no capture date, so "
                "time-based memories will be unreliable."
            )
        if report.with_people == 0:
            warnings.append(
                "No person data found in XMP sidecars. Person-based memories "
                "and person exclusions are unavailable. Use date-range and "
                "folder exclusions instead - they always work."
            )
    if report.long_paths:
        warnings.append(
            f"{len(report.long_paths)} long paths found (>=260 chars). On Windows, "
            "enable LongPathsEnabled or some files may be unreadable."
        )
    if report.xmp_unreadable:
        warnings.append(
            f"{report.xmp_unreadable} XMP sidecars are present but would not parse, so "
            "their people, keywords and descriptions were not read. Without this line "
            "the row above says the sidecars are there and the people row says nobody "
            "is tagged, which reads like missing data rather than broken files."
        )
    if report.orphan_sidecars:
        matched = max(report.json_sidecars - report.orphan_sidecars, 0)
        denom = matched + report.orphan_sidecars
        rate = round(100.0 * report.orphan_sidecars / denom, 1) if denom else 0.0
        warnings.append(
            f"INCOMPLETE EXPORT: {report.orphan_sidecars} metadata sidecars "
            f"({rate}%) have no matching photo. Those photos are almost certainly "
            "in Takeout archive parts you have not extracted yet. Extract every "
            "part into the SAME folder before indexing, or your library will be "
            "silently missing photos."
        )
    if report.json_sidecars and total and report.json_sidecars >= total * 0.25:
        warnings.append(
            f"{report.json_sidecars} Google JSON sidecars are present but not yet read. "
            "This looks like a Google Takeout export - run `rekindle enrich <folder>` "
            "after indexing to read the face tags, dates and album titles they carry."
        )
    if report.excluded_dirs:
        warnings.append(
            f"{report.excluded_dirs} files skipped in trash/system folders - "
            "deleted photos are never indexed."
        )
    return Diagnosis(report=report, warnings=warnings)


def render(diagnosis: Diagnosis, console: Console) -> None:
    r = diagnosis.report
    table = Table(title="Library report", show_header=True)
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_column("Coverage", justify="right")

    table.add_row("Files seen", str(r.files_seen), "")
    table.add_row("Photos indexed", str(r.media_indexed), "")
    table.add_row("With capture date", str(r.with_date), f"{diagnosis.pct(r.with_date)}%")
    table.add_row("With GPS", str(r.with_gps), f"{diagnosis.pct(r.with_gps)}%")
    table.add_row("With people", str(r.with_people), f"{diagnosis.pct(r.with_people)}%")
    table.add_row("With XMP sidecar", str(r.with_xmp), f"{diagnosis.pct(r.with_xmp)}%")
    if r.xmp_unreadable:
        table.add_row("[yellow]XMP sidecars unreadable[/yellow]", str(r.xmp_unreadable), "")
    table.add_row("Duplicates merged", str(r.duplicates_merged), "")
    table.add_row("Edited variants linked", str(r.edited_linked), "")
    table.add_row("Motion photo pairs", str(r.motion_pairs), "")
    table.add_row("JSON sidecars (unparsed)", str(r.json_sidecars), "")
    table.add_row("[yellow]Orphan sidecars[/yellow]", str(r.orphan_sidecars), "")
    table.add_row("Excluded (trash/system)", str(r.excluded_dirs), "")
    table.add_row("Skipped", str(r.total_skipped), "")
    # Indexed but undecodable: NOT part of `Skipped`, so the rows above still
    # add up to "Files seen". Shown because a library of corrupt files must
    # never look healthy.
    table.add_row("[yellow]Indexed but undecodable[/yellow]", str(r.undecodable), "")
    console.print(table)

    if r.skipped:
        console.print("\n[dim]Skipped by reason:[/dim]")
        for reason, count in sorted(r.skipped.items()):
            console.print(f"  {reason}: {count}")

    # Spec 5.0 requires doctor report anything unreadable. Collecting these and
    # never showing them would be the silent drop the whole design forbids.
    if r.unreadable:
        console.print(f"\n[red]Unreadable ({len(r.unreadable)}):[/red]")
        for path, reason in r.unreadable[:10]:
            console.print(f"  {path.name}: {reason}")
        if len(r.unreadable) > 10:
            console.print(f"  ... and {len(r.unreadable) - 10} more")

    if not HEIF_AVAILABLE:
        console.print(
            "\n[dim]HEIC support: disabled. Install with "
            f"{install_command_markup('heic')} if your library has iPhone photos.[/dim]"
        )

    for warning in diagnosis.warnings:
        console.print(f"\n[yellow]![/yellow] {warning}")


@dataclass(frozen=True)
class IndexDiagnosis:
    """What the STORED index holds. `diagnose()` above answers a different
    question - what a fresh scan of the filesystem can see - and after a
    successful enrich it would still report zero people forever, because a
    scan cannot see what a prior `enrich` run wrote to the database.
    """

    photos: int = 0
    with_date: int = 0
    with_gps: int = 0
    with_people: int = 0
    enriched: int = 0
    inherited: int = 0
    ambiguous: int = 0
    archived: int = 0
    conflicts: int = 0
    enriched_at: str | None = None
    index_root: str | None = None
    enrich_root: str | None = None
    warnings: list[str] = field(default_factory=list)

    def pct(self, n: int) -> float:
        return round(100.0 * n / self.photos, 1) if self.photos else 0.0


def diagnose_index(store: PhotoStore) -> IndexDiagnosis:
    """Read-only. Never writes - `store` is a `PhotoStore` the caller already
    opened; this function does not create or touch a database of its own.
    """
    counts = dict.fromkeys(
        (
            "photos",
            "with_date",
            "with_gps",
            "with_people",
            "enriched",
            "inherited",
            "ambiguous",
            "archived",
            "conflicts",
        ),
        0,
    )
    for photo in store.iter_photos():
        counts["photos"] += 1
        meta = photo.meta
        if meta.taken_at_utc and meta.tz_source is not TzSource.FILE_MTIME:
            counts["with_date"] += 1
        if meta.gps:
            counts["with_gps"] += 1
        if meta.people:
            counts["with_people"] += 1
        if photo.sidecar_match == "exact":
            counts["enriched"] += 1
        elif photo.sidecar_match == "inherited":
            counts["inherited"] += 1
        elif photo.sidecar_match == "ambiguous":
            counts["ambiguous"] += 1
        if meta.archived:
            counts["archived"] += 1
        if photo.metadata_conflict:
            counts["conflicts"] += 1

    enriched_at = store.get_meta("enriched_at")
    index_root = store.get_meta("index_root")
    enrich_root = store.get_meta("enrich_root")
    warnings: list[str] = []
    # `enriched_at` is the only thing that tells "enrich never ran" (None)
    # apart from "enrich ran and found nothing" (a timestamp, zero counts).
    # A count-based guard here ("with_people == 0") would conflate the two
    # and print this warning forever on a library that was correctly
    # enriched but genuinely has no face tags.
    if counts["photos"] and enriched_at is None:
        warnings.append(
            "This index has not been enriched. If it came from Google Takeout, "
            "`rekindle enrich <folder>` adds face tags, capture dates and album titles."
        )
    if index_root and enrich_root and index_root != enrich_root:
        warnings.append(
            f"Enrichment ran against a different folder ({enrich_root}) than the one "
            f"indexed ({index_root}). Photo paths are absolute, so almost nothing "
            "will have matched. Re-run both against the same folder."
        )
    if counts["ambiguous"]:
        warnings.append(
            f"{counts['ambiguous']} PHOTOS were deliberately not enriched because their "
            "sidecar could not be identified. Two different causes land here: (1) two or "
            "more sidecars named the photo and disagreed about it - on capture time, "
            "people, GPS, description or favourite; (2) a DIFFERENT photo that happens to "
            "share the filename also claimed the one sidecar, and nothing identified "
            "which of them it describes, so neither got it. `rekindle enrich` reports the "
            "two causes separately, and its own 'ambiguous' row counts SIDECARS, not "
            "photos - the two numbers measure different things and are not expected to "
            "match. This number is also small by construction: a disagreeing sidecar in a "
            "DIFFERENT folder is overridden rather than refused, and `rekindle enrich` "
            "reports those separately too. Do not read a low count here as no conflicts."
        )
    if counts["archived"]:
        warnings.append(
            f"{counts['archived']} photos are marked archived in Google Photos - the user "
            "deliberately hid them. They must not surface in a montage."
        )
    return IndexDiagnosis(
        **counts,
        enriched_at=enriched_at,
        index_root=index_root,
        enrich_root=enrich_root,
        warnings=warnings,
    )


def render_index(diagnosis: IndexDiagnosis, console: Console) -> None:
    table = Table(title="Index report", show_header=True)
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_column("Coverage", justify="right")
    d = diagnosis
    table.add_row("Photos in index", str(d.photos), "")
    table.add_row("With capture date", str(d.with_date), f"{d.pct(d.with_date)}%")
    table.add_row("With GPS", str(d.with_gps), f"{d.pct(d.with_gps)}%")
    table.add_row("With people", str(d.with_people), f"{d.pct(d.with_people)}%")
    table.add_row("Enriched from a sidecar", str(d.enriched), f"{d.pct(d.enriched)}%")
    table.add_row("Enriched via a derivative", str(d.inherited), f"{d.pct(d.inherited)}%")
    table.add_row("[yellow]Ambiguous photos (not enriched)[/yellow]", str(d.ambiguous), "")
    table.add_row("[yellow]Archived in Google Photos[/yellow]", str(d.archived), "")
    table.add_row("[yellow]EXIF/Google date conflicts[/yellow]", str(d.conflicts), "")
    console.print(table)
    console.print(f"\n[dim]Last enriched: {d.enriched_at or 'never'}[/dim]")
    for warning in d.warnings:
        console.print(f"\n[yellow]![/yellow] {warning}")


def render_enrich(report: EnrichReport, console: Console) -> None:
    table = Table(title="Enrichment report", show_header=True)
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for label, value in (
        ("JSON files seen", report.json_files_seen),
        ("Sidecars seen", report.sidecars_seen),
        ("  matched", report.matched),
        ("  superseded (another candidate won)", report.superseded),
        ("  [yellow]orphaned, vs. rows in the index[/yellow]", report.orphaned),
        ("  [yellow]ambiguous sidecars (refused)[/yellow]", report.ambiguous),
        ("  overridden by directory preference", report.directory_preference_broke_a_tie),
        ("  cross-photo collisions caught", report.cross_photo_collisions),
        ("Album metadata", report.album_metadata),
        ("Other JSON", report.other_json),
        ("Excluded (trash/system)", report.excluded_dirs),
        ("[yellow]Unparseable[/yellow]", report.unparseable),
        ("Photos enriched", report.photos_enriched),
        ("Derivatives enriched", report.derivatives_enriched),
        ("People added", report.people_added),
        ("Dates corrected", report.dates_corrected),
        ("GPS added", report.gps_added),
        ("Descriptions added", report.descriptions_added),
        ("Favourites added", report.favourites_added),
        ("Albums retitled", report.albums_retitled),
        ("[yellow]EXIF/Google conflicts[/yellow]", report.conflicts),
        ("EXIF/Google conflicts retracted", report.conflicts_retracted),
        ("Clustered dates suppressed", report.clustered_dates_suppressed),
        ("Sidecar title disagreed with filename", report.title_disagreements),
    ):
        table.add_row(label, str(value))
    console.print(table)

    # `title_disagreements` was counted and never shown. It is the measured
    # gap between a sidecar's own `title` field and the filename it actually
    # describes - the discovery this whole pass was rebuilt around, because
    # matching on `title` mis-paired 963 photos. Expected to be non-zero: it
    # counts every `(N)` sidecar, whose title omits the counter by design.
    console.print(
        f"\n[dim]{report.title_disagreements} sidecars name a file in their `title` "
        "field "
        "that is not the file they describe. Matching on `title` rather than on the "
        "sidecar's own filename would mis-pair every one of them; most are Takeout's "
        "`(N)` duplicates, whose title omits the counter.[/dim]"
    )

    # Report, never silently drop. If either identity fails, say so loudly -
    # a mismatch here is exactly the class of bug this whole pass was
    # redesigned to prevent.
    if report.json_files_seen != report.files_accounted:
        console.print(
            f"\n[red]ACCOUNTING BUG:[/red] {report.json_files_seen} JSON files seen but "
            f"{report.files_accounted} accounted for. Please file an issue."
        )
    if report.sidecars_seen != report.sidecars_accounted:
        console.print(
            f"\n[red]ACCOUNTING BUG:[/red] {report.sidecars_seen} sidecars seen but "
            f"{report.sidecars_accounted} accounted for. Please file an issue."
        )
    if report.album_title_collisions:
        console.print("\n[yellow]![/yellow] Album titles that would collide, left as folder names:")
        for folder, title in report.album_title_collisions:
            console.print(f"  {folder} -> {title}")
    if report.orphaned:
        console.print(
            f"\n[yellow]![/yellow] {report.orphaned} sidecars name a photo that has no row "
            "in the index. Some belong to archive parts you have not extracted; others are "
            "album copies of photos Takeout did not duplicate. Extract every part into "
            "the SAME folder and re-run if the number is large."
        )
        # Two honest numbers for one concept, not one: `doctor <root>` (a live
        # filesystem scan, BEFORE indexing) reports its own "Orphan sidecars"
        # row by comparing JSON sidecars against FILES ON DISK. This row
        # compares them against ROWS IN THE INDEX, which excludes whatever
        # that scan already dropped as non-media, undecodable, or excluded
        # (Trash/, etc). They measure different things about the same export
        # and will not match - reporting only one, unlabelled, would look
        # like a bug the first time a user runs both commands and compares.
        console.print(
            "  [dim]This is not the same number as `doctor`'s 'Orphan sidecars' row from "
            "a plain scan (files on disk, before indexing) - this one is measured against "
            "rows actually IN THE INDEX after indexing's own filtering. Both are correct "
            "for what they measure; they are expected to differ.[/dim]"
        )
    if report.directory_preference_broke_a_tie:
        console.print(
            f"\n[yellow]![/yellow] {report.directory_preference_broke_a_tie} photos had a "
            "sidecar in another album that DISAGREED with the one used; the same-directory "
            "copy was preferred (directory beats a distant disagreement by design). That is "
            "not the same as no conflict existing: compare it against the "
            f"'ambiguous sidecars (refused)' count above ({report.ambiguous}) - a "
            "disagreement in the SAME directory is refused outright, but a DISTANT one "
            "is silently "
            "overridden, so this number, not that one, is the true measure of disagreement."
        )
