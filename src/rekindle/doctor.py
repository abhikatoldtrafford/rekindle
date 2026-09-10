"""Coverage reporting. Tells the user what metadata they actually have
before they spend time indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from rekindle.meta.exif import HEIF_AVAILABLE
from rekindle.models import SourceReport

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
            f"{report.json_sidecars} Google JSON sidecars are present but not yet parsed. "
            "This looks like a Google Takeout export - the Takeout parser that reads those "
            "(face tags, descriptions) is not implemented yet."
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
            "`uv sync --extra heic` if your library has iPhone photos.[/dim]"
        )

    for warning in diagnosis.warnings:
        console.print(f"\n[yellow]![/yellow] {warning}")
