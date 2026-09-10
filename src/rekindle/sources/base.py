"""The Source protocol. Every library format normalises into Photo records."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rekindle.models import Photo, SourceReport


class Source(Protocol):
    name: str

    def scan(self, root: Path) -> tuple[list[Photo], SourceReport]:
        """Walk a library and return normalised photos plus an honest report.

        Implementations must count every skipped file with a reason. Silently
        dropping input is a bug.
        """
        ...
