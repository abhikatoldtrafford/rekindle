"""XMP sidecar reading.

Sidecars are the richest metadata available to a plain folder: Lightroom,
digiKam and osxphotos all write person names AND face regions here, which
EXIF cannot carry.

Parsed with defusedxml - XMP is arbitrary XML from an untrusted source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from defusedxml import ElementTree as DefusedET

from rekindle.models import FaceRegion

_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "mwg-rs": "http://www.metadataworkinggroup.com/schemas/regions/",
    "stArea": "http://ns.adobe.com/xmp/sType/Area#",
    "Iptc4xmpExt": "http://iptc.org/std/Iptc4xmpExt/2008-02-29/",
}

_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


@dataclass(frozen=True)
class XmpData:
    people: tuple[str, ...] = ()
    face_regions: tuple[FaceRegion, ...] = ()
    keywords: tuple[str, ...] = ()
    description: str | None = None


def find_sidecar(image_path: Path) -> Path | None:
    """`photo.jpg.xmp` is the common convention; `photo.xmp` is also used."""
    for candidate in (
        image_path.with_name(image_path.name + ".xmp"),
        image_path.with_suffix(".xmp"),
    ):
        if candidate.is_file():
            return candidate
    return None


def _bag_items(root, path: str) -> list[str]:
    out: list[str] = []
    for container in root.iterfind(f".//{path}", _NS):
        for li in container.iterfind(".//rdf:li", _NS):
            text = (li.text or "").strip()
            if text:
                out.append(text)
    return out


def _q(ns: str, name: str) -> str:
    return f"{{{_NS[ns]}}}{name}"


def _area_value(area, key: str) -> float | None:
    """MWG areas appear as attributes (Lightroom) or child elements (digiKam).

    Returns None when absent, so the caller can reject the region instead of
    defaulting a coordinate to 0.0.
    """
    raw = area.get(_q("stArea", key))
    if raw is None:
        child = area.find(f"stArea:{key}", _NS)
        raw = child.text if child is not None else None
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def read_xmp(path: Path) -> XmpData:
    try:
        root = DefusedET.parse(path).getroot()
    except Exception:
        return XmpData()

    people = _bag_items(root, "Iptc4xmpExt:PersonInImage")
    keywords = _bag_items(root, "dc:subject")

    # x-default is the canonical entry in an XMP language alternative. Taking
    # the first rdf:li returns whichever language happens to be serialised
    # first, which in a multi-language Lightroom catalog is arbitrary.
    alts = [n for n in root.iterfind(".//dc:description//rdf:li", _NS) if (n.text or "").strip()]
    chosen = next((n for n in alts if n.get(_XML_LANG) == "x-default"), alts[0] if alts else None)
    description = chosen.text.strip() if chosen is not None else None

    regions: list[FaceRegion] = []
    for li in root.iterfind(".//mwg-rs:RegionList//rdf:li", _NS):
        type_node = li.find("mwg-rs:Type", _NS)
        if type_node is not None and (type_node.text or "").strip().lower() != "face":
            continue
        name_node = li.find("mwg-rs:Name", _NS)
        area = li.find("mwg-rs:Area", _NS)
        if name_node is None or area is None:
            continue
        name = (name_node.text or "").strip()
        if not name:
            continue
        vals = {k: _area_value(area, k) for k in ("x", "y", "w", "h")}
        if any(v is None for v in vals.values()):
            # Never invent 0.0. A fabricated box at the origin with zero area
            # would be handed to face-aware cropping as if it were real.
            continue
        regions.append(FaceRegion(name=name, **vals))

    named = list(dict.fromkeys([*people, *(r.name for r in regions)]))
    return XmpData(
        people=tuple(named),
        face_regions=tuple(regions),
        keywords=tuple(keywords),
        description=description,
    )
