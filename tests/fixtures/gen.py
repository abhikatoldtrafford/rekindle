"""Synthetic photo libraries for tests.

CI has no photos, so every fixture is generated. The shapes here deliberately
reproduce the messes real libraries contain: duplicates across folders, edited
variants, sidecars, and files whose extensions lie.
"""

from __future__ import annotations

import io
import json
import struct
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.TiffImagePlugin import IFDRational

_XMP_TEMPLATE = """<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"
    xmlns:stArea="http://ns.adobe.com/xmp/sType/Area#"
    xmlns:stDim="http://ns.adobe.com/xmp/sType/Dimensions#"
    xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/">
{description}
{keywords}
{persons}
{regions}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


def make_jpeg(
    path: Path,
    *,
    size: tuple[int, int] = (64, 48),
    color: tuple[int, int, int] = (120, 80, 60),
    taken: datetime | None = None,
    offset: str | None = None,
    gps: tuple[float, float] | None = None,
    make: str | None = None,
    model: str | None = None,
) -> Path:
    """Write a small real JPEG, optionally with EXIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    exif = im.getexif()
    if make:
        exif[0x010F] = make
    if model:
        exif[0x0110] = model
    if taken or offset:
        sub = {}
        if taken:
            sub[0x9003] = taken.strftime("%Y:%m:%d %H:%M:%S")
        if offset:
            sub[0x9011] = offset
        exif[0x8769] = sub
    if gps:
        lat, lon = gps
        exif[0x8825] = {
            1: "N" if lat >= 0 else "S",
            2: _deg_to_dms(abs(lat)),
            3: "E" if lon >= 0 else "W",
            4: _deg_to_dms(abs(lon)),
        }
    im.save(path, "JPEG", exif=exif)
    return path


def _deg_to_dms(deg: float) -> tuple[IFDRational, IFDRational, IFDRational]:
    """EXIF rationals MUST be IFDRational.

    Passing raw (num, den) tuples makes Pillow raise
    `TypeError: bad operand type for abs(): 'tuple'` inside _limit_rational.
    Verified: IFDRational round-trips to 0.000000 degrees of error in both
    hemispheres, on Pillow 11.1.0, 11.3.0 and 12.3.0.
    """
    d = int(deg)
    m_full = (deg - d) * 60
    m = int(m_full)
    s = round((m_full - m) * 60 * 100)
    return (IFDRational(d, 1), IFDRational(m, 1), IFDRational(s, 100))


def make_corrupt_exif_jpeg(path: Path) -> Path:
    """A real, fully-openable JPEG whose embedded EXIF sub-IFD pointer is
    truncated - reproduces Pillow's `UserWarning: Corrupt EXIF data.
    Expecting to read N bytes but only got 0.` seen on a real Takeout export.

    The image itself decodes fine; only the tiny hand-crafted TIFF blob in
    its APP1 segment is broken, which is exactly the "damaged EXIF, healthy
    file" case read_exif must handle quietly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (8, 8), (1, 2, 3))
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    data = bytearray(buf.getvalue())

    exif_header = b"Exif\x00\x00"
    tiff = b"II*\x00\x08\x00\x00\x00"  # little-endian TIFF header, IFD0 at offset 8
    # IFD0's one entry is the EXIF sub-IFD pointer (tag 0x8769), aimed at an
    # offset past the end of this (deliberately short) TIFF blob.
    entry = struct.pack("<HHII", 0x8769, 4, 1, 26)
    ifd0 = struct.pack("<H", 1) + entry + struct.pack("<I", 0)
    app1_payload = exif_header + tiff + ifd0
    app1_segment = b"\xff\xe1" + struct.pack(">H", len(app1_payload) + 2) + app1_payload

    path.write_bytes(bytes(data[:2] + app1_segment + data[2:]))
    return path


def make_xmp_sidecar(
    image_path: Path,
    *,
    people: tuple[str, ...] | list[str] = (),
    regions: tuple[tuple[str, float, float, float, float], ...] = (),
    description: str | None = None,
    keywords: tuple[str, ...] | list[str] = (),
) -> Path:
    """Write `<image>.xmp` next to the image."""
    desc = ""
    if description:
        desc = (
            "    <dc:description><rdf:Alt><rdf:li xml:lang='x-default'>"
            f"{description}</rdf:li></rdf:Alt></dc:description>"
        )
    kw = ""
    if keywords:
        items = "".join(f"<rdf:li>{k}</rdf:li>" for k in keywords)
        kw = f"    <dc:subject><rdf:Bag>{items}</rdf:Bag></dc:subject>"
    persons = ""
    if people:
        items = "".join(f"<rdf:li>{p}</rdf:li>" for p in people)
        persons = (
            f"    <Iptc4xmpExt:PersonInImage><rdf:Bag>{items}</rdf:Bag></Iptc4xmpExt:PersonInImage>"
        )
    regs = ""
    if regions:
        items = "".join(
            "<rdf:li rdf:parseType='Resource'>"
            f"<mwg-rs:Name>{n}</mwg-rs:Name><mwg-rs:Type>Face</mwg-rs:Type>"
            f"<mwg-rs:Area stArea:x='{x}' stArea:y='{y}' "
            f"stArea:w='{w}' stArea:h='{h}' stArea:unit='normalized'/>"
            "</rdf:li>"
            for n, x, y, w, h in regions
        )
        regs = (
            "    <mwg-rs:Regions rdf:parseType='Resource'>"
            f"<mwg-rs:RegionList><rdf:Bag>{items}</rdf:Bag></mwg-rs:RegionList>"
            "</mwg-rs:Regions>"
        )
    out = image_path.with_name(image_path.name + ".xmp")
    out.write_text(
        _XMP_TEMPLATE.format(description=desc, keywords=kw, persons=persons, regions=regs),
        encoding="utf-8",
    )
    return out


def build_library(root: Path) -> Path:
    """A small library containing every mess we intend to handle."""
    root.mkdir(parents=True, exist_ok=True)
    year = root / "Photos from 2014"
    album = root / "Goa Trip"

    # Same bytes in both a year folder and an album folder.
    beach = make_jpeg(
        year / "IMG_0001.jpg",
        color=(10, 120, 200),
        taken=datetime(2014, 3, 21, 17, 45),
        offset="+05:30",
        gps=(15.2993, 74.1240),
        make="Canon",
        model="EOS R",
    )
    (album / "IMG_0001.jpg").parent.mkdir(parents=True, exist_ok=True)
    (album / "IMG_0001.jpg").write_bytes(beach.read_bytes())

    # An edited variant of the same original.
    make_jpeg(
        year / "IMG_0001-edited.jpg", color=(20, 130, 210), taken=datetime(2014, 3, 21, 17, 45)
    )

    # A Google-style JSON sidecar that must be ignored as media.
    (year / "IMG_0001.jpg.json").write_text(
        json.dumps({"photoTakenTime": {"timestamp": "1395423900"}}), encoding="utf-8"
    )

    # A photo with an XMP sidecar carrying people and face regions.
    portrait = make_jpeg(
        album / "IMG_0002.jpg", color=(200, 160, 140), taken=datetime(2014, 3, 22, 9, 0)
    )
    make_xmp_sidecar(
        portrait,
        people=["Alice", "Bob"],
        regions=(("Alice", 0.4, 0.35, 0.18, 0.24),),
        description="morning on the beach",
        keywords=("beach", "holiday"),
    )

    # A photo with no metadata at all.
    make_jpeg(year / "IMG_0003.jpg", color=(90, 90, 90))

    # An extension that lies: JPEG bytes named .heic
    make_jpeg(year / "actually_jpeg.heic", color=(30, 30, 30))

    # Not media.
    (root / "notes.txt").write_text("just some notes", encoding="utf-8")

    # --- shapes observed in a real 2,692-photo Takeout export ---

    # Deleted photos. Must never be indexed.
    make_jpeg(root / "Trash" / "deleted.jpg", color=(5, 5, 5))

    # A sidecar whose photo lives in an archive part the user never extracted.
    # 44% of sidecars were orphaned this way in the validation export.
    (year / "IMG_9999.jpg.supplemental-metadata.json").write_text(
        json.dumps({"title": "IMG_9999.jpg"}), encoding="utf-8"
    )

    # A motion photo: Google exports the video component as a separate .MP file
    # (ISO-BMFF, ftyp:isom) beside the still.
    make_jpeg(album / "PXL_0001.jpg", color=(70, 140, 90), taken=datetime(2025, 9, 6, 13, 3))
    (album / "PXL_0001.MP").write_bytes(
        b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 128
    )

    # An edited variant of the motion photo's STILL. Both the .jpg still and
    # the .MP video share the stem "PXL_0001" - proves the edited variant
    # links to the still's hash, never the video's, regardless of which one
    # the scan happens to visit first.
    make_jpeg(album / "PXL_0001-edited.jpg", color=(75, 145, 95), taken=datetime(2025, 9, 6, 13, 3))

    # The counter lands INSIDE the suffix when two photos share a filename.
    (year / "DSC_0880.JPG.supplemental-metadata(1).json").write_text(
        json.dumps({"title": "DSC_0880.JPG"}), encoding="utf-8"
    )

    # A duplicate where only the SECOND-visited copy carries an XMP sidecar -
    # the exact Takeout layout where an album copy has no sidecar of its own.
    # "AAA_First" sorts before "ZZZ_Second", so the plain copy is indexed
    # first and the metadata only shows up when the duplicate is merged.
    first_copy = make_jpeg(root / "AAA_First" / "IMG_8000.jpg", color=(11, 22, 33))
    second_copy = root / "ZZZ_Second" / "IMG_8000.jpg"
    second_copy.parent.mkdir(parents=True, exist_ok=True)
    second_copy.write_bytes(first_copy.read_bytes())
    make_xmp_sidecar(second_copy, people=["Carol"], description="second copy only")

    # An Apple edit sidecar (.aae) with no matching photo. _sidecar_target
    # only understands Google's JSON naming convention, so unlike a .json
    # sidecar this must NEVER be counted as an orphan (or as a JSON sidecar
    # at all) - an iPhone library emits one .AAE per edited photo, and
    # miscounting them would raise a false "missing archive parts" alarm on
    # a perfectly complete library.
    (year / "IMG_7000.aae").write_text("dummy Apple edit sidecar", encoding="utf-8")

    return root
