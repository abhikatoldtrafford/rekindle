# tests/test_xmp.py
from rekindle.meta.xmp import XmpData, find_sidecar, read_xmp
from tests.fixtures.gen import make_jpeg, make_xmp_sidecar


def test_finds_sidecar_named_image_dot_xmp(tmp_path):
    img = make_jpeg(tmp_path / "a.jpg")
    side = make_xmp_sidecar(img, people=["Alice"])
    assert find_sidecar(img) == side


def test_finds_sidecar_named_stem_dot_xmp(tmp_path):
    img = make_jpeg(tmp_path / "b.jpg")
    alt = tmp_path / "b.xmp"
    alt.write_text("<x/>", encoding="utf-8")
    assert find_sidecar(img) == alt


def test_returns_none_when_no_sidecar(tmp_path):
    img = make_jpeg(tmp_path / "c.jpg")
    assert find_sidecar(img) is None


def test_reads_people_description_and_keywords(tmp_path):
    img = make_jpeg(tmp_path / "d.jpg")
    side = make_xmp_sidecar(
        img,
        people=["Alice", "Bob"],
        description="morning on the beach",
        keywords=("beach", "holiday"),
    )
    d = read_xmp(side)
    assert sorted(d.people) == ["Alice", "Bob"]
    assert d.description == "morning on the beach"
    assert sorted(d.keywords) == ["beach", "holiday"]


def test_reads_face_regions_with_normalised_coordinates(tmp_path):
    img = make_jpeg(tmp_path / "e.jpg")
    side = make_xmp_sidecar(img, regions=(("Alice", 0.4, 0.35, 0.18, 0.24),))
    d = read_xmp(side)
    assert len(d.face_regions) == 1
    r = d.face_regions[0]
    assert r.name == "Alice"
    assert abs(r.x - 0.4) < 1e-6
    assert abs(r.h - 0.24) < 1e-6


def test_region_names_also_count_as_people(tmp_path):
    """A face region names a person even without PersonInImage."""
    img = make_jpeg(tmp_path / "f.jpg")
    side = make_xmp_sidecar(img, regions=(("Carol", 0.5, 0.5, 0.1, 0.1),))
    assert "Carol" in read_xmp(side).people


def test_area_as_child_elements_is_read_not_zeroed(tmp_path):
    """digiKam writes stArea as child elements, not attributes."""
    side = tmp_path / "d.xmp"
    side.write_text(
        """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"
    xmlns:stArea="http://ns.adobe.com/xmp/sType/Area#">
   <mwg-rs:Regions rdf:parseType='Resource'><mwg-rs:RegionList><rdf:Bag>
    <rdf:li rdf:parseType='Resource'>
      <mwg-rs:Name>Alice</mwg-rs:Name>
      <mwg-rs:Area rdf:parseType='Resource'>
        <stArea:x>0.4</stArea:x><stArea:y>0.35</stArea:y>
        <stArea:w>0.18</stArea:w><stArea:h>0.24</stArea:h>
      </mwg-rs:Area>
    </rdf:li>
   </rdf:Bag></mwg-rs:RegionList></mwg-rs:Regions>
  </rdf:Description></rdf:RDF></x:xmpmeta>""",
        encoding="utf-8",
    )
    r = read_xmp(side).face_regions
    assert len(r) == 1
    assert abs(r[0].x - 0.4) < 1e-6
    assert abs(r[0].h - 0.24) < 1e-6


def test_region_with_no_coordinates_is_rejected_not_zeroed(tmp_path):
    side = tmp_path / "e.xmp"
    side.write_text(
        """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"
    xmlns:stArea="http://ns.adobe.com/xmp/sType/Area#">
   <mwg-rs:Regions rdf:parseType='Resource'><mwg-rs:RegionList><rdf:Bag>
    <rdf:li rdf:parseType='Resource'>
      <mwg-rs:Name>Alice</mwg-rs:Name><mwg-rs:Area/>
    </rdf:li>
   </rdf:Bag></mwg-rs:RegionList></mwg-rs:Regions>
  </rdf:Description></rdf:RDF></x:xmpmeta>""",
        encoding="utf-8",
    )
    assert read_xmp(side).face_regions == ()


def test_description_prefers_x_default_language(tmp_path):
    side = tmp_path / "f.xmp"
    side.write_text(
        """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
   <dc:description><rdf:Alt>
     <rdf:li xml:lang='de'>Deutsch zuerst</rdf:li>
     <rdf:li xml:lang='x-default'>the real one</rdf:li>
   </rdf:Alt></dc:description>
  </rdf:Description></rdf:RDF></x:xmpmeta>""",
        encoding="utf-8",
    )
    assert read_xmp(side).description == "the real one"


def test_malformed_xmp_returns_empty_not_error(tmp_path):
    bad = tmp_path / "bad.xmp"
    bad.write_text("<not-closed>", encoding="utf-8")
    assert read_xmp(bad) == XmpData()


def test_missing_file_returns_empty_not_error(tmp_path):
    assert read_xmp(tmp_path / "nope.xmp") == XmpData()
