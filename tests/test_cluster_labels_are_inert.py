"""The cluster label is cosmetic, and this is what keeps it that way.

`rekindle semantic clusters` prints a "looks like" column: the vocabulary
phrase nearest each cluster's centroid. On the reference library it is often
right - cluster #31 really is green countryside, #50 really is food - and it
is sometimes badly wrong in a way that matters. **Cluster #16, 270 photographs
of institutional buildings with gardens, was labelled "a hospital or a
clinic", and contains no hospital.**

A hospital is a sensitive context in this project: `caption_vocab.toml` rule 4
forbids naming one, and `known-limitations.md` records this cluster as the
near-miss that rule exists for. A cosmetic label that lands on a
sensitive-context word stops being cosmetic the moment anything READS it.

Nothing does. The whole guarantee is that sentence, and until now the only
thing enforcing it was that sentence. These tests enforce it: the label may be
displayed, and it may not reach anything that decides what a user sees.

Scores range 0.127-0.282 and 12 of 57 clusters were labelled "a religious
ceremony or temple", so the vocabulary is absorbing whatever is nearest rather
than recognising anything. That is the measurement behind the rule.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "rekindle"

#: The one module allowed to read a label: the command that prints the table.
DISPLAY_ONLY = SRC / "semantic" / "cli.py"

#: Where the attribute is defined and assigned. Not a reader.
DEFINES = SRC / "semantic" / "cluster.py"

LABEL_ATTRS = {"label", "label_score"}


def _reads_label(path: Path) -> list[str]:
    """Lines in `path` that read `.label` or `.label_score` off something.

    An AST walk rather than a grep, so `labelled = []`, a `label=` keyword
    argument and the word "label" in a comment are not findings. Only an
    attribute ACCESS counts, which is what "reads it" means.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in LABEL_ATTRS:
            found.append(f"{path.name}:{node.lineno}")
    return found


def _python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def test_only_the_clusters_table_reads_a_cluster_label():
    """The structural version of "nothing may read it".

    A future change that wires the label into a caption, a memory title, a
    filename or an exclusion fails here and has to argue with this file
    first - which is the argument, written down, in the module docstring.
    """
    offenders = {}
    for path in _python_files():
        if path in (DISPLAY_ONLY, DEFINES):
            continue
        reads = _reads_label(path)
        if reads:
            offenders[path.name] = reads
    assert not offenders, (
        f"cluster labels are read outside the display: {offenders}. They are "
        "nearest-phrase matches with scores of 0.127-0.282, and one landed on "
        '"a hospital or a clinic" over 270 photographs of a campus. Nothing '
        "may decide anything with them - see this file's docstring."
    )


def test_the_memory_package_cannot_see_a_label_at_all():
    """Narrower and stronger, for the package that decides what a user sees.

    `rekindle.memory` builds titles, captions and specs. It never imports
    `rekindle.semantic` at all - the optional half is handed in through
    protocols - so a label cannot reach a memory even by accident. This pins
    the property directly rather than as a side effect of that boundary.
    """
    memory = sorted((SRC / "memory").rglob("*.py"))
    assert memory, "the memory package moved; this test needs updating"
    for path in memory:
        if "__pycache__" in path.parts:
            continue
        assert not _reads_label(path), f"{path.name} reads a cluster label"


def test_the_detector_finds_a_read_when_there_is_one():
    """The detector itself, before anything is concluded from it. A walk that
    found nothing everywhere would make both tests above pass while proving
    nothing."""
    assert _reads_label(DISPLAY_ONLY), "the clusters table does read the label"


@pytest.mark.parametrize("attr", sorted(LABEL_ATTRS))
def test_the_attributes_this_guards_still_exist(attr):
    """If `Cluster.label` is renamed, this file is guarding a name that no
    longer exists and would pass forever while the new one spread."""
    pytest.importorskip("numpy")
    from rekindle.semantic.cluster import Cluster

    assert attr in Cluster.__dataclass_fields__
