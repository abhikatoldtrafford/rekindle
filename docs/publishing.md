# Publishing to PyPI

**Nothing in this repository has been published, and no agent will publish it.**
This page is here so that the owner can, in one command, when they decide to.

Publishing is effectively permanent. A release can be *yanked* — hidden from
resolvers — but it can never be deleted, the version number can never be
reused, and the project name is claimed by whoever uploads first. It also needs
the owner's own PyPI account and an API token that no automation here has or
should have.

---

## The command

```bash
uv build && uv publish
```

**Exactly what that does, in order:**

1. `uv build` writes two files into `dist/`:
   - `rekindle-0.1.0-py3-none-any.whl` — the wheel, ~410 KB, pure Python, no
     compiled extensions, installable on any OS and any Python ≥ 3.12.
   - `rekindle-0.1.0.tar.gz` — the source distribution, ~860 KB: `src/`,
     `tests/`, the top-level Markdown, `docs/*.md`, the CI workflow,
     `pyproject.toml` and `uv.lock`. Nothing else — the sdist is a whitelist
     (`[tool.hatch.build.targets.sdist]`), and `tests/test_packaging.py`
     asserts that no path in it can carry a photo, an index or another agent's
     worktree.
2. `uv publish` uploads **both files** to `https://pypi.org/`, authenticating
   with `UV_PUBLISH_TOKEN` or a `~/.pypirc` entry. On success:
   - `pip install rekindle` starts working for everyone, worldwide, within a
     minute or two.
   - `https://pypi.org/project/rekindle/` appears, rendering `README.md` as
     its front page.
   - **The name `rekindle` becomes permanently yours**, and the version
     `0.1.0` becomes permanently used.

If it fails partway, nothing is half-published: PyPI accepts or rejects each
file whole, and re-running is safe for a file that did not land.

## Before running it

- [ ] `uv run pytest` is green.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean.
- [ ] `uv build` succeeds, and `uv run pytest tests/test_packaging.py` passes
      **with the freshly built wheel in `dist/`** — the wheel-content
      assertions skip when `dist/` is empty, so an empty `dist/` looks like a
      pass.
- [ ] `version` in `pyproject.toml` is the one you mean. It cannot be reused.
- [ ] `pipx install .` from a clean shell, then `rekindle --help` and
      `rekindle doctor <a folder of photos>`.
- [ ] The README renders. `python -m twine check dist/*` catches the common
      Markdown failures without uploading anything.

## Rehearsing it without consequences

TestPyPI is a separate index with separate accounts, and the name is free
there too (checked: HTTP 404 on `test.pypi.org/pypi/rekindle/json`,
2026-09-12).

```bash
uv publish --publish-url https://test.pypi.org/legacy/
pipx install --index-url https://test.pypi.org/simple/ \
             --pip-args '--extra-index-url https://pypi.org/simple/' rekindle
```

The second command needs the extra index because rekindle's dependencies
(pillow, typer, rich, defusedxml) live on real PyPI, not TestPyPI. **TestPyPI
releases are also permanent**, so rehearse with a version you do not intend to
ship — `0.1.0.dev1`, say.

## The name

`rekindle` was free on PyPI when last checked: HTTP 404 from
`https://pypi.org/pypi/rekindle/json`, re-verified 2026-09-12. That is a fact
with a shelf life. Check it again immediately before uploading; if somebody
has taken it, the upload fails with a 403 and the fix is a different `name` in
`pyproject.toml`, not a retry.

## One thing packaging cannot carry

`[tool.uv.sources]` points `torch` at PyTorch's CUDA index, because PyPI's
Windows and Linux torch wheels are CPU-only. **That instruction is a uv
setting and does not survive into wheel metadata.** Somebody who runs
`pipx install 'rekindle[semantic-gpu]'` from PyPI gets a CPU torch and no
warning from pip. `rekindle semantic doctor` reports it in as many words, and
the README says so under Install. There is no packaging fix; this is a
limitation of the wheel format, not of this configuration.
