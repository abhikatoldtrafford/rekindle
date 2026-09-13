# Publishing to PyPI

**`rekindle` is on PyPI. 0.1.0 and 0.1.1 are released and cannot be undone.**
This page describes how the next one goes out.

Publishing is effectively permanent. A release can be *yanked* — hidden from
resolvers — but it can never be deleted and the version number can never be
reused.

**There is no API token anywhere in this repository, and there does not need
to be.** Releases go out through PyPI Trusted Publishing (OIDC): the workflow
proves to PyPI that it is this repository running this workflow, and PyPI
issues a short-lived credential for that upload alone. Nothing to store,
nothing to rotate, nothing to leak. The account-scoped token used for the
first manual release should be revoked if it has not been already.

---

## The command

```bash
git tag v0.1.2 && git push origin v0.1.2
```

That is the whole of it. `.github/workflows/release.yml` fires on a `v*` tag
and will not publish unless, in this order: the suite passes on two operating
systems and on a minimal install, the tag matches `version` in
`pyproject.toml`, and a grep over the built sdist finds no personal data. Only
then does it upload, with `id-token: write` and no secret.

A tag that disagrees with `pyproject.toml` fails the job rather than
publishing something misnamed — which is the mistake this ordering exists to
prevent.

## Doing it by hand, if the workflow is broken

```bash
uv build && uv publish
```

This needs an API token, which is why it is the fallback and not the route.
Prefer fixing the workflow.

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
- [ ] `version` in `pyproject.toml` is the one you mean, and the tag matches
      it exactly. It cannot be reused, and the release workflow refuses a
      mismatch.
- [ ] `uv.lock` has been synced to the new version and committed. A stale lock
      is not caught by the tests and has shipped before.
- [ ] `pipx install .` from a clean shell, then `rekindle --help` and
      `rekindle doctor <a folder of photos>`.
- [ ] The README renders. `python -m twine check dist/*` catches the common
      Markdown failures without uploading anything.
- [ ] Every image in the README is an ABSOLUTE URL. A relative path renders on
      GitHub and is a broken image on the PyPI page; `tests/test_packaging.py`
      asserts it, including inside `<img src=...>`.

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

Taken, by us — `https://pypi.org/project/rekindle/`. This section used to say
it was free and to check again before uploading, which was right until
2026-09-12 and is now just wrong. Left here as the shape of a claim with a
shelf life: it needed a date on it, and it had one.

## One thing packaging cannot carry

`[tool.uv.sources]` points `torch` at PyTorch's CUDA index, because PyPI's
Windows and Linux torch wheels are CPU-only. **That instruction is a uv
setting and does not survive into wheel metadata.** Somebody who runs
`pipx install 'rekindle[semantic-gpu]'` from PyPI gets a CPU torch and no
warning from pip. `rekindle semantic doctor` reports it in as many words, and
the README says so under Install. There is no packaging fix; this is a
limitation of the wheel format, not of this configuration.
