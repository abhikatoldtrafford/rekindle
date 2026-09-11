"""Fetch model weights once, verify them, and never touch the network again.

THE CONTRACT
------------
`rekindle semantic setup` is the ONLY code in rekindle permitted to make a
network request. Every other entry point loads with `local_files_only=True`,
so a machine that has run setup works with the network unplugged, and a
machine that has not gets a message naming the command to run - never a
mid-run download of 1.7 GB the user did not ask for.

WHAT IS VERIFIED
----------------
Two independent things, and they answer different questions:

  * **The revision** is a 40-hex commit pinned in `registry.py`. It answers
    "am I getting the same model as the person who wrote the docs?" The Hub
    resolves a commit to content, so this is the reproducibility pin.
  * **The sha256 of every file**, against `model-locks.json`. It answers
    "did I get what the pin says, unmodified?" The lock file is generated
    from a real download by `scripts/lock_models.py` and committed, so a
    mismatch means the bytes on this machine differ from the bytes the
    project last saw - a corrupted download, a poisoned cache, or a Hub
    repository that force-pushed over its own history.

A file with no lock entry is reported as UNVERIFIED and, by default, that is
an error. `--allow-unlocked` downgrades it to a warning, for the one case it
exists for: generating the lock file in the first place.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from rekindle.semantic.availability import SemanticUnavailable
from rekindle.semantic.registry import (
    AESTHETIC_MODELS,
    EMBED_MODELS,
    FACE_MODELS,
    EmbedModel,
    FaceModel,
    HeadModel,
    Licence,
    RepoPin,
)

LOCK_FILE = Path(__file__).with_name("model-locks.json")

#: Read in 8 MiB blocks: a 1.7 GB safetensors file read byte-by-byte through
#: Python would take longer than downloading it.
_HASH_BLOCK = 8 * 1024 * 1024


class SetupError(RuntimeError):
    """Setup could not complete, and continuing would be unsafe."""


@dataclass(frozen=True)
class FetchedFile:
    repo_id: str
    revision: str
    filename: str
    path: Path
    size: int
    sha256: str
    verified: bool
    expected: str | None = None

    @property
    def mismatched(self) -> bool:
        return self.expected is not None and self.expected != self.sha256


@dataclass
class SetupReport:
    cache_dir: Path
    files: list[FetchedFile] = field(default_factory=list)
    licences: dict[str, Licence] = field(default_factory=dict)
    offline: bool = False

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def mismatches(self) -> list[FetchedFile]:
        return [f for f in self.files if f.mismatched]

    @property
    def unverified(self) -> list[FetchedFile]:
        return [f for f in self.files if f.expected is None]


def cache_dir_for(data_dir: Path) -> Path:
    """Where weights live. Inside the data directory, which is gitignored.

    `.gitignore` already carries `[Dd]ata/`, `models/`, `*.safetensors` and
    `*.onnx`, so a model cache here is covered four ways over. That redundancy
    is deliberate: this repo is public and a committed 1.7 GB weight file is
    not something a maintainer can take back.
    """
    return data_dir / "models"


def load_locks() -> dict[str, str]:
    """`"<repo>@<revision>/<file>" -> sha256`, or empty if no lock exists yet."""
    if not LOCK_FILE.is_file():
        return {}
    try:
        data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SetupError(f"{LOCK_FILE} is unreadable: {exc}") from exc
    files = data.get("files")
    if not isinstance(files, dict):
        raise SetupError(f"{LOCK_FILE} has no `files` object; it is not a lock file.")
    return {str(k): str(v) for k, v in files.items()}


def lock_key(pin: RepoPin, filename: str) -> str:
    return f"{pin.repo_id}@{pin.revision}/{filename}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_BLOCK):
            digest.update(chunk)
    return digest.hexdigest()


def _hub():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SemanticUnavailable(
            "downloading models needs huggingface_hub, from the 'semantic' "
            "extra (uv sync --extra semantic)"
        ) from exc
    return snapshot_download


def snapshot_dir(pin: RepoPin, cache_dir: Path, *, offline: bool = True) -> Path:
    """Local directory holding `pin`'s files. Never downloads when offline.

    `local_files_only=True` is the DEFAULT everywhere but `setup`. A library
    that silently downloads on first use cannot honestly claim to run offline,
    and the failure it produces when the network is missing arrives in the
    middle of a job rather than before it.
    """
    snapshot_download = _hub()
    try:
        return Path(
            snapshot_download(
                repo_id=pin.repo_id,
                revision=pin.revision,
                allow_patterns=list(pin.files),
                cache_dir=str(cache_dir),
                local_files_only=offline,
            )
        )
    except Exception as exc:  # noqa: BLE001 - hub raises a wide family
        if offline:
            raise SemanticUnavailable(
                f"{pin.repo_id}@{pin.revision[:8]} is not in {cache_dir}. "
                f"Run `rekindle semantic setup` first (it needs a network "
                f"connection once). Underlying error: {exc}"
            ) from exc
        raise SetupError(f"could not fetch {pin.repo_id}@{pin.revision[:8]}: {exc}") from exc


def fetch_pin(
    pin: RepoPin,
    cache_dir: Path,
    locks: dict[str, str],
    *,
    offline: bool = False,
) -> list[FetchedFile]:
    root = snapshot_dir(pin, cache_dir, offline=offline)
    out = []
    for filename in pin.files:
        path = root / filename
        if not path.is_file():
            raise SetupError(
                f"{pin.repo_id}@{pin.revision[:8]} was fetched but {filename} "
                f"is not at {path}. The repository layout has changed."
            )
        digest = sha256_of(path)
        expected = locks.get(lock_key(pin, filename))
        out.append(
            FetchedFile(
                repo_id=pin.repo_id,
                revision=pin.revision,
                filename=filename,
                path=path,
                size=path.stat().st_size,
                sha256=digest,
                verified=expected == digest,
                expected=expected,
            )
        )
    return out


def _pins_for(spec: EmbedModel | HeadModel | FaceModel, runtimes: Iterable[str]) -> list[RepoPin]:
    if isinstance(spec, EmbedModel):
        pins = []
        for runtime in runtimes:
            pin = spec.torch if runtime == "torch" else spec.onnx
            if pin is not None:
                pins.append(pin)
        if not pins:
            raise SetupError(
                f"model '{spec.key}' has no build for runtime(s) "
                f"{', '.join(runtimes)}; it offers {', '.join(spec.runtimes())}"
            )
        return pins
    return [spec.pin]


def run_setup(
    data_dir: Path,
    *,
    embed_keys: Iterable[str] = (),
    aesthetic_keys: Iterable[str] = (),
    face_keys: Iterable[str] = (),
    runtimes: Iterable[str] = ("torch",),
    offline: bool = False,
    allow_unlocked: bool = False,
) -> SetupReport:
    """Fetch and verify everything asked for. Idempotent; safe to re-run."""
    cache = cache_dir_for(data_dir)
    cache.mkdir(parents=True, exist_ok=True)
    locks = load_locks()
    report = SetupReport(cache_dir=cache, offline=offline)
    runtimes = tuple(runtimes)

    wanted: list[EmbedModel | HeadModel | FaceModel] = []
    for key in embed_keys:
        wanted.append(EMBED_MODELS[key])
    for key in aesthetic_keys:
        wanted.append(AESTHETIC_MODELS[key])
    for key in face_keys:
        wanted.append(FACE_MODELS[key])

    for spec in wanted:
        report.licences[spec.key] = spec.licence
        for pin in _pins_for(spec, runtimes):
            report.files.extend(fetch_pin(pin, cache, locks, offline=offline))

    if report.mismatches:
        lines = "\n".join(
            f"  {f.repo_id}/{f.filename}\n    expected {f.expected}\n    got      {f.sha256}"
            for f in report.mismatches
        )
        raise SetupError(
            "CHECKSUM MISMATCH. The bytes on this machine are not the bytes "
            "this project pinned. Do not use them.\n" + lines
        )
    if report.unverified and not allow_unlocked:
        names = ", ".join(f"{f.repo_id}/{f.filename}" for f in report.unverified)
        raise SetupError(
            f"{len(report.unverified)} file(s) have no entry in {LOCK_FILE.name} "
            f"and so could not be verified: {names}. Re-run with "
            "--allow-unlocked if you are deliberately generating the lock file."
        )
    return report


def write_lock(report: SetupReport, path: Path = LOCK_FILE) -> None:
    """Regenerate the lock file from what was actually fetched.

    Merges rather than replaces: a lock file is a cumulative record, and
    running setup for one model must not delete another model's entries.
    """
    existing = load_locks() if path.is_file() else {}
    for f in report.files:
        existing[f"{f.repo_id}@{f.revision}/{f.filename}"] = f.sha256
    payload = {
        "comment": (
            "sha256 of every pinned model file, as downloaded and verified. "
            "Regenerate with scripts/lock_models.py. A mismatch at setup time "
            "is an error, not a warning."
        ),
        "files": dict(sorted(existing.items())),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
