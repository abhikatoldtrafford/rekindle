"""Read a `torch.save`d state dict into numpy arrays, WITHOUT importing torch.

WHY THIS EXISTS
---------------
The LAION aesthetic head is 3.7 MB of weights distributed as a `.pth`. Loading
it the obvious way costs a 2.5 GB torch install, which defeats the purpose of
having a light `semantic` extra at all - the head is five matrix multiplies.

WHY IT IS SAFE
--------------
A `.pth` is a pickle, and unpickling arbitrary pickles executes arbitrary
code. `_Unpickler.find_class` here is an ALLOW-LIST of exactly the seven names
a plain tensor state dict needs. Anything else - `os.system`, `builtins.eval`,
a torch module reconstruction, a custom class - raises `UnpicklingError`
naming what it refused. This is strictly stronger than `torch.load`'s own
`weights_only=True`, which permits a much larger set.

FORMAT
------
`torch.save` (since 1.6) writes a zip:

    <archive>/data.pkl     the pickled object graph; tensors appear as
                           persistent ids ('storage', <dtype>, key, dev, numel)
    <archive>/data/<key>   that storage's raw little-endian bytes
    <archive>/version      "3\\n"

The legacy (pre-1.6) tar-ish format is NOT supported and is rejected by name,
because silently misreading weights produces a model that runs and is wrong.
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path
from typing import Any

#: torch storage class name -> (numpy dtype string, bytes per element).
_DTYPES: dict[str, tuple[str, int]] = {
    "FloatStorage": ("<f4", 4),
    "DoubleStorage": ("<f8", 8),
    "HalfStorage": ("<f2", 2),
    "LongStorage": ("<i8", 8),
    "IntStorage": ("<i4", 4),
    "ShortStorage": ("<i2", 2),
    "CharStorage": ("<i1", 1),
    "ByteStorage": ("<u1", 1),
    "BoolStorage": ("|b1", 1),
}

_ALLOWED: set[tuple[str, str]] = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_tensor"),
    *(("torch", name) for name in _DTYPES),
}


class TorchFileError(RuntimeError):
    """The file is not a state dict this reader will touch."""


class _Storage:
    """Placeholder standing in for a torch storage until a tensor claims it."""

    __slots__ = ("key", "dtype", "itemsize", "numel")

    def __init__(self, key: str, dtype: str, itemsize: int, numel: int) -> None:
        self.key = key
        self.dtype = dtype
        self.itemsize = itemsize
        self.numel = numel


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *_rest):
    return _TensorPlan(storage, storage_offset, tuple(size), tuple(stride))


class _TensorPlan:
    __slots__ = ("storage", "offset", "size", "stride")

    def __init__(self, storage: _Storage, offset: int, size: tuple, stride: tuple) -> None:
        self.storage = storage
        self.offset = offset
        self.size = size
        self.stride = stride


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(fh)
        self._archive = archive
        self._prefix = prefix

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(
                f"refusing to unpickle {module}.{name}: this reader allows only "
                "plain tensor state dicts. If you trust this file, load it with "
                "torch.load yourself."
            )
        if name in _DTYPES:
            return name  # the storage class is used only as a dtype tag
        if name == "OrderedDict":
            from collections import OrderedDict

            return OrderedDict
        return _rebuild_tensor_v2

    def persistent_load(self, pid: Any) -> _Storage:
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unsupported persistent id {pid!r}")
        _, storage_type, key, _location, numel = pid
        name = storage_type if isinstance(storage_type, str) else storage_type.__name__
        if name not in _DTYPES:
            raise pickle.UnpicklingError(f"unsupported storage type {name}")
        dtype, itemsize = _DTYPES[name]
        return _Storage(str(key), dtype, itemsize, int(numel))


def load_state_dict(path: Path) -> dict[str, Any]:
    """`{name: numpy array}` for every tensor in a `torch.save`d state dict."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded by callers
        raise TorchFileError("reading a .pth needs numpy") from exc

    if not zipfile.is_zipfile(path):
        raise TorchFileError(
            f"{path} is not a zip-format torch file. Files saved by torch < 1.6 "
            "use a legacy format this reader deliberately does not guess at."
        )
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pkl = next((n for n in names if n.endswith("data.pkl")), None)
        if pkl is None:
            raise TorchFileError(f"{path} has no data.pkl; it is not a torch save file.")
        prefix = pkl[: -len("data.pkl")]
        with archive.open(pkl) as fh:
            graph = _Unpickler(fh, archive, prefix).load()

        cache: dict[str, bytes] = {}

        def materialise(obj: Any) -> Any:
            if isinstance(obj, _TensorPlan):
                key = obj.storage.key
                if key not in cache:
                    with archive.open(f"{prefix}data/{key}") as fh:
                        cache[key] = fh.read()
                raw = cache[key]
                flat = np.frombuffer(raw, dtype=obj.storage.dtype)
                count = 1
                for d in obj.size:
                    count *= d
                start = obj.offset
                window = flat[start : start + count]
                if window.size != count:
                    raise TorchFileError(
                        f"{path}: tensor of {count} elements does not fit the "
                        f"{flat.size}-element storage it points into."
                    )
                array = np.lib.stride_tricks.as_strided(
                    window,
                    shape=obj.size,
                    strides=tuple(s * obj.storage.itemsize for s in obj.stride),
                )
                return np.array(array, copy=True)
            if isinstance(obj, dict):
                return {k: materialise(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(materialise(v) for v in obj)
            return obj

        result = materialise(graph)
    if not isinstance(result, dict):
        raise TorchFileError(f"{path} unpickled to {type(result).__name__}, not a dict.")
    return result
