"""Which device will actually run the model, and whether that is a surprise.

The rule this module exists to enforce: **a CPU fallback on a machine that has
an NVIDIA GPU is a defect, not a graceful degradation.** It is also close to
invisible - the only symptom is that embedding 18,000 photos takes nine hours
instead of eight minutes, which looks like "ML is slow" rather than "your
torch is the CPU build". So:

  * `--device auto` picks CUDA when torch can use it, CPU otherwise, and
    **warns loudly** when it picks CPU while `nvidia-smi` reports a GPU;
  * `--device cuda` REFUSES to run on CPU. If you asked for the GPU and did
    not get it, you want an error, not a slow success;
  * `--device cpu` is silent - you asked for it.

`nvidia-smi` is the cross-check because it is the only signal available
without torch: it ships with the driver on Windows and Linux, and its absence
is itself informative (no NVIDIA driver -> a CPU fallback is expected, not a
defect).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Literal

Device = Literal["auto", "cuda", "cpu"]

#: How long to wait for `nvidia-smi`. It normally answers in <1s; a hung call
#: (driver in a bad state) must not hang `rekindle embed`.
NVIDIA_SMI_TIMEOUT_S = 10.0


class DeviceUnavailable(RuntimeError):
    """A device was demanded explicitly and cannot be provided."""


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_mib: int
    driver: str
    compute_capability: str


@dataclass(frozen=True)
class DeviceReport:
    """Everything a user needs to tell "it used the GPU" from "it did not"."""

    requested: Device
    device: str
    torch_version: str | None = None
    torch_cuda_build: str | None = None
    torch_sees_cuda: bool = False
    gpus: tuple[GpuInfo, ...] = ()
    driver_gpus: tuple[GpuInfo, ...] = ()
    warnings: tuple[str, ...] = field(default=())

    @property
    def fell_back_to_cpu(self) -> bool:
        """CPU is in use while the machine visibly has a usable NVIDIA GPU."""
        return self.device == "cpu" and bool(self.driver_gpus)

    def lines(self) -> list[str]:
        out = [f"device: {self.device} (requested: {self.requested})"]
        if self.torch_version:
            build = self.torch_cuda_build or "CPU-only build"
            out.append(f"torch:  {self.torch_version} (CUDA build: {build})")
        else:
            out.append("torch:  not installed")
        for gpu in self.gpus or self.driver_gpus:
            out.append(
                f"gpu:    {gpu.name}, {gpu.vram_mib} MiB VRAM, "
                f"driver {gpu.driver}, compute {gpu.compute_capability}"
            )
        if not (self.gpus or self.driver_gpus):
            out.append("gpu:    none detected")
        out.extend(f"!       {w}" for w in self.warnings)
        return out


def _nvidia_smi_gpus() -> tuple[GpuInfo, ...]:
    """What the NVIDIA driver says is present, independent of torch."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return ()
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, resolved via which()
            [
                exe,
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if proc.returncode != 0:
        return ()
    gpus = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            vram = int(float(parts[1]))
        except ValueError:
            continue
        gpus.append(
            GpuInfo(name=parts[0], vram_mib=vram, driver=parts[2], compute_capability=parts[3])
        )
    return tuple(gpus)


def _torch_state() -> tuple[str | None, str | None, bool, tuple[GpuInfo, ...]]:
    try:
        import torch
    except ImportError:
        return None, None, False, ()
    version = str(torch.__version__)
    cuda_build = torch.version.cuda
    try:
        sees = bool(torch.cuda.is_available())
    except (RuntimeError, AssertionError):  # pragma: no cover - driver dependent
        sees = False
    gpus: list[GpuInfo] = []
    if sees:
        driver_versions = {g.name: g.driver for g in _nvidia_smi_gpus()}
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append(
                GpuInfo(
                    name=p.name,
                    vram_mib=p.total_memory // (1024 * 1024),
                    driver=driver_versions.get(p.name, "unknown"),
                    compute_capability=f"{p.major}.{p.minor}",
                )
            )
    return version, cuda_build, sees, tuple(gpus)


def resolve_device(requested: Device = "auto") -> DeviceReport:
    """Decide the device and say, honestly, what happened.

    Raises `DeviceUnavailable` when `requested == "cuda"` and CUDA is not
    usable. That is deliberate: the one thing this function must never do is
    quietly hand back a CPU to somebody who asked for a GPU.
    """
    version, cuda_build, sees, gpus = _torch_state()
    driver_gpus = _nvidia_smi_gpus()
    warnings: list[str] = []

    if requested == "cpu":
        return DeviceReport("cpu", "cpu", version, cuda_build, sees, (), driver_gpus, ())

    if sees:
        return DeviceReport(requested, "cuda", version, cuda_build, True, gpus, driver_gpus, ())

    reason = _explain_no_cuda(version, cuda_build, driver_gpus)
    if requested == "cuda":
        raise DeviceUnavailable(f"--device cuda was requested but torch cannot use CUDA. {reason}")
    if driver_gpus:
        names = ", ".join(g.name for g in driver_gpus)
        warnings.append(
            f"FALLING BACK TO CPU although this machine has a {names}. {reason} "
            "Embedding will be roughly two orders of magnitude slower."
        )
    return DeviceReport(
        requested, "cpu", version, cuda_build, False, (), driver_gpus, tuple(warnings)
    )


def _explain_no_cuda(
    version: str | None, cuda_build: str | None, driver_gpus: tuple[GpuInfo, ...]
) -> str:
    if version is None:
        return "torch is not installed; install the 'semantic-gpu' extra."
    if cuda_build is None:
        return (
            f"torch {version} is a CPU-only build (torch.version.cuda is None). "
            "On Windows the default PyPI wheel is CPU-only; install the "
            "'semantic-gpu' extra, which pins the CUDA wheel index."
        )
    if not driver_gpus:
        return (
            f"torch {version} is a CUDA {cuda_build} build but no NVIDIA GPU was "
            "found by nvidia-smi (no driver, or no NVIDIA hardware)."
        )
    return (
        f"torch {version} is a CUDA {cuda_build} build and nvidia-smi sees a GPU, "
        "but torch.cuda.is_available() is False - usually a driver too old for "
        f"CUDA {cuda_build}, or a GPU already claimed by another process."
    )
