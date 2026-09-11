"""Device resolution. The one rule: never quietly hand back a CPU.

These tests run on a CI machine with no torch AND on this developer machine
with an A4000, so every one of them patches the two probes rather than reading
the real environment. The probes themselves are thin wrappers over
`nvidia-smi` and `torch.cuda`; what is worth testing is the DECISION.
"""

from __future__ import annotations

import subprocess

import pytest

from rekindle.semantic import runtime
from rekindle.semantic.runtime import DeviceUnavailable, GpuInfo, resolve_device

A4000 = GpuInfo("NVIDIA RTX A4000", 16376, "551.61", "8.6")


@pytest.fixture
def machine(monkeypatch):
    """Describe a machine: what nvidia-smi sees and what torch reports."""

    def configure(*, driver_gpus=(), torch_version=None, cuda_build=None, sees=False):
        monkeypatch.setattr(runtime, "_nvidia_smi_gpus", lambda: tuple(driver_gpus))
        monkeypatch.setattr(
            runtime,
            "_torch_state",
            lambda: (torch_version, cuda_build, sees, tuple(driver_gpus) if sees else ()),
        )

    return configure


def test_cuda_is_used_when_torch_can_see_it(machine):
    machine(driver_gpus=(A4000,), torch_version="2.9.1+cu126", cuda_build="12.6", sees=True)
    report = resolve_device("auto")
    assert report.device == "cuda"
    assert not report.fell_back_to_cpu
    assert report.warnings == ()
    assert report.gpus[0].name == "NVIDIA RTX A4000"


def test_cpu_fallback_on_a_gpu_machine_is_reported_as_a_defect(machine):
    """The whole reason this module exists."""
    machine(driver_gpus=(A4000,), torch_version="2.9.1", cuda_build=None, sees=False)
    report = resolve_device("auto")
    assert report.device == "cpu"
    assert report.fell_back_to_cpu
    assert report.warnings
    assert "FALLING BACK TO CPU" in report.warnings[0]
    assert "RTX A4000" in report.warnings[0]


def test_the_cpu_only_wheel_is_named_in_the_explanation(machine):
    """Windows PyPI torch is CPU-only and the user has no other way to know."""
    machine(driver_gpus=(A4000,), torch_version="2.9.1", cuda_build=None, sees=False)
    report = resolve_device("auto")
    assert "CPU-only build" in report.warnings[0]
    assert "semantic-gpu" in report.warnings[0]


def test_requesting_cuda_refuses_rather_than_falling_back(machine):
    machine(driver_gpus=(A4000,), torch_version="2.9.1", cuda_build=None, sees=False)
    with pytest.raises(DeviceUnavailable, match="--device cuda"):
        resolve_device("cuda")


def test_requesting_cuda_without_torch_explains_the_extra(machine):
    machine()
    with pytest.raises(DeviceUnavailable, match="semantic-gpu"):
        resolve_device("cuda")


def test_cpu_on_a_machine_with_no_gpu_is_not_a_fallback(machine):
    """No NVIDIA hardware: CPU is the correct answer and must be silent."""
    machine(torch_version="2.9.1", cuda_build=None, sees=False)
    report = resolve_device("auto")
    assert report.device == "cpu"
    assert not report.fell_back_to_cpu
    assert report.warnings == ()


def test_asking_for_cpu_is_always_silent(machine):
    machine(driver_gpus=(A4000,), torch_version="2.9.1+cu126", cuda_build="12.6", sees=True)
    report = resolve_device("cpu")
    assert report.device == "cpu"
    assert report.warnings == ()


def test_a_cuda_build_that_cannot_init_blames_the_driver(machine):
    machine(driver_gpus=(A4000,), torch_version="2.9.1+cu126", cuda_build="12.6", sees=False)
    report = resolve_device("auto")
    assert "driver too old" in report.warnings[0]


def test_report_lines_name_the_gpu_and_the_build(machine):
    machine(driver_gpus=(A4000,), torch_version="2.9.1+cu126", cuda_build="12.6", sees=True)
    text = "\n".join(resolve_device("auto").lines())
    assert "cuda" in text
    assert "2.9.1+cu126" in text
    assert "16376 MiB" in text
    assert "551.61" in text


def test_report_lines_say_when_torch_is_absent(machine):
    machine()
    text = "\n".join(resolve_device("auto").lines())
    assert "torch:  not installed" in text
    assert "gpu:    none detected" in text


# ----------------------------------------------------- the nvidia-smi wrapper


def test_nvidia_smi_absent_means_no_gpus(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda _: None)
    assert runtime._nvidia_smi_gpus() == ()


def test_nvidia_smi_output_is_parsed(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 0, "NVIDIA RTX A4000, 16376, 551.61, 8.6\n", ""
        ),
    )
    assert runtime._nvidia_smi_gpus() == (A4000,)


def test_nvidia_smi_garbage_is_ignored_not_fatal(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "not, a, csv\nalso bad\n", ""),
    )
    assert runtime._nvidia_smi_gpus() == ()


def test_nvidia_smi_timeout_is_survived(monkeypatch):
    """A driver in a bad state must not hang an 18,000-photo embed run."""
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def boom(*_a, **_k):
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(runtime.subprocess, "run", boom)
    assert runtime._nvidia_smi_gpus() == ()


def test_nvidia_smi_nonzero_exit_is_survived(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 9, "", "driver error"),
    )
    assert runtime._nvidia_smi_gpus() == ()
