"""Accelerator detection for training/inference (CUDA / TPU / CPU).

Colab sessions can hand you a T4 GPU, a TPU v5e-1, or (rarely) nothing at
all. This module centralizes detection so the rest of the codebase never
has to guess, and so we never silently fall back to an expensive CPU run.
"""
from __future__ import annotations

import dataclasses
import warnings


@dataclasses.dataclass
class AcceleratorInfo:
    kind: str  # "cuda" | "tpu" | "cpu"
    name: str
    vram_gb: float | None = None
    recommended_precision: str = "fp32"

    def print_banner(self) -> None:
        print("=" * 60)
        if self.kind == "cuda":
            print(f"Accelerator: {self.name}")
            print(f"VRAM: {self.vram_gb:.1f} GB")
            print(f"Precision: {self.recommended_precision.upper()}")
        elif self.kind == "tpu":
            print(f"Accelerator: {self.name}")
            print(f"Precision: {self.recommended_precision.upper()}")
        else:
            print("Accelerator: CPU (no GPU/TPU detected)")
            print(f"Precision: {self.recommended_precision.upper()}")
        print("=" * 60)


def _detect_cuda() -> AcceleratorInfo | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    # T4 and other GPUs with < 20GB should use fp16 (no native bf16 support
    # on T4's Turing architecture); newer Ampere+ GPUs can use bf16.
    major, _minor = torch.cuda.get_device_capability(0)
    precision = "bf16" if major >= 8 else "fp16"
    return AcceleratorInfo(kind="cuda", name=name, vram_gb=vram_gb, recommended_precision=precision)


def _detect_tpu() -> AcceleratorInfo | None:
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
    except ImportError:
        return None
    try:
        device = xm.xla_device()
        name = xm.xla_device_hw(device)
    except Exception:
        return None
    return AcceleratorInfo(kind="tpu", name=f"TPU ({name})", vram_gb=None, recommended_precision="bf16")


def detect_accelerator(verbose: bool = True) -> AcceleratorInfo:
    """Detect the best available accelerator.

    Order of preference: CUDA GPU, then TPU (via torch_xla), then CPU.
    Never silently returns CPU without a loud warning, since training a
    diffusion LoRA on CPU is impractically slow.
    """
    info = _detect_cuda()
    if info is None:
        info = _detect_tpu()
    if info is None:
        warnings.warn(
            "No CUDA GPU or TPU detected. Falling back to CPU. Training on "
            "CPU is NOT recommended and will be extremely slow (hours per "
            "step for SDXL). Use this path only for pipeline smoke tests "
            "with tiny test models, never for real training.",
            stacklevel=2,
        )
        info = AcceleratorInfo(kind="cpu", name="CPU", vram_gb=None, recommended_precision="fp32")
    if verbose:
        info.print_banner()
    return info


if __name__ == "__main__":
    detect_accelerator()
