"""Checkpoint save/resume for LoRA training.

A checkpoint preserves everything needed to resume training after a
Colab disconnect: LoRA weights, optimizer state, LR scheduler state,
current step, and the training config used.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch


def save_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    unet_lora_state_dict: dict,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    config_dict: dict,
    text_encoder_lora_state_dict: dict | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    ckpt_path = checkpoint_dir / f"step-{step:07d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    torch.save(unet_lora_state_dict, ckpt_path / "unet_lora.pt")
    if text_encoder_lora_state_dict is not None:
        torch.save(text_encoder_lora_state_dict, ckpt_path / "text_encoder_lora.pt")
    torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")
    if lr_scheduler is not None:
        torch.save(lr_scheduler.state_dict(), ckpt_path / "lr_scheduler.pt")

    with open(ckpt_path / "training_state.json", "w") as f:
        json.dump({"step": step}, f)
    with open(ckpt_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    latest_link = checkpoint_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    try:
        latest_link.symlink_to(ckpt_path.name)
    except OSError:
        # Some filesystems (e.g. certain Colab/Drive mounts) don't support
        # symlinks; fall back to a plain text pointer file.
        (checkpoint_dir / "latest.txt").write_text(ckpt_path.name)

    return ckpt_path


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    latest_link = checkpoint_dir / "latest"
    if latest_link.exists():
        return (checkpoint_dir / latest_link.readlink()) if latest_link.is_symlink() else latest_link

    pointer = checkpoint_dir / "latest.txt"
    if pointer.exists():
        return checkpoint_dir / pointer.read_text().strip()

    candidates = sorted(checkpoint_dir.glob("step-*"))
    return candidates[-1] if candidates else None


def load_checkpoint(
    checkpoint_path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler=None,
    map_location: str = "cpu",
) -> dict:
    checkpoint_path = Path(checkpoint_path)

    unet_lora_state_dict = torch.load(checkpoint_path / "unet_lora.pt", map_location=map_location)

    text_encoder_lora_path = checkpoint_path / "text_encoder_lora.pt"
    text_encoder_lora_state_dict = (
        torch.load(text_encoder_lora_path, map_location=map_location) if text_encoder_lora_path.exists() else None
    )

    if optimizer is not None:
        optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt", map_location=map_location))
    if lr_scheduler is not None and (checkpoint_path / "lr_scheduler.pt").exists():
        lr_scheduler.load_state_dict(torch.load(checkpoint_path / "lr_scheduler.pt", map_location=map_location))

    with open(checkpoint_path / "training_state.json") as f:
        training_state = json.load(f)

    return {
        "unet_lora_state_dict": unet_lora_state_dict,
        "text_encoder_lora_state_dict": text_encoder_lora_state_dict,
        "step": training_state["step"],
    }


def sync_to_drive(local_dir: str | Path, drive_root: str | None) -> Path | None:
    """Copy outputs to a mounted Google Drive path, if configured and mounted."""
    if not drive_root:
        return None
    drive_path = Path(drive_root)
    if not drive_path.parent.exists():
        # /content/drive not mounted (e.g. running outside Colab) — skip.
        return None
    local_dir = Path(local_dir)
    dest = drive_path / local_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_dir, dest, dirs_exist_ok=True)
    return dest
