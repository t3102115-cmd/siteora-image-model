"""Export a trained Siteora LoRA as a standalone, versioned .safetensors
package with metadata — separate from the training checkpoint format used
by src/training/checkpointing.py (which also stores optimizer/scheduler
state for resuming training and isn't meant to be distributed).
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

from safetensors.torch import load_file

from training.config import SiteoraConfig


def _dataset_size(config: SiteoraConfig) -> int | None:
    metadata_path = Path(config.dataset.train_dir) / config.dataset.metadata_file
    if not metadata_path.exists():
        return None
    with open(metadata_path) as f:
        return sum(1 for line in f if line.strip())


def _software_versions() -> dict:
    versions = {}
    for pkg in ("torch", "diffusers", "transformers", "peft", "accelerate"):
        try:
            versions[pkg] = pkg_version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


def export_lora(
    lora_dir: str | Path,
    config: SiteoraConfig,
    version: str,
    dest_dir: str | Path = "models/export",
    training_step: int | None = None,
) -> Path:
    """Package a trained LoRA (as produced by `training.trainer.train()`,
    which calls `unet.save_lora_adapter()`) into a versioned export folder:

        models/export/<version>/
        ├── siteora_lora_<version>.safetensors
        └── metadata.json

    Validates the source file is actually a loadable safetensors file (not
    just present) before calling the export a success. Raises if the LoRA
    weights aren't where expected — never silently exports an empty package.
    """
    lora_dir = Path(lora_dir)
    src_file = lora_dir / "pytorch_lora_weights.safetensors"
    if not src_file.exists():
        raise FileNotFoundError(
            f"expected LoRA weights at {src_file} (from unet.save_lora_adapter()) — "
            f"found instead: {list(lora_dir.iterdir()) if lora_dir.exists() else 'directory does not exist'}"
        )

    # Validate it actually loads and is non-empty before calling this a success.
    state_dict = load_file(str(src_file))
    if len(state_dict) == 0:
        raise ValueError(f"{src_file} loaded but contains zero tensors — refusing to export an empty LoRA")
    num_params = sum(t.numel() for t in state_dict.values())

    out_dir = Path(dest_dir) / version
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"siteora_lora_{version}.safetensors"
    shutil.copyfile(src_file, out_file)

    metadata = {
        "name": f"siteora-image-{version}",
        "base_model": config.model.base_model,
        "base_model_license": "CreativeML Open RAIL++-M (see MODEL_LICENSE.md)",
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "target_modules": config.lora.target_modules,
        "num_lora_parameters": num_params,
        "training_resolution": config.model.resolution,
        "training_config_name": config.name,
        "training_steps": training_step if training_step is not None else config.training.max_train_steps,
        "learning_rate": config.training.learning_rate,
        "dataset_size": _dataset_size(config),
        "dataset_dir": config.dataset.train_dir,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_versions": _software_versions(),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return out_dir
