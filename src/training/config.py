"""Central training configuration (loaded from YAML, e.g. configs/standard.yaml)."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class ModelConfig:
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    revision: str | None = None
    resolution: int = 1024


@dataclasses.dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = dataclasses.field(
        default_factory=lambda: ["to_k", "to_q", "to_v", "to_out.0"]
    )


@dataclasses.dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 800
    save_every: int = 200
    eval_every: int = 200
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_grad_norm: float = 1.0
    seed: int = 42
    dataloader_num_workers: int = 2


@dataclasses.dataclass
class DatasetConfig:
    train_dir: str = "dataset"
    metadata_file: str = "metadata.jsonl"
    val_split: float = 0.05
    caption_column: str = "text"
    image_column: str = "file_name"


@dataclasses.dataclass
class PathsConfig:
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    lora_dir: str = "models/siteora-lora"
    sample_dir: str = "samples"
    log_dir: str = "logs"
    drive_root: str | None = "/content/drive/MyDrive/siteora-image-model"


@dataclasses.dataclass
class SiteoraConfig:
    name: str = "test"
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    lora: LoRAConfig = dataclasses.field(default_factory=LoRAConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    paths: PathsConfig = dataclasses.field(default_factory=PathsConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> "SiteoraConfig":
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return SiteoraConfig(
            name=raw.get("name", Path(path).stem),
            model=ModelConfig(**raw.get("model", {})),
            lora=LoRAConfig(**raw.get("lora", {})),
            training=TrainingConfig(**raw.get("training", {})),
            dataset=DatasetConfig(**raw.get("dataset", {})),
            paths=PathsConfig(**raw.get("paths", {})),
        )

    def validate(self) -> None:
        errors = []
        if self.training.batch_size < 1:
            errors.append("training.batch_size must be >= 1")
        if self.training.gradient_accumulation_steps < 1:
            errors.append("training.gradient_accumulation_steps must be >= 1")
        if self.lora.rank < 1:
            errors.append("lora.rank must be >= 1")
        if self.model.resolution % 8 != 0:
            errors.append("model.resolution must be divisible by 8")
        if self.training.mixed_precision not in ("no", "fp16", "bf16"):
            errors.append("training.mixed_precision must be one of: no, fp16, bf16")
        if not (0.0 <= self.dataset.val_split < 1.0):
            errors.append("dataset.val_split must be in [0, 1)")
        if errors:
            raise ValueError("Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors))
