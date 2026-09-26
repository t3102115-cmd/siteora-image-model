"""End-to-end test of training.trainer.train(), including checkpoint resume,
against the tiny public test model. This is the same orchestration function
the Colab notebook calls in Cell 10."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.config import DatasetConfig, LoRAConfig, ModelConfig, PathsConfig, SiteoraConfig, TrainingConfig
from training.trainer import train

TINY_MODEL = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.network


def _make_config(tmp_path: Path, max_train_steps: int) -> SiteoraConfig:
    return SiteoraConfig(
        model=ModelConfig(base_model=TINY_MODEL, resolution=128),
        lora=LoRAConfig(rank=4, alpha=4),
        training=TrainingConfig(
            learning_rate=1e-3, batch_size=1, gradient_accumulation_steps=1,
            max_train_steps=max_train_steps, save_every=1, eval_every=1, mixed_precision="no",
            gradient_checkpointing=False, use_8bit_adam=False, lr_scheduler="constant",
            lr_warmup_steps=0, seed=0, dataloader_num_workers=0,
        ),
        dataset=DatasetConfig(train_dir=str(REPO_ROOT / "dataset" / "test"), metadata_file="metadata.jsonl", val_split=0.2),
        paths=PathsConfig(
            output_dir=str(tmp_path / "outputs"), checkpoint_dir=str(tmp_path / "checkpoints"),
            lora_dir=str(tmp_path / "lora"), sample_dir=str(tmp_path / "samples"), log_dir=str(tmp_path / "logs"),
            drive_root=None,
        ),
    )


def test_train_runs_and_exports_lora(tmp_path):
    config = _make_config(tmp_path, max_train_steps=2)
    lora_dir = train(config, resume=False)
    assert lora_dir.exists()
    assert any(lora_dir.iterdir())


def test_train_resumes_from_checkpoint(tmp_path):
    config = _make_config(tmp_path, max_train_steps=2)
    train(config, resume=False)

    checkpoint_dir = Path(config.paths.checkpoint_dir)
    assert (checkpoint_dir / "step-0000002").exists()

    config.training.max_train_steps = 4
    train(config, resume=True)
    assert (checkpoint_dir / "step-0000004").exists()
