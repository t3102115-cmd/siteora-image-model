"""Model loading, LoRA injection, and checkpoint save/resume, against the
tiny public test model (fast enough to run on CPU in CI)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from training import checkpointing
from training.accelerator import AcceleratorInfo
from training.config import DatasetConfig, LoRAConfig, ModelConfig, PathsConfig, SiteoraConfig, TrainingConfig
from training.trainer import attach_lora, count_parameters, load_base_model

TINY_MODEL = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"

pytestmark = pytest.mark.network  # these tests download a tiny model from HF


@pytest.fixture(scope="module")
def cpu_accel_info():
    return AcceleratorInfo(kind="cpu", name="CPU", vram_gb=None, recommended_precision="fp32")


@pytest.fixture(scope="module")
def base_config(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("model_test")
    return SiteoraConfig(
        model=ModelConfig(base_model=TINY_MODEL, resolution=128),
        lora=LoRAConfig(rank=4, alpha=4),
        training=TrainingConfig(mixed_precision="no", use_8bit_adam=False),
        dataset=DatasetConfig(),
        paths=PathsConfig(
            output_dir=str(tmp / "outputs"), checkpoint_dir=str(tmp / "checkpoints"),
            lora_dir=str(tmp / "lora"), sample_dir=str(tmp / "samples"), log_dir=str(tmp / "logs"),
            drive_root=None,
        ),
    )


def test_load_base_model(base_config, cpu_accel_info):
    components = load_base_model(base_config, cpu_accel_info)
    assert components.unet is not None
    assert components.vae is not None
    assert components.is_sdxl is True


def test_lora_injection_adds_trainable_params(base_config, cpu_accel_info):
    components = load_base_model(base_config, cpu_accel_info)
    trainable_before, _ = count_parameters(components.unet)
    assert trainable_before == 0
    components.unet = attach_lora(components.unet, base_config)
    trainable_after, total = count_parameters(components.unet)
    assert trainable_after > 0
    assert trainable_after < total


def test_checkpoint_save_and_resume(base_config, cpu_accel_info, tmp_path):
    components = load_base_model(base_config, cpu_accel_info)
    components.unet = attach_lora(components.unet, base_config)
    trainable_params = [p for p in components.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    lora_state = get_peft_model_state_dict(components.unet)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_path = checkpointing.save_checkpoint(ckpt_dir, 5, lora_state, optimizer, None, {"x": 1})
    assert (ckpt_path / "unet_lora.pt").exists()
    assert (ckpt_path / "training_state.json").exists()

    latest = checkpointing.find_latest_checkpoint(ckpt_dir)
    assert latest == ckpt_path

    optimizer2 = torch.optim.AdamW(trainable_params, lr=1e-3)
    state = checkpointing.load_checkpoint(latest, optimizer=optimizer2)
    assert state["step"] == 5
    set_peft_model_state_dict(components.unet, state["unet_lora_state_dict"])  # must not raise


def test_config_validation_rejects_bad_config():
    with pytest.raises(ValueError):
        bad = SiteoraConfig()
        bad.lora.rank = 0
        bad.validate()
