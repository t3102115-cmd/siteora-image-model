"""Exports a LoRA from the tiny test model, then loads it back INDEPENDENTLY
(fresh process state, via the public diffusers pipeline API) and generates
an image — the same path a real Siteora LoRA would go through."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference.generate import SiteoraImageGenerator
from training.accelerator import AcceleratorInfo
from training.config import LoRAConfig, ModelConfig, SiteoraConfig
from training.trainer import attach_lora, load_base_model

TINY_MODEL = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def exported_lora_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("export")
    config = SiteoraConfig(model=ModelConfig(base_model=TINY_MODEL, resolution=128), lora=LoRAConfig(rank=4, alpha=4))
    accel_info = AcceleratorInfo(kind="cpu", name="CPU")
    components = load_base_model(config, accel_info)
    components.unet = attach_lora(components.unet, config)

    lora_dir = tmp / "siteora-lora"
    lora_dir.mkdir()
    components.unet.save_lora_adapter(str(lora_dir))
    return lora_dir


def test_lora_exports_files(exported_lora_dir):
    files = list(exported_lora_dir.iterdir())
    assert len(files) > 0


def test_lora_loads_independently_and_generates(exported_lora_dir):
    generator = SiteoraImageGenerator(base_model=TINY_MODEL, lora_path=str(exported_lora_dir), device="cpu")
    assert generator.lora_loaded

    result = generator.generate(
        "a modern SaaS website hero image",
        width=128,
        height=128,
        num_inference_steps=2,
        seed=123,
        lora_strength=0.8,
    )
    assert result.image.size == (128, 128)
    assert result.seed == 123
    assert result.lora_strength == 0.8


def test_seed_reproducibility_on_cpu(exported_lora_dir):
    generator = SiteoraImageGenerator(base_model=TINY_MODEL, lora_path=str(exported_lora_dir), device="cpu")
    img1 = generator.generate("a test prompt", width=128, height=128, num_inference_steps=2, seed=999).image
    img2 = generator.generate("a test prompt", width=128, height=128, num_inference_steps=2, seed=999).image
    import numpy as np

    assert np.array_equal(np.array(img1), np.array(img2))
