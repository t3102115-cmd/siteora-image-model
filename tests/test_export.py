import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.accelerator import AcceleratorInfo
from training.config import LoRAConfig, ModelConfig, SiteoraConfig
from training.export import export_lora
from training.trainer import attach_lora, load_base_model

TINY_MODEL = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def trained_lora_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("export_src")
    config = SiteoraConfig(model=ModelConfig(base_model=TINY_MODEL, resolution=128), lora=LoRAConfig(rank=4, alpha=4))
    accel_info = AcceleratorInfo(kind="cpu", name="CPU")
    components = load_base_model(config, accel_info)
    components.unet = attach_lora(components.unet, config)

    lora_dir = tmp / "raw_lora"
    lora_dir.mkdir()
    components.unet.save_lora_adapter(str(lora_dir))
    return lora_dir, config


def test_export_lora_produces_safetensors_and_metadata(trained_lora_dir, tmp_path):
    lora_dir, config = trained_lora_dir
    out_dir = export_lora(lora_dir, config, version="v0.1-test", dest_dir=tmp_path / "export")

    safetensors_files = list(out_dir.glob("*.safetensors"))
    assert len(safetensors_files) == 1
    assert safetensors_files[0].name == "siteora_lora_v0.1-test.safetensors"

    metadata_path = out_dir / "metadata.json"
    assert metadata_path.exists()
    import json

    with open(metadata_path) as f:
        metadata = json.load(f)
    assert metadata["base_model"] == TINY_MODEL
    assert metadata["lora_rank"] == 4
    assert metadata["num_lora_parameters"] > 0


def test_export_lora_raises_on_missing_weights(tmp_path):
    config = SiteoraConfig()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        export_lora(empty_dir, config, version="v0.1", dest_dir=tmp_path / "export")
