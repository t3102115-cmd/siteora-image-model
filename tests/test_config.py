import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.config import SiteoraConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["test", "standard", "extended"])
def test_shipped_configs_are_valid(name):
    cfg = SiteoraConfig.from_yaml(REPO_ROOT / "configs" / f"{name}.yaml")
    cfg.validate()  # must not raise


def test_invalid_resolution_rejected():
    cfg = SiteoraConfig()
    cfg.model.resolution = 513
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_batch_size_rejected():
    cfg = SiteoraConfig()
    cfg.training.batch_size = 0
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_precision_rejected():
    cfg = SiteoraConfig()
    cfg.training.mixed_precision = "fp8"
    with pytest.raises(ValueError):
        cfg.validate()
