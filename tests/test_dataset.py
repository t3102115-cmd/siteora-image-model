import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset.preprocessing import dataset_statistics, resize_and_crop, train_val_split
from dataset.validation import require_valid_dataset, validate_dataset


@pytest.fixture
def tiny_dataset(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    records = []
    for i in range(4):
        img = Image.new("RGB", (100, 60), (i * 10, 0, 0))
        fname = f"img_{i}.jpg"
        img.save(images_dir / fname)
        records.append({"file_name": fname, "text": f"caption number {i}"})
    with open(tmp_path / "metadata.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return tmp_path


def test_validate_dataset_all_valid(tiny_dataset):
    report = validate_dataset(tiny_dataset)
    assert report.total_records == 4
    assert report.valid == 4
    assert report.invalid == 0


def test_validate_dataset_missing_file(tiny_dataset):
    with open(tiny_dataset / "metadata.jsonl", "a") as f:
        f.write(json.dumps({"file_name": "missing.jpg", "text": "x"}) + "\n")
    report = validate_dataset(tiny_dataset)
    assert report.total_records == 5
    assert len(report.missing_files) == 1


def test_validate_dataset_missing_caption(tiny_dataset):
    with open(tiny_dataset / "metadata.jsonl", "a") as f:
        f.write(json.dumps({"file_name": "img_0.jpg", "text": ""}) + "\n")
    report = validate_dataset(tiny_dataset)
    assert len(report.missing_captions) == 1


def test_validate_dataset_corrupted_file(tiny_dataset):
    bad_path = tiny_dataset / "images" / "corrupt.jpg"
    bad_path.write_bytes(b"not an image")
    with open(tiny_dataset / "metadata.jsonl", "a") as f:
        f.write(json.dumps({"file_name": "corrupt.jpg", "text": "x"}) + "\n")
    report = validate_dataset(tiny_dataset)
    assert len(report.corrupted_files) == 1


def test_require_valid_dataset_raises_on_bad_data(tiny_dataset):
    with open(tiny_dataset / "metadata.jsonl", "a") as f:
        f.write(json.dumps({"file_name": "missing.jpg", "text": "x"}) + "\n")
    with pytest.raises(ValueError):
        require_valid_dataset(tiny_dataset)


def test_require_valid_dataset_passes_on_good_data(tiny_dataset):
    report = require_valid_dataset(tiny_dataset)
    assert report.is_valid


def test_resize_and_crop_produces_square():
    img = Image.new("RGB", (400, 200), (1, 2, 3))
    out = resize_and_crop(img, 128)
    assert out.size == (128, 128)


def test_train_val_split(tiny_dataset):
    train, val = train_val_split(tiny_dataset, val_split=0.5, seed=0)
    assert len(train) + len(val) == 4
    assert (tiny_dataset / "train.jsonl").exists()
    assert (tiny_dataset / "val.jsonl").exists()


def test_dataset_statistics(tiny_dataset):
    stats = dataset_statistics(tiny_dataset)
    assert stats["num_records"] == 4
    assert stats["avg_width"] == 100
    assert stats["avg_height"] == 60
