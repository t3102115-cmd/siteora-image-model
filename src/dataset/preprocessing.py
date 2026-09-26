"""Dataset preprocessing utilities: resize/crop/split/stats.

Preprocessing NEVER overwrites or deletes original files. Processed
images are always written to a separate output directory.
"""
from __future__ import annotations

import json
import random
import shutil
import statistics
from pathlib import Path

from PIL import Image


def resize_and_crop(image: Image.Image, target_size: int) -> Image.Image:
    """Resize so the short side == target_size, then center-crop to a square.

    This is the standard preprocessing for SD/SDXL-style training at a
    fixed square resolution (matches diffusers' text_to_image_lora examples).
    """
    image = image.convert("RGB")
    w, h = image.size
    scale = target_size / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    return image.crop((left, top, left + target_size, top + target_size))


def preprocess_dataset(
    src_dir: str | Path,
    dst_dir: str | Path,
    target_size: int = 1024,
    metadata_file: str = "metadata.jsonl",
    images_subdir: str = "images",
) -> int:
    """Resize/crop every image in src_dir into dst_dir, copying metadata as-is.

    Returns the number of images processed. Source files are untouched.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_images = dst_dir / images_subdir
    dst_images.mkdir(parents=True, exist_ok=True)

    records = []
    with open(src_dir / metadata_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    processed = 0
    for rec in records:
        file_name = rec["file_name"]
        src_path = src_dir / images_subdir / file_name
        if not src_path.exists():
            src_path = src_dir / file_name
        if not src_path.exists():
            continue
        with Image.open(src_path) as img:
            out = resize_and_crop(img, target_size)
        dst_path = dst_images / file_name
        out.save(dst_path)
        processed += 1

    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(dst_dir / metadata_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return processed


def train_val_split(
    dataset_dir: str | Path,
    val_split: float = 0.05,
    metadata_file: str = "metadata.jsonl",
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split metadata records into train/val, writing train.jsonl and val.jsonl."""
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / metadata_file) as f:
        records = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split)) if len(shuffled) > 1 else 0
    val_records = shuffled[:n_val]
    train_records = shuffled[n_val:]

    with open(dataset_dir / "train.jsonl", "w") as f:
        for rec in train_records:
            f.write(json.dumps(rec) + "\n")
    with open(dataset_dir / "val.jsonl", "w") as f:
        for rec in val_records:
            f.write(json.dumps(rec) + "\n")

    return train_records, val_records


def dataset_statistics(dataset_dir: str | Path, metadata_file: str = "metadata.jsonl", images_subdir: str = "images") -> dict:
    """Compute basic dataset statistics without modifying anything."""
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / metadata_file) as f:
        records = [json.loads(line) for line in f if line.strip()]

    widths, heights, caption_lens = [], [], []
    for rec in records:
        file_name = rec.get("file_name")
        text = rec.get("text", "")
        caption_lens.append(len(text))
        path = dataset_dir / images_subdir / file_name if file_name else None
        if path and path.exists():
            with Image.open(path) as img:
                widths.append(img.width)
                heights.append(img.height)

    return {
        "num_records": len(records),
        "avg_caption_length": statistics.mean(caption_lens) if caption_lens else 0,
        "median_caption_length": statistics.median(caption_lens) if caption_lens else 0,
        "avg_width": statistics.mean(widths) if widths else 0,
        "avg_height": statistics.mean(heights) if heights else 0,
        "min_resolution": (min(widths), min(heights)) if widths else None,
        "max_resolution": (max(widths), max(heights)) if widths else None,
    }
