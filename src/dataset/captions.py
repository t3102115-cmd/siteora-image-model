"""Caption utilities: metadata generation and manual-review helpers.

This project does NOT auto-generate captions with a captioning model by
default (to avoid inaccurate captions going straight into training
unreviewed). It provides the scaffolding to add one (e.g. BLIP-2) behind
an explicit review step, and to hand-author/curate metadata.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path


def write_metadata(records: list[dict], dataset_dir: str | Path, metadata_file: str = "metadata.jsonl") -> None:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_dir / metadata_file, "w") as f:
        for rec in records:
            if "file_name" not in rec or "text" not in rec:
                raise ValueError(f"record missing file_name/text: {rec}")
            f.write(json.dumps(rec) + "\n")


def generate_metadata_from_captions_dir(
    images_dir: str | Path,
    captions_dir: str | Path | None = None,
    caption_ext: str = ".txt",
) -> list[dict]:
    """Build metadata records from a directory of images plus one .txt caption
    per image (same basename). This is the common manual-captioning layout.

    Raises if any image is missing its caption file — captions must be
    reviewed by a human before training, per project policy.
    """
    images_dir = Path(images_dir)
    captions_dir = Path(captions_dir) if captions_dir else images_dir

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    records = []
    missing = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue
        caption_path = captions_dir / (img_path.stem + caption_ext)
        if not caption_path.exists():
            missing.append(img_path.name)
            continue
        text = caption_path.read_text(encoding="utf-8").strip()
        if not text:
            missing.append(img_path.name)
            continue
        records.append({"file_name": img_path.name, "text": text})

    if missing:
        raise ValueError(
            f"{len(missing)} image(s) missing a non-empty caption file: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    return records


def review_captions(dataset_dir: str | Path, metadata_file: str = "metadata.jsonl", n: int = 10) -> list[dict]:
    """Return a random-ish sample of records for manual review (no mutation)."""
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / metadata_file) as f:
        records = [json.loads(line) for line in f if line.strip()]
    step = max(1, len(records) // n)
    return records[::step][:n]
