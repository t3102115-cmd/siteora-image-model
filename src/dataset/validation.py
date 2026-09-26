"""Dataset validation utilities.

Never silently drops or mutates the user's original files: validation is
purely read-only and reports problems for the caller to act on.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError


@dataclasses.dataclass
class ValidationIssue:
    file_name: str
    reason: str


@dataclasses.dataclass
class ValidationReport:
    total_records: int = 0
    valid: int = 0
    missing_files: list[ValidationIssue] = dataclasses.field(default_factory=list)
    corrupted_files: list[ValidationIssue] = dataclasses.field(default_factory=list)
    missing_captions: list[ValidationIssue] = dataclasses.field(default_factory=list)
    resolutions: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    caption_lengths: list[int] = dataclasses.field(default_factory=list)

    @property
    def invalid(self) -> int:
        return len(self.missing_files) + len(self.corrupted_files) + len(self.missing_captions)

    @property
    def is_valid(self) -> bool:
        return self.total_records > 0 and self.invalid == 0

    def summary(self) -> str:
        avg_len = sum(self.caption_lengths) / len(self.caption_lengths) if self.caption_lengths else 0.0
        lines = [
            f"Total records:      {self.total_records}",
            f"Valid images:       {self.valid}",
            f"Invalid images:     {self.invalid}",
            f"  Missing files:    {len(self.missing_files)}",
            f"  Corrupted files:  {len(self.corrupted_files)}",
            f"  Missing captions: {len(self.missing_captions)}",
            f"Average caption length (chars): {avg_len:.1f}",
        ]
        if self.resolutions:
            widths = sorted(w for w, _ in self.resolutions)
            heights = sorted(h for _, h in self.resolutions)
            lines.append(
                f"Resolution range: {widths[0]}x{heights[0]} .. {widths[-1]}x{heights[-1]}"
            )
        return "\n".join(lines)


def load_metadata(dataset_dir: str | Path, metadata_file: str = "metadata.jsonl") -> list[dict]:
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / metadata_file
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")
    records = []
    with open(metadata_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{metadata_path}:{line_no}: invalid JSON: {e}") from e
    return records


def validate_dataset(
    dataset_dir: str | Path,
    metadata_file: str = "metadata.jsonl",
    image_column: str = "file_name",
    caption_column: str = "text",
    images_subdir: str = "images",
) -> ValidationReport:
    """Validate an image/caption dataset without modifying anything."""
    dataset_dir = Path(dataset_dir)
    records = load_metadata(dataset_dir, metadata_file)
    report = ValidationReport(total_records=len(records))

    for rec in records:
        file_name = rec.get(image_column)
        text = rec.get(caption_column)

        if not file_name:
            report.missing_files.append(ValidationIssue("<unknown>", "record missing file_name field"))
            continue

        image_path = dataset_dir / images_subdir / file_name
        if not image_path.exists():
            image_path = dataset_dir / file_name  # fall back to flat layout
        if not image_path.exists():
            report.missing_files.append(ValidationIssue(file_name, "file does not exist"))
            continue

        try:
            with Image.open(image_path) as img:
                img.verify()
            with Image.open(image_path) as img:
                report.resolutions.append(img.size)
        except (UnidentifiedImageError, OSError) as e:
            report.corrupted_files.append(ValidationIssue(file_name, str(e)))
            continue

        if not text or not text.strip():
            report.missing_captions.append(ValidationIssue(file_name, "empty or missing caption"))
            continue

        report.caption_lengths.append(len(text))
        report.valid += 1

    return report


def require_valid_dataset(dataset_dir: str | Path, **kwargs) -> ValidationReport:
    """Validate and raise loudly if the dataset is not trainable.

    Training must fail clearly on a bad dataset rather than silently
    training on garbage.
    """
    report = validate_dataset(dataset_dir, **kwargs)
    if report.total_records == 0:
        raise ValueError(f"Dataset at {dataset_dir} contains no records in metadata.jsonl")
    if report.valid == 0:
        raise ValueError(
            f"Dataset at {dataset_dir} has zero valid image/caption pairs "
            f"out of {report.total_records} records:\n{report.summary()}"
        )
    if report.invalid > 0:
        raise ValueError(
            f"Dataset at {dataset_dir} has {report.invalid} invalid record(s). "
            f"Fix these before training:\n{report.summary()}"
        )
    return report
