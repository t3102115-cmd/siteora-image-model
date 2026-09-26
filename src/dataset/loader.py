"""PyTorch Dataset for image/caption pairs, matching diffusers' SDXL LoRA
training conventions (original size / crop coords for SDXL micro-conditioning).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SiteoraImageCaptionDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        metadata_file: str = "metadata.jsonl",
        images_subdir: str = "images",
        resolution: int = 1024,
        image_column: str = "file_name",
        caption_column: str = "text",
        center_crop: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.images_subdir = images_subdir
        self.resolution = resolution
        self.image_column = image_column
        self.caption_column = caption_column

        with open(self.dataset_dir / metadata_file) as f:
            self.records = [json.loads(line) for line in f if line.strip()]

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, file_name: str) -> Path:
        p = self.dataset_dir / self.images_subdir / file_name
        if p.exists():
            return p
        return self.dataset_dir / file_name

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        file_name = rec[self.image_column]
        text = rec[self.caption_column]
        path = self._resolve_path(file_name)

        with Image.open(path) as img:
            img = img.convert("RGB")
            original_size = img.size  # (w, h), for SDXL micro-conditioning

        # Resize short side to resolution, record crop offsets for SDXL.
        w, h = original_size
        scale = self.resolution / min(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        resized = Image.open(path).convert("RGB").resize((new_w, new_h), Image.LANCZOS)
        left = max(0, (new_w - self.resolution) // 2)
        top = max(0, (new_h - self.resolution) // 2)

        pixel_values = self.image_transforms.transforms[2](  # ToTensor
            resized.crop((left, top, left + self.resolution, top + self.resolution))
        )
        pixel_values = self.image_transforms.transforms[3](pixel_values)  # Normalize

        return {
            "pixel_values": pixel_values,
            "text": text,
            "original_size": torch.tensor([h, w]),
            "crop_top_left": torch.tensor([top, left]),
            "target_size": torch.tensor([self.resolution, self.resolution]),
        }


def collate_fn(batch: list[dict]) -> dict:
    pixel_values = torch.stack([b["pixel_values"] for b in batch]).to(memory_format=torch.contiguous_format).float()
    return {
        "pixel_values": pixel_values,
        "text": [b["text"] for b in batch],
        "original_size": torch.stack([b["original_size"] for b in batch]),
        "crop_top_left": torch.stack([b["crop_top_left"] for b in batch]),
        "target_size": torch.stack([b["target_size"] for b in batch]),
    }
