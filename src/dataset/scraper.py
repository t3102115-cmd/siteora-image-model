"""Dataset scraper — pulls commercially-licensed stock photos from the
Pexels API and appends them to dataset/metadata.jsonl.

This is deliberately NOT a generic web scraper. Scraping arbitrary websites
would pull in images of unknown or likely-prohibited license, which is
exactly the risk dataset/README.md warns about: this LoRA is meant for
commercial use, so every training image needs a license that permits that.

Pexels' license (verified 2026-09-26 at https://www.pexels.com/license/)
explicitly permits: free use, commercial use, and modification, with no
attribution required. It disallows: reselling unaltered copies, implying
endorsement by people/brands shown, using images as a trademark/logo, and
redistributing photos on other stock platforms. Training a LoRA is a
modifying/transformative use and fits within "you can modify the photos and
videos from Pexels" — this repo is not attempting to resell or redistribute
the source photos themselves.

Requires a free API key from https://www.pexels.com/api/ — set it via the
PEXELS_API_KEY environment variable, a Colab secret, or pass api_key=.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

PEXELS_API_URL = "https://api.pexels.com/v1/search"

# query -> caption template. Covers the website categories and styles from
# src/evaluation/evaluate.py's EVAL_PROMPTS, so scraped training data lines
# up with what the model will be evaluated on. {alt} is filled in with
# Pexels' own alt-text for the photo when available.
DEFAULT_QUERIES: dict[str, str] = {
    "modern saas office": "a modern SaaS office interior, professional technology photography, {alt}",
    "restaurant interior luxury": "a luxury restaurant interior, warm ambient lighting, professional architectural photography, {alt}",
    "modern real estate house": "a modern luxury house exterior for a real estate website, professional architectural photography, {alt}",
    "fashion editorial studio": "a premium fashion editorial photograph, studio lighting, {alt}",
    "technology abstract background": "an abstract futuristic technology background, {alt}",
    "product photography minimal": "a minimalist product photography shot on a clean background, {alt}",
    "travel resort beach": "a scenic travel resort photograph, professional travel photography, {alt}",
    "modern architecture building": "a striking modern architecture building, wide composition, {alt}",
    "professional business meeting": "a professional business meeting, corporate photography style, {alt}",
    "healthcare clinic modern": "a modern healthcare clinic interior, clean professional photography, {alt}",
}

LICENSE_NOTE = (
    "Pexels License (free for commercial use, modification allowed, no attribution "
    "required; do not resell unaltered, imply endorsement, or use as a trademark) "
    "- https://www.pexels.com/license/"
)


def _build_caption(query: str, alt_text: str | None, template: str) -> str:
    alt = (alt_text or "").strip()
    caption = template.format(alt=alt)
    caption = caption.replace(" ,", ",").replace(",,", ",")
    return caption.rstrip(", ").strip()


def _load_existing_ids(license_path: Path) -> set[int]:
    if not license_path.exists():
        return set()
    ids = set()
    with open(license_path) as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["pexels_id"])
    return ids


def scrape_pexels(
    queries: dict[str, str] | None = None,
    per_query: int = 20,
    dataset_dir: str | Path = "dataset",
    images_subdir: str = "images",
    metadata_file: str = "metadata.jsonl",
    license_manifest_file: str = "license_manifest.jsonl",
    api_key: str | None = None,
    orientation: str = "landscape",
    request_delay_s: float = 0.5,
    max_pages: int = 20,
) -> int:
    """Search Pexels for each query and append new images to the dataset.

    - Never overwrites or deletes existing dataset content (appends only).
    - Deduplicates by Pexels photo id, tracked in license_manifest_file, so
      re-running the scraper is safe and only fetches new images.
    - Records a per-image license/attribution entry, even though Pexels does
      not require attribution, for your own compliance record-keeping.
    - Stops after `max_pages` per query even if `per_query` isn't reached
      (e.g. because most results on each page are already-scraped
      duplicates) — bounds worst-case API calls when re-running against a
      near-exhausted query, rather than paging until Pexels returns empty.

    Returns the number of NEW images added.
    """
    api_key = api_key or os.environ.get("PEXELS_API_KEY")
    if not api_key:
        raise ValueError(
            "No Pexels API key found. Get a free key at https://www.pexels.com/api/ "
            "then pass api_key=... or set the PEXELS_API_KEY environment variable."
        )

    queries = queries or DEFAULT_QUERIES
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / images_subdir
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = dataset_dir / metadata_file
    license_path = dataset_dir / license_manifest_file
    existing_ids = _load_existing_ids(license_path)

    new_records: list[dict] = []
    new_license_entries: list[dict] = []
    headers = {"Authorization": api_key}

    for query, template in queries.items():
        page = 1
        collected = 0
        while collected < per_query and page <= max_pages:
            resp = requests.get(
                PEXELS_API_URL,
                headers=headers,
                params={
                    "query": query,
                    "per_page": min(per_query - collected, 80),
                    "page": page,
                    "orientation": orientation,
                },
                timeout=30,
            )
            resp.raise_for_status()
            photos = resp.json().get("photos", [])
            if not photos:
                break

            for photo in photos:
                if collected >= per_query:
                    break
                photo_id = photo["id"]
                if photo_id in existing_ids:
                    continue

                img_bytes = requests.get(photo["src"]["large2x"], timeout=30).content
                file_name = f"pexels_{photo_id}.jpg"
                (images_dir / file_name).write_bytes(img_bytes)

                new_records.append({"file_name": file_name, "text": _build_caption(query, photo.get("alt"), template)})
                new_license_entries.append(
                    {
                        "pexels_id": photo_id,
                        "file_name": file_name,
                        "query": query,
                        "photographer": photo.get("photographer"),
                        "photographer_url": photo.get("photographer_url"),
                        "source_url": photo.get("url"),
                        "license": LICENSE_NOTE,
                    }
                )
                existing_ids.add(photo_id)
                collected += 1
                time.sleep(request_delay_s)

            page += 1

    if new_records:
        with open(metadata_path, "a") as f:
            for rec in new_records:
                f.write(json.dumps(rec) + "\n")
        with open(license_path, "a") as f:
            for entry in new_license_entries:
                f.write(json.dumps(entry) + "\n")

    return len(new_records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query", type=int, default=20, help="images to fetch per search query")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--api-key", default=None, help="defaults to PEXELS_API_KEY env var")
    args = parser.parse_args()

    n = scrape_pexels(per_query=args.per_query, dataset_dir=args.dataset_dir, api_key=args.api_key)
    print(f"Added {n} new images to {args.dataset_dir}/metadata.jsonl")


if __name__ == "__main__":
    main()
