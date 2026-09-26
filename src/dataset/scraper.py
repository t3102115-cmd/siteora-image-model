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


def _log(msg: str) -> None:
    # Plain print(), not the `logging` module: Jupyter/Colab cell output
    # capture and stale cached module imports make `logging` handlers
    # unreliable to see live in a notebook. print(flush=True) always shows
    # up immediately in a cell, no configuration required.
    print(f"[scraper] {msg}", flush=True)


PEXELS_API_URL = "https://api.pexels.com/v1/search"

# query -> caption template. Covers the website categories and styles from
# src/evaluation/evaluate.py's EVAL_PROMPTS, so scraped training data lines
# up with what the model will be evaluated on. {alt} is filled in with
# Pexels' own alt-text for the photo when available.
DEFAULT_QUERIES: dict[str, str] = {
    # --- original 10 ---
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
    # --- SaaS / technology / AI startups ---
    "startup team coding": "a startup team working on laptops, modern tech office photography, {alt}",
    "data center servers": "a modern data center with server racks, cool blue technology lighting, {alt}",
    "coding on laptop closeup": "a close-up of code on a laptop screen, modern technology photography, {alt}",
    "futuristic circuit board": "a futuristic glowing circuit board macro shot, technology background, {alt}",
    "cloud computing network": "an abstract cloud computing network visualization, futuristic technology style, {alt}",
    "software developer desk setup": "a clean modern software developer desk setup, minimal technology photography, {alt}",
    "cybersecurity lock digital": "a digital cybersecurity concept with glowing lock icon, futuristic style, {alt}",
    "robotics ai lab": "a modern robotics and AI research lab, clean futuristic photography, {alt}",
    # --- finance ---
    "modern bank office interior": "a modern bank office interior, professional corporate photography, {alt}",
    "finance stock chart screen": "a financial stock chart on a screen, professional business photography, {alt}",
    "business handshake office": "a professional handshake in a modern office, corporate photography style, {alt}",
    "investment growth concept": "an abstract investment growth concept image, clean corporate style, {alt}",
    # --- healthcare ---
    "doctor patient consultation": "a doctor consulting with a patient, professional healthcare photography, {alt}",
    "modern hospital hallway": "a clean modern hospital hallway, professional healthcare photography, {alt}",
    "medical equipment closeup": "a close-up of modern medical equipment, clean clinical photography, {alt}",
    # --- restaurants / food ---
    "gourmet food plating": "a beautifully plated gourmet dish, professional food photography, {alt}",
    "chef cooking kitchen": "a chef cooking in a professional restaurant kitchen, editorial food photography, {alt}",
    "coffee shop cozy interior": "a cozy modern coffee shop interior, warm inviting photography, {alt}",
    "outdoor restaurant patio": "an outdoor restaurant patio at golden hour, editorial photography, {alt}",
    # --- hotels / travel ---
    "luxury hotel lobby": "a luxury hotel lobby interior, elegant professional photography, {alt}",
    "hotel room modern minimal": "a modern minimal hotel room interior, clean professional photography, {alt}",
    "infinity pool resort": "an infinity pool at a luxury resort, scenic travel photography, {alt}",
    "airplane window travel": "a view from an airplane window, travel photography style, {alt}",
    "mountain landscape scenic": "a scenic mountain landscape, wide travel photography composition, {alt}",
    # --- real estate / architecture ---
    "modern apartment interior": "a modern minimalist apartment interior, professional architectural photography, {alt}",
    "luxury living room interior": "a luxury living room interior, warm professional architectural photography, {alt}",
    "glass office building exterior": "a modern glass office building exterior, wide architectural photography, {alt}",
    "minimalist kitchen design": "a minimalist modern kitchen design, clean architectural photography, {alt}",
    "rooftop city skyline": "a rooftop view of a city skyline at dusk, wide architectural photography, {alt}",
    # --- e-commerce / product ---
    "product on white background": "a professional product photograph on a clean white background, {alt}",
    "flat lay product shot": "a flat lay product photography shot, minimal styled composition, {alt}",
    "cosmetics product photography": "an elegant cosmetics product photography shot, studio lighting, {alt}",
    "shoes product studio shot": "a pair of shoes on a clean studio background, product photography, {alt}",
    "packaging design mockup": "a clean modern product packaging mockup, studio photography, {alt}",
    # --- fashion / editorial ---
    "fashion model street style": "a fashion model in street style clothing, editorial photography, {alt}",
    "fashion accessories flatlay": "a fashion accessories flat lay, minimal editorial styling, {alt}",
    "model portrait studio light": "a fashion model portrait with dramatic studio lighting, {alt}",
    # --- professional services / agencies / portfolios ---
    "creative agency workspace": "a creative agency workspace with designers collaborating, modern office photography, {alt}",
    "lawyer office professional": "a professional lawyer's office interior, corporate photography style, {alt}",
    "consultant presenting meeting": "a consultant presenting to a business team, corporate photography, {alt}",
    "designer portfolio desk": "a designer's desk with a portfolio and sketches, creative workspace photography, {alt}",
    # --- education ---
    "modern classroom students": "a modern classroom with students learning, bright professional photography, {alt}",
    "university campus building": "a university campus building exterior, wide architectural photography, {alt}",
    "online learning laptop": "a person taking an online course on a laptop, modern lifestyle photography, {alt}",
    # --- fitness ---
    "modern gym interior": "a modern gym interior with equipment, clean fitness photography, {alt}",
    "yoga studio calm interior": "a calm minimalist yoga studio interior, soft natural lighting, {alt}",
    "person running outdoors": "a person running outdoors at sunrise, energetic fitness photography, {alt}",
    # --- automotive ---
    "luxury car studio shot": "a luxury car in a studio photography shot, dramatic lighting, {alt}",
    "car driving mountain road": "a car driving along a scenic mountain road, cinematic automotive photography, {alt}",
    "electric car charging": "an electric car charging at a modern charging station, clean technology photography, {alt}",
    # --- abstract / backgrounds / 3D-style ---
    "abstract gradient background": "a smooth abstract gradient background, modern minimal style, {alt}",
    "geometric shapes 3d render": "an abstract 3D geometric shapes render, futuristic clean style, {alt}",
    "minimal texture background": "a minimal textured background with soft lighting, clean modern style, {alt}",
    "neon light abstract": "an abstract neon light photography composition, futuristic style, {alt}",
    "nature abstract macro": "an abstract macro photograph of a natural texture, soft artistic style, {alt}",
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
    verbose: bool = True,
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
    - With `verbose=True` (the default), prints progress per query/page/image
      and download failures, so a long-running scrape isn't a silent black
      box in a Colab cell. Set `verbose=False` for quiet/scripted use.

    Returns the number of NEW images added.
    """
    log = _log if verbose else (lambda msg: None)

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
    log(f"Starting scrape: {len(queries)} quer(y/ies), {per_query} images/query, {len(existing_ids)} already in dataset")

    new_records: list[dict] = []
    new_license_entries: list[dict] = []
    skipped_duplicates = 0
    failed_downloads = 0
    headers = {"Authorization": api_key}

    for query_idx, (query, template) in enumerate(queries.items(), 1):
        log(f"[{query_idx}/{len(queries)}] query='{query}': searching...")
        page = 1
        collected = 0
        query_duplicates = 0
        while collected < per_query and page <= max_pages:
            log(f"  page {page} (have {collected}/{per_query} for this query)")
            try:
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
            except requests.RequestException as e:
                log(f"  search request failed on page {page}: {e} — stopping this query")
                break

            photos = resp.json().get("photos", [])
            if not photos:
                log(f"  no more results for '{query}'")
                break

            for photo in photos:
                if collected >= per_query:
                    break
                photo_id = photo["id"]
                if photo_id in existing_ids:
                    skipped_duplicates += 1
                    query_duplicates += 1
                    continue

                try:
                    img_bytes = requests.get(photo["src"]["large2x"], timeout=30).content
                except requests.RequestException as e:
                    failed_downloads += 1
                    log(f"  failed to download photo {photo_id}: {e} — skipping")
                    continue

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
                log(f"  saved {file_name} ({collected}/{per_query})")
                time.sleep(request_delay_s)

            page += 1

        if page > max_pages and collected < per_query:
            log(
                f"  WARNING: query '{query}' stopped at max_pages={max_pages} with only "
                f"{collected}/{per_query} new images (saw {query_duplicates} duplicates) — "
                f"try a different query or raise max_pages"
            )
        log(f"[{query_idx}/{len(queries)}] query='{query}' done: {collected} new images")

    if new_records:
        with open(metadata_path, "a") as f:
            for rec in new_records:
                f.write(json.dumps(rec) + "\n")
        with open(license_path, "a") as f:
            for entry in new_license_entries:
                f.write(json.dumps(entry) + "\n")

    log(
        f"Done: {len(new_records)} new images added, {skipped_duplicates} duplicates skipped, "
        f"{failed_downloads} downloads failed. Total in dataset: {len(existing_ids)}"
    )
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
