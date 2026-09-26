"""Tests scraper logic (caption building, dedup, append-only writes) against
a mocked Pexels API — no real network call or API key needed."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset.scraper import _build_caption, scrape_pexels


def test_build_caption_fills_alt_text():
    caption = _build_caption("modern saas office", "people working at desks", "a modern office, {alt}")
    assert caption == "a modern office, people working at desks"


def test_build_caption_handles_missing_alt():
    caption = _build_caption("x", None, "a photo of x, {alt}")
    assert caption == "a photo of x"


def _fake_photo(photo_id: int) -> dict:
    return {
        "id": photo_id,
        "alt": f"alt text {photo_id}",
        "photographer": "Jane Doe",
        "photographer_url": "https://pexels.com/@jane",
        "url": f"https://pexels.com/photo/{photo_id}",
        "src": {"large2x": f"https://images.pexels.com/{photo_id}.jpg"},
    }


def test_scrape_pexels_writes_metadata_and_dedupes(tmp_path):
    queries = {"modern saas office": "a modern office, {alt}"}

    mock_image_resp = MagicMock()
    mock_image_resp.content = b"fake-jpeg-bytes"

    def make_search_side_effect(pages: list[list[dict]]):
        # Real Pexels pagination: each page has its own results, and the
        # page past the last one returns an empty list — this is what makes
        # the scraper's pagination loop terminate.
        def side_effect(url, *args, **kwargs):
            if url == "https://api.pexels.com/v1/search":
                page_num = kwargs["params"]["page"]
                resp = MagicMock()
                resp.raise_for_status = lambda: None
                photos = pages[page_num - 1] if page_num <= len(pages) else []
                resp.json.return_value = {"photos": photos}
                return resp
            return mock_image_resp

        return side_effect

    with patch("dataset.scraper.requests.get") as mock_get:
        mock_get.side_effect = make_search_side_effect([[_fake_photo(1), _fake_photo(2)]])
        n = scrape_pexels(queries=queries, per_query=2, dataset_dir=tmp_path, api_key="fake-key", request_delay_s=0)

    assert n == 2
    metadata_path = tmp_path / "metadata.jsonl"
    with open(metadata_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 2
    assert records[0]["text"] == "a modern office, alt text 1"
    assert (tmp_path / "images" / "pexels_1.jpg").read_bytes() == b"fake-jpeg-bytes"

    license_path = tmp_path / "license_manifest.jsonl"
    with open(license_path) as f:
        licenses = [json.loads(line) for line in f if line.strip()]
    assert len(licenses) == 2
    assert licenses[0]["photographer"] == "Jane Doe"

    # Re-running with the same photos should add nothing (dedup by pexels id),
    # and must terminate via max_pages rather than looping forever, since
    # every page keeps handing back the same two already-seen photos.
    with patch("dataset.scraper.requests.get") as mock_get:
        mock_get.side_effect = make_search_side_effect(
            [[_fake_photo(1), _fake_photo(2)]] * 3  # same duplicates on every page
        )
        n_second = scrape_pexels(
            queries=queries, per_query=2, dataset_dir=tmp_path, api_key="fake-key", request_delay_s=0, max_pages=3
        )
    assert n_second == 0
    with open(metadata_path) as f:
        assert len([l for l in f if l.strip()]) == 2  # unchanged, not duplicated


def test_scrape_pexels_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    import pytest

    with pytest.raises(ValueError):
        scrape_pexels(dataset_dir=tmp_path, api_key=None)
