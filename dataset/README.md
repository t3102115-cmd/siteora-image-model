# Dataset

## Structure

```
dataset/
├── images/
│   ├── image_000001.jpg
│   └── ...
└── metadata.jsonl
```

Each line of `metadata.jsonl` is a JSON object:

```json
{"file_name": "image_000001.jpg", "text": "modern luxury restaurant interior with warm ambient lighting, professional architectural photography"}
```

`file_name` is resolved relative to `images/` first, falling back to the dataset root (flat layout also supported).

## Building a dataset

1. Collect images licensed for training + commercial derivative use (see licensing note below — this matters because the
   trained LoRA is a derivative work).
2. Caption each image manually, or via `dataset/captions.py::generate_metadata_from_captions_dir`, which expects one
   `<image>.txt` file per image and **fails loudly** if any caption is missing — captions are never auto-filled silently.
3. Run `colab/dataset.ipynb` or `src/dataset/inspect_cli.py <dataset_dir>` to validate before training.

## Caption guidance

Write captions that describe subject, style, environment, lighting, and composition — see the main
[README](../README.md#captions) and project spec for examples of good vs. bad captions. Bad: `"car"`. Better: a full
sentence describing subject, lighting, composition, and photography style.

## Licensing

**No sample dataset is bundled with this repo.** Any images you add must be licensed for commercial derivative use
(training a commercial LoRA on images without a license that permits this is a legal risk to Siteora). Track the
license source for every batch of images you add (stock license, public-domain source, self-shot, synthetic, etc.) —
this repo does not track per-image licenses for you.

The tiny placeholder images under `dataset/test/` are procedurally generated solid-color rectangles created for pipeline
smoke-testing only (see `tests/smoke_test.py`) — they contain no copyrighted content and produce no meaningful training
signal; do not use them as a real training set.
