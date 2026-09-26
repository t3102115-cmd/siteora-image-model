# Siteora Image Model

A standalone text-to-image model for generating website visuals (hero images, product
shots, backgrounds, illustrations) — built by parameter-efficient (LoRA) post-training
of a pretrained open-weight diffusion model.

**This repo is intentionally standalone.** It does not integrate with, call, or modify
any part of Siteora (frontend, backend, editor, auth, storage, billing, etc.). It
produces a small LoRA adapter file that a future, separate task will wire into a
Siteora image-generation API.

```
PRETRAINED MODEL (SDXL Base 1.0)
        ↓
Siteora dataset (image/caption pairs)
        ↓
LoRA fine-tuning (PEFT + diffusers)
        ↓
Siteora LoRA adapter
        ↓
Text prompt → generated image (PNG)
```

## Tested vs. Expected vs. Not tested

This section exists because the project brief explicitly forbids false claims. Read it
before trusting any other section.

| Claim | Status |
|---|---|
| Accelerator detection (CUDA/TPU/CPU) | **Tested** on CPU (this dev machine has no GPU). CUDA path is implemented per documented `torch.cuda` API but not run against a real GPU. |
| Config loading + validation for `test`/`standard`/`extended` | **Tested** — all three configs load and validate (`tests/test_config.py`). |
| Dataset validation (missing files/captions, corrupted images) | **Tested** — `tests/test_dataset.py`. |
| Base model load, LoRA injection, checkpoint save/resume | **Tested end-to-end on CPU** using the tiny public test checkpoint `hf-internal-testing/tiny-stable-diffusion-xl-pipe`, standing in for real SDXL (`tests/test_model_and_checkpoint.py`, `tests/smoke_test.py`). This validates the *code path*, not real SDXL numerics. |
| LoRA export → independent reload → generation, with adjustable strength and seed reproducibility | **Tested on CPU** with the tiny test checkpoint (`tests/test_inference.py`). |
| Real SDXL training on a T4 (Colab) | **Not tested.** No CUDA GPU was available in the environment this repo was built in. `configs/standard.yaml`'s numbers (steps, batch size, LR) are reasoned defaults based on published SDXL LoRA fine-tuning guidance, not measured on hardware. |
| TPU v5e-1 training | **Not tested; not even attempted end-to-end.** `src/training/accelerator.py` can detect a TPU via `torch_xla`, and `Accelerator` from `accelerate` is TPU-aware, but the SDXL LoRA trainer has not been run on TPU hardware, and `torch_xla` was not available to install/test here. Treat the TPU path as "should plausibly work via `accelerate`" — not verified. |
| VRAM usage, steps/sec, checkpoint size, inference time on a T4 | **Not measured** (no T4 available). Do not quote numbers for these until measured on real Colab hardware. |
| Image quality / prompt adherence of a real Siteora LoRA | **Not evaluated** — no real training run has produced a real Siteora LoRA yet. The evaluation *harness* (fixed prompts, base-vs-LoRA comparison, notebook) is built and ready to use once a real run happens. |
| Base model license (commercial use permitted) | **Verified** by reading the model card + `LICENSE.md` on Hugging Face directly — see [`MODEL_LICENSE.md`](MODEL_LICENSE.md). |

**In short: the pipeline is real, complete, and mechanically verified end-to-end on
CPU with a tiny stand-in model. It has not yet been run with the real SDXL base model
on a T4, because no GPU was available while building this repo.** The Colab notebooks
in `colab/` are what you run to do that; Cell 9 of `train_lora.ipynb` runs the same
smoke test against whatever model you configure, on whatever accelerator Colab gives
you, before you commit to a long run.

## Architecture

- **Base model:** [Stable Diffusion XL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
  (`stabilityai/stable-diffusion-xl-base-1.0`), a ~2.6B-parameter UNet latent diffusion
  model with dual text encoders (OpenCLIP-ViT/G + CLIP-ViT/L). See [`RESEARCH.md`](RESEARCH.md)
  for the comparison against SD 1.5 and FLUX that led to this choice.
- **Post-training:** LoRA adapters (via `peft`) injected into the UNet's attention
  projection layers (`to_k`, `to_q`, `to_v`, `to_out.0`). The base model's weights stay
  frozen; only the LoRA adapter (a few million parameters, not billions) is trained and
  exported. Text encoders are frozen (not LoRA-tuned) in this implementation.
- **Why LoRA over full fine-tuning:** full fine-tuning of a 2.6B UNet does not fit in
  16GB alongside optimizer state even in fp16, and isn't necessary — LoRA is the
  standard, well-supported approach for style/domain adaptation of SDXL.

## Hardware

- **Primary target: NVIDIA T4 (16GB), Google Colab.** `src/training/accelerator.py`
  detects it, reports name/VRAM/recommended precision (fp16 on T4's Turing architecture
  — no native bf16), and the training loop uses fp16 mixed precision, gradient
  checkpointing, 8-bit Adam (`bitsandbytes`, CUDA-only), and gradient accumulation to
  fit LoRA training in that budget. **Not yet run on an actual T4** (see table above).
- **TPU v5e-1:** detected via `torch_xla` if present; `accelerate`'s `Accelerator`
  abstracts most of the training loop across CUDA/TPU/CPU. **Not tested.** If you try
  it and it breaks, that is expected until someone runs and fixes it on real TPU
  hardware — this repo does not claim otherwise.
- **CPU:** `detect_accelerator()` warns loudly and refuses to pretend CPU is a good
  idea for real training. It's used here only for the pipeline smoke test with a tiny
  test model (seconds, not hours).

## Dataset

See [`dataset/README.md`](dataset/README.md) for the full spec. Summary:

```
dataset/
├── images/
│   └── image_000001.jpg ...
└── metadata.jsonl   # {"file_name": "...", "text": "caption..."}
```

Tools in `src/dataset/`:
- `validation.py` — checks for missing files, missing captions, corrupted images;
  `require_valid_dataset()` raises loudly rather than silently training on bad data.
- `preprocessing.py` — resize/crop to a square training resolution (writes to a
  **separate** output directory, never overwrites originals), train/val split,
  dataset statistics.
- `captions.py` — build `metadata.jsonl` from an `images/` + `.txt`-per-image layout;
  fails if any caption is missing rather than silently skipping or inventing one.
- `loader.py` — PyTorch `Dataset` producing pixel tensors plus SDXL's
  micro-conditioning (`original_size`, `crop_top_left`, `target_size`).

No training images are bundled with this repo (see dataset/README.md's licensing
note) except tiny procedurally-generated solid-color images under `dataset/test/`,
used purely for the automated pipeline smoke test.

### Captions

Write full sentences covering subject, style, environment, lighting, and composition.
Bad: `"car"`. Better: `"A premium black sports car driving along a winding mountain
road at sunset, cinematic commercial photography, dramatic warm lighting, wide
composition with negative space on the left."`

## Training

Three configs in `configs/`, chosen by `--config` / edited in the notebook:

| Config | Purpose | Steps | Resolution | LoRA rank | Status |
|---|---|---|---|---|---|
| `test.yaml` | Verify the pipeline works end-to-end | 20 | 512 | 4 | **Tested mechanically** (via the smoke test's equivalent path) |
| `standard.yaml` | Produce a genuinely useful Siteora LoRA | 3000 | 1024 | 16 | Not yet run — reasoned defaults, see table above |
| `extended.yaml` | Maximum-quality experimentation | 8000 | 1024 | 32 | Not yet run; may require multiple resumed Colab sessions |

### Colab quickstart

Open `colab/train_lora.ipynb` and run cells top to bottom:

1. Install dependencies
2. Mount Google Drive (optional, for checkpoint persistence)
3. Detect accelerator
4. Load a config (`configs/test.yaml` to start)
5. Point at your dataset
6. Validate the dataset (fails loudly if broken)
7. Load the base model
8. Attach LoRA
9. **Run the smoke test** (`tests/smoke_test.py`) — always do this before a long run
10. Train
11. View TensorBoard metrics
12. Generate evaluation images
13. Export the LoRA

### Checkpointing & resume

Colab disconnects are expected. `src/training/checkpointing.py` saves LoRA weights +
optimizer state + LR scheduler state + step count + config to
`checkpoints/<config>/step-XXXXXXX/`, and `training.trainer.train(config, resume=True)`
(the default) automatically resumes from the latest checkpoint if one exists.
`sync_to_drive()` mirrors checkpoints to `paths.drive_root` (default
`/content/drive/MyDrive/siteora-image-model/`) when Drive is mounted.

## Inference

```python
from inference.generate import generate_image

image = generate_image(
    prompt="A premium AI startup office at night, futuristic but professional",
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    lora_path="models/siteora-lora-standard",
    seed=12345,
    lora_strength=1.0,
)
image.save("out.png")
```

Or the class form for repeated generations without reloading the model:
`inference.generate.SiteoraImageGenerator`. Supports prompt, negative prompt,
width/height, steps, guidance scale, seed, and LoRA strength (tested range 0.5–1.25).
See `colab/inference.ipynb` — it works standalone, without having run training in the
same session, as long as a LoRA has been exported.

Exact reproducibility from a seed can vary across hardware, precision, PyTorch/CUDA
versions, and sampler; seed reproducibility was verified only in-process, on CPU, with
the tiny test model (`tests/test_inference.py::test_seed_reproducibility_on_cpu`).

## Evaluation

`src/evaluation/evaluate.py` defines a fixed, reusable evaluation prompt set
(`EVAL_PROMPTS`) covering SaaS, restaurant, real estate, fashion, travel,
e-commerce, architecture, technology, professional services, and abstract imagery,
across composition types (wide hero, centered, left/right subject, negative space,
close-up, full scene). `run_evaluation()` generates and saves one image per prompt
plus a `manifest.json`; `build_comparison_report()` lists what was generated across
labeled runs (e.g. `base`, `checkpoint_1`, `final`) for side-by-side human judgment.
See `colab/evaluate.ipynb`. This harness does not compute an automated "quality
score" — per project policy, quality/composition/prompt-adherence judgments are made
by a person looking at the images, not fabricated by a metric.

## Model export & versioning

`unet.save_lora_adapter(lora_dir)` (called at the end of `training.trainer.train()`)
exports **only the LoRA adapter**, not the base model — a few MB to tens of MB,
depending on rank, not gigabytes. Metadata to record alongside a release (not yet
automated — do this manually until an integration task adds it): base model + version,
LoRA rank, training resolution, dataset version, training steps, learning rate, date,
and the versions in `requirements.txt`. Suggested version naming:
`siteora-image-v0.1`, `v0.2`, `v1.0`, matching the project's phase.

## Project structure

```
siteora-image-model/
├── colab/                    # train_lora.ipynb, inference.ipynb, dataset.ipynb, evaluate.ipynb
├── src/
│   ├── training/              # config, accelerator detection, trainer, checkpointing
│   ├── inference/              # generate.py — standalone generation interface
│   ├── dataset/                # loader, preprocessing, validation, captions
│   └── evaluation/             # fixed-prompt evaluation harness
├── configs/                   # test.yaml, standard.yaml, extended.yaml
├── dataset/                    # README + test/ (smoke-test images only)
├── tests/                       # pytest suite + smoke_test.py
├── outputs/ checkpoints/ models/ samples/ evaluations/   # gitignored, generated at runtime
├── requirements.txt
├── MODEL_LICENSE.md            # base model license, verified from source
├── RESEARCH.md                 # base model comparison / selection rationale
└── README.md
```

## Running the tests locally

```bash
pip install -r requirements.txt pytest
python -m pytest tests/ -v         # unit tests (downloads a ~few-MB tiny test model)
python tests/smoke_test.py         # full pipeline smoke test, same tiny model
```

All 22 unit tests and the smoke test pass on CPU as of this writing (see table above
for exactly what that does and doesn't prove).

## Limitations

- No real SDXL training run has been performed; `standard.yaml`/`extended.yaml`'s
  hyperparameters are untested defaults, not measured-good values.
- No T4 or TPU performance numbers exist yet (VRAM, steps/sec, wall-clock time).
- No real Siteora dataset is bundled; you must build and license one
  (see `dataset/README.md`).
- Text encoders are frozen; only the UNet gets a LoRA. This is standard practice for
  SDXL LoRA but means text-encoder-level style shifts are out of scope for this
  implementation.
- TPU support is unverified — treat it as experimental.
- This repo does not connect to Siteora in any way, by design (see task scope).

## Reproduction

1. `pip install -r requirements.txt`
2. Build a dataset per `dataset/README.md` (or use `dataset/test/` to just exercise
   the pipeline).
3. `python -m pytest tests/` and `python tests/smoke_test.py` to confirm your
   environment works.
4. Open `colab/train_lora.ipynb` on a Colab T4 runtime, pick a config, and run cells
   top to bottom (see "Training" above).
5. Open `colab/inference.ipynb` to generate images from the exported LoRA.
6. Open `colab/evaluate.ipynb` to compare base vs. LoRA output on the fixed prompt set.
