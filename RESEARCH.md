# Base Model Selection — Research Notes

This document records the Phase 1 research comparing candidate pretrained open-weight
text-to-image models before committing to one, per the project's requirement not to
just pick the largest model.

## Candidates considered

| Model | Params (denoiser) | License | Commercial use | LoRA/PEFT support | Realistic T4 (16GB) fit |
|---|---|---|---|---|---|
| **Stable Diffusion XL Base 1.0** | ~2.6B (UNet) | CreativeML Open RAIL++-M | **Yes**, with use-based restrictions (see [MODEL_LICENSE.md](MODEL_LICENSE.md)) | Mature — first-class `diffusers` + `peft` LoRA training examples, huge community tooling (kohya_ss, ai-toolkit) | Yes — widely documented (incl. by Hugging Face) as trainable via LoRA/DreamBooth-LoRA on a single T4 with fp16 + gradient checkpointing + 8-bit Adam |
| Stable Diffusion 1.5 | ~860M (UNet) | CreativeML OpenRAIL-M | Yes, same style of use-based restrictions | Extremely mature, the original LoRA training ecosystem | Yes, easily — smaller and faster than SDXL, but weaker text understanding and lower native output quality/resolution (512 vs 1024) |
| FLUX.1-dev | 12B (rectified-flow transformer) | **FLUX.1-dev Non-Commercial License** (Black Forest Labs) | **No** — non-commercial only, unless a separate paid commercial license is purchased from BFL | Growing (ai-toolkit, diffusers `train_dreambooth_lora_flux.py`), but memory-hungry | Not realistically on a stock T4 without aggressive NF4/8-bit quantization; even then it is tight and this repo has not tested it |
| FLUX.1-schnell | 12B (distilled rectified-flow transformer) | Apache 2.0 | Yes | Exists but training a step-distilled model with LoRA is less standard/robust and less documented for quality fine-tuning | Same 12B memory footprint problem as FLUX.1-dev on a T4; not selected |

## Decision: Stable Diffusion XL Base 1.0

**Selected as the primary base model.** Reasoning:

1. **License** — CreativeML Open RAIL++-M explicitly permits commercial use and hosting
   as a service, which matches Siteora's stated future intent. FLUX.1-dev's license
   rules it out outright for a commercial product without a separate paid license from
   Black Forest Labs, which is out of scope for this task to negotiate.
2. **T4 compatibility** — SDXL LoRA fine-tuning on a 16GB T4 is a well-trodden,
   documented path (fp16, gradient checkpointing, 8-bit Adam, batch size 1 + gradient
   accumulation). FLUX's 12B parameter count makes a T4 fit unproven and risky to
   promise without hardware to test it on.
3. **Quality/text understanding** — SDXL is a significant step up from SD 1.5 in both
   image fidelity and prompt adherence, and natively trains/generates at 1024px, which
   matters for website hero imagery.
4. **Ecosystem maturity** — `diffusers`' official `train_text_to_image_lora_sdxl.py`
   example is the direct basis for `src/training/trainer.py` in this repo; PEFT's
   `LoraConfig` + `get_peft_model_state_dict`/`set_peft_model_state_dict` are
   first-class for SDXL's UNet.

SD 1.5 is kept as a documented fallback: if standard-config SDXL training turns out to
be impractically slow/tight on an actual T4 (not yet measured — see README "Tested vs
Expected"), switching `configs/*.yaml`'s `model.base_model` to
`runwayml/stable-diffusion-v1-5`-style checkpoints is a one-line config change, since
`src/training/trainer.py` branches on `"xl" in model_id.lower()` to select the
single-text-encoder (SD1.5) vs dual-text-encoder (SDXL) code path.

## What was NOT tested

Per project policy against false claims: this repo has NOT run a real training
comparison between SDXL and FLUX/SD1.5 on real hardware. The table above reflects
published documentation, license text, and architecture parameter counts, not
head-to-head measurements. A CPU-only pipeline smoke test (see
[`tests/smoke_test.py`](tests/smoke_test.py) and the README) validates that the SDXL
code path works mechanically, using a tiny public test checkpoint, not real SDXL
weights (too large/slow to run end-to-end on the CPU-only machine this repo was
developed on).
