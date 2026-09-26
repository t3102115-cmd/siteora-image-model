# Base Model License

Verified 2026-09-26 by reading the model card and `LICENSE.md` directly from
[huggingface.co/stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

```
Base model:      Stable Diffusion XL (SDXL) Base
Version:         1.0
License:         CreativeML Open RAIL++-M License (dated July 26, 2023)
Commercial use:  Permitted. The license grants a perpetual, worldwide, non-exclusive,
                 royalty-free, irrevocable copyright + patent license, and explicitly
                 permits hosting the Model or Derivatives as a Third-Party service
                 (SaaS / API-based / web access) — this covers Siteora's intended
                 future use (an API in front of this model).
Derivative/fine-tuning permissions:
                 Fine-tuning and LoRA adapters ARE "Derivatives of the Model" under
                 this license's definition. You may create and distribute them.
Attribution requirements:
                 Any Distribution of the Model or a Derivative must include a copy of
                 this License and give notice to downstream recipients that it is
                 subject to the license's use-based restrictions (Section III,
                 paragraph 4.a). Copyright/patent/attribution notices must be retained.
Other restrictions:
                 "Use-based restrictions" (Attachment A) prohibit specific categories
                 of use regardless of commercial/non-commercial status — e.g. generating
                 or facilitating illegal content, exploiting/harming minors, generating
                 disinformation, fully automated legal/medical/immigration decisions
                 without human oversight, and similar harm categories. ANY Derivative
                 (including the Siteora LoRA) MUST carry forward at least these same
                 use-based restrictions to its own downstream users — this needs to be
                 reflected in Siteora's own terms of service once this model is exposed
                 to Siteora users through the future API.
```

## Why this matters for Siteora

Siteora intends commercial use (websites for paying customers) and eventual hosting
as an API. The CreativeML Open RAIL++-M License was specifically designed to permit
exactly this (unlike, for example, FLUX.1-dev's non-commercial research license — see
`RESEARCH.md` for why FLUX.1-dev was ruled out). The only ongoing obligation is
carrying the use-based restrictions through to whatever terms Siteora publishes for
end users of the image-generation feature — this is a product/legal task for the
(out-of-scope) integration phase, not a training-pipeline concern, but is flagged here
so it isn't lost.

## LoRA adapter license

The LoRA adapter weights produced by this repo are themselves a "Derivative of the
Model" and are subject to the same CreativeML Open RAIL++-M terms above — they are
not separately licensed. This repo does not itself redistribute the adapter; whoever
distributes a trained `models/siteora-lora-*/` directory is responsible for including
a copy of this license alongside it, per the terms above.

## Training images

The base model's license does not cover the license of images you add to `dataset/`.
See [`dataset/README.md`](dataset/README.md) — that is a separate, per-image licensing
question you must resolve for your training data (this repo does not track or verify
per-image licenses).
