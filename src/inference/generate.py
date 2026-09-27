"""Standalone inference interface for the Siteora image model.

Deliberately shaped like a future API call — `generate_image(prompt, ...)`
returns a PIL.Image — so a later (out-of-scope) HTTP API can wrap this
function directly without restructuring it.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from PIL import Image


@dataclasses.dataclass
class GenerationResult:
    image: Image.Image
    seed: int
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    steps: int
    guidance_scale: float
    lora_strength: float


class SiteoraImageGenerator:
    """Loads the base model once and the Siteora LoRA on top of it.

    Usage:
        gen = SiteoraImageGenerator(base_model="stabilityai/stable-diffusion-xl-base-1.0",
                                     lora_path="models/siteora-lora-standard")
        result = gen.generate("A premium AI startup office at night", seed=12345)
        result.image.save("out.png")
    """

    def __init__(
        self,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora_path: str | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        scheduler: str = "euler",
        num_threads: int | None = None,
    ):
        """
        scheduler: "euler" (SDXL's default, needs ~30-50 steps for good quality)
            or "dpm++" (DPMSolverMultistepScheduler with Karras sigmas, reaches
            comparable quality in ~15-20 steps -- meaningfully faster on CPU,
            where every step is expensive). Not benchmarked against each other
            in this repo; "dpm++ needs fewer steps" is standard, widely-used
            diffusion sampling knowledge, not a measurement made here.
        num_threads: for CPU inference, pins PyTorch's intra-op thread count
            to the deployment machine's actual CPU core count (torch's
            default can over- or under-subscribe on a VPS with an unusual
            core count). Ignored on CUDA.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            # fp16 has no real speed benefit on CPU (most CPUs lack fast fp16
            # kernels and diffusers/torch often upcast anyway) and can even be
            # slower than fp32 -- only use fp16 on CUDA.
            dtype = torch.float16 if device == "cuda" else torch.float32

        if device == "cpu" and num_threads is not None:
            torch.set_num_threads(num_threads)

        self.device = device
        self.dtype = dtype
        self.pipe = StableDiffusionXLPipeline.from_pretrained(base_model, dtype=dtype)
        if scheduler == "dpm++":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config, use_karras_sigmas=True
            )
        elif scheduler != "euler":
            raise ValueError(f"unknown scheduler '{scheduler}', expected 'euler' or 'dpm++'")
        self.pipe.to(device)
        self.lora_loaded = False

        if lora_path is not None:
            self.load_lora(lora_path)

    def load_lora(self, lora_path: str, adapter_name: str = "siteora") -> None:
        # Siteora LoRAs are exported UNet-only via `UNet2DConditionModel.save_lora_adapter`
        # (see src/training/trainer.py), which writes unprefixed PEFT keys —
        # loading through the generic `pipe.load_lora_weights` (which expects
        # a per-component key prefix) silently finds zero matching keys, so
        # we load directly onto the UNet with prefix=None instead.
        weight_name = "pytorch_lora_weights.safetensors"
        weight_path = Path(lora_path) / weight_name
        self.pipe.unet.load_lora_adapter(
            lora_path,
            adapter_name=adapter_name,
            weight_name=weight_name if weight_path.exists() else None,
            prefix=None,
        )
        self._adapter_name = adapter_name
        self.lora_loaded = True

    def set_lora_strength(self, strength: float) -> None:
        if not self.lora_loaded:
            raise RuntimeError("no LoRA loaded; call load_lora() first")
        self.pipe.unet.set_adapters([self._adapter_name], weights=[strength])

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        seed: int | None = None,
        lora_strength: float = 1.0,
    ) -> GenerationResult:
        if self.lora_loaded:
            self.set_lora_strength(lora_strength)

        if seed is None:
            seed = torch.seed() % (2**32)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        return GenerationResult(
            image=image,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=num_inference_steps,
            guidance_scale=guidance_scale,
            lora_strength=lora_strength if self.lora_loaded else 0.0,
        )


def generate_image(
    prompt: str,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    lora_path: str | None = None,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    lora_strength: float = 1.0,
    save_path: str | Path | None = None,
) -> Image.Image:
    """Convenience one-shot function matching the shape a future API would call."""
    generator = SiteoraImageGenerator(base_model=base_model, lora_path=lora_path)
    result = generator.generate(prompt, width=width, height=height, seed=seed, lora_strength=lora_strength)
    if save_path is not None:
        result.image.save(save_path)
    return result.image
