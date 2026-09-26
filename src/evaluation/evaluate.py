"""Fixed-prompt evaluation harness: generate images for a reusable prompt
set across the base model and each checkpoint / final LoRA, for
side-by-side comparison. Does not compute an automatic "quality score" —
per project policy (#20), quality/prompt-adherence/composition judgments
are made by a human looking at the saved images, not fabricated by a metric.
"""
from __future__ import annotations

import json
from pathlib import Path

from inference.generate import SiteoraImageGenerator

# Reusable evaluation prompt set (#21): covers Siteora's target website
# categories and composition types.
EVAL_PROMPTS: list[dict] = [
    {"id": "saas_hero", "category": "SaaS", "composition": "wide hero",
     "prompt": "A premium SaaS website hero image with a futuristic technology aesthetic, wide composition and empty space for headline text."},
    {"id": "restaurant_interior", "category": "Restaurant", "composition": "full scene",
     "prompt": "A luxury modern restaurant interior for a restaurant website, warm lighting, editorial architectural photography."},
    {"id": "real_estate_house", "category": "Real estate", "composition": "wide hero",
     "prompt": "A modern luxury oceanfront house for a real estate website, professional architectural photography, wide composition."},
    {"id": "fashion_editorial", "category": "Fashion", "composition": "centered",
     "prompt": "A premium fashion editorial photograph for a luxury clothing brand website, centered subject, studio lighting."},
    {"id": "ai_startup_abstract", "category": "Abstract", "composition": "negative space",
     "prompt": "An abstract futuristic technology background for an AI startup landing page, large negative space on the right."},
    {"id": "tech_left_subject", "category": "Technology", "composition": "left subject",
     "prompt": "A sleek modern laptop on a minimal desk, subject positioned on the left third of the frame, clean background for a technology website."},
    {"id": "travel_right_subject", "category": "Travel", "composition": "right subject",
     "prompt": "A scenic tropical beach resort at sunset for a travel website, subject on the right third, empty sky on the left for text."},
    {"id": "ecommerce_product", "category": "E-commerce", "composition": "close-up",
     "prompt": "A close-up product photography shot of a premium wristwatch on a clean gradient background for an e-commerce website."},
    {"id": "architecture_wide", "category": "Architecture", "composition": "wide hero",
     "prompt": "A striking modern glass office building for an architecture firm website, wide composition, dramatic sky."},
    {"id": "professional_services", "category": "Professional services", "composition": "full scene",
     "prompt": "A confident professional in a modern office consulting with a client, corporate photography style, for a professional services website."},
]


def run_evaluation(
    label: str,
    output_root: str | Path,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    lora_path: str | None = None,
    lora_strength: float = 1.0,
    seed: int = 12345,
    num_inference_steps: int = 30,
    width: int = 1024,
    height: int = 1024,
    prompts: list[dict] | None = None,
) -> Path:
    """Generate images for every EVAL_PROMPTS entry and save under
    evaluation/<label>/, plus a manifest.json describing the run."""
    prompts = prompts or EVAL_PROMPTS
    out_dir = Path(output_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = SiteoraImageGenerator(base_model=base_model, lora_path=lora_path)

    manifest = {
        "label": label,
        "base_model": base_model,
        "lora_path": lora_path,
        "lora_strength": lora_strength if lora_path else None,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "results": [],
    }

    for item in prompts:
        result = generator.generate(
            item["prompt"],
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
            lora_strength=lora_strength,
        )
        file_name = f"{item['id']}.png"
        result.image.save(out_dir / file_name)
        manifest["results"].append({**item, "file_name": file_name})

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return out_dir


def build_comparison_report(evaluation_root: str | Path, labels: list[str]) -> str:
    """Produce a plain-text index comparing manifests across evaluation runs
    (e.g. base / checkpoint_1 / checkpoint_2 / final). This lists what was
    generated where; it does not itself judge image quality."""
    evaluation_root = Path(evaluation_root)
    lines = ["# Siteora LoRA Evaluation Report", ""]
    for label in labels:
        manifest_path = evaluation_root / label / "manifest.json"
        if not manifest_path.exists():
            lines.append(f"## {label}: NOT RUN (no manifest found)")
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        lines.append(f"## {label}")
        lines.append(f"- base_model: {manifest['base_model']}")
        lines.append(f"- lora_path: {manifest['lora_path']}")
        lines.append(f"- lora_strength: {manifest['lora_strength']}")
        lines.append(f"- seed: {manifest['seed']}")
        lines.append(f"- images: {len(manifest['results'])}")
        lines.append("")
    return "\n".join(lines)
