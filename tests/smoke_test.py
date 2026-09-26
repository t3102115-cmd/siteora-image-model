"""End-to-end smoke test of the training pipeline mechanics.

Runs the FULL pipeline — dataset load, model load, LoRA attach, one batch,
forward pass, loss, backward pass, optimizer step, checkpoint save,
checkpoint resume, and image generation — using a tiny public test model
(`hf-internal-testing/tiny-stable-diffusion-xl-pipe`) on CPU.

This validates the CODE PATH, not model quality: the tiny model produces
noise images and the loss numbers are meaningless. Real training requires
the actual SDXL base model on a CUDA GPU (see README "Tested vs Expected").
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dataset.validation import require_valid_dataset
from training import checkpointing
from training.config import DatasetConfig, LoRAConfig, ModelConfig, PathsConfig, SiteoraConfig, TrainingConfig
from training.trainer import (
    attach_lora,
    build_dataloaders,
    count_parameters,
    load_base_model,
    training_step,
)
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict


def make_smoke_config(tmp_root: Path) -> SiteoraConfig:
    return SiteoraConfig(
        name="smoke",
        model=ModelConfig(base_model="hf-internal-testing/tiny-stable-diffusion-xl-pipe", resolution=128),
        lora=LoRAConfig(rank=4, alpha=4, dropout=0.0, target_modules=["to_k", "to_q", "to_v", "to_out.0"]),
        training=TrainingConfig(
            learning_rate=1e-3,
            batch_size=1,
            gradient_accumulation_steps=1,
            max_train_steps=2,
            save_every=1,
            eval_every=1,
            mixed_precision="no",
            gradient_checkpointing=False,
            use_8bit_adam=False,
            lr_scheduler="constant",
            lr_warmup_steps=0,
            seed=0,
            dataloader_num_workers=0,
        ),
        dataset=DatasetConfig(train_dir="dataset/test", metadata_file="metadata.jsonl", val_split=0.2),
        paths=PathsConfig(
            output_dir=str(tmp_root / "outputs"),
            checkpoint_dir=str(tmp_root / "checkpoints"),
            lora_dir=str(tmp_root / "lora"),
            sample_dir=str(tmp_root / "samples"),
            log_dir=str(tmp_root / "logs"),
            drive_root=None,
        ),
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tmp_root = repo_root / "outputs" / "_smoke_test"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)

    print("[1/9] Validating dataset...")
    report = require_valid_dataset(repo_root / "dataset" / "test")
    print(report.summary())

    config = make_smoke_config(tmp_root)
    config.validate()

    print("[2/9] Loading base model (tiny test model)...")
    from training.accelerator import detect_accelerator

    accel_info = detect_accelerator()
    components = load_base_model(config, accel_info)

    print("[3/9] Attaching LoRA adapter...")
    components.unet = attach_lora(components.unet, config)
    trainable, total = count_parameters(components.unet)
    print(f"    trainable params: {trainable:,} / total: {total:,}")
    assert trainable > 0, "LoRA attach produced zero trainable parameters"

    print("[4/9] Building dataloader and loading one batch...")
    train_loader, val_loader = build_dataloaders(config)
    batch = next(iter(train_loader))
    assert batch["pixel_values"].shape[-1] == config.model.resolution

    print("[5/9] Forward pass + loss...")
    device = torch.device("cpu")
    components.vae.to(device)
    components.text_encoder_one.to(device)
    if components.text_encoder_two is not None:
        components.text_encoder_two.to(device)
    components.unet.to(device)
    loss = training_step(components, batch, device, torch.float32)
    print(f"    loss = {loss.item():.4f}")
    assert torch.isfinite(loss)

    print("[6/9] Backward pass + optimizer step...")
    trainable_params = [p for p in components.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.training.learning_rate)
    optimizer.zero_grad()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in trainable_params if p.grad is not None]
    assert len(grad_norms) > 0 and any(g > 0 for g in grad_norms), "no gradients flowed to LoRA params"
    optimizer.step()
    print(f"    {len(grad_norms)} LoRA tensors received gradients")

    print("[7/9] Saving checkpoint...")
    lora_state_dict = get_peft_model_state_dict(components.unet)
    ckpt_path = checkpointing.save_checkpoint(
        config.paths.checkpoint_dir, step=1, unet_lora_state_dict=lora_state_dict,
        optimizer=optimizer, lr_scheduler=None, config_dict={"smoke": True},
    )
    assert ckpt_path.exists()
    print(f"    saved to {ckpt_path}")

    print("[8/9] Resuming from checkpoint...")
    optimizer2 = torch.optim.AdamW(trainable_params, lr=config.training.learning_rate)
    state = checkpointing.load_checkpoint(ckpt_path, optimizer=optimizer2)
    set_peft_model_state_dict(components.unet, state["unet_lora_state_dict"])
    assert state["step"] == 1
    print("    resume OK, step =", state["step"])

    print("[9/9] Generating an image with the (tiny) pipeline...")
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.model.base_model,
        unet=components.unet,
        vae=components.vae,
        text_encoder=components.text_encoder_one,
        text_encoder_2=components.text_encoder_two,
        tokenizer=components.tokenizer_one,
        tokenizer_2=components.tokenizer_two,
    )
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator().manual_seed(42)
    image = pipe(
        "a modern SaaS website hero image", num_inference_steps=2, height=128, width=128, generator=generator
    ).images[0]
    sample_dir = Path(config.paths.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_dir / "smoke_test_output.png"
    image.save(out_path)
    assert out_path.exists()
    print(f"    saved image to {out_path}")

    print("\nSMOKE TEST PASSED — pipeline mechanics verified on CPU with a tiny test model.")
    print("This does NOT validate real image quality or T4/SDXL performance.")


if __name__ == "__main__":
    main()
