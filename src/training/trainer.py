"""LoRA training loop for the Siteora image model.

Built on Hugging Face diffusers + PEFT, following the structure of
diffusers' official `train_text_to_image_lora_sdxl.py` example, adapted to:
  - our config system (src/training/config.py)
  - our dataset loader (src/dataset/loader.py)
  - our checkpoint format (src/training/checkpointing.py)
  - CUDA / TPU / CPU accelerator detection (src/training/accelerator.py)

This module intentionally does not reimplement diffusion math that
diffusers already provides (noise scheduler, UNet, VAE) — it wires
those together with a LoRA adapter and a training loop.
"""
from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, PretrainedConfig

from dataset.loader import SiteoraImageCaptionDataset, collate_fn
from training import checkpointing
from training.accelerator import AcceleratorInfo, detect_accelerator
from training.config import SiteoraConfig


def _import_text_encoder_class(pretrained_model_name_or_path: str, revision: str | None, subfolder: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    raise ValueError(f"unsupported text encoder architecture: {model_class}")


@dataclasses.dataclass
class TrainingComponents:
    """Everything loaded once at the start of training/smoke-test, so the
    caller (or the notebook) can inspect/reuse them without reloading."""

    tokenizer_one: object
    tokenizer_two: object | None
    text_encoder_one: object
    text_encoder_two: object | None
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler
    is_sdxl: bool


def load_base_model(config: SiteoraConfig, accel_info: AcceleratorInfo) -> TrainingComponents:
    model_id = config.model.base_model
    revision = config.model.revision
    is_sdxl = "xl" in model_id.lower()

    tokenizer_one = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision, use_fast=False)
    text_encoder_cls_one = _import_text_encoder_class(model_id, revision, "text_encoder")
    text_encoder_one = text_encoder_cls_one.from_pretrained(model_id, subfolder="text_encoder", revision=revision)

    tokenizer_two = None
    text_encoder_two = None
    if is_sdxl:
        tokenizer_two = AutoTokenizer.from_pretrained(
            model_id, subfolder="tokenizer_2", revision=revision, use_fast=False
        )
        text_encoder_cls_two = _import_text_encoder_class(model_id, revision, "text_encoder_2")
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            model_id, subfolder="text_encoder_2", revision=revision
        )

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", revision=revision)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze everything; only LoRA adapter params on the UNet will train.
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    if text_encoder_two is not None:
        text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    return TrainingComponents(
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        vae=vae,
        unet=unet,
        noise_scheduler=noise_scheduler,
        is_sdxl=is_sdxl,
    )


def attach_lora(unet: UNet2DConditionModel, config: SiteoraConfig) -> UNet2DConditionModel:
    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
    )
    unet.add_adapter(lora_config)
    return unet


def count_parameters(unet: UNet2DConditionModel) -> tuple[int, int]:
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    return trainable, total


def _encode_prompt_sdxl(components: TrainingComponents, prompts: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds_list = []
    pooled_prompt_embeds = None
    tokenizers = [components.tokenizer_one, components.tokenizer_two]
    text_encoders = [components.text_encoder_one, components.text_encoder_two]

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = outputs[0]
        prompt_embeds = outputs.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def _encode_prompt_sd15(components: TrainingComponents, prompts: list[str], device) -> torch.Tensor:
    text_inputs = components.tokenizer_one(
        prompts,
        padding="max_length",
        max_length=components.tokenizer_one.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return components.text_encoder_one(text_inputs.input_ids.to(device))[0]


def training_step(components: TrainingComponents, batch: dict, device, weight_dtype) -> torch.Tensor:
    """One forward pass + loss computation. Shared by smoke test and trainer."""
    # VAE is kept in fp32 for numerical stability (standard SDXL practice), so its
    # input must be fp32 too; the resulting latents are cast to weight_dtype
    # afterward for the UNet forward pass.
    pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)

    with torch.no_grad():
        latents = components.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * components.vae.config.scaling_factor
    latents = latents.to(weight_dtype)

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(
        0, components.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
    ).long()
    noisy_latents = components.noise_scheduler.add_noise(latents, noise, timesteps)

    if components.is_sdxl:
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_sdxl(components, batch["text"], device)
        add_time_ids = torch.cat(
            [batch["original_size"], batch["crop_top_left"], batch["target_size"]], dim=1
        ).to(device, dtype=weight_dtype)
        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
        model_pred = components.unet(
            noisy_latents.to(weight_dtype),
            timesteps,
            encoder_hidden_states=prompt_embeds.to(weight_dtype),
            added_cond_kwargs=added_cond_kwargs,
        ).sample
    else:
        encoder_hidden_states = _encode_prompt_sd15(components, batch["text"], device)
        model_pred = components.unet(
            noisy_latents.to(weight_dtype), timesteps, encoder_hidden_states=encoder_hidden_states.to(weight_dtype)
        ).sample

    if components.noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif components.noise_scheduler.config.prediction_type == "v_prediction":
        target = components.noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"unsupported prediction type: {components.noise_scheduler.config.prediction_type}")

    return F.mse_loss(model_pred.float(), target.float(), reduction="mean")


def build_dataloaders(config: SiteoraConfig) -> tuple[DataLoader, DataLoader | None]:
    dataset = SiteoraImageCaptionDataset(
        dataset_dir=config.dataset.train_dir,
        metadata_file=config.dataset.metadata_file,
        resolution=config.model.resolution,
        image_column=config.dataset.image_column,
        caption_column=config.dataset.caption_column,
    )
    if config.dataset.val_split > 0 and len(dataset) > 1:
        n_val = max(1, int(len(dataset) * config.dataset.val_split))
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(config.training.seed)
        )
    else:
        train_set, val_set = dataset, None

    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.training.dataloader_num_workers,
    )
    val_loader = (
        DataLoader(val_set, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn)
        if val_set is not None
        else None
    )
    return train_loader, val_loader


def train(config: SiteoraConfig, resume: bool = True) -> Path:
    config.validate()
    accel_info = detect_accelerator()

    set_seed(config.training.seed)

    mixed_precision = config.training.mixed_precision if accel_info.kind != "cpu" else "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_dir=config.paths.log_dir,
    )

    components = load_base_model(config, accel_info)
    components.unet = attach_lora(components.unet, config)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    components.vae.to(accelerator.device, dtype=torch.float32)  # VAE stays fp32 for stability
    components.text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    if components.text_encoder_two is not None:
        components.text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    components.unet.to(accelerator.device, dtype=weight_dtype)
    # Re-enable fp32 for LoRA params so gradients don't underflow in fp16.
    for p in components.unet.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    if config.training.gradient_checkpointing:
        components.unet.enable_gradient_checkpointing()

    trainable_params = [p for p in components.unet.parameters() if p.requires_grad]
    trainable_count, total_count = count_parameters(components.unet)

    use_8bit = config.training.use_8bit_adam and accel_info.kind == "cuda"
    if use_8bit:
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(trainable_params, lr=config.training.learning_rate)

    train_loader, val_loader = build_dataloaders(config)

    num_update_steps_per_epoch = max(1, math.ceil(len(train_loader) / config.training.gradient_accumulation_steps))
    from diffusers.optimization import get_scheduler

    lr_scheduler = get_scheduler(
        config.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.lr_warmup_steps,
        num_training_steps=config.training.max_train_steps,
    )

    components.unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        components.unet, optimizer, train_loader, lr_scheduler
    )

    start_step = 0
    checkpoint_dir = Path(config.paths.checkpoint_dir)
    if resume:
        latest = checkpointing.find_latest_checkpoint(checkpoint_dir)
        if latest is not None:
            state = checkpointing.load_checkpoint(latest, optimizer=optimizer, lr_scheduler=lr_scheduler)
            unwrapped = accelerator.unwrap_model(components.unet)
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(unwrapped, state["unet_lora_state_dict"])
            start_step = state["step"]
            print(f"Resumed from checkpoint {latest} at step {start_step}")

    print("=" * 60)
    print(f"GPU/Accelerator: {accel_info.name}")
    print(f"VRAM: {accel_info.vram_gb:.1f} GB" if accel_info.vram_gb else "VRAM: n/a")
    print(f"Precision: {accelerator.mixed_precision}")
    print(f"Resolution: {config.model.resolution}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"LoRA rank: {config.lora.rank}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Total parameters: {total_count:,}")
    print("=" * 60)

    global_step = start_step
    t_start = time.time()
    components.unet.train()

    progress_target = config.training.max_train_steps
    data_iter = iter(train_loader)
    while global_step < progress_target:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        with accelerator.accumulate(components.unet):
            loss = training_step(components, batch, accelerator.device, weight_dtype)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, config.training.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            elapsed = time.time() - t_start
            accelerator.log({"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

            if global_step % max(1, config.training.save_every) == 0 or global_step == progress_target:
                unwrapped = accelerator.unwrap_model(components.unet)
                lora_state_dict = get_peft_model_state_dict(unwrapped)
                checkpointing.save_checkpoint(
                    checkpoint_dir,
                    global_step,
                    lora_state_dict,
                    optimizer,
                    lr_scheduler,
                    dataclasses.asdict(config) if dataclasses.is_dataclass(config) else {},
                )
                checkpointing.sync_to_drive(checkpoint_dir, config.paths.drive_root)
                print(f"step {global_step}/{progress_target}  loss={loss.item():.4f}  elapsed={elapsed:.1f}s")

    lora_dir = Path(config.paths.lora_dir)
    lora_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(components.unet)
    unwrapped.save_lora_adapter(str(lora_dir))
    accelerator.end_training()
    return lora_dir
