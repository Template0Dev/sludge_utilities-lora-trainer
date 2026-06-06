from __future__ import annotations

import os
import shutil
from inspect import signature
from pathlib import Path
from typing import Any

import yaml

from lora_trainer.config import AppConfig
from lora_trainer.dataset import DatasetManifest, load_training_records, load_validation_records, write_manifest


def adapter_output_dir(config: AppConfig, dataset_name: str) -> Path:
    return config.output.adapter_root / dataset_name


def train_adapter(config: AppConfig, dataset_root: Path, manifest: DatasetManifest, *, config_path: Path) -> Path:
    output_dir = adapter_output_dir(config, manifest.dataset_name)
    if output_dir.exists():
        if not config.output.overwrite:
            raise RuntimeError(f"Adapter output already exists: {output_dir}. Set output.overwrite=true to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.training.set_unsloth_env_flags:
        for key, val in config.training.unsloth_env_flags.items():
            os.environ[key] = str(val)

    if config.training.unsloth_moe_backend:
        os.environ["UNSLOTH_MOE_BACKEND"] = config.training.unsloth_moe_backend

    if config.training.qlora_4bit:
        print("Warning: 4-bit QLoRA is explicit opt-in and may be limited for MoE/VL models.")

    try:
        _train_with_unsloth(config, dataset_root, output_dir)
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth training dependencies are not installed. Run this command on a Linux NVIDIA GPU host after uv sync."
        ) from exc

    shutil.copyfile(config_path, output_dir / "run_config.yaml")
    write_manifest(manifest, output_dir / "dataset_manifest.json")
    return output_dir


def _train_with_unsloth(config: AppConfig, dataset_root: Path, output_dir: Path) -> None:
    from unsloth import FastVisionModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    train_dataset = Dataset.from_list(load_training_records(dataset_root))
    eval_records = load_validation_records(dataset_root)
    eval_dataset = Dataset.from_list(eval_records) if eval_records else None

    model, processor = FastVisionModel.from_pretrained(
        config.model.id,
        max_seq_length=config.training.max_seq_length,
        load_in_4bit=config.training.qlora_4bit,
        use_gradient_checkpointing="unsloth" if config.training.gradient_checkpointing else False,
    )
    model = FastVisionModel.get_peft_model(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_steps=config.training.max_steps,
        learning_rate=config.training.learning_rate,
        bf16=True,
        fp16=False,
        seed=config.training.seed,
        remove_unused_columns=False,
        dataset_text_field="",
        **sft_length_kwargs(SFTConfig, config.training.max_seq_length),
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def write_config_snapshot(config: AppConfig, output_path: Path) -> None:
    output_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True), encoding="utf-8")


def sft_length_kwargs(sft_config_cls: type[Any], max_seq_length: int) -> dict[str, int]:
    params = signature(sft_config_cls).parameters
    if "max_length" in params:
        return {"max_length": max_seq_length}
    if "max_seq_length" in params:
        return {"max_seq_length": max_seq_length}
    raise RuntimeError("TRL SFTConfig does not expose max_length or max_seq_length")
