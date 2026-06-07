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

    if config.training_misc_options.env_flags.unsloth.set_unsloth_env_flags:
        for key, val in config.training_misc_options.env_flags.unsloth.flags.items():
            os.environ[key] = str(val)

    if config.training_misc_options.backend_params.unsloth_moe_backend:
        os.environ["UNSLOTH_MOE_BACKEND"] = config.training_misc_options.backend_params.unsloth_moe_backend

    if config.training_misc_options.qlora_4bit:
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

    import builtins

    # Patch Qwen MoE configs and inject modeling classes into builtins to avoid NameErrors/AttributeErrors during dynamic Unsloth compilation
    try:
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
        Qwen3_5MoeTextConfig.intermediate_size = property(
            lambda self: getattr(self, "moe_intermediate_size", None)
        )
    except ImportError:
        pass

    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts, Qwen3_5MoeSparseMoeBlock
        builtins.Qwen3_5MoeExperts = Qwen3_5MoeExperts
        builtins.Qwen3_5MoeSparseMoeBlock = Qwen3_5MoeSparseMoeBlock
    except (ImportError, AttributeError):
        pass

    try:
        from transformers.models.qwen2_5_moe.configuration_qwen2_5_moe import Qwen2_5MoeConfig
        Qwen2_5MoeConfig.intermediate_size = property(
            lambda self: getattr(self, "moe_intermediate_size", None)
        )
    except ImportError:
        pass

    try:
        from transformers.models.qwen2_5_moe.modeling_qwen2_5_moe import Qwen2_5MoeExperts, Qwen2_5MoeSparseMoeBlock
        builtins.Qwen2_5MoeExperts = Qwen2_5MoeExperts
        builtins.Qwen2_5MoeSparseMoeBlock = Qwen2_5MoeSparseMoeBlock
    except (ImportError, AttributeError):
        pass

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    train_dataset = Dataset.from_list(load_training_records(dataset_root))
    eval_records = load_validation_records(dataset_root)
    eval_dataset = Dataset.from_list(eval_records) if eval_records else None

    model_kwargs = build_model_load_kwargs(config)
    model, processor = FastVisionModel.from_pretrained(
        config.model.id,
        **model_kwargs,
    )
    patch_model_runtime_config(model, config)
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

    save_policy = config.training_misc_options.save_settings.save_policy
    if save_policy == "final":
        save_strategy = "no"
        save_steps = None
    elif save_policy == "steps":
        save_strategy = "steps"
        save_steps = config.training_misc_options.save_settings.save_steps
    elif save_policy == "percent":
        save_strategy = "steps"
        save_steps = max(
            1,
            int(
                config.training_hyper_params.max_steps
                * (config.training_misc_options.save_settings.save_percentage / 100.0)
            ),
        )
    else:
        raise ValueError(f"Unknown save_policy: {save_policy}")

    sft_kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "per_device_train_batch_size": config.training_hyper_params.batch_size,
        "gradient_accumulation_steps": config.training_hyper_params.gradient_accumulation_steps,
        "max_steps": config.training_hyper_params.max_steps,
        "learning_rate": config.training_hyper_params.learning_rate,
        "logging_steps": 1,
        "bf16": True,
        "fp16": False,
        "seed": config.training_hyper_params.seed,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "save_strategy": save_strategy,
        "save_total_limit": config.training_misc_options.save_settings.save_total_limit,
        **sft_length_kwargs(SFTConfig, config.training_hyper_params.max_seq_length),
    }
    if save_steps is not None:
        sft_kwargs["save_steps"] = save_steps

    callbacks = []
    if config.training_hyper_params.early_stopping:
        if eval_dataset is None:
            print("Warning: early_stopping is enabled, but no validation dataset was loaded. Skipping early stopping configuration.")
        else:
            from transformers import EarlyStoppingCallback
            sft_kwargs.update({
                "eval_strategy": "steps",
                "eval_steps": save_steps,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            })
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config.training_hyper_params.early_stopping_patience,
                    early_stopping_threshold=config.training_hyper_params.early_stopping_threshold,
                )
            )

    args = SFTConfig(**sft_kwargs)
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=UnslothVisionDataCollator(
            model,
            processor,
            resize=config.training_misc_options.images_params.resize_policy.resize,
            resize_dimension=config.training_misc_options.images_params.resize_policy.resize_dimension,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        callbacks=callbacks,
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


def build_model_load_kwargs(config: AppConfig) -> dict[str, Any]:
    return {
        "max_seq_length": config.training_hyper_params.max_seq_length,
        "load_in_4bit": config.training_misc_options.qlora_4bit,
        "use_gradient_checkpointing": "unsloth" if config.training_misc_options.gradient_checkpointing else False,
        "attn_implementation": config.training_misc_options.backend_params.attn_implementation,
        "experts_implementation": config.training_misc_options.experts_implementation,
    }


def patch_model_runtime_config(model: Any, config: AppConfig) -> None:
    impl = config.training_misc_options.experts_implementation
    if hasattr(model, "config"):
        model.config._experts_implementation = impl
        if hasattr(model.config, "language_config") and model.config.language_config is not None:
            model.config.language_config._experts_implementation = impl
