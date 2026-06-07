from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    id: str = "Qwen/Qwen3.6-35B-A3B"


class DatasetConfig(BaseModel):
    input_dir: Path = Path("input")
    archive: Path | None = None
    extracted_root: Path = Path(".cache/datasets")


class OutputConfig(BaseModel):
    adapter_root: Path = Path("output/adapters")
    overwrite: bool = False


class LoraConfig(BaseModel):
    precision: Literal["bf16"] = "bf16"
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "gate_up_proj",
        ]
    )


class TrainingHyperParams(BaseModel):
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_steps: int = 60
    max_seq_length: int = 2048
    seed: int = 3407
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    @field_validator("max_seq_length")
    @classmethod
    def validate_max_seq_length(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_seq_length must be positive")
        if value > 3072:
            raise ValueError("max_seq_length must be <= 3072 for the conservative memory profile")
        return value


class SaveSettings(BaseModel):
    save_policy: Literal["final", "steps", "percent"] = "percent"
    save_percentage: float = 10.0
    save_steps: int = 50
    save_total_limit: int = 1


class EvaluationSettings(BaseModel):
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    eval_steps: int = 50


class UnslothEnvFlags(BaseModel):
    set_unsloth_env_flags: bool = False
    flags: dict[str, str] = Field(
        default_factory=lambda: {
            "UNSLOTH_RETURN_LOGITS": "1",
            "UNSLOTH_COMPILE_DISABLE": "1",
            "UNSLOTH_DISABLE_FAST_GENERATION": "1",
            "UNSLOTH_ENABLE_LOGGING": "1",
            "UNSLOTH_FORCE_FLOAT32": "1",
            "UNSLOTH_STUDIO_DISABLED": "1",
            "UNSLOTH_COMPILE_DEBUG": "1",
            "UNSLOTH_COMPILE_MAXIMUM": "0",
            "UNSLOTH_COMPILE_IGNORE_ERRORS": "1",
            "UNSLOTH_FULLGRAPH": "0",
            "UNSLOTH_DISABLE_AUTO_UPDATES": "1",
        }
    )


class EnvFlagsConfig(BaseModel):
    unsloth: UnslothEnvFlags = Field(default_factory=UnslothEnvFlags)


class BackendParams(BaseModel):
    unsloth_moe_backend: str | None = None
    attn_implementation: str | None = None


class ResizePolicy(BaseModel):
    resize: int | str | None = "min"
    resize_dimension: str | None = "max"


class ImagesParams(BaseModel):
    resize_policy: ResizePolicy = Field(default_factory=ResizePolicy)


class TrainingMiscOptions(BaseModel):
    gradient_checkpointing: bool = True
    qlora_4bit: bool = False
    experts_implementation: str = "eager"
    save_settings: SaveSettings = Field(default_factory=SaveSettings)
    evaluation_settings: EvaluationSettings = Field(default_factory=EvaluationSettings)
    env_flags: EnvFlagsConfig = Field(default_factory=EnvFlagsConfig)
    backend_params: BackendParams = Field(default_factory=BackendParams)
    images_params: ImagesParams = Field(default_factory=ImagesParams)


class ExportConfig(BaseModel):
    mode: Literal["adapter_only"] = "adapter_only"

    @field_validator("mode", mode="before")
    @classmethod
    def only_adapter_mode(cls, value: object) -> object:
        if value != "adapter_only":
            raise ValueError("v1 supports only adapter_only export; merged/GGUF export is out of scope")
        return value


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    training_hyper_params: TrainingHyperParams = Field(default_factory=TrainingHyperParams)
    training_misc_options: TrainingMiscOptions = Field(default_factory=TrainingMiscOptions)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @model_validator(mode="after")
    def validate_config(self) -> "AppConfig":
        if self.export.mode != "adapter_only":
            raise ValueError("Only adapter_only export is supported")
        if self.training_hyper_params.early_stopping and self.training_misc_options.save_settings.save_policy == "final":
            raise ValueError("early_stopping requires save_policy to be 'steps' or 'percent' (cannot be 'final' because load_best_model_at_end is required).")
        return self


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise ValueError(f"Config file does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid config {path}: {exc}") from exc


def ensure_runtime_dirs(config: AppConfig) -> None:
    config.dataset.input_dir.mkdir(parents=True, exist_ok=True)
    config.dataset.extracted_root.mkdir(parents=True, exist_ok=True)
    config.output.adapter_root.mkdir(parents=True, exist_ok=True)
