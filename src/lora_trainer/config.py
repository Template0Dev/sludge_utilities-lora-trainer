from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    id: str = "Qwen/Qwen3.6-35B-A3B-FP8"


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


class TrainingConfig(BaseModel):
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    max_seq_length: int = 2048
    max_steps: int = 60
    learning_rate: float = 2e-4
    seed: int = 3407
    qlora_4bit: bool = False
    unsloth_moe_backend: str | None = None

    @field_validator("max_seq_length")
    @classmethod
    def validate_max_seq_length(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_seq_length must be positive")
        if value > 3072:
            raise ValueError("max_seq_length must be <= 3072 for the conservative memory profile")
        return value


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
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @model_validator(mode="after")
    def ensure_adapter_only(self) -> "AppConfig":
        if self.export.mode != "adapter_only":
            raise ValueError("Only adapter_only export is supported")
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
