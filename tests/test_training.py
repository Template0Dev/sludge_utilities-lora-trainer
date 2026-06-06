import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lora_trainer.config import AppConfig
from lora_trainer.dataset import DatasetManifest
from lora_trainer.training import build_model_load_kwargs, patch_model_runtime_config, sft_length_kwargs, train_adapter


class CurrentSFTConfig:
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length


class LegacySFTConfig:
    def __init__(self, max_seq_length: int) -> None:
        self.max_seq_length = max_seq_length


def test_sft_length_kwargs_uses_current_trl_max_length() -> None:
    assert sft_length_kwargs(CurrentSFTConfig, 2048) == {"max_length": 2048}


def test_sft_length_kwargs_supports_legacy_trl_max_seq_length() -> None:
    assert sft_length_kwargs(LegacySFTConfig, 2048) == {"max_seq_length": 2048}


def test_build_model_load_kwargs_sets_eager_experts_backend_by_default() -> None:
    config = AppConfig()

    kwargs = build_model_load_kwargs(config)

    assert kwargs["max_seq_length"] == 2048
    assert kwargs["use_gradient_checkpointing"] == "unsloth"
    assert kwargs["experts_implementation"] == "eager"


def test_patch_model_runtime_config_sets_nested_experts_backend() -> None:
    nested = SimpleNamespace(_experts_implementation="grouped_mm")
    model = SimpleNamespace(config=SimpleNamespace(language_config=nested))
    config = AppConfig()

    patch_model_runtime_config(model, config)

    assert model.config._experts_implementation == "eager"
    assert nested._experts_implementation == "eager"


@patch("lora_trainer.training._train_with_unsloth")
def test_train_adapter_sets_unsloth_env_flags(mock_train, tmp_path: Path) -> None:
    config = AppConfig(
        model={"id": "Qwen/Qwen3.6-35B-A3B-FP8"},
        dataset={"input_dir": tmp_path / "input", "extracted_root": tmp_path / "cache"},
        output={"adapter_root": tmp_path / "output"},
        training={
            "set_unsloth_env_flags": True,
            "unsloth_env_flags": {
                "UNSLOTH_TEST_VAR_1": "abc",
                "UNSLOTH_TEST_VAR_2": "xyz",
            }
        }
    )
    manifest = DatasetManifest(
        dataset_name="test_ds",
        root=tmp_path / "extracted",
        split_counts={"train": 10},
        image_count=5,
    )
    config_path = tmp_path / "train.yaml"
    config_path.write_text("dummy yaml content", encoding="utf-8")

    # Clear keys from os.environ first to be safe
    for k in ["UNSLOTH_TEST_VAR_1", "UNSLOTH_TEST_VAR_2"]:
        os.environ.pop(k, None)

    train_adapter(config, tmp_path / "extracted", manifest, config_path=config_path)

    assert os.environ.get("UNSLOTH_TEST_VAR_1") == "abc"
    assert os.environ.get("UNSLOTH_TEST_VAR_2") == "xyz"
    assert mock_train.called


@patch("lora_trainer.training._train_with_unsloth")
def test_train_adapter_does_not_set_unsloth_env_flags_if_disabled(mock_train, tmp_path: Path) -> None:
    config = AppConfig(
        model={"id": "Qwen/Qwen3.6-35B-A3B-FP8"},
        dataset={"input_dir": tmp_path / "input", "extracted_root": tmp_path / "cache"},
        output={"adapter_root": tmp_path / "output"},
        training={
            "set_unsloth_env_flags": False,
            "unsloth_env_flags": {
                "UNSLOTH_TEST_VAR_1": "abc",
                "UNSLOTH_TEST_VAR_2": "xyz",
            }
        }
    )
    manifest = DatasetManifest(
        dataset_name="test_ds",
        root=tmp_path / "extracted",
        split_counts={"train": 10},
        image_count=5,
    )
    config_path = tmp_path / "train.yaml"
    config_path.write_text("dummy yaml content", encoding="utf-8")

    for k in ["UNSLOTH_TEST_VAR_1", "UNSLOTH_TEST_VAR_2"]:
        os.environ.pop(k, None)

    train_adapter(config, tmp_path / "extracted", manifest, config_path=config_path)

    assert "UNSLOTH_TEST_VAR_1" not in os.environ
    assert "UNSLOTH_TEST_VAR_2" not in os.environ
    assert mock_train.called
