from pathlib import Path

import pytest
import yaml

from lora_trainer.config import load_config


def test_load_config_defaults_dataset_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"id": "Qwen/Qwen3.6-35B-A3B-FP8"},
                "dataset": {},
                "output": {},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.model.id == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert config.dataset.input_dir == Path("input")
    assert config.output.adapter_root == Path("output/adapters")
    assert config.training_hyper_params.batch_size == 1
    assert config.lora.precision == "bf16"
    assert config.training_hyper_params.max_seq_length == 2048
    assert config.training_misc_options.experts_implementation == "eager"
    assert config.training_misc_options.save_settings.save_percentage == 10.0


def test_load_config_rejects_merged_export_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"id": "Qwen/Qwen3.6-35B-A3B-FP8"},
                "export": {"mode": "merged"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="adapter_only"):
        load_config(config_path)


def test_load_config_rejects_max_seq_length_above_conservative_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"id": "Qwen/Qwen3.6-35B-A3B-FP8"},
                "training_hyper_params": {"max_seq_length": 4096},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_seq_length"):
        load_config(config_path)


def test_load_config_unsloth_env_flags(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"id": "Qwen/Qwen3.6-35B-A3B-FP8"},
                "training_misc_options": {
                    "env_flags": {
                        "unsloth": {
                            "set_unsloth_env_flags": True,
                            "flags": {
                                "UNSLOTH_COMPILE_DISABLE": "0",
                                "UNSLOTH_DISABLE_FAST_GENERATION": "0",
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.training_misc_options.env_flags.unsloth.set_unsloth_env_flags is True
    assert config.training_misc_options.env_flags.unsloth.flags["UNSLOTH_COMPILE_DISABLE"] == "0"
    assert config.training_misc_options.env_flags.unsloth.flags["UNSLOTH_DISABLE_FAST_GENERATION"] == "0"


def test_load_config_unsloth_env_flags_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"id": "Qwen/Qwen3.6-35B-A3B-FP8"},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.training_misc_options.env_flags.unsloth.set_unsloth_env_flags is False
    assert config.training_misc_options.env_flags.unsloth.flags["UNSLOTH_COMPILE_DISABLE"] == "1"


def test_load_config_early_stopping_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"id": "Qwen/Qwen3.6-35B-A3B-FP8"},
                "training_hyper_params": {
                    "early_stopping": True,
                    "early_stopping_patience": 5,
                    "early_stopping_threshold": 0.01,
                },
                "training_misc_options": {
                    "evaluation_settings": {
                        "eval_strategy": "epoch",
                        "eval_steps": 100,
                    },
                    "save_settings": {
                        "save_steps": 100,
                        "save_total_limit": 2,
                        "save_percentage": 15.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.training_hyper_params.early_stopping is True
    assert config.training_hyper_params.early_stopping_patience == 5
    assert config.training_hyper_params.early_stopping_threshold == 0.01
    assert config.training_misc_options.evaluation_settings.eval_strategy == "epoch"
    assert config.training_misc_options.evaluation_settings.eval_steps == 100
    assert config.training_misc_options.save_settings.save_steps == 100
    assert config.training_misc_options.save_settings.save_total_limit == 2
    assert config.training_misc_options.save_settings.save_percentage == 15.0


def test_load_config_rejects_early_stopping_with_final_save_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"id": "Qwen/Qwen3.6-35B-A3B-FP8"},
                "training_hyper_params": {
                    "early_stopping": True,
                },
                "training_misc_options": {
                    "save_settings": {
                        "save_policy": "final",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="early_stopping requires save_policy"):
        load_config(config_path)
