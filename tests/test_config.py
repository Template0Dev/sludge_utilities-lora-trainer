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
    assert config.training.batch_size == 1
    assert config.lora.precision == "bf16"


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
