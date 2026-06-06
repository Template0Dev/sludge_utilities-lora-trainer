from pathlib import Path

import pytest

from lora_trainer.dataset import DatasetSelectionError, dataset_name_from_archive, discover_dataset_archive


def test_single_archive_is_selected_automatically(tmp_path: Path) -> None:
    archive = tmp_path / "lba_analysis.zip"
    archive.write_bytes(b"")

    selected = discover_dataset_archive(input_dir=tmp_path, explicit_archive=None, interactive=False)

    assert selected == archive


def test_multiple_archives_fail_without_interactive_choice(tmp_path: Path) -> None:
    (tmp_path / "a.zip").write_bytes(b"")
    (tmp_path / "b.zip").write_bytes(b"")

    with pytest.raises(DatasetSelectionError, match="multiple"):
        discover_dataset_archive(input_dir=tmp_path, explicit_archive=None, interactive=False)


def test_dataset_name_uses_archive_stem() -> None:
    assert dataset_name_from_archive(Path("input/lba_analysis.zip")) == "lba_analysis"
