import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from lora_trainer.dataset import DatasetValidationError, extract_dataset_zip, validate_extracted_dataset


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), "red").save(path)


def _record(image_path: str, assistant_text: str = "```json\n{\"lba_mark\": 3}\n```") -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Выполни ЛБА по фотографии."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


def test_validate_dataset_accepts_messages_with_image_and_fenced_json(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    _write_png(root / "images-augmented/src-10k_augmented_ЛБА [2885-2890]_aug_0.png")
    _write_jsonl(root / "train.jsonl", [_record("images-augmented/src-10k_augmented_ЛБА [2885-2890]_aug_0.png")])
    _write_jsonl(root / "train-augmented.jsonl", [_record("images-augmented/src-10k_augmented_ЛБА [2885-2890]_aug_0.png")])
    _write_jsonl(root / "validation.jsonl", [_record("images-augmented/src-10k_augmented_ЛБА [2885-2890]_aug_0.png")])

    manifest = validate_extracted_dataset(root, dataset_name="lba_analysis")

    assert manifest.dataset_name == "lba_analysis"
    assert manifest.split_counts["train"] == 1
    assert manifest.split_counts["train-augmented"] == 1
    assert manifest.split_counts["validation"] == 1
    assert manifest.image_count == 1


def test_validate_dataset_rejects_missing_image_reference(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    _write_jsonl(root / "train.jsonl", [_record("images/missing.png")])
    _write_jsonl(root / "train-augmented.jsonl", [])
    _write_jsonl(root / "validation.jsonl", [])

    with pytest.raises(DatasetValidationError, match="missing image"):
        validate_extracted_dataset(root, dataset_name="bad")


def test_extract_dataset_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "no")

    with pytest.raises(DatasetValidationError, match="unsafe"):
        extract_dataset_zip(archive, tmp_path / "cache", dataset_name="bad")
