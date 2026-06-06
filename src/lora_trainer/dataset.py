from __future__ import annotations

import json
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


REQUIRED_JSONL_FILES = ("train.jsonl", "train-augmented.jsonl", "validation.jsonl")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class DatasetSelectionError(RuntimeError):
    pass


class DatasetValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetManifest:
    dataset_name: str
    root: Path
    split_counts: dict[str, int]
    image_count: int
    image_references: list[str] = field(default_factory=list)


def dataset_name_from_archive(archive: Path) -> str:
    return archive.stem


def discover_dataset_archive(
    *,
    input_dir: Path,
    explicit_archive: Path | None,
    interactive: bool | None = None,
) -> Path:
    if explicit_archive is not None:
        archive = explicit_archive
        if not archive.exists():
            raise DatasetSelectionError(f"Dataset archive does not exist: {archive}")
        if archive.suffix.lower() != ".zip":
            raise DatasetSelectionError(f"Dataset archive must be .zip: {archive}")
        return archive

    archives = sorted(input_dir.glob("*.zip"))
    if not archives:
        raise DatasetSelectionError(f"No .zip dataset archives found in {input_dir}")
    if len(archives) == 1:
        return archives[0]

    is_interactive = sys.stdin.isatty() if interactive is None else interactive
    if not is_interactive:
        raise DatasetSelectionError(
            f"Found multiple dataset archives in {input_dir}; provide --dataset or dataset.archive"
        )

    print("Select dataset archive:")
    for index, archive in enumerate(archives, start=1):
        print(f"{index}. {archive.name}")
    while True:
        raw = input("Dataset number: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if 1 <= choice <= len(archives):
            return archives[choice - 1]
        print("Choice out of range.")


def extract_dataset_zip(archive: Path, extracted_root: Path, *, dataset_name: str) -> Path:
    destination = extracted_root / dataset_name
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise DatasetValidationError(f"unsafe zip member path: {member.filename}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)

    _flatten_single_top_level_directory(destination)
    return destination


def _flatten_single_top_level_directory(destination: Path) -> None:
    children = list(destination.iterdir())
    filtered_children = [
        c for c in children
        if c.name not in ("__MACOSX", ".DS_Store") and not c.name.startswith(".")
    ]
    if len(filtered_children) == 1 and filtered_children[0].is_dir():
        sub_dir = filtered_children[0]
        for item in sub_dir.iterdir():
            target_path = destination / item.name
            if target_path.exists():
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
            shutil.move(str(item), str(target_path))
        shutil.rmtree(sub_dir)



def validate_extracted_dataset(root: Path, *, dataset_name: str) -> DatasetManifest:
    if not root.exists():
        raise DatasetValidationError(f"Dataset root does not exist: {root}")

    split_counts: dict[str, int] = {}
    seen_images: set[str] = set()
    references: list[str] = []
    for jsonl_name in REQUIRED_JSONL_FILES:
        path = root / jsonl_name
        if not path.exists():
            raise DatasetValidationError(f"Required file is missing: {jsonl_name}")
        split_name = path.stem
        records = _read_jsonl(path)
        split_counts[split_name] = len(records)
        for line_number, record in records:
            for image_ref in _validate_record(record, source=path.name, line_number=line_number):
                resolved = _resolve_image(root, image_ref, source=path.name, line_number=line_number)
                seen_images.add(str(resolved.relative_to(root).as_posix()))
                references.append(image_ref)

    return DatasetManifest(
        dataset_name=dataset_name,
        root=root,
        split_counts=split_counts,
        image_count=len(seen_images),
        image_references=references,
    )


def load_training_records(root: Path) -> list[dict]:
    records: list[dict] = []
    for filename in ("train.jsonl", "train-augmented.jsonl"):
        for line_number, record in _read_jsonl(root / filename):
            copied = dict(record)
            copied["_metadata"] = {"source_file": filename, "line_number": line_number}
            records.append(copied)
    return records


def load_validation_records(root: Path) -> list[dict]:
    return [record for _, record in _read_jsonl(root / "validation.jsonl")]


def write_manifest(manifest: DatasetManifest, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(
            {
                "dataset_name": manifest.dataset_name,
                "root": str(manifest.root),
                "split_counts": manifest.split_counts,
                "image_count": manifest.image_count,
                "image_references": manifest.image_references,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[tuple[int, dict]]:
    records: list[tuple[int, dict]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"{path.name}:{index}: invalid JSONL: {exc}") from exc
        if not isinstance(value, dict):
            raise DatasetValidationError(f"{path.name}:{index}: each JSONL line must be an object")
        records.append((index, value))
    return records


def _validate_record(record: dict, *, source: str, line_number: int) -> list[str]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise DatasetValidationError(f"{source}:{line_number}: messages must be a non-empty list")

    image_refs: list[str] = []
    has_assistant_text = False
    for message in messages:
        if not isinstance(message, dict):
            raise DatasetValidationError(f"{source}:{line_number}: message must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise DatasetValidationError(f"{source}:{line_number}: invalid role {role!r}")
        content = message.get("content")
        if not isinstance(content, list) or not content:
            raise DatasetValidationError(f"{source}:{line_number}: content must be a non-empty list")
        has_user_payload = False
        for item in content:
            if not isinstance(item, dict):
                raise DatasetValidationError(f"{source}:{line_number}: content item must be an object")
            item_type = item.get("type")
            if item_type == "image":
                image = item.get("image")
                if not isinstance(image, str) or not image:
                    raise DatasetValidationError(f"{source}:{line_number}: image item needs non-empty image path")
                image_refs.append(image)
                has_user_payload = True
            elif item_type == "text":
                text = item.get("text")
                if not isinstance(text, str):
                    raise DatasetValidationError(f"{source}:{line_number}: text item needs text string")
                if role == "assistant" and text.strip():
                    has_assistant_text = True
                if role == "user" and text.strip():
                    has_user_payload = True
            else:
                raise DatasetValidationError(f"{source}:{line_number}: unsupported content type {item_type!r}")
        if role == "user" and not has_user_payload:
            raise DatasetValidationError(f"{source}:{line_number}: user message needs text or image content")

    if not image_refs:
        raise DatasetValidationError(f"{source}:{line_number}: record needs at least one image")
    if not has_assistant_text:
        raise DatasetValidationError(f"{source}:{line_number}: assistant message needs text answer")
    return image_refs


def _resolve_image(root: Path, image_ref: str, *, source: str, line_number: int) -> Path:
    if Path(image_ref).is_absolute():
        raise DatasetValidationError(f"{source}:{line_number}: image path must be relative: {image_ref}")
    resolved = (root / image_ref).resolve()
    root_resolved = root.resolve()
    if root_resolved not in resolved.parents:
        raise DatasetValidationError(f"{source}:{line_number}: image path escapes dataset root: {image_ref}")
    if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
        raise DatasetValidationError(f"{source}:{line_number}: unsupported image extension: {image_ref}")
    if not resolved.exists():
        raise DatasetValidationError(f"{source}:{line_number}: missing image: {image_ref}")
    return resolved
