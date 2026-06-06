from __future__ import annotations

from pathlib import Path

import typer

from lora_trainer.config import ensure_runtime_dirs, load_config
from lora_trainer.dataset import (
    DatasetSelectionError,
    DatasetValidationError,
    dataset_name_from_archive,
    discover_dataset_archive,
    extract_dataset_zip,
    validate_extracted_dataset,
)
from lora_trainer.preflight import PreflightError, run_preflight
from lora_trainer.training import adapter_output_dir, train_adapter


app = typer.Typer(no_args_is_help=True)


def _prepare_dataset(config_path: Path, dataset: Path | None) -> tuple:
    config = load_config(config_path)
    ensure_runtime_dirs(config)
    archive = discover_dataset_archive(
        input_dir=config.dataset.input_dir,
        explicit_archive=dataset or config.dataset.archive,
        interactive=None,
    )
    dataset_name = dataset_name_from_archive(archive)
    root = extract_dataset_zip(archive, config.dataset.extracted_root, dataset_name=dataset_name)
    manifest = validate_extracted_dataset(root, dataset_name=dataset_name)
    return config, archive, root, manifest


@app.command("inspect-config")
def inspect_config(
    config_path: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    dataset: Path | None = typer.Option(None, "--dataset", file_okay=True, dir_okay=False),
) -> None:
    try:
        config = load_config(config_path)
        ensure_runtime_dirs(config)
        archive = discover_dataset_archive(
            input_dir=config.dataset.input_dir,
            explicit_archive=dataset or config.dataset.archive,
            interactive=False,
        )
        name = dataset_name_from_archive(archive)
    except (ValueError, DatasetSelectionError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)

    typer.echo(f"model: {config.model.id}")
    typer.echo(f"dataset_archive: {archive}")
    typer.echo(f"dataset_name: {name}")
    typer.echo(f"extracted_dir: {config.dataset.extracted_root / name}")
    typer.echo(f"adapter_output: {adapter_output_dir(config, name)}")
    typer.echo("export_mode: adapter_only")


@app.command("validate-dataset")
def validate_dataset(
    config_path: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    dataset: Path | None = typer.Option(None, "--dataset", file_okay=True, dir_okay=False),
) -> None:
    try:
        _, archive, root, manifest = _prepare_dataset(config_path, dataset)
    except (ValueError, DatasetSelectionError, DatasetValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    typer.echo(f"dataset_archive: {archive}")
    typer.echo(f"extracted_dir: {root}")
    typer.echo(f"split_counts: {manifest.split_counts}")
    typer.echo(f"image_count: {manifest.image_count}")


@app.command("preflight")
def preflight(
    config_path: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
) -> None:
    try:
        load_config(config_path)
        for message in run_preflight():
            typer.echo(message)
    except (ValueError, PreflightError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)


@app.command("train")
def train(
    config_path: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    dataset: Path | None = typer.Option(None, "--dataset", file_okay=True, dir_okay=False),
) -> None:
    try:
        config, _, root, manifest = _prepare_dataset(config_path, dataset)
        output_dir = train_adapter(config, root, manifest, config_path=config_path)
    except (ValueError, DatasetSelectionError, DatasetValidationError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    typer.echo(f"adapter_output: {output_dir}")
