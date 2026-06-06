from pathlib import Path

from typer.testing import CliRunner

from lora_trainer.cli import app


def test_inspect_config_emits_resolved_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    archive = input_dir / "lba_analysis.zip"
    archive.write_bytes(b"")
    config_path.write_text(
        f"""
model:
  id: Qwen/Qwen3.6-35B-A3B-FP8
dataset:
  input_dir: {input_dir.as_posix()}
output:
  adapter_root: {(tmp_path / "output" / "adapters").as_posix()}
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["inspect-config", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "lba_analysis.zip" in result.stdout
    assert "output/adapters/lba_analysis" in result.stdout.replace("\\", "/")
