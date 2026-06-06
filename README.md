# LoRA Trainer

CLI project for training separate PEFT LoRA adapters from multimodal JSONL zip datasets with Unsloth.

## TL;DR

Target runtime is a Linux host with a high-VRAM NVIDIA GPU.

```bash
uv venv --python python3.13
uv sync
uv sync --group gpu
./download_dataset.sh
uv run lora-trainer preflight --config configs/train.yaml
uv run lora-trainer validate-dataset --config configs/train.yaml
uv run lora-trainer train --config configs/train.yaml
```

If `input/` contains exactly one `.zip`, it is used automatically. If several archives are present, the CLI asks which one to use. Non-interactive runs should pass `--dataset input/name.zip` or set `dataset.archive` in YAML.

The adapter is saved separately from the base model:

```text
output/adapters/<dataset_name>/
  adapter_model.safetensors
  adapter_config.json
  run_config.yaml
  dataset_manifest.json
```

For `input/lba_analysis.zip`, the adapter directory is `output/adapters/lba_analysis/`.

## Dataset Format

Each dataset archive must be a `.zip` containing:

```text
train.jsonl
train-augmented.jsonl
validation.jsonl
images/ or images_augmented/ or images-augmented/
```

Each JSONL line is a UTF-8 JSON object in multimodal chat format:

```json
{"messages":[{"role":"user","content":[{"type":"image","image":"images-augmented/example.png"},{"type":"text","text":"Выполни ЛБА по фотографии."}]},{"role":"assistant","content":[{"type":"text","text":"```json\n{\"lba_mark\":3,\"lba_color\":\"БЖ\",\"lba_type\":\"МБ\"}\n```"}]}]}
```

Validation checks required JSONL files, JSON syntax, `messages`, roles, content item types, relative image paths, image existence, image extensions, and split counts. Assistant responses are trained as text exactly as provided, including fenced JSON blocks.

## Commands

```bash
uv run lora-trainer inspect-config --config configs/train.yaml
uv run lora-trainer validate-dataset --config configs/train.yaml
uv run lora-trainer preflight --config configs/train.yaml
uv run lora-trainer train --config configs/train.yaml
uv run lora-trainer train --config configs/train.yaml --dataset input/lba_analysis.zip
```

## Training Notes

v1 uses the Unsloth vision path with `FastVisionModel`, `SFTTrainer`, and `UnslothVisionDataCollator`. It does not use text-only `FastLanguageModel`.

Defaults are conservative:

- Qwen/Qwen3.6-35B-A3B-FP8
- bf16 LoRA
- batch size 1
- gradient accumulation enabled
- gradient checkpointing enabled
- max sequence length 2048 by default, validated to stay at or below 3072
- adapter-only export

4-bit QLoRA is available only as explicit opt-in via `training.qlora_4bit: true` and emits a warning because MoE/VL support depends on the selected Unsloth/model path.

The project explicitly rejects Python `>=3.14` and CUDA `13.2` in preflight policy. CUDA 12.4+ is recommended; CUDA 13.0 may work where supported by the installed PyTorch/Unsloth stack.

## Development

Windows development is intended for static/unit tests and dataset/config validation.

```bash
uv sync
uv run pytest
```
