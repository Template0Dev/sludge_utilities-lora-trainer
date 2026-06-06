#!/usr/bin/env bash
set -euo pipefail

mkdir -p input

read -r -p "Dataset download URL: " dataset_url
if [[ -z "${dataset_url}" ]]; then
  echo "Dataset URL is required." >&2
  exit 1
fi

read -r -p "Save name without extension: " save_name
if [[ -z "${save_name}" ]]; then
  echo "Save name is required." >&2
  exit 1
fi

save_name="${save_name%.zip}"
target="input/${save_name}.zip"

if [[ -e "${target}" ]]; then
  read -r -p "${target} exists. Overwrite? [y/N]: " overwrite
  case "${overwrite}" in
    y|Y|yes|YES) ;;
    *) echo "Cancelled."; exit 0 ;;
  esac
fi

curl -L --fail --output "${target}" "${dataset_url}"
echo "Saved dataset archive to ${target}"
