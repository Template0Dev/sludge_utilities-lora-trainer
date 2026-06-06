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

# Verify if the downloaded file is a valid zip archive
PY_CMD=""
if command -v python3 &>/dev/null; then
  PY_CMD="python3"
elif command -v python &>/dev/null; then
  PY_CMD="python"
fi

if [[ -n "${PY_CMD}" ]]; then
  if ! "${PY_CMD}" -c "import zipfile, sys; sys.exit(0 if zipfile.is_zipfile('${target}') else 1)" 2>/dev/null; then
    echo "=========================================================================" >&2
    echo "ERROR: The downloaded file is not a valid zip archive!" >&2
    echo "This usually happens when the URL is a cloud sharing page (e.g. Google Drive," >&2
    echo "Yandex Disk, Dropbox, or Hugging Face) instead of a direct download link." >&2
    echo "=========================================================================" >&2
    echo "Preview of downloaded file content:" >&2
    "${PY_CMD}" -c "import sys; f=open(sys.argv[1], 'rb'); chunk=f.read(500); print(chunk.decode('utf-8', errors='ignore'))" "${target}" >&2
    echo "" >&2
    echo "=========================================================================" >&2
    rm -f "${target}"
    exit 1
  fi
fi

echo "Saved dataset archive to ${target}"
