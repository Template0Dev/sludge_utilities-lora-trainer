from __future__ import annotations

import re
import subprocess
import sys


class PreflightError(RuntimeError):
    pass


def check_python_version(version: tuple[int, int, int] | None = None) -> None:
    current = version or sys.version_info[:3]
    if current < (3, 11, 0) or current >= (3, 14, 0):
        raise PreflightError(f"Python 3.11-3.13 is required, got {current[0]}.{current[1]}.{current[2]}")


def check_cuda_version(cuda_version: str | None) -> None:
    if cuda_version is None:
        return
    normalized = cuda_version.strip()
    if normalized.startswith("13.2"):
        raise PreflightError("CUDA 13.2 is explicitly unsupported by this project policy")


def detect_cuda_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", result.stdout)
    return match.group(1) if match else None


def run_preflight() -> list[str]:
    messages: list[str] = []
    check_python_version()
    messages.append(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}: OK")
    cuda_version = detect_cuda_version()
    check_cuda_version(cuda_version)
    messages.append(f"CUDA {cuda_version}: OK" if cuda_version else "CUDA: not detected")
    return messages
