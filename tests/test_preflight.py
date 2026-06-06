import pytest

from lora_trainer.preflight import PreflightError, check_python_version, check_cuda_version


def test_preflight_rejects_python_314() -> None:
    with pytest.raises(PreflightError, match="Python"):
        check_python_version((3, 14, 0))


def test_preflight_rejects_cuda_132() -> None:
    with pytest.raises(PreflightError, match="13.2"):
        check_cuda_version("13.2")
