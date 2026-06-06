from lora_trainer.training import sft_length_kwargs


class CurrentSFTConfig:
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length


class LegacySFTConfig:
    def __init__(self, max_seq_length: int) -> None:
        self.max_seq_length = max_seq_length


def test_sft_length_kwargs_uses_current_trl_max_length() -> None:
    assert sft_length_kwargs(CurrentSFTConfig, 2048) == {"max_length": 2048}


def test_sft_length_kwargs_supports_legacy_trl_max_seq_length() -> None:
    assert sft_length_kwargs(LegacySFTConfig, 2048) == {"max_seq_length": 2048}
