import pytest

pytest.importorskip("torch")

from mini_transformer.config import ModelConfig
from mini_transformer.model import MiniTransformerLM


def test_parameter_counts_strictly_increase() -> None:
    dimensions = [(64, 2, 2, 256), (128, 4, 4, 512), (256, 6, 8, 1024)]
    counts = [
        MiniTransformerLM(
            ModelConfig(vocab_size=65, d_model=d, n_layers=layers, n_heads=heads, d_ff=ff)
        ).count_parameters()
        for d, layers, heads, ff in dimensions
    ]
    assert counts[0] < counts[1] < counts[2]
