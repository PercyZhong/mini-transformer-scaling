import pytest

torch = pytest.importorskip("torch")

from mini_transformer.config import ModelConfig
from mini_transformer.model import MiniTransformerLM


def test_forward_shape_and_finite_single_step_loss() -> None:
    model = MiniTransformerLM(
        ModelConfig(vocab_size=23, context_length=12, d_model=24, n_layers=1, n_heads=3, d_ff=48)
    )
    inputs = torch.randint(0, 23, (2, 12))
    targets = torch.randint(0, 23, (2, 12))
    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 12, 23)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()


def test_heads_must_divide_model_dimension() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(vocab_size=10, d_model=10, n_heads=3)
