import torch

from mini_transformer.config import ModelConfig
from mini_transformer.model import MiniTransformerLM


def test_future_token_cannot_change_past_logits() -> None:
    torch.manual_seed(7)
    model = MiniTransformerLM(
        ModelConfig(
            vocab_size=17, context_length=8, d_model=16, n_layers=2,
            n_heads=4, d_ff=32, dropout=0.0,
        )
    ).eval()
    original = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = original.clone()
    changed[0, 4:] = torch.tensor([9, 10])
    original_logits, _ = model(original)
    changed_logits, _ = model(changed)
    torch.testing.assert_close(original_logits[:, :4], changed_logits[:, :4], rtol=0, atol=1e-6)
