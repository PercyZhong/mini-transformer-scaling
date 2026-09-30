from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mini_transformer.config import ModelConfig
from mini_transformer.data import InfiniteBatchIterator, WindowDataset
from mini_transformer.model import MiniTransformerLM
from mini_transformer.train import load_checkpoint, save_checkpoint


def test_checkpoint_restores_step_parameters_and_optimizer(tmp_path: Path) -> None:
    config = ModelConfig(vocab_size=11, context_length=4, d_model=8, n_layers=1, n_heads=2, d_ff=16)
    model = MiniTransformerLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    batches = InfiniteBatchIterator(WindowDataset(list(range(11)) * 4, 4), 2, 42)
    inputs, targets = next(batches)
    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()
    optimizer.step()
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model, optimizer, scaler, batches, 7, 2.5, 0.1, ["<unk>"])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    restored_step, restored_best = load_checkpoint(
        path, model, optimizer, scaler, batches, torch.device("cpu")
    )
    assert restored_step == 7
    assert restored_best == 2.5
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key])
    assert optimizer.state_dict()["state"]
