"""Small decoder-only Transformer experiments."""

from .config import ModelConfig, TrainingConfig
from .model import MiniTransformerLM

__all__ = ["MiniTransformerLM", "ModelConfig", "TrainingConfig"]
__version__ = "0.1.0"
