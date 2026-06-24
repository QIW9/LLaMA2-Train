import pytest
import torch

from config import ModelConfig
from transformer import Transformer


@pytest.fixture
def model() -> Transformer:
    return Transformer(ModelConfig())


class TestTransformer:
    def test_num_parameters_positive(self, model: Transformer) -> None:
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0

    def test_inference_logits_shape(self, model: Transformer) -> None:
        config = ModelConfig()
        batch_size = 1
        seq_len = 50

        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        out = model(x)

        assert out.logits.shape == (batch_size, 1, config.vocab_size)
        assert model.last_loss is None

    def test_training_logits_shape_and_loss(self, model: Transformer) -> None:
        config = ModelConfig()
        batch_size = 1
        seq_len = 50

        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        out = model(x, targets)

        assert out.logits.shape == (batch_size, seq_len, config.vocab_size)
        assert model.last_loss is not None
        assert model.last_loss.shape == (batch_size * seq_len,)
