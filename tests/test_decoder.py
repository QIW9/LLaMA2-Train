import pytest
import torch

from config import ModelConfig
from decoder import DecoderLayer


@pytest.fixture
def decoder_layer() -> DecoderLayer:
    return DecoderLayer(0, ModelConfig())


class TestDecoderLayer:
    def test_output_shape(self, decoder_layer: DecoderLayer) -> None:
        config = ModelConfig()
        batch_size = 1
        seq_len = 50
        dim = config.dim
        head_dim = dim // config.n_heads

        x = torch.randn(batch_size, seq_len, dim)
        freqs_cos, freqs_sin = decoder_layer.attention.precompute_freqs_cis(head_dim, seq_len)

        out = decoder_layer(x, freqs_cos, freqs_sin)

        assert out.shape == (batch_size, seq_len, dim)

    def test_output_dtype_matches_input(self, decoder_layer: DecoderLayer) -> None:
        config = ModelConfig()
        x = torch.randn(2, 16, config.dim, dtype=torch.float32)
        head_dim = config.dim // config.n_heads
        freqs_cos, freqs_sin = decoder_layer.attention.precompute_freqs_cis(head_dim, 16)

        out = decoder_layer(x, freqs_cos, freqs_sin)

        assert out.dtype == x.dtype
