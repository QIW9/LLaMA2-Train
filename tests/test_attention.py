import pytest
import torch

from attention import Attention
from config import ModelConfig


@pytest.fixture
def attention() -> Attention:
    config = ModelConfig()
    return Attention(config)


class TestPrecomputeFreqsCis:
    def test_output_shapes(self, attention: Attention) -> None:
        head_dim, seq_len = 48, 50
        cos, sin = attention.precompute_freqs_cis(head_dim, seq_len)
        assert cos.shape == (seq_len, head_dim // 2)
        assert sin.shape == cos.shape

    def test_cos_sin_bounded(self, attention: Attention) -> None:
        cos, sin = attention.precompute_freqs_cis(64, 32)
        assert cos.abs().max() <= 1.0
        assert sin.abs().max() <= 1.0


class TestReshapeForBroadcast:
    def test_broadcast_shape(self, attention: Attention) -> None:
        x = torch.randn(2, 50, 6, 24)
        freqs = torch.randn(50, 24)
        out = attention.reshape_for_broadcast(x, freqs)
        assert out.shape == (1, 50, 1, 24)


class TestApplyRotaryEmb:
    def test_output_shapes_match_input(self, attention: Attention) -> None:
        xq = torch.randn(1, 50, 6, 48)
        xk = torch.randn(1, 50, 6, 48)
        cos, sin = attention.precompute_freqs_cis(48, 50)

        xq_out, xk_out = attention.apply_rotary_emb(xq, xk, cos, sin)

        assert xq_out.shape == xq.shape
        assert xk_out.shape == xk.shape

    def test_preserves_dtype(self, attention: Attention) -> None:
        xq = torch.randn(1, 16, 4, 32, dtype=torch.float16)
        xk = torch.randn(1, 16, 4, 32, dtype=torch.float16)
        cos, sin = attention.precompute_freqs_cis(32, 16)

        xq_out, xk_out = attention.apply_rotary_emb(xq, xk, cos, sin)

        assert xq_out.dtype == torch.float16
        assert xk_out.dtype == torch.float16


class TestRepeatKv:
    def test_no_repeat_when_n_rep_is_one(self, attention: Attention) -> None:
        x = torch.randn(2, 10, 4, 32)
        out = attention.repeat_kv(x, n_rep=1)
        assert torch.equal(out, x)

    def test_repeat_expands_head_dim(self, attention: Attention) -> None:
        x = torch.randn(2, 10, 2, 32)
        n_rep = 4
        out = attention.repeat_kv(x, n_rep=n_rep)
        assert out.shape == (2, 10, 8, 32)

    def test_repeated_heads_are_identical(self, attention: Attention) -> None:
        x = torch.randn(1, 4, 2, 8)
        out = attention.repeat_kv(x, n_rep=3)
        # head0 → 索引 0,1,2；head1 → 索引 3,4,5
        assert torch.equal(out[:, :, 0], out[:, :, 1])
        assert torch.equal(out[:, :, 0], out[:, :, 2])
        assert torch.equal(out[:, :, 3], out[:, :, 4])
        assert torch.equal(out[:, :, 3], out[:, :, 5])
