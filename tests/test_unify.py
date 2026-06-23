import torch
from typing import cast

from unify import RMSNorm


class TestRMSNorm:
    def test_output_shape(self) -> None:
        x = torch.randn(1, 10, 10)
        norm: RMSNorm = RMSNorm(10)
        output = cast(torch.Tensor, norm(x))
        assert output.shape == x.shape

    def test_preserves_dtype(self) -> None:
        x = torch.randn(2, 8, 16, dtype=torch.float16)
        norm: RMSNorm = RMSNorm(16)
        output = cast(torch.Tensor, norm(x))
        assert output.dtype == x.dtype

    def test_normalized_rms_is_approximately_one(self) -> None:
        x = torch.randn(1, 4, 8)
        norm: RMSNorm = RMSNorm(8)
        normed = cast(torch.Tensor, norm(x) / norm.weight)
        rms = torch.sqrt(normed.pow(2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)
