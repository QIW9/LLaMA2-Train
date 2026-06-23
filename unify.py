from torch.nn.parameter import Parameter


from torch._tensor import Tensor


import torch.nn as nn

import torch
from typing import final, override

@final
class RMSNorm(nn.Module):
    # rmsnorm 归一化
    # 公式: output = x / sqrt(mean(x^2) + eps) * weight
    # 归一化主要是为了防止多层layer导致数据爆炸

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps: float = eps
        self.weight: Parameter = nn.Parameter(data=torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # 计算RMSNorm的核心部分
        return x * torch.rsqrt(input=x.pow(2).mean(-1, keepdim=True) + self.eps)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: Tensor = self._norm(x=x.float()).type_as(other=x)
        return output * self.weight.type_as(other=x)
    