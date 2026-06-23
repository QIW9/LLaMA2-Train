import torch.nn as nn

import torch

class RMSNorm(nn.Module):
    # rmsnorm 归一化
    # 公式: output = x / sqrt(mean(x^2) + eps) * weight
    # 归一化主要是为了防止多层layer导致数据爆炸

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算RMSNorm的核心部分
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self,x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight.type_as(x)
    