"""
MLP（多层感知机）模块 —— 也叫"前馈神经网络"或 FFN（Feed-Forward Network）

通俗理解：
  你已经有了 attention 的输出（每个词都看过了上下文），
  但还不够，还需要一段"思考消化"的过程，MLP 就是做这个的。
  它把 attention 的输出再加工一遍，提取更深层的模式。

  形象比喻：attention 是"大家开会讨论，互相交换信息"，
           MLP 是"每个人回自己工位，独自消化整理刚才的信息"。

本文件实现的是 SwiGLU 风格的 MLP，被 LLaMA 等主流大模型采用。
"""

from torch import nn
import torch
from config import ModelConfig
import torch.nn.functional as F


class MLP(nn.Module):
    """
    SwiGLU 风格的前馈神经网络

    数据流向：
       输入 x，形状为 (batch, seq_len, dim)
          │
          ├──→ w1（线性变换：dim → hidden_dim）──→ SiLU 激活 ──┐
          │                                                      ├──→ 逐元素相乘 ──→ w2（线性变换：hidden_dim → dim）──→ Dropout ──→ 输出
          └──→ w3（线性变换：dim → hidden_dim，充当"门控"）────┘

    关键思路：
      - w1 负责"提取特征"，然后过 SiLU 激活函数（类似给特征加个非线性的开关）
      - w3 负责"控制信息流"，作为一道门（gate），决定哪些特征该通过、哪些该拦住
      - 两者逐元素相乘，相当于"门控筛选"：只有 w1 和 w3 都认为重要的信息才会保留
      - w2 把筛选后的信息压缩回原来的维度 dim，方便和下一层对接
      - Dropout 随机丢弃一部分神经元，防止模型"死记硬背"训练数据（过拟合）
    """

    def __init__(self,
                 dim: int,                          # 输入/输出的特征维度（词的向量长度）
                 hidden_dim: int | None = None,      # 中间层的宽度。None 则自动计算
                 multiple_of: int = 256,             # hidden_dim 会向上取整为它的倍数（给硬件加速用的）
                 dropout: float = 0.0):              # Dropout 比例，0 表示不丢弃
        super().__init__()

        # --- 自动计算中间层宽度（如果用户没指定）---
        if hidden_dim is None:
            # 第一步：先膨胀到 4 倍
            hidden_dim = 4 * dim                        # 例：dim=512 → 2048
            # 第二步：取 2/3，这是一种经验公式，让中间层不至于太大
            hidden_dim = int(2 * hidden_dim / 3)         # 例：2048 * 2/3 ≈ 1365
            # 第三步：向上取整到 multiple_of 的倍数（保证计算效率）
            # 例：1365 → 向上取整到 256 的倍数 → 1536
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # --- 三个线性层（全连接层）---
        # w1: "上投影"，把输入从 dim 放大到 hidden_dim，然后过 SiLU 激活
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # w2: "下投影"，把 hidden_dim 压缩回 dim
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # w3: "门控投影"，和 w1 维度一样，但不加激活函数，直接作为"门"和 w1 的输出相乘
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        # Dropout 层，训练时随机"关掉"一些神经元，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 —— 数据通过 MLP 的完整过程

        参数:
            x: 输入张量，形状 (batch_size, seq_len, dim)
               通俗理解：一批句子，每个句子 seq_len 个词，每个词用一个 dim 维向量表示

        返回:
            输出张量，形状和 x 一样 (batch_size, seq_len, dim)
            内容已经经过"门控 + 非线性变换"，表示更深层的语义信息
        """
        # 拆解版（便于理解每一步在干什么）：
        # gate   = F.silu(self.w1(x))   # w1 放大维度 + SiLU 非线性激活 → 作为"被筛选的内容"
        # up     = self.w3(x)            # w3 放大维度（无激活）           → 作为"门的开关程度"
        # hidden = gate * up             # 逐元素相乘：门控筛选
        # output = self.w2(hidden)       # w2 压缩回原维度
        # return self.dropout(output)    # Dropout 正则化

        # 等价的一行写法：
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))