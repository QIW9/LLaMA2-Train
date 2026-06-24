from torch import nn
from config import ModelConfig
from attention import Attention
from mlp import MLP
import torch
from unify import RMSNorm

class DecoderLayer(nn.Module):
    """
    模型的"一层思考"

    你也可以把它理解成"一层阅读理解"，每一层做两件事：
      ① 看上下文（attention）：这个词和前后词的关系是什么？
      ② 消化理解（MLP）：看完关系后自己动脑想明白

    N 层 DecoderLayer 摞起来就是整个模型的大脑。
    每层做的事情一样，但每层的参数不同，学到的内容也不同。
    """

    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        # 从配置里抄一份参数过来，方便用
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.hidden_dim = config.hidden_dim
        self.multiple_of = config.multiple_of
        self.dropout = config.dropout
        self.norm_eps = config.norm_eps

        # 第①步：看上下文（注意力机制）
        self.attention = Attention(config)

        # 第②步：消化理解（前馈网络）
        self.feed_forward = MLP(self.dim, self.hidden_dim, self.multiple_of, self.dropout)

        self.layer_id = layer_id  # 第几层（编号而已，不影响计算）
        # 两个"数值收束器"，分别在 attention 前和 MLP 前把数据压一压，防止数值爆炸
        self.attention_norm = RMSNorm(self.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, self.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        """
        一层处理的全过程：

        输入 x → 归一化 → 注意力（看上下文） → 和原始输入相加（残差连接）
              → 归一化 → MLP（消化理解）     → 和上一步结果相加（残差连接）
              → 输出

        为什么每次都要"和输入相加"（残差连接）？
          想象你在修改一份文档：
            - 如果不加残差：你只能重写整篇文档，改错了就全毁了
            - 加残差：你只需要在原文上做小修改，原文的好内容不会丢
          这样模型训练更稳定，层数堆再多也不容易崩。
        """
        # 子步骤1：看上下文
        #   attention_norm(x)：先归一化 → 进注意力 → 结果加到原始输入上
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)

        # 子步骤2：消化理解
        #   ffn_norm(h)：先归一化 → 进 MLP → 结果加到上一步输出上
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
