"""
注意力机制 —— 模型的"眼睛"

通俗理解：
  当你读一句话"我昨天吃了苹果"，读到"苹果"时，
  你的大脑会自动关注前面的"吃了"（谁吃？），而不是"昨天"（什么时候？）。
  
  注意力机制就是让模型自动学会"看一句话时，每个词应该重点关注哪些其他词"。

核心公式（先有个印象）：
  分数 = Q（我要查什么）× K（别人有什么标签）
  分数越大 → 越相关 → 权重越大 → V（实际信息）被提取得越多

这个文件实现了两个关键技术：
  1. GQA：省显存的注意力（多个查询头共享一组键值头）
  2. RoPE：给每个词打上"位置标签"，让模型知道词的顺序
"""

import math
from typing import final
from torch import nn
import torch

from config import ModelConfig


@final
class Attention(nn.Module):
    """
    注意力模块

    数据流（以一句话为例，"今天 天气 真好"）：
      输入 3 个词 → 每个词算出 Q（查询）、K（标签）、V（内容）
      → Q×K 算出词与词的关联度 → 根据关联度加权取 V 的信息
      → 输出每个词"看完上下文后"的新表示
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads: int = config.n_heads         # 有多少个"注意力头"（多头 = 多个视角同时看）
        self.n_kv_heads: int = config.n_kv_heads   # 键值头数（可以比 n_heads 少，省显存）
        self.head_dim: int = config.dim // config.n_heads  # 每个头的维度
        self.max_seq_len: int = config.max_seq_len

        # n_heads 必须是 n_kv_heads 的整数倍（比如 12 个 Q 头共享 3 个 KV 头，每个 KV 头被 4 个 Q 头共享）
        assert config.n_heads % self.n_kv_heads == 0

        model_parallel_size = 1
        self.n_local_heads = self.n_heads // model_parallel_size     # 本地的 Q 头数
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size  # 本地的 KV 头数
        # 每个 KV 头要被几个 Q 头共享（n_rep=4 意思是 1 个 KV 头服务 4 个 Q 头）
        self.n_rep = self.n_heads // self.n_local_heads

        # ======== 四个变换矩阵 ========
        # 它们的作用：把输入的一串数字（dim维），投射到不同的"视角"
        #
        # Q（Query，查询）：  "我是'苹果'这个词，我想知道前面谁跟我有关"
        # K（Key，标签）：    "我是'吃了'这个词，我身上贴着'动作'的标签"
        # V（Value，内容）：  "我是'吃了'这个词，我的实际含义是xxx"
        #
        # Q 和 K 做点积 = 看"查询"和"标签"匹配程度 → 匹配度高的 V 内容被提取
        # 打个比方：图书馆查资料，Q 是你要找的关键词，K 是每本书的标签，V 是书的内容
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)      # 生成 Q
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)   # 生成 K（头更少，省显存）
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)   # 生成 V（头更少，省显存）
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)      # 把所有头的结果拼回去

        # 训练时随机丢弃，防止死记硬背
        self.att_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # 检查 PyTorch 是否内置了 Flash Attention（一种高速注意力算法）
        # 有就用快的，没有就用自己手写的（慢但能用）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
        if not self.flash:
            print("WARNING: Flash Attention not available, using slow attention")
            # 如果没有 Flash Attention，自己手写时需要一个"因果遮罩"
            # 作用：让"今天"不能偷看"天气"（生成式模型只能看过去，不能看未来）
            mask = torch.full((1, 1, self.max_seq_len, self.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)  # 上三角设为 -inf（不可见）
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        """
        前向传播：输入一批词的向量表示，输出它们"看完上下文后"的新表示

        流程：
        输入 → 算出Q、K、V → 加上位置信息 → Q×K算关联度 → 加权取V → 拼回头 → 输出
        """
        bsz, seqlen, _ = x.shape  # 批次数, 句子长度, 向量维度

        # 第1步：用三个矩阵分别生成 Q、K、V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 第2步：调整形状，让每个"头"独立
        # 比如 12 个头，就切成 12 份，每份单独做注意力
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 第3步：给 Q 和 K 加上位置信息（RoPE）
        # 不加的话模型分不清"饭吃我"和"我吃饭"，因为字一样但顺序不同
        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 第4步：GQA —— 把少的 KV 头复制成和 Q 头一样多
        # 比如 Q 有 12 个头，KV 只有 3 个头 → 每个 KV 头复制 4 份
        xk = self.repeat_kv(xk, self.n_rep)
        xv = self.repeat_kv(xv, self.n_rep)

        # 第5步：调整维度顺序，准备做矩阵乘法
        # 从 [batch, seqlen, n_heads, head_dim] → [batch, n_heads, seqlen, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 第6步：核心计算 —— Q × K^T → 注意力分数 → 加权 V
        if self.flash:
            # 方式A：Flash Attention（快，省显存）
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True  # 因果遮罩：每个词只能看它前面的词
            )
        else:
            # 方式B：手写版（慢，但不需要额外依赖）
            # Q×K^T / sqrt(head_dim)：算相似度，除以 sqrt 防止数字太大
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            # 加上因果遮罩（当前词不能看未来词）
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            # softmax：把分数变成概率（总和=1）
            scores = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # 用概率加权取 V 的信息
            output = torch.matmul(scores, xv)

        # 第7步：把多个头拼回去
        # 从 [batch, n_heads, seqlen, head_dim] → [batch, seqlen, n_heads*head_dim]
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 第8步：最终变换 + dropout，得到输出
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

    # ============================================================
    # GQA（分组查询注意力）—— 省显存的技巧
    # ============================================================
    def repeat_kv(self, x: torch.Tensor, n_rep: int):
        """
        把少的 KV 头复制成和 Q 头一样多。

        为什么需要？
          标准的全头注意力：Q 有 12 个头，K/V 也有 12 个头 → 一对一配对
          分组注意力（GQA）：Q 有 12 个头，K/V 只有 3 个头 → 每 4 个 Q 头共享 1 组 K/V

        好处：
          K/V 头少了 → 推理时缓存的 K/V 小了 4 倍 → 显存省了 4 倍
          实验证明：效果几乎不下降

        示例（Q 有 8 个头，KV 只有 2 个头，n_rep=4）：
          K: [头A] [头B]           → 复制后 → [头A][头A][头A][头A][头B][头B][头B][头B]
          V: [头A] [头B]           → 同理

        用 expand 而不是 repeat：
          expand 只是"换个视角看同一块数据"，不真正复制内存，0 开销
        """
        bs, kv_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x  # 不需要复制，直接返回

        return (
            x[:, :, :, None, :]                                   # 插入一个维度
            .expand(bs, kv_len, n_kv_heads, n_rep, head_dim)      # 广播（不拷贝内存）
            .reshape(bs, kv_len, n_kv_heads * n_rep, head_dim)    # 合并维度
        )

    # ============================================================
    # RoPE（旋转位置编码）—— 给每个词打"位置标签"
    # ============================================================
    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        """
        提前算好所有位置的"旋转角度表"。

        通俗理解：
          每个词都有一个"位置标签"，这个标签是一个角度。
          - 位置 0 的词转 0°
          - 位置 1 的词转 30°
          - 位置 2 的词转 60°
          ...

          两个词的点积会自动包含它们的"角度差"：
            位置差越小 → 角度差越小 → 点积越大 → 它们越相关
            位置差越大 → 角度差越大 → 点积越小 → 它们越不相关

          这就让模型天然知道了"距离近的词更相关"。

        原理：
          给每个维度对分配一个不同的"旋转速度"：
            高频维度（转得快）→ 捕捉精细的局部关系
            低频维度（转得慢）→ 捕捉大范围的长距离关系

        公式：
          对于位置 pos，第 i 个维度对旋转角度 = pos × (1 / 10000^(2i/dim))

        返回值：
          freqs_cos: cos(角度),  形状 [序列长度, dim//2]
          freqs_sin: sin(角度),  形状 [序列长度, dim//2]
        """
        # 算每个维度对的基础频率：theta 越大频率越低，转得越慢
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        # 位置编号: 0, 1, 2, ..., end-1
        t = torch.arange(end, device=freqs.device)
        # 每个位置 × 每个频率 = 角度矩阵 [end, dim//2]
        freqs = torch.outer(t, freqs).float()
        # 取 cos 和 sin
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return freqs_cos, freqs_sin

    def reshape_for_broadcast(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """
        把 2D 的角度表 [seqlen, dim//2] 变形为 4D [1, seqlen, 1, dim//2]，
        这样就能和 Q/K 的形状 [batch, seqlen, n_heads, dim//2] 做逐元素乘法（自动广播）。
        """
        ndim = x.ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        """
        给 Q 和 K 施加旋转（加上位置信息）。

        原理：把每个词的头维度两两分组，每组看成一个 2D 向量，
        然后按该位置的角度旋转这个向量。

        旋转公式（2D 旋转矩阵）：
          新实部 = 实部 × cos(θ) - 虚部 × sin(θ)
          新虚部 = 实部 × sin(θ) + 虚部 × cos(θ)

        为什么只转 Q 和 K，不转 V？
          位置信息是通过 Q×K 点积引入的。
          Q 转了角度 pos_m，K 转了角度 pos_n，
          点积结果 ≈ cos(pos_m - pos_n) × 内容 → 自动包含相对位置！

        V 存的是"实际内容"，不需要位置信息（位置已经在 Q×K 里体现了）。
        """
        # 把 head_dim 的最后两维度拆成 (dim//2, 2)，看成实部和虚部
        xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
        xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

        # 广播 cos/sin 表和 Q/K 同形状
        freqs_sin = self.reshape_for_broadcast(xq_r, freqs_sin)
        freqs_cos = self.reshape_for_broadcast(xq_r, freqs_cos)

        # 2D 旋转：把 (实部, 虚部) 向量旋转 θ 角度
        xq_r_out = xq_r * freqs_cos - xq_i * freqs_sin   # 旋转后的实部
        xq_i_out = xq_r * freqs_sin + xq_i * freqs_cos   # 旋转后的虚部
        xk_r_out = xk_r * freqs_cos - xk_i * freqs_sin
        xk_i_out = xk_r * freqs_sin + xk_i * freqs_cos

        # 把实部虚部拼回去
        xq_out = torch.stack([xq_r_out, xq_i_out], dim=-1).flatten(3)
        xk_out = torch.stack([xk_r_out, xk_i_out], dim=-1).flatten(3)

        return xq_out.type_as(xq), xk_out.type_as(xk)
