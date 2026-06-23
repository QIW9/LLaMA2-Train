from torch import nn
import torch

from config import ModelConfig
class Attention(nn.Module):
    """
    Attention 模块: 实现了多头注意力机制的两个关键技术:
      1. GQA (Grouped Query Attention)  —— 通过 repeat_kv 实现，减少 KV 头数以节省显存
      2. RoPE (Rotary Position Embedding) —— 旋转位置编码，将位置信息编码为旋转角度注入 Q/K

    Attention 核心公式（Scaled Dot-Product Attention）:
        Attention(Q, K, V) = softmax( Q·K^T / √d_k ) · V

    其中:
        Q ∈ R^{n_heads × seq_len × d_k}    Query 矩阵（"我要查什么"）
        K ∈ R^{n_heads × seq_len × d_k}    Key   矩阵（"我有什么标签"）
        V ∈ R^{n_heads × seq_len × d_v}    Value 矩阵（"我的实际内容"）
        d_k = head_dim                     缩放因子，防止点积过大导致 softmax 梯度消失
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_kv_heads = config.n_kv_heads
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.max_seq_len = config.max_seq_len
        self.dropout = config.dropout
        self.flash_attn = config.flash_attn
        self.norm_eps = config.norm_eps
        self.hidden_dim = config.hidden_dim
        self.multiple_of = config.multiple_of
    # =========================================================================
    #  一、GQA (Grouped Query Attention) —— 分组查询注意力
    # =========================================================================
    def repeat_kv(self, x: torch.Tensor, n_rep: int):
        """
        将 KV 头复制 n_rep 次，使 KV 头数与 Q 头数对齐。

        背景:
          - 标准多头注意力: n_q_heads = n_kv_heads，每个 Q 头对应一个独立的 KV 头
          - GQA: n_q_heads > n_kv_heads，多个 Q 头共享一组 KV 头
          - 优势: 大幅减少 KV 缓存，推理时更省显存，实验证明性能损失很小

        数学描述:
          给定 x ∈ R^{bs × kv_len × n_kv_heads × head_dim},
          输出 x' ∈ R^{bs × kv_len × (n_kv_heads × n_rep) × head_dim}
          其中 x'[..., i*n_rep:(i+1)*n_rep, :] = x[..., i:i+1, :]  (每组复制 n_rep 次)

        示例 (n_kv_heads=2, n_rep=4):
          输入:  [头A, 头B]           (2 个头)
          输出:  [头A, 头A, 头A, 头A, 头B, 头B, 头B, 头B]  (8 个头)

        形状变换流程:
          [bs, kv_len, n_kv_heads, head_dim]              # 原始 KV
          → [bs, kv_len, n_kv_heads, 1, head_dim]         # 插入复制维度 (unsqueeze)
          → [bs, kv_len, n_kv_heads, n_rep, head_dim]    # 沿复制维度广播 (expand)
          → [bs, kv_len, n_kv_heads*n_rep, head_dim]     # 合并维度 (reshape)

        参数:
          x:      KV 张量, shape = [batch_size, kv_len, n_kv_heads, head_dim]
          n_rep:  每个 KV 头的复制次数, n_rep = n_q_heads / n_kv_heads
        返回:
          扩展后的 KV 张量, shape = [batch_size, kv_len, n_q_heads, head_dim]
        """
        bs, kv_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x  # 标准多头注意力，无需复制

        return (
            # 步骤 1: unsqueeze —— 在第 3 维后插入一个维度（复制槽位）
            # [bs, kv_len, n_kv_heads, head_dim] → [bs, kv_len, n_kv_heads, 1, head_dim]
            x[:, :, :, None, :]
            # 步骤 2: expand —— 沿新维度广播 n_rep 次（不拷贝内存，仅改变视图）
            # [bs, kv_len, n_kv_heads, 1, head_dim] → [bs, kv_len, n_kv_heads, n_rep, head_dim]
            .expand(bs, kv_len, n_kv_heads, n_rep, head_dim)
            # 步骤 3: reshape —— 合并 n_kv_heads 和 n_rep，得到完整的 n_q_heads
            # [bs, kv_len, n_kv_heads, n_rep, head_dim] → [bs, kv_len, n_kv_heads*n_rep, head_dim]
            .reshape(bs, kv_len, n_kv_heads * n_rep, head_dim)
        )

    # =========================================================================
    #  二、RoPE (Rotary Position Embedding) —— 旋转位置编码
    # =========================================================================
    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        """
        预计算 RoPE 所需的 cos 和 sin 查找表。

        RoPE 核心思想:
          将位置信息编码为 2D 平面上的旋转角度，直接注入 Q 和 K。
          当 Q 和 K 做点积时，旋转角度会自动转化为相对位置信息:

            Q_pos_i · K_pos_j = f(内容相似度, 位置差 i-j)

          位置越近 → 角度差越小 → 点积越大 → 注意力权重越高
          位置越远 → 角度差越大 → 点积越小 → 注意力权重越低

        频率计算（每个维度对的旋转速度不同）:
          Θ = { θ_i = theta^{-2i/d} | i = 0, 1, ..., d/2 - 1 }

          其中:
            theta = 10000.0 (默认，来自原始 Transformer 论文)
            d = dim (head_dim)
            i = 维度对的索引 (共 d/2 对)

          θ_0 = 1.0          → 高频，旋转快（捕捉局部模式）
          θ_{d/2-1} = 1/theta → 低频，旋转慢（捕捉长程依赖）

        旋转角度表:
          对每个位置 pos 和每个维度对 i:
            角度 = pos × θ_i

          最终得二维表:
            freqs[pos][i] = pos × θ_i    (shape: [seq_len, dim/2])

        返回:
          freqs_cos: cos 值表, shape = [end, dim//2]
          freqs_sin: sin 值表, shape = [end, dim//2]
        """
        # 步骤 1: 计算各维度对的频率 θ_i
        # torch.arange(0, dim, 2) → [0, 2, 4, ..., dim-2], 共 dim//2 个值
        # 除以 dim 并取前 dim//2 个值 → [0/d, 2/d, 4/d, ..., (dim-2)/d]
        # theta^{...} → 负指数衰减, 使得高频到低频递减
        # 取倒数 → 频率值 θ_i (高频→大值, 低频→小值)
        #
        # 公式: θ_i = 1 / (theta^{2i/d}) = theta^{-2i/d}
        # 结果 shape: [dim//2]
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                       [: (dim // 2)].float() / dim))

        # 步骤 2: 生成位置索引 t = [0, 1, 2, ..., end-1]
        # shape: [end]
        t = torch.arange(end, device=freqs.device)

        # 步骤 3: 计算旋转角度矩阵（外积）
        # outer(t, freqs) → t × freqs^T
        # freqs[pos][i] = pos × θ_i
        # shape: [end, dim//2]
        #
        # 示意图 (dim//2=4, end=5):
        #        θ₀     θ₁     θ₂     θ₃
        # pos0: 0×θ₀   0×θ₁   0×θ₂   0×θ₃
        # pos1: 1×θ₀   1×θ₁   1×θ₂   1×θ₃
        # pos2: 2×θ₀   2×θ₁   2×θ₂   2×θ₃
        # pos3: 3×θ₀   3×θ₁   3×θ₂   3×θ₃
        # pos4: 4×θ₀   4×θ₁   4×θ₂   4×θ₃
        freqs = torch.outer(t, freqs).float()

        # 步骤 4: 计算 cos 和 sin 表
        # 每个位置 pos 和维度对 i 都有唯一的角度值
        freqs_sin = torch.sin(freqs)  # sin(pos × θ_i)
        freqs_cos = torch.cos(freqs)  # cos(pos × θ_i)
        return freqs_cos, freqs_sin

    def reshape_for_broadcast(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """
        将 2D 的 cos/sin 表广播到 4D 张量 x 的形状。

        背景:
          - cos/sin 是二维表: [seq_len, dim//2]
          - QK 是四维张量: [batch, seq_len, n_heads, dim//2]
          - 需要通过广播机制让 cos/sin 可以逐元素与 QK 相乘

        PyTorch 广播规则:
          从右往左对齐维度, 缺失的维度自动补 1:
          原始:            [seq_len, dim//2]
          目标 (view后):   [1,       seq_len, 1,      dim//2]
          QK:              [batch,   seq_len, n_heads, dim//2]
          → 广播后:        [batch,   seq_len, n_heads, dim//2]  ✓

        参数:
          x:         参考张量 (xq_r 或 xk_r), 用于确定目标形状
                     shape = [batch, seq_len, n_heads, dim//2]
          freqs_cis: cos 或 sin 表
                     shape = [seq_len, dim//2]
        返回:
          广播后的 cos/sin, view 为 [1, seq_len, 1, dim//2]
        """
        ndim = x.ndim  # 张量维度数: 4
        assert 0 <= 1 < ndim
        # 验证形状匹配: cos/sin 表的行数=seq_len, 列数=head_dim//2
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        # 构造 view 形状: 只在 seq_len 维度(第1维)和最后一维保持原样, 其余补 1
        # 例如 x.shape = [2, 32, 8, 64] → shape = [1, 32, 1, 64]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor
    ):
        """
        对 Query 和 Key 应用旋转位置编码 (RoPE)。

        核心 —— 2D 平面旋转公式:
          将 head_dim 两两配对，每对 (x_r, x_i) 看作 2D 平面上的向量。
          以角度 θ = pos × freq 旋转该向量:

            [ x_r' ]   [ cos(θ)  -sin(θ) ] [ x_r ]
            [ x_i' ] = [ sin(θ)   cos(θ) ] [ x_i ]

          展开:
            x_r' = x_r × cos(θ) - x_i × sin(θ)    ← 旋转后的实部
            x_i' = x_r × sin(θ) + x_i × cos(θ)    ← 旋转后的虚部

        RoPE 的性质（基于复数乘法）:
          设 Q_pos_m 旋转了 m×θ, K_pos_n 旋转了 n×θ,
          则 Q_pos_m · K_pos_n ∝ cos((m-n)×θ) + 其它项
          → 点积天然包含相对位置 (m-n) 信息!

        步骤概览:
          1. 将 head_dim 拆分成 dim//2 对, 每对看作复数的 (实部, 虚部)
          2. 广播 cos/sin 表到 QK 同形状
          3. 执行 2D 旋转变换
          4. 将实部虚部拼回原始形状

        参数:
          xq:        Query 张量, shape = [batch, seq_len, n_heads, head_dim]
          xk:        Key   张量, shape = [batch, seq_len, n_kv_heads, head_dim]
          freqs_cos: 预计算的 cos 表, shape = [seq_len, head_dim//2]
          freqs_sin: 预计算的 sin 表, shape = [seq_len, head_dim//2]
        返回:
          (xq_out, xk_out): 旋转后的 Query 和 Key, 形状与输入一致
        """
        # --- 步骤 1: 将 head_dim 拆分成 dim//2 对, 每对 (实部, 虚部) ---
        # xq.shape = [bs, seq_len, n_heads, head_dim=128]
        # reshape → [bs, seq_len, n_heads, 64, 2]
        # unbind(-1) → 沿最后一维拆分, 得到实部和虚部
        # xq_r, xq_i shape 均为 [bs, seq_len, n_heads, 64]
        xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
        xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

        # --- 步骤 2: 广播 cos/sin 表到 QK 同形状 ---
        # view 前: [seq_len, 64] → view 后: [1, seq_len, 1, 64]
        # 乘法时自动广播到 [bs, seq_len, n_heads, 64]
        freqs_sin = self.reshape_for_broadcast(xq_r, freqs_sin)
        freqs_cos = self.reshape_for_broadcast(xq_r, freqs_cos)

        # --- 步骤 3: 执行 2D 旋转变换 ---
        # 对每一对 (x_r, x_i) 应用旋转:
        #   x_r' = x_r × cos(θ) - x_i × sin(θ)
        #   x_i' = x_r × sin(θ) + x_i × cos(θ)
        #
        # 解释: 这相当于将向量 (x_r, x_i) 在 2D 平面上逆时针旋转了角度 θ
        # θ = pos × freq, 所以不同位置的词旋转不同角度
        xq_r_out = xq_r * freqs_cos - xq_i * freqs_sin
        xq_i_out = xq_r * freqs_sin + xq_i * freqs_cos
        xk_r_out = xk_r * freqs_cos - xk_i * freqs_sin
        xk_i_out = xk_r * freqs_sin + xk_i * freqs_cos

        # --- 步骤 4: 拼回原始形状 ---
        # stack(..., dim=-1) → [bs, seq_len, n_heads, 64, 2]
        # flatten(3) → [bs, seq_len, n_heads, 128]
        # 实现: 将最后一维交错排列 (r0,i0, r1,i1, r2,i2, ...)
        xq_out = torch.stack([xq_r_out, xq_i_out], dim=-1).flatten(3)
        xk_out = torch.stack([xk_r_out, xk_i_out], dim=-1).flatten(3)

        # --- 步骤 5: 转回原始数据类型并返回 ---
        # .type_as() 确保输出与输入的数据类型一致（如 bfloat16）
        return xq_out.type_as(xq), xk_out.type_as(xk)
