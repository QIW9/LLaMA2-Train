from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    """
    Tiny-K 模型配置类，继承自 HuggingFace 的 PretrainedConfig。

    负责定义模型的所有超参数（hyperparameters），包括:
      - 模型结构参数 (维度、层数、头数等)
      - 训练参数 (dropout、norm_eps 等)
      - 推理参数 (max_seq_len、flash_attn 等)

    继承 PretrainedConfig 的好处:
      - 自动支持 save_pretrained() / from_pretrained() 读写 config.json
      - 与 HuggingFace Trainer、Pipeline 等无缝集成

    模型架构参考: LLaMA-style Decoder-only Transformer
    """

    # model_type 用于 HuggingFace AutoModel 注册和识别
    # 可通过 AutoConfig.from_pretrained(..., model_type='Tiny-K') 加载
    model_type = 'Tiny-K'

    def __init__(
            self,
            # ============================================================
            # 模型核心结构参数
            # ============================================================

            # dim: 模型隐藏层维度 (hidden_size / d_model)
            # 决定了整个网络的宽度，所有层的输入输出都保持这个维度
            # 默认 768 对应 BERT-base 规模，越大模型能力越强但也越慢
            dim: int = 768,

            # n_layers: Transformer 层数 (decoder block 的层数)
            # 决定了网络的深度，每层包含 Self-Attention + FFN
            # 12 层是小型模型的常见配置 (如 TinyLlama-1.1B)
            # 越大模型能学到更深层的语义, 但训练/推理成本线性增长
            n_layers: int = 12,

            # n_heads: 多头注意力的 Query 头数
            # 每个头关注输入的不同部分（多视角），各头独立计算 Attention
            # 每头维度 head_dim = dim / n_heads (这里 768/12 = 64)
            # 头数越多 → 每个头的粒度越细 → 信息捕捉更丰富
            n_heads: int = 12,

            # n_kv_heads: 多头注意力的 Key/Value 头数 (用于 GQA)
            # GQA (Grouped Query Attention):
            #   - 当 n_kv_heads < n_heads 时: 多个 Q 头共享一组 KV 头 (省显存)
            #   - 当 n_kv_heads = n_heads 时: 退化为标准多头注意力 (MHA)
            #   - 当 n_kv_heads = 1    时: 退化为多查询注意力 (MQA, 最省显存)
            #
            #   MHA vs GQA vs MQA 对比:
            #   类型        n_kv_heads     KV 缓存     效果
            #   MHA         = n_heads      最大          最好
            #   GQA         2~8            中等          接近 MHA
            #   MQA         1              最小          略差
            #
            #   KV 缓存大小 = 2 × n_layers × n_kv_heads × head_dim × max_seq_len
            #   以本配置为例: 2 × 12 × 12 × 64 × 512 ≈ 9.4 MB (MHA)
            n_kv_heads: int = 12,

            # ============================================================
            # 词表与 FFN 参数
            # ============================================================

            # vocab_size: 词表大小
            # 决定了 Embedding 层和 LM Head 的参数量:
            #   Embedding 参数量 = vocab_size × dim
            #   本配置: 6144 × 768 ≈ 4.7M 参数
            # 词表越大 → 能表示更多 token → 编码效率更高 (但参数量增加)
            vocab_size: int = 6144,

            # hidden_dim: FFN (前馈网络) 的中间层维度
            # FFN 结构: x → Linear(dim, hidden_dim) → SiLU → Linear(hidden_dim, dim)
            # 通常 hidden_dim ≈ 2.7~4 倍的 dim (SwiGLU 结构)
            # 若为 None: 自动计算为满足 multiple_of 约束的最接近值
            #
            #   auto_hidden_dim(dim) = round_up( 8/3 × dim, multiple_of )
            #   本配置: round_up(8/3 × 768, 64) = round_up(2048, 64) = 2048
            #
            # 为什么是 8/3?
            #   SwiGLU 有三个权重矩阵 (W_gate, W_up, W_down), 相比标准 FFN (W1, W2)
            #   参数量多 50%, 所以 dim→8/3×dim 来保持总参数量与标准 FFN 一致
            hidden_dim: int = None,

            # multiple_of: FFN 隐藏层维度的对齐倍数
            # hidden_dim 会被补齐到该值的整数倍，方便 GPU 硬件加速
            # 常见的对齐倍数: 64, 128, 256 (取决于 GPU 的 Tensor Core 要求)
            multiple_of: int = 64,

            # ============================================================
            # 归一化与正则化参数
            # ============================================================

            # norm_eps: RMSNorm 的 epsilon, 防止除零
            # RMSNorm(x) = x / sqrt(mean(x²) + norm_eps) × γ
            # 默认 1e-5, 比 LayerNorm 常用的 1e-12 略大
            norm_eps: float = 1e-5,

            # dropout: 训练时的 Dropout 比例
            # 0.0 表示不使用 dropout (预训练阶段常见的做法)
            # 微调时可能需要设为 0.1 防止过拟合
            dropout: float = 0.0,

            # ============================================================
            # 推理与优化参数
            # ============================================================

            # max_seq_len: 最大序列长度 (即上下文窗口大小)
            # 决定了模型能同时处理的 token 数量上限
            # 注意: RoPE 的频率表在训练时就由 max_seq_len 决定
            # 推理时可扩展（如 NTK-aware、YaRN 等方法）
            max_seq_len: int = 512,

            # flash_attn: 是否使用 FlashAttention 加速
            # FlashAttention 是一种 IO-aware 的 Attention 算法:
            #   - 原理: 利用 GPU SRAM 做分块计算, 减少 HBM 读写次数
            #   - 效果: 2~4 倍速度提升, 内存占用 O(N) 替代 O(N²)
            #   - 数学等价: 结果与标准 Attention 完全一致 (非近似算法)
            # 需要安装: pip install flash-attn --no-build-isolation
            flash_attn: bool = True,

            # **kwargs: 传递给父类 PretrainedConfig 的其他参数
            # 如 pad_token_id、bos_token_id、eos_token_id 等
            **kwargs
    ):
        # 将参数保存为实例属性
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn

        super().__init__(**kwargs)
