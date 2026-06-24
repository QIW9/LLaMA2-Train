from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from config import ModelConfig
from decoder import DecoderLayer
import torch
from typing import Optional
from torch import nn
from unify import RMSNorm
import math
import torch.nn.functional as F

class Transformer(PreTrainedModel):
    config_class = ModelConfig
    last_loss: Optional[torch.Tensor]

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # 保存一份配置，方便后面随时查看（比如词表多大、模型几层）
        self.args = config
        self.vocab_size = config.vocab_size  # 词表里有多少个"字"
        self.n_layers = config.n_layers      # 模型有多少层（层数越多越聪明，但也越慢）

        # ============================================================
        # 搭建模型的"骨架"——就像搭积木，按顺序把零件拼起来
        # ============================================================

        # ① 查表器：给每个"字"一个编号，查出这个字对应的数字表示（一组768个小数,向量化）
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.args.dim)

        # ② 随机丢弃：训练时随机扔掉一些信息，强迫模型不要死记硬背，学会举一反三，防止过拟合
        self.dropout = nn.Dropout(self.args.dropout)

        # ③ 思考层 × N：模型的核心，每一层都在做"理解上下文并加工信息"
        #    就像读文章时反复揣摩每句话的意思，层数越多理解越深
        self.layers = nn.ModuleList([DecoderLayer(layer_id, self.args) for layer_id in range(self.n_layers)])

        # ④ 数值收束：把经过N层处理后的数据"压一压"，让数值保持在合理范围内
        self.norm = RMSNorm(self.args.dim, self.args.norm_eps)

        # ⑤ 预测头：把处理完的信息转成"每个字分别得多少分"
        #    比如词表有6000个字，就输出6000个分数，分数越高代表越可能是下一个字
        self.output = nn.Linear(self.args.dim, self.vocab_size, bias=False)

        # ============================================================
        # 省参数的技巧：输入查表和输出预测共用同一张分数表
        # 好处：省了一半参数，而且模型效果反而更好
        # ============================================================
        self.tok_embeddings.weight = self.output.weight

        # ============================================================
        # 位置编码：让模型知道每个字在句子中的第几个位置
        # "我吃饭" 和 "饭吃我" 字一样但顺序不同，意思完全不同
        # 提前算好位置信息存起来，后面每层直接用
        # ============================================================
        freqs_cos, freqs_sin = self.layers[0].attention.precompute_freqs_cis(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos)  # 存为固定数据（不参与训练）
        self.register_buffer("freqs_sin", freqs_sin)

        # ============================================================
        # 初始化参数：相当于给模型一个"出厂设置"
        # 所有参数从一个正态分布里随机取值，让模型从零开始学习
        # ============================================================
        self.apply(self._init_weights)

        # 对最后两层（w3 和 wo）做特殊处理，给更小的初始值
        # 理由：这两层是残差路径的出口，初始值太大会导致训练不稳定
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

        # ============================================================
        # 杂项
        # ============================================================
        self.last_loss = None                        # 记录最近一次的错误程度
        self.OUT = CausalLMOutputWithPast()          # 预分配输出容器，省内存
        self._no_split_modules = [name for name, _ in self.named_modules()]  # 多GPU时不让切分

    def _init_weights(self, module: nn.Module) -> None:
        """
        赋予随机的初始值（从均值为0、标准差0.02的正态分布里抽取）。
        具备了学习的潜力。
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)  # 权重随机
            if module.bias is not None:
                module.bias.data.zero_(module.bias)          # 偏置归零
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)  # 查表器也随机

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> CausalLMOutputWithPast:
        """
        让模型"读"一段文字，输出它对下一个字的预测。

        两种模式：
        ① 训练模式（传了targets）→ 每个位置都预测下一个字，和正确答案对比算差距
        ② 推理模式（没传targets）→ 只看最后一个字，猜下一个字是什么

        通俗流程：
        输入一段话 → 查表转数字 → 随机丢点信息 → 经过N层思考加工
        → 数值收束 → 输出每个字的分数 → 训练时算差距 / 推理时取最后一个
        """
        # 兼容 HuggingFace 的传参习惯（它可能用 input_ids 和 labels）
        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'labels' in kwargs:
            targets = kwargs['labels']

        _bsz, seqlen = tokens.shape

        # 第1步：把每个字的编号（整数）查表转成一组小数（768个），方便计算机计算
        h = self.tok_embeddings(tokens)
        # 训练时随机丢掉一些信息，防止死记硬背
        h = self.dropout(h)

        # 第2步：取出当前句子长度对应的"位置标签"
        # 告诉模型"这个字在第几个位置"，帮助理解语序
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]
        
        # 第3步：依次经过每一层思考
        # 每一层都会：查看上下文（这个字前后有什么字）→ 对信息做加工变换
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        
        # 第4步：数值收束，防止数字过大或过小
        h = self.norm(h)

        # 第5步：输出预测
        if targets is not None:
            # ===== 训练模式 =====
            # 对句子中每个位置都预测"下一个字该是什么"，输出每个字的得分
            logits = self.output(h)  # 形状: [批次数, 句子长度, 词表大小]
            # 和正确答案对比，算出差了多少（差距越大说明预测越不准）
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # 把所有位置摊平
                targets.view(-1),                   # 正确答案也摊平
                ignore_index=0,                     # 忽略编号为0的占位符
                reduction='none'                    # 保留每个位置的差距，不加总
            )
        else:
            # ===== 推理模式 =====
            # 只看最后一个字，猜下一个字是什么（自回归：每次只往后猜一个字）
            logits = self.output(h[:, [-1], :])  # 形状: [批次数, 1, 词表大小]
            self.last_loss = None  # 推理时不需要算差距
        
        # 打包输出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        让模型"写作文"——给它一个开头，它一个字一个字地往后接。
        
        整个过程就像一个循环：
        ① 把已有文字喂给模型 → ② 模型猜下一个字 → ③ 把猜的字接在后面
        → ④ 回到①，直到字数够了或者生成了结束符
        
        参数：
        - idx: 开头文字（比如"今天天气"）
        - stop_id: 结束符的编号，生成这个就停（类似句号）
        - max_new_tokens: 最多生成多少个字
        - temperature: 控制随机程度
            0 = 每次都选最有把握的字（保守，结果固定）
            大于0 = 允许偶尔选第二第三候选（有创意，每次结果可能不同）
            数值越大越天马行空
        - top_k: 只从得分前k高的字里选，防止选到不靠谱的字
        """
        index = idx.shape[1]  # 记住开头有几个字，最后只返回新写的部分
        for _ in range(max_new_tokens):
            # 如果文字太长，只保留最后一段（模型一次只能看有限长度）
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]

            # 让模型读一遍当前文字，拿到最后一个位置对每个字的"把握程度"
            logits = self(idx_cond).logits
            logits = logits[:, -1, :]  # 只要最后一个字的预测

            if temperature == 0.0:
                # 方式A：直接选得分最高的字（保守策略）
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 方式B：按概率随机抽（有创意的策略）
                logits = logits / temperature  # 温度越高，各个字的概率越接近
                if top_k is not None:
                    # 只保留前k个得分最高的字，其余设为"不可能"
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)       # 把分数转成概率（总和=100%）
                idx_next = torch.multinomial(probs, num_samples=1)  # 按概率抽一个字

            # 抽到了结束符就停
            if idx_next == stop_id:
                break

            # 把新字接到句尾
            idx = torch.cat((idx, idx_next), dim=1)

        # 只返回新写的部分（去掉开头那段）
        return idx[:, index:]