"""
分词器训练脚本 —— 教模型"认识字"

这个文件的作用：
  模型不能直接理解"你好世界"这样的汉字，它只能理解数字。
  分词器的工作就是把文字切成小片段，每个片段给一个编号。

  比如："你好世界" → [234, 567, 890]（三个编号）

  这段代码做的事：
    1. 读取一堆文本数据（jsonl格式，每行一个JSON）
    2. 用 BPE 算法自动学习怎么切分文字
    3. 生成一个"词典"文件，保存到磁盘

通俗理解：
  BPE算法像一个聪明的压缩算法——常见字单独一个编号（比如"的"），
  罕见字用字母或偏旁拼起来（比如"鎏"可能拆成两个编号）。
  这样词表不用太大，但什么字都能表示。
"""

import json
import os
import sys
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer as TokenizerModel,
)
from tokenizers.normalizers import NFKC
from tqdm import tqdm
from typing import Generator


class TokenizerTrainer:
    """
    分词器训练器

    使用方式：
      t = TokenizerTrainer("原始数据.jsonl", "保存目录/")
      t.train_tokenizer(vocab_size=8192)  # 训练一个8192词的词典
    """

    def __init__(self, file_path: str, save_dir: str):
        """
        file_path: 训练数据文件路径（jsonl格式，每行 {"text": "一段文字"}）
        save_dir:  训练好的分词器保存到哪个目录
        """
        self.file_path = file_path
        self.save_dir = save_dir

    def read_texts_from_jsonl(self, show_progress: bool = True) -> Generator[str, None, None]:
        """
        从 jsonl 文件里一行一行读出文字内容。

        文件格式示例：
          {"text": "今天天气真好"}
          {"text": "机器学习很有趣"}
          ...

        用生成器（yield）的好处：不一次性全读进内存，省内存。
        """
        file_size = os.path.getsize(self.file_path)
        line_num = 0

        with open(self.file_path, 'r', encoding='utf-8') as f:
            pbar = tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc="[1/2] 读取语料",
                dynamic_ncols=True,
                disable=not show_progress,
            )
            try:
                for line_num, line in enumerate(f, 1):
                    pbar.update(len(line.encode("utf-8")))
                    try:
                        data = json.loads(line)
                        if 'text' not in data:
                            raise KeyError(f"Missing 'text' field in line {line_num}")
                        yield data['text']
                    except json.JSONDecodeError:
                        tqdm.write(f"Error decoding JSON in line {line_num}")
                        continue
                    except KeyError as e:
                        tqdm.write(str(e))
                        continue
            finally:
                pbar.close()

        if show_progress:
            print(f"语料读取完成，共 {line_num:,} 行", flush=True)

    def create_tokenizer_config(self) -> None:
        """
        生成分词器的配置文件。

        生成两个文件：
          1. tokenizer_config.json —— 主配置，定义特殊符号、对话模板等
          2. special_tokens_map.json —— 特殊符号的对照表

        对话模板（chat_template）的作用：
          当模型用于聊天时，会自动把对话历史格式化成规定格式。
          比如：
            用户说："你好"
            → 会被转成：<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n
        """
        config = {
            "add_bos_token": False,     # 不自动加开头符号
            "add_eos_token": False,     # 不自动加结尾符号
            "add_prefix_space": False,  # 不在文本前加空格
            "bos_token": "<|im_start|>",   # 对话开始符号
            "eos_token": "<|im_end|>",     # 对话结束符号
            "pad_token": "<|im_end|>",     # 填充符号（用结束符号代替）
            "unk_token": "<unk>",          # 未知字符的符号
            "model_max_length": 1000000000000000019884624838656,  # 最大长度（极大值，相当于不限）
            "clean_up_tokenization_spaces": False,
            "tokenizer_class": "PreTrainedTokenizerFast",
            # 对话模板：自动把 [{role: "user", content: "你好"}] 格式化成模型能读的文本
            "chat_template": (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'user' %}"
                "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'assistant' %}"
                "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\n' }}"
                "{% endif %}"
            )
        }

        # 保存主配置文件
        with open(os.path.join(self.save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        # 保存特殊符号映射
        special_tokens_map = {
            "bos_token": "<|im_start|>",
            "eos_token": "<|im_end|>",
            "unk_token": "<unk>",
            "pad_token": "<|im_end|>",
            "additional_special_tokens": ["<s>", "</s>"]
        }
        with open(os.path.join(self.save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

    def train_tokenizer(self, vocab_size: int = 8192) -> None:
        """
        核心：训练分词器。

        过程：
          1. 读取训练数据
          2. 用 BPE 算法统计哪些字/片段出现频率最高
          3. 把高频片段收录进词典（共 vocab_size=8192 个）
          4. 保存词典文件

        vocab_size=8192 的意思是：
          最终词典里有 8192 个"词条"（包括单个字、常见词组、特殊符号等）。
        """
        os.makedirs(self.save_dir, exist_ok=True)

        # 初始化分词器，使用 BPE（字节对编码）算法，未知字符用 <unk> 表示
        tokenizer = TokenizerModel(models.BPE(unk_token="<unk>"))
        tokenizer.normalizer = NFKC()  # 文本规范化（全角转半角、繁体转简体等）
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # 字节级预处理
        tokenizer.decoder = decoders.ByteLevel()  # 解码器：把编号还原成文字

        # 定义特殊符号及其编号
        # <unk>         = 0  （不认识的字都用它）
        # <s>           = 1  （句子开头）
        # </s>          = 2  （句子结尾）
        # <|im_start|>  = 3  （对话轮次开始）
        # <|im_end|>    = 4  （对话轮次结束）
        special_tokens = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"]

        # 配置 BPE 训练器
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,       # 词典大小
            special_tokens=special_tokens,  # 特殊符号（不会参与 BPE 合并）
            min_frequency=2,             # 片段至少出现2次才收录（过滤掉只出现1次的）
            show_progress=True,          # 显示训练进度条
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # 初始化字母表（256个字节）
        )

        # 开始训练：喂数据，自动学习切分规则
        file_size_gb = os.path.getsize(self.file_path) / (1024 ** 3)
        print(
            f"开始训练 tokenizer | 语料: {self.file_path} | "
            f"大小: {file_size_gb:.2f} GB | 词表: {vocab_size}",
            flush=True,
        )
        print("[2/2] BPE 训练（下方进度条为词表合并阶段）", flush=True)
        texts = self.read_texts_from_jsonl(show_progress=True)
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # 验证特殊符号的编号是否正确
        try:
            assert tokenizer.token_to_id("<unk>") == 0
            assert tokenizer.token_to_id("<s>") == 1
            assert tokenizer.token_to_id("</s>") == 2
            assert tokenizer.token_to_id("<|im_start|>") == 3
            assert tokenizer.token_to_id("<|im_end|>") == 4
        except AssertionError as e:
            print("Special tokens mapping error:", e)
            raise

        # 保存分词器到文件
        tokenizer.save(os.path.join(self.save_dir, "tokenizer.json"))

        # 生成配置文件
        self.create_tokenizer_config()
        print(f"Tokenizer saved to {self.save_dir}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    corpus_path = "/Users/qiwang/Downloads/mobvoi_seq_monkey_general_open_corpus.jsonl"
    save_dir = os.path.join(os.path.dirname(__file__), "tokenizer")
    vocab_size = 6144

    trainer = TokenizerTrainer(corpus_path, save_dir)
    trainer.train_tokenizer(vocab_size=vocab_size)
