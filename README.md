# ZzkMind

一个基于 PyTorch、从零实现的 Decoder-only 小型大语言模型项目。

## 项目简介

ZzkMind 是一个从零实现的小型大语言模型项目，目标是逐步复现 LLM 的核心组成，包括：

- tokenizer
- dataset / dataloader
- embedding
- self-attention
- FFN
- normalization
- training
- generation

当前项目已经完成了字符级 tokenizer、语言模型数据集构造、Input Embedding、Multi-Head Self-Attention、FeedForward、RMSNorm、TransformerBlock，以及模型前向传播测试。

后续将继续补齐训练闭环、文本生成，以及更高级的扩展能力，如 RoPE、MoE、SFT 和 RLHF。

---

## 当前进度

### 已完成
- [x] 字符级 Tokenizer
- [x] 语言模型数据集构造（next-token prediction）
- [x] Input Embedding（Token Embedding + Position Embedding）
- [x] Multi-Head Self-Attention
- [x] Causal Mask
- [x] FeedForward Network
- [x] RMSNorm
- [x] TransformerBlock（Pre-Norm + Residual）
- [x] ZzkModel 主干组装
- [x] 前向传播 Shape 测试

### 进行中
- [ ] Training Loop
- [ ] Cross Entropy Loss
- [ ] 小数据集过拟合测试
- [ ] 文本生成（Generate）

### 计划中
- [ ] RoPE
- [ ] 更好的 Tokenizer（BPE / SentencePiece）
- [ ] Checkpoint 保存与加载
- [ ] 评测脚本
- [ ] MoE
- [ ] SFT / RLHF

---

## 项目结构

```text
ZzkMind/
├── config/
├── configs/
│   └── tiny.yaml
├── data/
├── dataset/
│   ├── tokenizer.py
│   ├── loader.py
│   └── text_dataset.py
├── model/
│   └── ZzkModel.py
├── debug.py
├── main.py
└── README.md


## 核心模块说明


- **CharTokenizer**: builds a character-level vocabulary from raw text and provides `encode/decode`.
- **LMDataset**: constructs next-token prediction samples using a sliding window.
- **InputEmbedding**: combines token embedding and positional embedding.
- **SelfAttention**: implements causal multi-head self-attention.
- **FeedForward**: position-wise MLP for token-wise nonlinear transformation.
- **RMSNorm**: normalization over hidden dimensions.
- **TransformerBlock**: Pre-Norm residual block combining attention and FFN.
- **ZzkModel**: stacks multiple Transformer blocks and projects to vocabulary logits.

## 一个最小示例
README 里放一小段最小测试代码会很加分。

```md
## Example

```python
import torch
from config import load_config
from model.ZzkModel import ZzkModel

config = load_config("configs/tiny.yaml")
model = ZzkModel(config.model)

idx = torch.randint(0, config.model.vocab_size, (2, 8), dtype=torch.long)
logits = model(idx)

print(logits.shape)