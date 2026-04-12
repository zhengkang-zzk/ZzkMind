# ZzkMind

一个基于 PyTorch、从零实现并逐步扩展的 Decoder-only 小型语言模型项目。

---

## 项目简介

ZzkMind 是一个面向学习与实践的大语言模型项目，目标不是直接调用现成训练框架，而是从底层开始，逐步实现一个最小可运行的语言模型系统，并在此基础上不断扩展更现代的结构与训练流程。

当前项目已经完成了从 **tokenizer、dataset、embedding、self-attention、FFN、RMSNorm、TransformerBlock** 到 **训练、验证、checkpoint 保存、文本生成** 的完整闭环，并进一步将位置编码从可学习绝对位置编码升级为 **RoPE**，同时保留了 **YaRN scaling** 的兼容接口。

---

## 当前进度

### 已完成
- [x] 字符级 Tokenizer（早期版本）
- [x] 基于本地 Hugging Face tokenizer 的多语种分词接口
- [x] Language Modeling Dataset（next-token prediction）
- [x] Input Embedding
- [x] Multi-Head Self-Attention
- [x] Causal Mask
- [x] FeedForward Network
- [x] RMSNorm
- [x] TransformerBlock（Pre-Norm + Residual）
- [x] ZzkModel 主干组装
- [x] Forward shape test
- [x] Tiny corpus overfit
- [x] Training loop
- [x] Validation loss
- [x] Checkpoint save / load
- [x] Text generation
- [x] RoPE
- [x] YaRN-compatible scaling interface

### 进行中
- [ ] 更系统的实验记录与结果对比
- [ ] 更长文本上的训练与生成测试
- [ ] 更完整的生成策略对比（greedy / top-k / temperature）

### 计划中
- [ ] 更规范的实验配置管理
- [ ] 更完整的日志与可视化
- [ ] 更大规模语料训练
- [ ] SwiGLU / weight tying 等结构升级
- [ ] SFT / 对话式数据训练
- [ ] 更高级的对齐与后训练方法

---

## 项目目标

这个项目的核心目标有两个：

1. **从零理解大语言模型的基本结构与训练流程**
2. **在最小可运行版本上逐步演进到更接近现代 LLM 的实现**

相比直接使用成熟框架，本项目更强调：

- 模型结构本身的理解
- 数据流与张量形状的理解
- 训练与生成闭环的搭建
- 从简单版本逐步扩展到更复杂能力

---

## 当前模型结构

当前版本的模型主干为一个最小 Decoder-only Transformer，主要包含：

- Token Embedding
- Multi-Head Self-Attention
- RoPE 位置编码
- RMSNorm
- FeedForward
- Residual Connection
- Final Norm
- LM Head

整体结构可以概括为：

```text
Input Tokens
-> Token Embedding
-> N x TransformerBlock
   -> RMSNorm
   -> Self-Attention + RoPE
   -> Residual
   -> RMSNorm
   -> FFN
   -> Residual
-> Final RMSNorm
-> LM Head
-> Logits