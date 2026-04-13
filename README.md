# ZzkMind

一个基于 PyTorch、从零实现并逐步扩展的 decoder-only 小型语言模型项目。

---

## 项目简介

ZzkMind 是一个面向学习和实验的语言模型项目。目标不是直接调用成熟训练框架，而是从 tokenizer、dataset、模型结构、训练循环、checkpoint、评估和文本生成开始，一步步搭出一个可以运行、可以调试、可以继续扩展的小型 LLM 系统。

当前版本已经形成了完整闭环：

- 使用本地 Hugging Face tokenizer 分词
- 支持 `.txt` 和本地 `.jsonl` 预训练数据
- 构建 next-token prediction 数据集
- 训练 decoder-only Transformer
- 支持 RoPE、KV cache、SwiGLU FFN
- 支持可选 MoE FeedForward
- 支持 shared expert、top-k routing 和负载均衡辅助损失
- 支持 checkpoint 保存、加载、中断续训
- 支持自动评测和人工 prompt 评测

---

## 当前进度

### 已完成

- [x] 本地 Hugging Face tokenizer 封装
- [x] Language Modeling Dataset
- [x] Decoder-only Transformer 主干
- [x] Token Embedding
- [x] Multi-Head Self-Attention
- [x] Causal Mask
- [x] RoPE 位置编码
- [x] YaRN-compatible RoPE scaling 配置接口
- [x] RMSNorm
- [x] SwiGLU FeedForward
- [x] TransformerBlock，使用 Pre-Norm + Residual
- [x] KV cache 生成接口
- [x] 可选 MoE FeedForward
- [x] MoE shared expert
- [x] MoE top-k routing
- [x] MoE 负载均衡辅助损失
- [x] 预训练脚本 `train_pretrain.py`
- [x] Validation loss / Validation perplexity
- [x] Checkpoint save / load
- [x] `Ctrl + C` 中断保存 `interrupted.pt`
- [x] 交互式 eval：自动评测 / 人工输入 prompt

---

## 当前配置

当前 [configs/tiny.yaml](configs/tiny.yaml) 是一个 tiny MoE 实验配置：

```yaml
model:
  hidden_size: 128
  intermediate_size: 256
  num_attention_heads: 4
  num_hidden_layers: 4
  max_position_embeddings: 128
  use_moe: true
  moe_num_experts: 4
  moe_top_k: 2
  moe_capacity_factor: 1.25
  moe_use_shared_expert: true

train:
  batch_size: 8
  lr: 0.0003
  num_epochs: 5
  moe_aux_loss_weight: 0.01
  eval_max_batches: 100

data:
  train_path: "dataset/pretrain_t2t_mini.jsonl"
  jsonl_text_field: "text"
  max_records: 500
  max_chars: 500000
  stride: 128
```

注意：

- 本地 JSONL 数据集不上传 GitHub，已通过 `.gitignore` 忽略。
- `stride: 128` 表示样本窗口基本不重叠，训练速度比 `stride: 1` 更高。
- `max_records` 和 `max_chars` 用来先做小规模实验，确认流程跑通后再逐步调大。
- 如果加载旧 dense FFN checkpoint，需要把 `use_moe` 改回 `false`，否则模型结构不匹配。

---

## 模型结构

当前模型结构：

```text
Input Tokens
-> Token Embedding
-> N x TransformerBlock
   -> RMSNorm
   -> Self-Attention + RoPE
   -> KV cache, only during generation
   -> Residual
   -> RMSNorm
   -> FeedForward
      -> Dense SwiGLU, when use_moe=false
      -> MoE + shared expert, when use_moe=true
   -> Residual
-> Final RMSNorm
-> LM Head
-> Logits
```

核心文件：

- [model/ZzkModel.py](model/ZzkModel.py)：模型结构，包括 attention、RoPE、KV cache、SwiGLU 和 MoE。
- [train_pretrain.py](train_pretrain.py)：预训练入口，支持读取 `.txt` 和本地 `.jsonl`。
- [eval.py](eval.py)：评估入口，支持自动评测和人工 prompt 评测。
- [config.py](config.py)：配置 dataclass。
- [configs/tiny.yaml](configs/tiny.yaml)：当前 tiny 实验配置。
- [dataset/loader.py](dataset/loader.py)：next-token dataset，支持 `stride`。
- [dataset/tokenizer.py](dataset/tokenizer.py)：本地 tokenizer 封装。

---

## MoE 设计

MoE FeedForward 由三部分组成：

- Router / Gate：给每个 token 计算专家概率。
- Routed experts：每个 token 只选择 top-k 个专家参与计算。
- Shared expert：所有 token 都会经过的共享专家，提供一条稳定的 dense FFN 通路。

训练时总损失为：

```text
total_loss = lm_loss + moe_aux_loss_weight * aux_loss
```

其中：

- `lm_loss` 是 next-token prediction 的交叉熵损失。
- `aux_loss` 是 MoE 的负载均衡辅助损失。
- `moe_aux_loss_weight` 当前配置为 `0.01`。

当前 `aux_loss` 包含两项：

- `load_loss`：约束 top-k 后实际分配给各专家的 token 数更均衡。
- `importance_loss`：约束 router softmax 概率不要长期偏向少数专家。

如果 `use_moe=false`，模型走普通 dense SwiGLU FFN，`aux_loss` 为 0。

---

## 数据集

当前预训练数据从本地 JSONL 读取：

```yaml
data:
  train_path: "dataset/pretrain_t2t_mini.jsonl"
  jsonl_text_field: "text"
```

JSONL 每一行应该是一个 JSON 对象，例如：

```json
{"text": "这里是一条用于预训练的文本。"}
```

本地大数据集不会上传 GitHub。`.gitignore` 中已经忽略：

```gitignore
dataset/*.jsonl
dataset/**/*.jsonl
```

如果你换了数据集，需要确认：

- 每行是合法 JSON。
- 文本字段名和 `jsonl_text_field` 一致。
- 先用较小的 `max_records` / `max_chars` 测试流程。
- 数据中如果有大量 Markdown 表格、分隔符、重复模板，tiny 模型很容易优先学到这些浅层格式。

---

## 训练

运行预训练：

```powershell
.\.venv\Scripts\python.exe train_pretrain.py
```

训练脚本会：

1. 读取 [configs/tiny.yaml](configs/tiny.yaml)
2. 读取 `.jsonl` 或 `.txt` 数据
3. 使用本地 tokenizer 编码文本
4. 按 `train_val_split` 切分 train / val
5. 使用 `LMDataset` 构造 next-token 样本
6. 训练模型
7. 每个 epoch 保存 `checkpoints/last.pt`
8. 当验证集 loss 更低时保存 `checkpoints/best.pt`

### epoch 是什么

一个 epoch 表示训练脚本把当前构造出来的训练集完整跑一遍。

当前样本构造方式是：

```text
x = tokens[start : start + seq_len]
y = tokens[start + 1 : start + seq_len + 1]
```

也就是输入一段 token，目标是预测这段 token 整体右移一位后的下一个 token。

当前 `seq_len = max_position_embeddings = 128`，所以每条训练样本长度是 128 token。

当前 `batch_size = 8`，所以每个 step 读取：

```text
8 条样本 x 128 token = 1024 个输入 token
```

如果 `stride = 128`，相邻样本基本不重叠；如果 `stride = 1`，会产生大量重叠窗口，数据更多但训练更慢。

### 中断训练

训练过程中可以按：

```text
Ctrl + C
```

脚本会保存：

```text
checkpoints/interrupted.pt
```

### 继续训练

在 [configs/tiny.yaml](configs/tiny.yaml) 中设置：

```yaml
train:
  resume_from: "checkpoints/interrupted.pt"
```

然后重新运行：

```powershell
.\.venv\Scripts\python.exe train_pretrain.py
```

注意：`num_epochs` 表示目标总 epoch 数。如果 checkpoint 已经训练到第 5 个 epoch，而 `num_epochs: 5`，脚本不会继续训练。需要把 `num_epochs` 调大，例如改成 `10`。

---

## 评估与生成

运行：

```powershell
.\.venv\Scripts\python.exe eval.py
```

脚本会加载：

```text
checkpoints/best.pt
```

然后提示选择模式：

```text
请选择评测模式：
1. 自动评测：计算 val loss / ppl，并运行默认 prompts
2. 人工评测：手动输入 prompt；直接回车退出
请输入 1 或 2：
```

模式说明：

- 输入 `1`：自动评测，计算 validation loss / perplexity，并运行默认 prompts。
- 输入 `2`：人工评测，可以手动输入 prompt；如果直接回车，则退出。

生成相关参数目前写在 [eval.py](eval.py) 顶部：

```python
CHECKPOINT_PATH = "checkpoints/best.pt"
MAX_NEW_TOKENS = 40
TEMPERATURE = 0.8
TOP_K = 5
GREEDY = False
```

当前生成阶段使用 KV cache，但 cache 是累积式 cache，不是 sliding window cache，所以 prompt + generated tokens 的总长度不能超过 `max_position_embeddings`。

---

## 关于输出很多 `|`

当前实验中出现过模型连续生成很多 `|` 的情况。这个现象不是单纯“训练不够”，更像是 tiny 模型学到了数据里的局部格式模式。

定位结果：

- tokenizer 中 `▁` 是 SentencePiece 的词边界 token。
- 当前数据里有 Markdown 表格，例如 `姓名 | 职位 | 联系方式`。
- 在训练样本中，`▁` 后面接 `|` 的条件频率比较高。
- 在 `|` 后面又经常接 `▁`。

因此模型容易进入：

```text
▁ -> | -> ▁ -> | -> ▁ -> |
```

这种局部循环。

建议处理顺序：

1. 清洗数据中的 Markdown 表格和大量 `|` 分隔符。
2. 继续扩大训练数据规模，让模型看到更丰富的自然语言。
3. 在生成阶段加入 repetition penalty 或 no-repeat 规则。
4. 必要时在 eval 采样前临时屏蔽 `|` token，作为调试手段。

---

## 下一步

建议按这个顺序推进：

1. 清洗当前 JSONL 中的表格、分隔符、重复模板，减少 `|` 循环。
2. 扩大 `max_records` / `max_chars`，跑更长一点的 MoE tiny 预训练。
3. 在训练日志里增加 MoE 路由统计，例如每个 expert 的 token 占比、平均 router probability、被丢弃 token 数。
4. 给 eval 增加 repetition penalty，改善早期模型的重复输出。
5. 实现 weight tying，让 `lm_head.weight` 和 token embedding 权重共享。
6. 记录 dense FFN 和 MoE FFN 的对比实验，包括 train loss、val loss、ppl、生成样例和训练速度。
7. 尝试更大的 `max_position_embeddings` 和更长 `seq_len`，观察训练稳定性和生成质量。

---

## Git 提交建议

本次改动可以使用：

```powershell
git add README.md eval.py train_pretrain.py train.py config.py configs/tiny.yaml dataset/loader.py dataset/tokenizer.py model/ZzkModel.py .gitignore
git commit -m "update pretraining and eval workflow docs"
git push
```
