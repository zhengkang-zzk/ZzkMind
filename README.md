# ZzkMind

一个基于 PyTorch、从零实现并逐步扩展的 Decoder-only 小型语言模型项目。

---

## 项目简介

ZzkMind 是一个面向学习与实践的大语言模型项目。目标不是直接调用成熟训练框架，而是从底层开始，逐步实现一个最小可运行的语言模型系统，并在这个基础上不断加入更现代的结构。

当前版本已经形成了从 tokenizer、dataset、模型 forward、训练、验证、checkpoint 保存、checkpoint 加载到文本生成抽查的完整闭环。模型侧已经包含 RoPE、KV cache、SwiGLU FFN，以及可选的 MoE FeedForward。

---

## 当前进度

### 已完成

- [x] 基于本地 Hugging Face tokenizer 的分词接口
- [x] Language Modeling Dataset，使用 next-token prediction
- [x] Decoder-only Transformer 主干
- [x] Token Embedding
- [x] Multi-Head Self-Attention
- [x] Causal Mask
- [x] RoPE 位置编码
- [x] YaRN-compatible RoPE scaling 配置接口
- [x] RMSNorm
- [x] SwiGLU FeedForward
- [x] TransformerBlock，使用 Pre-Norm + Residual
- [x] KV cache 推理加速接口
- [x] 可选 MoE FeedForward
- [x] MoE shared expert
- [x] MoE top-k routing
- [x] MoE 负载均衡辅助损失
- [x] Training loop
- [x] Validation loss / Validation perplexity
- [x] Checkpoint save / load
- [x] Eval pipeline，包含验证集评估和固定 prompt 生成抽查

### 当前配置状态

当前 [configs/tiny.yaml](configs/tiny.yaml) 中已经开启 MoE：

```yaml
model:
  use_moe: true
  moe_num_experts: 4
  moe_top_k: 2
  moe_expert_capacity:
  moe_capacity_factor: 1.25
  moe_use_shared_expert: true

train:
  moe_aux_loss_weight: 0.01
```

如果要加载旧的 dense FFN checkpoint，需要把 `use_moe` 改回 `false`。开启 MoE 后 FFN 参数结构会变化，旧 dense checkpoint 不能作为同结构模型直接 strict load。

---

## 当前模型结构

当前模型是一个小型 Decoder-only Transformer：

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

- [model/ZzkModel.py](model/ZzkModel.py)：模型结构，包括 attention、RoPE、KV cache、SwiGLU、MoE。
- [train.py](train.py)：训练入口，计算语言模型交叉熵损失，并在 MoE 开启时加入负载均衡辅助损失。
- [eval.py](eval.py)：评估入口，加载 checkpoint，计算验证集 loss / ppl，并使用 KV cache 生成文本。
- [config.py](config.py)：配置 dataclass。
- [configs/tiny.yaml](configs/tiny.yaml)：当前 tiny 实验配置。

---

## MoE 设计

MoE FeedForward 由三部分组成：

- Router / Gate：给每个 token 计算专家概率。
- Routed experts：每个 token 只选择 top-k 个专家参与计算。
- Shared expert：所有 token 都会经过的共享专家，提供稳定的 dense FFN 通路。

训练时总损失为：

```text
total_loss = lm_loss + moe_aux_loss_weight * aux_loss
```

其中：

- `lm_loss` 是 next-token prediction 的交叉熵损失。
- `aux_loss` 是 MoE 的负载均衡辅助损失。
- `moe_aux_loss_weight` 当前默认为 `0.01`。

当前 `aux_loss` 包含两项：

- `load_loss`：约束 top-k 后实际分配给各专家的 token 数更均衡。
- `importance_loss`：约束 router softmax 概率不要长期偏向少数专家。

如果 `use_moe=false`，模型仍走普通 dense SwiGLU FFN，`aux_loss` 为 0。

---

## 训练

运行训练：

```powershell
.\.venv\Scripts\python.exe train.py
```

训练脚本会：

- 读取 `configs/tiny.yaml`
- 加载本地 tokenizer
- 构造 next-token dataset
- 按比例切分 train / val
- 训练模型
- 每个 epoch 保存 `last.pt`
- 当验证 loss 更低时保存 `best.pt`

---

## 评估与生成

运行评估：

```powershell
.\.venv\Scripts\python.exe eval.py
```

评估脚本会：

- 加载 `checkpoints/best.pt`
- 重建验证集
- 输出 validation loss 和 perplexity
- 使用固定 prompt 做生成抽查
- 生成阶段使用 KV cache

注意：`eval.py` 使用当前配置重建模型。如果 checkpoint 是 dense FFN 训练出来的，而当前配置 `use_moe=true`，需要先关闭 MoE 或重新训练 MoE checkpoint。

---

## 下一步

建议接下来按这个顺序推进：

1. 跑一轮 MoE tiny 训练，确认 `moe_aux_loss` 是否下降，生成结果是否正常。
2. 记录 dense FFN 和 MoE FFN 的对比实验，包括 train loss、val loss、ppl、生成样例和训练速度。
3. 给 MoE 增加路由统计日志，例如每个 expert 的 token 占比、平均 router probability、被丢弃 token 数。
4. 实现 weight tying，让 `lm_head.weight` 和 token embedding 权重共享。
5. 增加更稳定的 checkpoint 兼容策略，例如从 dense checkpoint 迁移 attention、embedding、norm、lm_head 到 MoE 模型。
6. 扩展数据集，测试更长文本和更大的 `max_position_embeddings`。
7. 整理实验记录文件，形成可复现的实验表格。

---

## Git 提交建议

本次改动可以使用：

```powershell
git add README.md model/ZzkModel.py train.py config.py configs/tiny.yaml eval.py
git commit -m "add optional moe feedforward with shared expert"
git push
```
