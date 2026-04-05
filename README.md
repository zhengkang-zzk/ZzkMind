# 🧠 ZzkMind

一个基于 PyTorch 的极简大语言模型学习项目。本项目参考 MiniMind，旨在从零开始、不依赖庞大的第三方库，手写实现一个最小可运行的 LLM 训练与推理流程，深入理解大模型的底层逻辑。

## 🎯 项目愿景

化繁为简，剥离工业级框架的冗余代码，用最清晰的结构呈现大模型（Decoder-only）的心脏：**数据流转、因果自注意力机制（Causal Self-Attention）以及自回归训练循环。**

## 🚀 进展与路线图 (Roadmap)

### 🟢 阶段一：基础设施与数据流 (已完成)
- [x] **配置管理**: 实现基于 `YAML` + `dataclass` 的强类型配置系统 (`config.py`)。
- [x] **极简分词器**: 实现 `CharTokenizer`，支持动态从训练语料构建无 OOV 的专属字符级词表。
- [x] **因果语言模型数据集**: 实现 `LMDataset`，基于滑动窗口策略自动构建 $X$ (输入) 和 $Y$ (偏移 1 位的标签)，完美对接 PyTorch `DataLoader`。

### 🟡 阶段二：模型核心骨架 (进行中)
- [x] **嵌入层**: 实现 Token Embedding 与绝对位置编码 (Positional Encoding) 的融合。
- [x] **多头注意力机制**: 手写 `CausalSelfAttention`，精准实现 QKV 切分矩阵乘法与基于 `masked_fill` 的下三角因果掩码 (防止穿越偷看未来 Token)。
- [x] **线性输出头**: 实现 `LM Head`，将高维隐藏状态映射回词表概率分布 (Logits)。
- [ ] **前馈神经网络**: 实现 `FeedForward` 模块。
- [ ] **Transformer 组装**: 整合 Attention、FFN 与 RMSNorm/LayerNorm，搭建完整的 `TransformerBlock`。

### ⚪ 阶段三：训练与推理 (待启动)
- [ ] **Trainer 训练器**: 实现基于 `CrossEntropyLoss` 的前向/反向传播与梯度更新。
- [ ] **过拟合测试**: 在极小批量数据上验证 Loss 迅速下降至 0，确保模型架构无 Bug。
- [ ] **文本生成**: 实现基于 `Top-K` 或 `Temperature` 采样的自回归文本生成 (Next-token prediction)。

## 📂 项目结构

\`\`\`text
ZzkMind/
├─ configs/
│  └─ tiny.yaml         # 极简模型架构与训练参数配置
├─ dataset/
│  ├─ text_dataset.py   # 纯文本读取加载
│  ├─ tokenizer.py      # CharTokenizer 字符级分词器
│  └─ loader.py         # LMDataset 与张量切片逻辑
├─ model/
│  └─ ZzkModel.py       # 模型核心组件 (Attention, Embedding, Head 等)
├─ trainer/             # 训练循环控制 (TODO)
├─ scripts/             # 评测与生成脚本 (TODO)
├─ out/                 # 模型权重与日志输出目录
├─ config.py            # 配置解析入口
├─ main.py              # 项目主入口与联调测试脚本
├─ pyproject.toml
└─ README.md
\`\`\`

## 🛠️ 快速测试

目前项目已打通“数据加载 -> 模型前向传播”的测试链路。你可以运行 `main.py` 观察张量在模型内部的形状变换：

```bash
python main.py