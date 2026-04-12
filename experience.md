## Exp 001 - CharTokenizer + Learned Position Embedding

### 配置
- tokenizer: CharTokenizer
- position: learned
- seq_len: 32
- hidden_size: 128
- num_hidden_layers: 2
- num_attention_heads: 4
- intermediate_size: 256
- batch_size: 8
- lr: 1e-3
- epochs: 5

### 结果
[summary] tokenizer=xlm-roberta-base-local, position=rope+yarn, seq_len=32, hidden_size=128, layers=4, train_loss=0.0406, val_loss=0.0247, val_ppl=1.0250

### 生成
- prompt: 今天天气
- output: 今天天气很好，我们开始训练一个最小语言模型...

### 观察
- 训练稳定，loss 下降正常
- 生成可读，但有重复现象