import sys
import os
# 将当前文件的上一级目录（ZzkMind根目录）加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import math
from config import ModelConfig


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        assert config.hidden_size % config.num_attention_heads == 0
        self.hidden_size = config.hidden_size
        self.num_attention_head = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.num_attention_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_attention_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_attention_head, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        casualmask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        scores = scores.masked_fill(~casualmask, float('-inf'))
        prob = torch.softmax(scores, dim=-1)
        out = prob @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


class ZzkModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 1. 词嵌入层：把 Token ID 变成向量
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # TODO: 2. 这里未来会加入 Transformer Blocks (Attention + FFN)
        # self.blocks = nn.ModuleList([...])
        
        # 3. 输出层：把隐藏状态映射回词表大小，用于计算概率
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, idx):
        # idx 形状: (batch_size, seq_len)
        
        # x 形状: (batch_size, seq_len, hidden_size)
        x = self.token_embedding(idx) 
        
        # TODO: 通过 Transformer Blocks
        
        # logits 形状: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x) 
        return logits