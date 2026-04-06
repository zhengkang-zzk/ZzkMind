import sys
import os
# 将当前文件的上一级目录（ZzkMind根目录）加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import math
from config import ModelConfig

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x) -> torch.Tensor:
        return self.weight * (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)).type_as(x)

class InputEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len = x.shape

        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError("seq_len exceeds max_position_embeddings")

        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb
        return self.dropout(x)


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        assert config.hidden_size % config.num_attention_heads == 0
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        scores = scores.masked_fill(~causal_mask, float('-inf'))
        prob = torch.softmax(scores, dim=-1)
        prob = self.attn_dropout(prob)
        out = prob @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.resid_dropout(self.o_proj(out))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        x = self.fc1(x)      # (B, T, intermediate_size)
        x = self.act(x)
        x = self.fc2(x)      # (B, T, hidden_size)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = SelfAttention(config)
        self.ffn = FeedForward(config)
        self.norm2 = RMSNorm(config.hidden_size)

    def forward(self, x):
        # 注意力残差
        x = x + self.attn(self.norm1(x))

        # FFN残差
        x = x + self.ffn(self.norm2(x))

        return x

class ZzkModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embedding = InputEmbedding(config)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits