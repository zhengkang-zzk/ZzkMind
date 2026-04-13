import sys
import os
# 将当前文件的上一级目录（ZzkMind根目录）加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import ModelConfig
from typing import Optional

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
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        tok_emb = self.token_embedding(x)
        return self.dropout(tok_emb)


def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    q, k: (B, H, T, D)
    cos, sin: (T, D)
    """
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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

        freqs_cos, freqs_sin = precompute_freqs(
            dim=self.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_base,
            rope_scaling=config.rope_scaling
        )

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, x, past_key_value=None, use_cache=False):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        past_len = 0 if past_key_value is None else past_key_value[0].size(-2)

        cos = self.freqs_cos[past_len: past_len + seq_len].to(device=x.device, dtype=q.dtype)
        sin = self.freqs_sin[past_len: past_len + seq_len].to(device=x.device, dtype=q.dtype)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)  # (B, H, T_total, D)
            v = torch.cat([past_v, v], dim=-2)

        present_key_value = (k, v) if use_cache else None



        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)

        total_len = k.size(-2)  # T_past + T_new

        q_positions = past_len + torch.arange(seq_len, device=x.device)   # (T_new,)
        k_positions = torch.arange(total_len, device=x.device)            # (T_total,)

        causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        # (T_new, T_total)

        scores = scores.masked_fill(
            ~causal_mask.unsqueeze(0).unsqueeze(0),
            float("-inf")
        )

        prob = torch.softmax(scores, dim=-1)
        prob = self.attn_dropout(prob)
        out = prob @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        out = self.resid_dropout(self.o_proj(out))

        return out, present_key_value


class SwiGLUExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # SwiGLU: 三个线性层
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: (B, T, hidden_size)
        gate = F.silu(self.gate_proj(x))     # (B, T, intermediate_size)
        up = self.up_proj(x)                 # (B, T, intermediate_size)
        x = gate * up                        # 逐元素门控
        x = self.down_proj(x)                # (B, T, hidden_size)
        x = self.dropout(x)
        return x


class MoEFeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.expert_capacity = config.moe_expert_capacity
        self.capacity_factor = config.moe_capacity_factor

        if self.num_experts <= 0:
            raise ValueError("moe_num_experts must be greater than 0")
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError("moe_top_k must be in [1, moe_num_experts]")

        # Router: 给每个 token 打分，决定它应该交给哪些稀疏专家处理。
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(config) for _ in range(self.num_experts)
        ])
        # Shared expert 始终处理所有 token，给 MoE 一条稳定的 dense FFN 通路。
        self.shared_expert = (
            SwiGLUExpert(config) if config.moe_use_shared_expert else None
        )

    def _capacity(self, num_tokens: int) -> int:
        if self.expert_capacity is not None:
            return max(1, self.expert_capacity)
        # 默认容量按平均负载放大一点，减少热门专家过载时被丢弃的 token。
        avg_tokens = num_tokens * self.top_k / self.num_experts
        return max(1, math.ceil(self.capacity_factor * avg_tokens))

    def _load_balance_loss(self, probs, topk_indices):
        if not self.training:
            return probs.new_zeros(())

        # selected 统计“实际被 top-k 选中”的专家，probs 统计 router 的软概率。
        selected = torch.zeros_like(probs)
        selected.scatter_(1, topk_indices, 1.0)

        importance = probs.mean(dim=0)
        load = selected.mean(dim=0)
        # 鼓励“router 概率”和“实际分配 token 数”都更均匀，避免所有 token 挤到少数专家。
        load_loss = self.num_experts * torch.sum(importance * load) / self.top_k
        importance_loss = torch.var(importance) * self.num_experts
        return load_loss + importance_loss

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.reshape(batch_size * seq_len, hidden_size)
        num_tokens = x_flat.size(0)

        logits = self.gate(x_flat)
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        # top-k 内重新归一化，使被选中的专家权重和为 1，输出尺度更稳定。
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        aux_loss = self._load_balance_loss(probs, topk_indices)

        # 展平为 token-expert 对，后面按 expert 分组批量计算。
        flat_expert_indices = topk_indices.reshape(-1)
        flat_expert_weights = topk_weights.reshape(-1)
        token_indices = torch.arange(num_tokens, device=x.device)
        token_indices = token_indices[:, None].expand(-1, self.top_k).reshape(-1)

        capacity = self._capacity(num_tokens)
        routed_output = torch.zeros_like(x_flat)

        for expert_idx, expert in enumerate(self.experts):
            expert_mask = flat_expert_indices == expert_idx
            expert_tokens = token_indices[expert_mask]
            expert_weights = flat_expert_weights[expert_mask]

            if expert_tokens.numel() == 0:
                continue

            if expert_tokens.numel() > capacity:
                # 超容量时保留 router 权重最高的 token，防止单个专家处理过多 token。
                keep = torch.argsort(expert_weights, descending=True)[:capacity]
                expert_tokens = expert_tokens[keep]
                expert_weights = expert_weights[keep]

            expert_output = expert(x_flat[expert_tokens].unsqueeze(1)).squeeze(1)
            expert_output = expert_output * expert_weights.unsqueeze(-1)
            routed_output.index_add_(0, expert_tokens, expert_output)

        if self.shared_expert is not None:
            # 共享专家不参与稀疏路由，始终补充一条通用变换路径。
            routed_output = routed_output + self.shared_expert(x_flat.unsqueeze(1)).squeeze(1)

        return routed_output.view(batch_size, seq_len, hidden_size), aux_loss


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_moe = config.use_moe
        if self.use_moe:
            self.moe = MoEFeedForward(config)
        else:
            self.hidden_size = config.hidden_size
            self.intermediate_size = config.intermediate_size
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
            self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.use_moe:
            return self.moe(x)

        # Dense 路径保持原来的参数命名，use_moe=False 时旧 checkpoint 可以继续加载。
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)
        return x, x.new_zeros(())

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = SelfAttention(config)
        self.ffn = FeedForward(config)
        self.norm2 = RMSNorm(config.hidden_size)

    def forward(self, x, past_key_value=None, use_cache=False):

        attn_out, present_key_value = self.attn(
            self.norm1(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        # 注意力残差
        x = x + attn_out

        # FFN残差
        ffn_out, aux_loss = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x, present_key_value, aux_loss

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


    def forward(self, x, past_key_values=None, use_cache=False, return_aux_loss=False):
        x = self.embedding(x)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        new_past_key_values = []
        aux_losses = []

        for i in range(len(self.layers)):
            x, present, aux_loss = self.layers[i](
                x,
                past_key_value=past_key_values[i],
                use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(present)
            aux_losses.append(aux_loss)

        x = self.norm(x)
        logits = self.lm_head(x)
        # 每层 FFN 都可能返回 MoE 辅助损失；dense FFN 返回 0。
        aux_loss = torch.stack(aux_losses).sum()

        if use_cache and return_aux_loss:
            return logits, new_past_key_values, aux_loss
        if use_cache:
            return logits, new_past_key_values
        if return_aux_loss:
            return logits, aux_loss
        return logits
