import torch
import torch.nn as nn
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = Q.reshape(*x.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        K = K.reshape(*x.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        V = V.reshape(*x.shape[:-1], self.num_heads, self.d_v).transpose(-3, -2)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        mask = torch.tril(mask)
        multihead = scaled_dot_product_attention(Q, K, V, mask)
        multihead = multihead.transpose(-3, -2).reshape(x.shape)
        attention = self.o_proj(multihead)
        return attention
