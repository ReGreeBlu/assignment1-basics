import torch
import torch.nn as nn
from cs336_basics.multihead_self_attention_with_rope import CausalMultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.positionwise_feedforward import SwiGLU
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(self, x, token_positions=None) -> torch.Tensor:
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        result = x + self.attn(self.ln1(x), token_positions)
        result = result + self.ffn(self.ln2(result))
        return result