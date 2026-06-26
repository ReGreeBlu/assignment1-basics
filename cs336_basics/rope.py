import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        cos = torch.zeros(max_seq_len, d_k//2)
        sin = torch.zeros(max_seq_len, d_k//2)
        for i in range(max_seq_len):
            rotate_theta = torch.Tensor([i/(theta**((2*k-2)/d_k)) for k in range(1, d_k//2+1)])
            cos[i] = torch.cos(rotate_theta)
            sin[i] = torch.sin(rotate_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.d_k = d_k
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        y = torch.zeros_like(x)
        for i in range(seq_len):
            for k in range(self.d_k//2):
                p = x[..., i, 2*k:2*k+2]
                cos_val = self.cos[token_positions[..., i], k]
                sin_val = self.sin[token_positions[..., i], k]
                y[..., i, 2*k] = p[..., 0]*cos_val - p[..., 1]*sin_val
                y[..., i, 2*k+1] = p[..., 0]*sin_val + p[..., 1]*cos_val
        return y

