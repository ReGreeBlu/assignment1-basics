import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        rotate_theta = torch.arange(max_seq_len).unsqueeze(-1) * torch.tensor([1/(theta**((2*k-2)/d_k)) for k in range(1, d_k//2+1)]).unsqueeze(0)
        cos = torch.cos(rotate_theta)
        sin = torch.sin(rotate_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos_val = self.cos[token_positions]
        sin_val = self.sin[token_positions]
        y_even = x_even*cos_val - x_odd*sin_val
        y_odd = x_even*sin_val + x_odd*cos_val
        y = torch.stack((y_even, y_odd), dim=-1).flatten(start_dim=-2, end_dim=-1)
        return y

