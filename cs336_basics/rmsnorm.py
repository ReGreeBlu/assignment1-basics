import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (torch.mean(x**2, dim=-1, keepdim=True)+self.eps) ** 0.5
        rmsnorm = x / rms * self.weight
        return rmsnorm.to(in_dtype)