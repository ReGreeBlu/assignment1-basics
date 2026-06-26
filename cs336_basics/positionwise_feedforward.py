import torch
import torch.nn as nn
from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = int((8/3*d_model)//64*64)
        self.d_model = d_model
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.w1(x)
        silu_y1 = y1*torch.sigmoid(y1)
        y3 = self.w3(x)
        return self.w2(silu_y1*y3)

