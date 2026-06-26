import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        W = torch.zeros([out_features, in_features])
        sigma = (2/(in_features+out_features)) ** 0.5
        nn.init.trunc_normal_(W, 0, sigma, -3*sigma, 3*sigma)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       y = torch.matmul(x, self.W.T)
       return y
