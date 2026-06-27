import torch
def silu(x: torch.Tensor):
    return x*torch.sigmoid(x)