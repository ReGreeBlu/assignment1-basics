import torch

def gradient_clipping(params, M, eps=1e-6):
    params = list(params)
    l2_norm = 0
    for p in params:
        if p.grad is None:
            continue
        l2_norm += torch.sum(p.grad.data ** 2)
    l2_norm = torch.sqrt(l2_norm)
    if l2_norm >= M:
        for p in params:
            if p.grad is None:
                continue
            p.grad.data *= M / (l2_norm + eps)
