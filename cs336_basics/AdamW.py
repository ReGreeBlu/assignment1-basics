from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas,
                    "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta_1 = group["betas"][0]
            beta_2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or 1.
                m = state.get("m", 0) # Get momentum from the state, or 0.
                v = state.get("v", 0) # Get variance from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                lr_adjusted = lr * math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
                p.data -= lr * weight_decay * p.data # Update weight tensor in-place.
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * (grad ** 2)
                p.data -= lr_adjusted * m / torch.sqrt(v + eps) # Update weight tensor in-place.
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1 # Increment iteration number.
        return loss