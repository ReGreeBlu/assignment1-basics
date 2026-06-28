import numpy as np
import torch
from cs336_basics.data_loading import get_batch
from cs336_basics.cross_entropy import cross_entropy

def evaluate(model: torch.nn.Module, dataset, batch_size, context_length, device, num_eval=10) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        loss_valid = 0
        for i in range(num_eval):
            in_indices, targets = get_batch(dataset, batch_size, context_length, device)
            logits = model(in_indices)
            loss_valid += cross_entropy(logits, targets)
        loss_valid /= num_eval
    model.train()
    return loss_valid
