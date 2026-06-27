import numpy as np
import torch

def get_batch(dataset, batch_size, context_length, device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = np.zeros((batch_size, context_length))
    targets = np.zeros((batch_size, context_length))
    n = dataset.shape[0]
    m = context_length
    sample_pos = torch.randint(0, n-m, (batch_size,))
    for i in range(batch_size):
        inputs[i] = dataset[sample_pos[i]:sample_pos[i]+m]
        targets[i] = dataset[sample_pos[i]+1:sample_pos[i]+m+1]
    inputs = torch.Tensor(inputs).to(dtype=torch.long, device=device)
    targets = torch.Tensor(targets).to(dtype=torch.long, device=device)
    return (inputs, targets)