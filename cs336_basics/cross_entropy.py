import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    n = logits.shape[0]
    log_sum_exp_value = torch.logsumexp(logits, dim=-1)
    targets_index = targets.reshape(-1, 1)
    targets_value = torch.gather(logits, -1, targets_index).squeeze(-1)
    loss = torch.mean(log_sum_exp_value - targets_value)
    return loss
