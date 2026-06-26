import torch

def softmax(x: torch.Tensor, dim_index: int):
    value_max = torch.max(x, dim=dim_index, keepdim=True).values
    exp_value = torch.exp(x - value_max)
    sum_exp_value = torch.sum(exp_value, dim=dim_index, keepdim=True)
    return exp_value/sum_exp_value




