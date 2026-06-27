import torch

def cross_entropy(o, x):
    m = x.shape[-1]
    loss = 0
    value_max = torch.max(o, dim=-1, keepdim=True).values
    o_subtract_max = o - value_max
    exp_value = torch.exp(o_subtract_max)
    sum_exp_value = torch.sum(exp_value, dim=-1)
    for i in range(m):
        loss += torch.log(sum_exp_value[i]) - o_subtract_max[i][x[i]]
    return loss/m
