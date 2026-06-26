import torch
from cs336_basics.softmax import softmax

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    K_trans = torch.transpose(K, dim0=-1, dim1=-2)
    dot_product = torch.matmul(Q, K_trans)/(d_k ** 0.5)
    if mask is not None:
        dot_product = dot_product.masked_fill(~mask, float('-inf'))
    attention = torch.matmul(softmax(dot_product, dim_index=-1), V)
    return attention
