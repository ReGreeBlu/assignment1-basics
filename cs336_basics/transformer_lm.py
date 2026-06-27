import torch
import torch.nn as nn
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length)
            for i in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)  
    def forward(self, in_indices, token_positions=None) -> torch.Tensor: 
        '''
        in_indices (Int[Tensor, "batch_size sequence_length"]):
            Tensor with input indices to run the language model on.
            Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.
        '''
        seq_len = in_indices.shape[-1]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=in_indices.device)
        in_features = self.token_embeddings(in_indices)
        for layer in self.layers:
            in_features = layer(in_features, token_positions)
        result = self.lm_head(self.ln_final(in_features))
        return result