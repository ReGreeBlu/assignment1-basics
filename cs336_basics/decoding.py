import torch
from cs336_basics.softmax import softmax
from cs336_basics.tokenizer import Tokenizer

# Top-p sampling (nucleus sampling)
def get_nucleus_sampling_token(logits: torch.Tensor, temperature: float, threshold: float) -> torch.Tensor:
    probs = softmax(logits/temperature, dim=-1)
    sorted_probs_decreasing = torch.sort(probs, dim=-1, descending=True).values
    cum_sum_probs = torch.cumsum(sorted_probs_decreasing, dim=-1)
    thresh_index = torch.argmax((cum_sum_probs>threshold).int())
    thresh_prob = sorted_probs_decreasing[thresh_index]
    probs[probs<thresh_prob] = 0
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

def decode(tokenizer: Tokenizer, special_tokens: list[str], 
           model: torch.nn.Module, input_tokens: list[int],
           max_num_tokens: int, context_length: int, device: str,
           temperature, threshold) -> list[int]:
    inputs = torch.tensor(input_tokens, device=device)
    num_tokens = inputs.shape[-1]
    special_token_ids = [tokenizer.token_to_ID[token.encode("utf-8")]
                     for token in special_tokens]
    for seq_len in range(num_tokens, max_num_tokens):
        if seq_len >= context_length:
            logits = model(inputs[seq_len-context_length:seq_len])
            next_token = get_nucleus_sampling_token(logits[context_length-1], temperature=temperature, threshold=threshold)
        else:
            logits = model(inputs)
            next_token = get_nucleus_sampling_token(logits[seq_len-1], temperature=temperature, threshold=threshold)
        inputs = torch.cat([inputs, next_token], dim=-1)
        if next_token.item() in special_token_ids:
            break
    output_tokens = inputs.tolist()
    return output_tokens
