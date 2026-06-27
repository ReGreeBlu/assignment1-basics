def parameters_accounting(vocab_size, context_length, num_layers,
                          d_model, num_heads, d_ff):
#     Model weights: 
#    - `token_embeddings.weight`
#      - Shape is (vocab_size, d_model).
#    - `layers.{num_layers}.attn.q_proj.weight`
#      - Shape is (num_heads * (d_model / num_heads), d_model).
#    - `layers.{num_layers}.attn.k_proj.weight`
#      - Shape is (num_heads * (d_model / num_heads), d_model).
#    - `layers.{num_layers}.attn.v_proj.weight`
#      - Shape is (num_heads * (d_model / num_heads), d_model).
#    - `layers.{num_layers}.attn.output_proj.weight`
#      - Shape is ((d_model / num_heads) * num_heads, d_model).
#    - `layers.{num_layers}.ln1.weight`
#      - Shape is (d_model,).
#    - `layers.{num_layers}.ffn.w1.weight`
#      - Shape is (d_ff, d_model).
#    - `layers.{num_layers}.ffn.w2.weight`
#      - Shape is (d_model, d_ff).
#    - `layers.{num_layers}.ffn.w3.weight`
#      - Shape is (d_ff, d_model).
#    - `layers.{num_layers}.ln2.weight`
#      - Shape is (d_model,).
#    - `ln_final.weight`
#      - Shape is (d_model, ).
#    - `lm_head.weight`
#      - Shape is (vocab_size, d_model).
    num_parameters_layer = 4*num_heads*(d_model//num_heads)*d_model + 2*d_model + 3*d_model*d_ff
    num_parameters = 2*vocab_size*d_model + num_layers*num_parameters_layer + d_model
    return num_parameters

def FLOP_accounting(m, n, p):
    return 2*m*n*p

def model_analysis(model_name, vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    print(model_name)
    d_k = d_v = d_model//num_heads
    each_layer_FLOPs = 0

    num_FLOPs = 3 * FLOP_accounting(context_length, d_model, num_heads*d_k)
    print("Q, K, V projection: %d"%num_FLOPs)
    each_layer_FLOPs += num_FLOPs

    num_FLOPs = num_heads * FLOP_accounting(context_length, d_k, context_length)
    print("Attention scores (all heads): %d"%num_FLOPs)
    each_layer_FLOPs += num_FLOPs

    num_FLOPs = num_heads * FLOP_accounting(context_length, context_length, d_v)
    print("Attention output (all heads): %d"%num_FLOPs)
    each_layer_FLOPs += num_FLOPs

    num_FLOPs = FLOP_accounting(context_length, num_heads*d_v, d_model)
    print("Output projection: %d"%num_FLOPs)
    each_layer_FLOPs += num_FLOPs

    num_FLOPs = 3 * FLOP_accounting(context_length, d_model, d_ff)
    print("FFN: %d"%num_FLOPs)
    each_layer_FLOPs += num_FLOPs

    total_FLOPs = num_layers * each_layer_FLOPs

    num_FLOPs = FLOP_accounting(context_length, d_model, vocab_size)
    print("Output embedding: %d"%num_FLOPs)
    total_FLOPs += num_FLOPs
    print("Summed up: %d"%total_FLOPs)
    print("Proportion of each component:")

    num_FLOPs = 0
    num_FLOPs += num_layers * 3 * FLOP_accounting(context_length, d_model, num_heads*d_k)
    num_FLOPs += num_layers * num_heads * FLOP_accounting(context_length, d_k, context_length)
    num_FLOPs += num_layers * num_heads * FLOP_accounting(context_length, context_length, d_v)
    num_FLOPs += num_layers * FLOP_accounting(context_length, num_heads*d_v, d_model)
    print(f"Causal Multi-Head Self-Attention with RoPE: {num_FLOPs/total_FLOPs:.2%}")

    num_FLOPs = num_layers * 3 * FLOP_accounting(context_length, d_model, d_ff)
    print(f"Position-Wise Feed-Forward: {num_FLOPs/total_FLOPs:.2%}")

    num_FLOPs = FLOP_accounting(context_length, d_model, vocab_size)
    print(f"Linear (Output Embedding): {num_FLOPs/total_FLOPs:.2%}\n")

# -------------------(a)-------------------
# GPT-2 XL
vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 4288
num_parameters = parameters_accounting(vocab_size, context_length,
                                       num_layers, d_model, num_heads, d_ff)
print(f"Total number of parameters: {num_parameters}")
print(f"Required memory: {4*num_parameters} B\n")

# -------------------(b)-------------------
# GPT-2 XL
model_name = "GPT-2 XL"
vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 4288

model_analysis(model_name, vocab_size, context_length,
               num_layers, d_model, num_heads, d_ff)

# -------------------(d)-------------------
# GPT-2 small
model_name = "GPT-2 small"
vocab_size = 50257
context_length = 1024
num_layers = 12
d_model = 768
num_heads = 12
d_ff = 2048

model_analysis(model_name, vocab_size, context_length,
               num_layers, d_model, num_heads, d_ff)
# GPT-2 medium
model_name = "GPT-2 medium"
vocab_size = 50257
context_length = 1024
num_layers = 24
d_model = 1024
num_heads = 16
d_ff = 2752

model_analysis(model_name, vocab_size, context_length,
               num_layers, d_model, num_heads, d_ff)
# GPT-2 large
model_name = "GPT-2 large"
vocab_size = 50257
context_length = 1024
num_layers = 36
d_model = 1280
num_heads = 20
d_ff = 3392

model_analysis(model_name, vocab_size, context_length,
               num_layers, d_model, num_heads, d_ff)
# GPT-2 XL
model_name = "GPT-2 XL"
vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 4288

model_analysis(model_name, vocab_size, context_length,
               num_layers, d_model, num_heads, d_ff)

# -------------------(e)-------------------
# GPT-2 XL (updated)
model_name = "GPT-2 XL (updated)"
vocab_size = 50257
context_length = 16384
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 4288

model_analysis(model_name, vocab_size, context_length,
               num_layers, d_model, num_heads, d_ff)