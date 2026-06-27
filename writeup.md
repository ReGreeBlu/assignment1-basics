### Problem (train_bpe_tinystories): BPE Training on TinyStories

(a) 

time: 25 minutes

memory: 10 GB

longest token: < accomplishment>

```bash
     1495.36 real      1487.05 user         6.77 sys
         10758242304  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             1776898  page reclaims
                 963  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   1  messages sent
                   1  messages received
                   1  signals received
                 246  voluntary context switches
               37508  involuntary context switches
           181603710  instructions retired
            96825550  cycles elapsed
            12599656  peak memory footprint
```

(b) 

cProfile result on TinyStoriesV2-GPT4-valid.txt, vocabulary size 10000:

```bash
         150824515 function calls (150824406 primitive calls) in 19.819 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
9745/9744    8.533    0.001   15.379    0.002 {built-in method builtins.max}
148599155    6.846    0.000    6.846    0.000 train_bpe.py:24(<lambda>)
    27631    2.724    0.000    2.726    0.000 __init__.py:921(_keep_positive)
    27631    0.753    0.000    0.753    0.000 {method 'findall' of '_regex.Pattern' objects}
    27631    0.235    0.000    2.964    0.000 __init__.py:928(__iadd__)
     9743    0.200    0.000   15.642    0.002 train_bpe.py:10(token_merge)
    27631    0.171    0.000    0.171    0.000 {built-in method _collections._count_elements}
        1    0.074    0.074   19.816   19.816 train_bpe.py:66(train_bpe)
    27632    0.032    0.000    0.130    0.000 _main.py:459(_compile)
    55423    0.028    0.000    0.081    0.000 enum.py:1609(__and__)
```

Profiling shows the bottleneck shifted to the argmax over pair_counts.

Scanning all pairs on every merge step costs ~78% of total runtime.

A heap with lazy deletion would further optimize the algorithm.

### Problem (tokenizer_experiments): Experiments with tokenizers

(a)

Compression ratio on TinyStories (valid): 4.010

Compression ratio on OWT (valid): 4.558

(b)

Compression ratio on OWT (valid) with TinyStories tokenizer: 3.318

Tokenizing OWT with the TinyStories tokenizer yields lower compression ratio than that with the OWT tokenizer, because the TinyStories tokenizer was trained on simple children's stories and lacks tokens for the longer, varied words common in OWT.

(c)

Total bytes: 37,242.000 B

Throughput of the OWT tokenizer: 1,671,109.556 B/sec

Estimated time of tokenizing the Pile dataset: 147 hours

(d)

Since the maximum vocabulary size is 32,000, all token IDs fit within `uint16` (range 0–65,535).

`uint16` can save half the storage of `int32` with no loss of information.

### Problem (transformer_accounting): Transformer LM resource accounting

(a)

Model weights: 

            - `token_embeddings.weight`
                - Shape is (vocab_size, d_model).
   - `layers.{num_layers}.attn.q_proj.weight`
        - Shape is (num_heads * (d_model / num_heads), d_model).
   - `layers.{num_layers}.attn.k_proj.weight`
        - Shape is (num_heads * (d_model / num_heads), d_model).
   - `layers.{num_layers}.attn.v_proj.weight`
        - Shape is (num_heads * (d_model / num_heads), d_model).
   - `layers.{num_layers}.attn.output_proj.weight`
        - Shape is ((d_model / num_heads) * num_heads, d_model).
   - `layers.{num_layers}.ln1.weight`
        - Shape is (d_model,).
   - `layers.{num_layers}.ffn.w1.weight`
        - Shape is (d_ff, d_model).
   - `layers.{num_layers}.ffn.w2.weight`
        - Shape is (d_model, d_ff).
   - `layers.{num_layers}.ffn.w3.weight`
        - Shape is (d_ff, d_model).
   - `layers.{num_layers}.ln2.weight`
        - Shape is (d_model,).
   - `ln_final.weight`
        - Shape is (d_model, ).
   - `lm_head.weight`
        - Shape is (vocab_size, d_model).

GPT-2 XL-sized model:

+ vocab_size: 50,257

+ context_length: 1,024

+ num_layers: 48

+ d_model: 1,600

+ num_heads: 25

+ d_ff: 4,288

Total number of parameters: 1640452800

Memory required to load the model: 6.11 GB

(b)

In each Transformer block:

+ Q, K, V projection
  + d_k = d_v = d_model / num_heads
  + Q, K: (seq_len, d_model) * (d_model, num_heads * d_k)
  + V:  (seq_len, d_model) * (d_model, num_heads * d_v)
  + Total number of FLOPs: 15728640000
+ Attention scores
  + (seq_len, d_k) * (d_k, seq_len)
  + num_heads times
  + Total number of FLOPs (all heads): 3355443200
+ Attention output
  + (seq_len, seq_len) * (seq_len, d_v)
  + num_heads times
  + Total number of FLOPs (all heads): 3355443200
+ Output projection
  + (seq_len, num_head * d_v) * (num_head * d_v, d_model)
  + Total number of FLOPs: 5242880000
+ FFN
  + W1, W3: (seq_len, d_model) * (d_model, d_ff)
  + W2: (seq_len, d_ff) * (d_ff, d_model)
  + Total number of FLOPs: 42152755200

The matrix multiplies above will run num_layers times

Global: 

+ Output embedding
  + (seq_len, d_model) * (d_model, vocab_size)
  + Total number of FLOPs: 164682137600

Summed up FLOPs: 3,516,769,894,400 

(c)

FFN of all layers takes up the biggest part of the total FLOPs.

Number of FLOPs in FFN (all layers): 2,023,332,249,600 (57.53%)

(d)

GPT-2 small:

+ model size

  + vocab_size: 50,257

  + context_length: 1,024

  + num_layers: 12

  + d_model: 768

  + num_heads: 12

  + d_ff: 2,048

+ Proportion of each component
  + Causal Multi-Head Self-Attention with RoPE: 33.13%
  + Position-Wise Feed-Forward: 39.76%
  + Linear (Output Embedding): 27.10%

GPT-2 medium:

+ model size

  + vocab_size: 50,257

  + context_length: 1,024

  + num_layers: 24

  + d_model: 1,024

  + num_heads: 16

  + d_ff: 2,752

+ Proportion of each component

  + Causal Multi-Head Self-Attention with RoPE: 37.25%
  + Position-Wise Feed-Forward: 50.05%
  + Linear (Output Embedding): 12.70%

GPT-2 large:

+ model size

  + vocab_size: 50,257

  + context_length: 1,024

  + num_layers: 36

  + d_model: 1,280

  + num_heads: 20

  + d_ff: 3,392

+ proportion of each component

  + Causal Multi-Head Self-Attention with RoPE: 38.25%
  + Position-Wise Feed-Forward: 54.30%
  + Linear (Output Embedding): 7.45%

GPT-2 XL:

+ model size:
  + vocab_size: 50,257
  + context_length: 1,024
  + num_layers: 48
  + d_model: 1,600
  + num_heads: 25
  + d_ff: 4,288
+ proportion of each component
  + Causal Multi-Head Self-Attention with RoPE: 37.78%
  + Position-Wise Feed-Forward: 57.53%
  + Linear (Output Embedding): 4.68%

In conclusion, as model size grows larger, the proportion of Position-Wise Feed-Forward increases, while the proportion of Linear (Output Embedding) decreases significantly, and the proportion of Causal Multi-Head Self-Attention with RoPE holds steady.

(e)

GPT-2 XL (updated):

+ model size:
  + vocab_size: 50,257
  + context_length: 16,384
  + num_layers: 48
  + d_model: 1,600
  + num_heads: 25
  + d_ff: 4,288
+ proportion of each component
  + Causal Multi-Head Self-Attention with RoPE: 73.79%
  + Position-Wise Feed-Forward: 24.24%
  + Linear (Output Embedding): 1.97%

The total FLOPs grows from 3,516,769,894,400 to 133,577,729,638,400.

The proportion of Causal Multi-Head Self-Attention with RoPE becomes the most significant, since the number of FLOPs in Attention scores scales quadratically with context length.









