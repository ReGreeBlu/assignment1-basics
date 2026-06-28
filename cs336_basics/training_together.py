import argparse
import numpy as np
import os
import torch
import time
from cs336_basics.data_loading import get_batch
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.adamw import AdamW
from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.checkpointing import save_checkpoint

parser = argparse.ArgumentParser()
# model hyperparameters
parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--context_length", type=int, default=256)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--num_heads", type=int, default=16)
parser.add_argument("--d_ff", type=int, default=1344)
parser.add_argument("--rope_theta", type=float, default=10000)

# optimizer hyperparameters
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--beta_1", type=float, default=0.9)
parser.add_argument("--beta_2", type=float, default=0.999)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--weight_decay", type=float, default=1e-2)

# scheduler hyperparameters
parser.add_argument("--max_learning_rate", type=float, default=1)
parser.add_argument("--min_learning_rate", type=float, default=0.1)
parser.add_argument("--warmup_iters", type=int, default=200)
parser.add_argument("--cosine_cycle_iters", type=int, default=5000)

# training hyperparameters
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device", type=str, default="mps")
parser.add_argument("--num_steps", type=int, default=5000)
parser.add_argument("--max_l2_norm", type=float, default=1.0)
parser.add_argument("--eps_gradient_clipping", type=float, default=1e-6)
parser.add_argument("--checkpoint_path", type=str, default="output/checkpoints")


arg = parser.parse_args()

# model
vocab_size = arg.vocab_size
context_length = arg.context_length
d_model = arg.d_model
num_layers = arg.num_layers
num_heads = arg.num_heads
d_ff = arg.d_ff
rope_theta = arg.rope_theta

# optimizer
lr = arg.lr
betas = (arg.beta_1, arg.beta_2)
eps = arg.eps
weight_decay = arg.weight_decay

# train
batch_size = arg.batch_size
device = arg.device
num_steps = arg.num_steps
max_l2_norm = arg.max_l2_norm
eps_gradient_clipping = arg.eps_gradient_clipping
checkpoint_path = arg.checkpoint_path

# scheduler
max_learning_rate = arg.max_learning_rate
min_learning_rate = arg.min_learning_rate
warmup_iters = arg.warmup_iters
cosine_cycle_iters = arg.cosine_cycle_iters

# periodically logging training and validation performance
logging_interval = 100
checkpoint_interval = 1000
eval_interval = 100

filename_train = "output/tinystories_train_tokens.npy"
dataset_train = np.load(filename_train, mmap_mode="r")

filename_valid = "output/tinystories_valid_tokens.npy"
dataset_valid = np.load(filename_valid, mmap_mode="r")

model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
model = model.to(device)

print(f"model.device: {next(model.parameters()).device}")

optimizer = AdamW(model.parameters(), betas=betas, eps=eps, weight_decay=weight_decay)

time_data_loading = 0
time_forward = 0
time_backward = 0
time_gradient_clipping = 0
time_optimizer_step = 0

for step in range(num_steps):

    if step in range(10, 20):
        torch.mps.synchronize()
        start_time = time.perf_counter()

    in_indices, targets = get_batch(dataset_train, batch_size, context_length, device)

    if step in range(10, 20):
        torch.mps.synchronize()
        end_time = time.perf_counter()
        time_data_loading += end_time - start_time
        start_time = time.perf_counter()

    logits = model(in_indices)
    loss = cross_entropy(logits, targets)

    if step in range(10, 20):
        torch.mps.synchronize()
        end_time = time.perf_counter()
        time_forward += end_time - start_time
        start_time = time.perf_counter()

    optimizer.zero_grad()
    loss.backward()

    if step in range(10, 20):
        torch.mps.synchronize()
        end_time = time.perf_counter()
        time_backward += end_time - start_time
        start_time =time.perf_counter()
    
    gradient_clipping(model.parameters(), max_l2_norm, eps_gradient_clipping)

    if step in range(10, 20):
        torch.mps.synchronize()
        end_time = time.perf_counter()
        time_gradient_clipping += end_time - start_time
    
    lr =  get_lr_cosine_schedule(step, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()

    if step % logging_interval == 0:
        loss_train = loss.item()

        print(f"step = {step: 5d}, loss_train = {loss_train: .6f}")
    if step % checkpoint_interval == 0:
        out = os.path.join(checkpoint_path, f"step_{step}.pt")
        save_checkpoint(model, optimizer, step, out)
    if step == 20:
        break

print(f"Elapsed time (data loading): {time_data_loading/10:.3f} sec")
print(f"Elapsed time (forward): {time_forward/10:.3f} sec")
print(f"Elapsed time (backward): {time_backward/10:.3f} sec")
print(f"Elapsed time (gradient clipping): {time_gradient_clipping/10:.3f} sec")

