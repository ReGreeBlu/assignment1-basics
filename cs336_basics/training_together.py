import argparse
import numpy as np
import os
import csv
import time
from cs336_basics.data_loading import get_batch
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.adamw import AdamW
from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.checkpointing import save_checkpoint
from cs336_basics.evaluate import evaluate


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
parser.add_argument("--lr", type=float)
parser.add_argument("--beta_1", type=float, default=0.9)
parser.add_argument("--beta_2", type=float, default=0.999)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--weight_decay", type=float, default=1e-2)

# scheduler hyperparameters
parser.add_argument("--max_learning_rate", type=float, default=1e-3)
parser.add_argument("--min_learning_rate", type=float, default=1e-4)
parser.add_argument("--warmup_iters", type=int)
parser.add_argument("--cosine_cycle_iters", type=int)

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

if lr is None:
    lr = max_learning_rate
if arg.warmup_iters is None:
    warmup_iters = num_steps * 0.05
if arg.cosine_cycle_iters is None:
    cosine_cycle_iters = num_steps

# periodically logging training and validation performance
logging_interval = 100
eval_interval = 200
checkpoint_interval = 1000

filename_train = "output/tinystories_train_tokens.npy"
dataset_train = np.load(filename_train, mmap_mode="r")

filename_valid = "output/tinystories_valid_tokens.npy"
dataset_valid = np.load(filename_valid, mmap_mode="r")

model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
model = model.to(device)

print(f"model.device: {next(model.parameters()).device}")

optimizer = AdamW(model.parameters(), betas=betas, eps=eps, weight_decay=weight_decay)
valid_log = []
train_log = []
start_time = time.perf_counter()

for step in range(num_steps):

    in_indices, targets = get_batch(dataset_train, batch_size, context_length, device)

    logits = model(in_indices)
    loss = cross_entropy(logits, targets)

    optimizer.zero_grad()
    loss.backward()

    gradient_clipping(model.parameters(), max_l2_norm, eps_gradient_clipping)

    lr =  get_lr_cosine_schedule(step, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()

    if step % logging_interval == 0:
        loss_train = loss.item()
        elapsed = time.perf_counter() - start_time
        train_log.append((step, elapsed, loss_train))
        print(f"step = {step:5d}, loss_train = {loss_train:.6f}")
    if step % eval_interval == 0:
        loss_valid = evaluate(model, dataset_valid, batch_size, context_length, device).item()
        elapsed = time.perf_counter() - start_time
        valid_log.append((step, elapsed, loss_valid))
        print(f"validation: step = {step:5d}, loss_valid = {loss_valid:.6f}")
    if step % checkpoint_interval == 0:
        out = os.path.join(checkpoint_path, f"step_{step}.pt")
        save_checkpoint(model, optimizer, step, out)

with open("output/tinystories_train_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "time", "loss"])
    writer.writerows(train_log)

with open("output/tinystories_valid_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "time", "loss"])
    writer.writerows(valid_log)