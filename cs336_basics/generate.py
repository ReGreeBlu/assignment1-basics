import argparse
import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.decoding import decode
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import load_checkpoint

vocab_path = "output/tinystories_train_vocab.json"
merges_path = "output/tinystories_train_merges.json"
special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

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

# decoding hyperparameters
parser.add_argument("--max_num_tokens", type=int, default=1000)
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--threshold", type=float, default=0.6)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)

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

# training
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

# decoding
max_num_tokens = arg.max_num_tokens
temperature = arg.temperature
threshold = arg.threshold
checkpoint = arg.checkpoint
prompt = arg.prompt

if arg.lr is None:
    lr = max_learning_rate
if arg.warmup_iters is None:
    warmup_iters = num_steps * 0.05
if arg.cosine_cycle_iters is None:
    cosine_cycle_iters = num_steps

model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
model = model.to(device)

optimizer = AdamW(model.parameters(), betas=betas, eps=eps, weight_decay=weight_decay)

step = load_checkpoint(checkpoint, model, optimizer)

model.eval()
with torch.no_grad():
    input_tokens = tokenizer.encode(prompt)
    output_tokens = decode(tokenizer, special_tokens,
                           model, input_tokens,
                           max_num_tokens, context_length, device,
                           temperature, threshold)
    response = tokenizer.decode(output_tokens)
    print(response)
