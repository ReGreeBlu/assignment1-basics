from cs336_basics.tokenizer import Tokenizer
import time
import numpy as np

# ----------------------(a)----------------------
# vocab_path = "output/vocab_tinystories_valid.json"
# merges_path = "output/merges_tinystories_valid.json"
# special_tokens = ["<|endoftext|>"]

# tokenizer_ts = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

# input_path = "data/TinyStoriesV2-GPT4-valid.txt"
# with open(input_path, "r") as f:
#     input_text = f.read()
# split_text = [text for text in input_text.split("<|endoftext|>") if text.strip()]
# sample_size = 10
# sample_text = split_text[:sample_size]

# bytes_count = 0
# token_count = 0
# for text in sample_text:
#     bytes_count += len(text.encode("utf-8"))
#     token_count += len(tokenizer_ts.encode(text))
# print("Compression ratio on TinyStories (valid): %.3f"%(bytes_count/token_count))

# vocab_path = "output/vocab_owt_valid.json"
# merges_path = "output/merges_owt_valid.json"
# special_tokens = []

# tokenizer_owt = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

# input_path = "data/owt_valid.txt"
# with open(input_path, "r") as f:
#     input_text = f.read()
# split_text = [text for text in input_text.split("\n\n") if text.strip()]
# sample_size = 10
# sample_text = split_text[:sample_size]

# bytes_count = 0
# token_count = 0
# for text in sample_text:
#     bytes_count += len(text.encode("utf-8"))
#     token_count += len(tokenizer_owt.encode(text))

# print("Compression ratio on OWT (valid): %.3f"%(bytes_count/token_count))

# ----------------------(b)----------------------
# input_path = "data/owt_valid.txt"
# with open(input_path, "r") as f:
#     input_text = f.read()
# split_text = [text for text in input_text.split("\n\n") if text.strip()]
# sample_size = 10
# sample_text = split_text[:sample_size]

# bytes_count = 0
# token_count = 0
# for text in sample_text:
#     bytes_count += len(text.encode("utf-8"))
#     token_count += len(tokenizer_ts.encode(text))

# print("Compression ratio on OWT (valid) with TinyStories tokenizer: %.3f"%(bytes_count/token_count))

# ----------------------(c)----------------------
# vocab_path = "output/vocab_owt_valid.json"
# merges_path = "output/merges_owt_valid.json"
# special_tokens = []

# tokenizer_owt = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

# input_path = "data/owt_valid.txt"
# with open(input_path, "r") as f:
#     input_text = f.read()
# split_text = [text for text in input_text.split("\n\n") if text.strip()]
# sample_size = 100
# sample_text = split_text[:sample_size]

# bytes_count = 0
# for text in sample_text:
#     bytes_count += len(text.encode("utf-8"))
# start = time.perf_counter()
# for text in sample_text:
#     tokenizer_owt.encode(text)
# end = time.perf_counter()
# elapsed_time = end-start
# print(f"Total bytes: {bytes_count:.3f}")
# print(f"Throughput of the OWT tokenizer: {bytes_count/elapsed_time:.3f}")

# ----------------------(d)----------------------
# -----------------TinyStoriesV2-GPT4-valid-----------------
vocab_path = "output/vocab_tinystories_valid.json"
merges_path = "output/merges_tinystories_valid.json"
special_tokens = ["<|endoftext|>"]

tokenizer_ts = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

input_path = "data/TinyStoriesV2-GPT4-valid.txt"
with open(input_path, "r") as f:
    total_ids = list(tokenizer_ts.encode_iterable(f))
arr = np.array(total_ids, dtype=np.uint16)
np.save("output/tinystories_valid_tokens.npy", arr)
print(f"tinystories_valid: Saved {len(total_ids)} tokens")

# -----------------owt_valid-----------------
vocab_path = "output/vocab_owt_valid.json"
merges_path = "output/merges_owt_valid.json"
special_tokens = []

tokenizer_owt = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
input_path = "data/owt_valid.txt"
with open(input_path, "r") as f:
    total_ids = list(tokenizer_owt.encode_iterable(f))
arr = np.array(total_ids, dtype=np.uint16)
np.save("output/owt_valid_tokens.npy", arr)
print(f"owt_valid: Saved {len(total_ids)} tokens")

# -----------------TinyStoriesV2-GPT4-train-----------------
vocab_path = "output/vocab_tinystories_train.json"
merges_path = "output/merges_tinystories_train.json"
special_tokens = ["<|endoftext|>"]

tokenizer_ts = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

input_path = "data/TinyStoriesV2-GPT4-train.txt"
with open(input_path, "r") as f:
    total_ids = list(tokenizer_ts.encode_iterable(f))
arr = np.array(total_ids, dtype=np.uint16)
np.save("output/tinystories_train_tokens.npy", arr)
print(f"tinystories_train: Saved {len(total_ids)} tokens")