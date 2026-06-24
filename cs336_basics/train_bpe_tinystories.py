from cs336_basics.train_bpe import train_bpe
import json

# vocab, merges = train_bpe(
#     input_path = "data/TinyStoriesV2-GPT4-train.txt",
#     vocab_size = 10000,
#     special_tokens = ["<|endoftext|>"],
# )

# serial_vocab = {}
# for token_ID, token in vocab.items():
#     serial_vocab[token_ID] = list(token)
# with open("cs336_basics/vocab_train.json", "w") as f:
#     json.dump(serial_vocab, f)

# serial_merges = []
# for pair in merges:
#     serial_merges.append((list(pair[0]), list(pair[1])))
# with open("cs336_basics/merges_train.json", "w") as f:
#     json.dump(serial_merges, f)

with open("cs336_basics/vocab_train.json", "r") as f:
    vocab = json.load(f)
longest_token_ID = max(vocab, key=lambda p: len(vocab[p]))
longest_token = vocab[longest_token_ID]
longest_token_str = bytes(longest_token).decode("utf-8")
print("["+longest_token_str+"]")

