from cs336_basics.train_bpe import train_bpe
import cProfile


input_path = "data/TinyStoriesV2-GPT4-valid.txt"
vocab_size = 10000
special_tokens = ["<|endoftext|>"]

cProfile.run('train_bpe(input_path, vocab_size, special_tokens)', sort="tottime")
