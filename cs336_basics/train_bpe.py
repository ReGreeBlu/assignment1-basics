import regex
import collections

def pre_tokenize(text):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    words = regex.findall(PAT, text)
    word_counts = collections.Counter(words)
    return word_counts

def token_merge(
        token_counts: dict[tuple[bytes, ...], int],
            # {word: frequency}
        token_ID: int,
        vocab: dict[int, bytes],
            # {token_ID, token}
        merges: list[tuple[bytes, bytes]],
            # [pair]
        pair_counts: dict[tuple[bytes, bytes], int],
            # {pair: frequency}
        pair_index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
            # {pair: word_set}
):
    
    merge_pair = max(pair_counts, key = lambda p: (pair_counts[p], p))
    token1 = merge_pair[0]
    token2 = merge_pair[1]
    new_token = token1 + token2

    for word in set(pair_index[merge_pair]):
        word_len = len(word)
        freq = token_counts[word]

        for i in range(word_len-1):
            pair = (word[i], word[i+1])
            pair_counts[pair] -= freq
            pair_index[pair].discard(word)
        
        new_word = []
        i = 0
        while i < word_len:
            if i < word_len-1 and merge_pair == (word[i], word[i+1]):
                new_word.append(new_token)
                i += 2           
            else:
                new_word.append(word[i])
                i += 1
            
        new_word = tuple(new_word)
        new_word_len = len(new_word)
        for i in range(new_word_len-1):
            pair = (new_word[i], new_word[i+1])
            pair_counts[pair] += freq
            pair_index[pair].add(new_word)

        del token_counts[word]
        token_counts[new_word] = freq
    
    del pair_index[merge_pair]
    del pair_counts[merge_pair]

    vocab[token_ID] = new_token
    token_ID += 1
    merges.append((token1, token2))
    return token_counts, token_ID, vocab, merges, pair_counts, pair_index

def train_bpe(input_path, vocab_size, special_tokens):
    vocab = dict()
    token_ID = 0
    for sp_token in special_tokens:
        vocab[token_ID] = sp_token.encode("utf-8")
        token_ID += 1
    for i in range(256):
        vocab[token_ID] = bytes([i])
        token_ID += 1

    with open(input_path, "r", encoding="utf-8") as f:
        input_text = f.read()
    special_tokens_set = "|".join([regex.escape(token) for token in special_tokens])
    split_input_text = regex.split(special_tokens_set, input_text)

    total_word_counts = collections.Counter()
    for text in split_input_text:
        word_counts = pre_tokenize(text)
        total_word_counts += word_counts

    token_counts = dict()
    merges = list()
    for word, freq in total_word_counts.items():   
        word_bytes = tuple([bytes([byte]) for byte in word.encode("utf-8")])
        token_counts[word_bytes] = freq

    pair_counts = collections.defaultdict(int)
    pair_index = collections.defaultdict(set)
    for word, freq in token_counts.items():
        for i in range(len(word)-1):
            pair = (word[i], word[i+1])
            pair_counts[pair] += freq
            pair_index[pair].add(word)

    while len(vocab) < vocab_size:
        token_counts, token_ID, vocab, merges, pair_counts, pair_index \
            = token_merge(token_counts, token_ID, vocab, merges, pair_counts, pair_index)
    if __name__ == "__main__":
        for idx, pair in enumerate(merges):
            print(f"{idx} merge: {pair}")
    return vocab, merges

if __name__ == "__main__":
    input_path = "data/test.txt"
    train_bpe(input_path, 290, ["<|endoftext|>"])





    