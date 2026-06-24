import json
import regex
from collections.abc import Iterable
from collections.abc import Iterator

class Tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.token_to_ID = {v: k for k, v in vocab.items()}
        if self.special_tokens:
            self.special_tokens_set = set(self.special_tokens)
            self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
            self.split_pattern = "(" + "|".join([regex.escape(token) for token in self.special_tokens_sorted]) + ")"
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r") as f:
            serial_vocab = json.load(f)
        vocab = dict()
        for token_ID, token in serial_vocab.items():
            vocab[int(token_ID)] = bytes(token)
        
        with open(merges_filepath, "r") as f:
            serial_merges = json.load(f)
        merges = []
        for pair in serial_merges:
            merges.append((bytes(pair[0]), bytes(pair[1])))

        return cls(vocab, merges, special_tokens)
    
    def _encode_chunk(self, text: str):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        words = regex.findall(PAT, text)
        ids = []
        for word_str in words:   # word: str -> list[bytes, ...]
            word = [bytes([byte]) for byte in word_str.encode("utf-8")]
            for pair in self.merges:
                new_token = pair[0] + pair[1]
                word_len = len(word)
                new_word = []
                i = 0
                while i < word_len:
                    if i+1 < word_len and pair == (word[i], word[i+1]):
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word
            for token in word:
                ids.append(self.token_to_ID[token])
        return ids

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            split_text = regex.split(self.split_pattern, text)
            ids = []
            for chunk in split_text:
                if chunk in self.special_tokens_set:
                    sp_token = chunk.encode("utf-8")
                    ids.append(self.token_to_ID[sp_token])
                else:
                    ids += self._encode_chunk(chunk)
        else:
            ids = self._encode_chunk(text)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            ids = self.encode(line)
            yield from ids

    def decode(self, ids: list[int]) -> str:
        token_list = []
        for token_ID in ids:
            token_list.append(self.vocab[token_ID])
        text_bytes = b"".join(token_list)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

if __name__ == "__main__":
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at', 11: b'<|endoftext|>'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    text = "the cat ate<|endoftext|>the cat ate"
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    print(tokenizer.encode(text))
