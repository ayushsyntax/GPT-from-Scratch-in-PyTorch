# tokenizer.py

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, tokens: list[int]) -> str:
        return ''.join([self.itos[i] for i in tokens])

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
