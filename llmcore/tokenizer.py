"""Day 1 — tokenizer.

Start char-level: the vocabulary is just the set of unique characters in the corpus.
Two lookups (char->id, id->char) and two methods (encode, decode). That's it. You can
swap in BPE later, but char-level lets you train something that works on day 1.
"""

from __future__ import annotations


class CharTokenizer:
    """Maps characters <-> integer ids.

    Attributes you'll want:
        stoi: dict[str, int]   # char -> id
        itos: dict[int, str]   # id -> char
        vocab_size: int
    """
    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {}
        self.itos = {}

        for i, s in enumerate(self.chars):
            self.stoi[s] = i
            self.itos[i] = s

        self.vocab_size = len(self.chars)

    def encode(self, s: str) -> list[int]:
        """Text -> list of token ids."""
        return [self.stoi[letter] for letter in s]

    def decode(self, ids: list[int]) -> str:
        """List of token ids -> text. Must round-trip: decode(encode(s)) == s."""
        return "".join(self.itos[idx] for idx in ids)

if __name__ == "__main__":
    tok = CharTokenizer(open("data/input.txt").read())
    s = "First Citizen:"
    print(tok.encode(s))
    print(tok.decode(tok.encode(s)))
    assert tok.decode(tok.encode(s)) == s