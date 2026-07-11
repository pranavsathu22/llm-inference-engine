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
        # TODO(day1): build the sorted set of unique chars, then stoi/itos/vocab_size.
        raise NotImplementedError

    def encode(self, s: str) -> list[int]:
        """Text -> list of token ids."""
        # TODO(day1)
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """List of token ids -> text. Must round-trip: decode(encode(s)) == s."""
        # TODO(day1)
        raise NotImplementedError
