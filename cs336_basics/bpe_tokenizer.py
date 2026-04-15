import os
import time
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
import regex as re
import collections

from cs336_basics.pretokenization_example import pretokenize_file


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError

@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges:  list[tuple[bytes, bytes]]
    special_tokens: list[str]

class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params
        self._pretoken_pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            re.IGNORECASE,
        )
        self._bytes_to_indices = {value: key for key, value in self.params.vocab.items()}
        self._special_tokens = self.params.special_tokens or []
        self._special_token_to_id = {
            tok: self._bytes_to_indices[tok.encode("utf-8")]
            for tok in self._special_tokens
        }
        self._id_to_special_token = {
            token_id: token
            for token, token_id in self._special_token_to_id.items()
        }
        if self._special_tokens:
            ordered_special_tokens = sorted(self._special_tokens, key=len, reverse=True)
            self._special_token_pattern = re.compile(
                "|".join(re.escape(token) for token in ordered_special_tokens)
            )
        else:
            self._special_token_pattern = None

    def _encode_pretoken(self, string: str) -> list[int]:
        indices = [self._bytes_to_indices[bytes([byte])] for byte in string.encode("utf-8")]
        # Note: this is a very slow implementation
        for new_index, pair in enumerate(self.params.merges):
            idx_pair = (self._bytes_to_indices[pair[0]], self._bytes_to_indices[pair[1]])
            indices = merge(indices, idx_pair, new_index + 256)
        return indices

    def _encode_ordinary_text(self, string: str) -> list[int]:
        indices: list[int] = []
        for pretoken in self._pretoken_pattern.findall(string):
            indices.extend(self._encode_pretoken(pretoken))
        return indices

    def encode(self, string: str) -> list[int]:
        if not self._special_token_pattern:
            return self._encode_ordinary_text(string)

        indices: list[int] = []
        start = 0
        for match in self._special_token_pattern.finditer(string):
            if match.start() > start:
                indices.extend(self._encode_ordinary_text(string[start:match.start()]))
            indices.append(self._special_token_to_id[match.group(0)])
            start = match.end()

        if start < len(string):
            indices.extend(self._encode_ordinary_text(string[start:]))

        return indices

    def decode(self, indices: list[int]) -> str:
        def decode_bytes(value: bytes) -> str:
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                # Single-token decode may contain byte fragments (e.g. b"\xc3")
                # that are not valid UTF-8 by themselves.
                return value.decode("utf-8", errors="replace")

        decoded_parts: list[str] = []
        byte_buffer = bytearray()

        for index in indices:
            special_token = self._id_to_special_token.get(index)
            if special_token is not None:
                if byte_buffer:
                    decoded_parts.append(decode_bytes(bytes(byte_buffer)))
                    byte_buffer.clear()
                decoded_parts.append(special_token)
                continue

            token_bytes = self.params.vocab.get(index)
            if token_bytes is None:
                raise KeyError(f"Unknown token index: {index}")
            byte_buffer.extend(token_bytes)

        if byte_buffer:
            decoded_parts.append(decode_bytes(bytes(byte_buffer)))

        return "".join(decoded_parts)


def train_tokenizer(input_path: (str | os.PathLike), vocab_size: int, special_tokens: list[str])->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    special_tokens = special_tokens or []

    print("start pretokenization")
    start_time = time.time()
    pre_tokens = pretokenize_file(input_path)
    end_time = time.time()
    print(f"pretokenized {len(pre_tokens)} tokens in {end_time - start_time} seconds")
    special_token_bytes = {tuple(token.encode("utf-8")) for token in special_tokens}
    word_counts: dict[tuple[int, ...], int] = {
        token: freq
        for token, freq in pre_tokens.items()
        if token not in special_token_bytes
    }

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    num_merges = vocab_size - len(special_tokens) - 256
    if num_merges > 0:
        for i in range(num_merges):
            print(f"iteration{i} of {num_merges}")
            pair_counts: dict[tuple[int, int], int] = collections.defaultdict(int)
            for word, freq in word_counts.items():
                for pair in zip(word, word[1:]):
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            best_pair = max(
                pair_counts.items(),
                key=lambda item: (
                    item[1],
                    vocab[item[0][0]],
                    vocab[item[0][1]],
                ),
            )[0]
            left_id, right_id = best_pair
            new_id = len(vocab)
            vocab[new_id] = vocab[left_id] + vocab[right_id]
            merges.append((vocab[left_id], vocab[right_id]))

            updated_word_counts: dict[tuple[int, ...], int] = collections.defaultdict(int)
            for word, freq in word_counts.items():
                merged_word = tuple(merge(list(word), best_pair, new_id))
                updated_word_counts[merged_word] += freq
            word_counts = dict(updated_word_counts)

    for special_token in special_tokens:
        special_token_bytes_value = special_token.encode("utf-8")
        if special_token_bytes_value not in vocab.values():
            vocab[len(vocab)] = special_token_bytes_value

    return vocab, merges
