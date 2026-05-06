import heapq
import json
import os
import time
from abc import ABC
from dataclasses import dataclass
import regex as re
import collections
from collections.abc import Iterable, Iterator
from functools import lru_cache

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


def merge_tuple(indices: tuple[int, ...], pair: tuple[int, int], new_index: int) -> tuple[int, ...]:
    """Tuple-native version of `merge` to avoid list conversions in training."""
    new_indices: list[int] = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return tuple(new_indices)


@dataclass(frozen=True)
class _PairHeapEntry:
    count: int
    pair: tuple[int, int]
    left_bytes: bytes
    right_bytes: bytes

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _PairHeapEntry):
            return NotImplemented
        # Reverse order so heapq(min-heap) picks:
        # max(count), then max(left_bytes), then max(right_bytes).
        if self.count != other.count:
            return self.count > other.count
        if self.left_bytes != other.left_bytes:
            return self.left_bytes > other.left_bytes
        return self.right_bytes > other.right_bytes


class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
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
    _STATE_VERSION = 1

    def __init__(self, params: BPETokenizerParams):
        self.params = params
        self._pretoken_pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            re.IGNORECASE,
        )
        self._bytes_to_indices = {value: key for key, value in self.params.vocab.items()}
        self._merge_pair_to_rank: dict[tuple[int, int], int] = {}
        self._merge_pair_to_new_id: dict[tuple[int, int], int] = {}
        for rank, pair in enumerate(self.params.merges):
            left_id = self._bytes_to_indices[pair[0]]
            right_id = self._bytes_to_indices[pair[1]]
            new_id = self._bytes_to_indices[pair[0] + pair[1]]
            idx_pair = (left_id, right_id)
            self._merge_pair_to_rank[idx_pair] = rank
            self._merge_pair_to_new_id[idx_pair] = new_id
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

    def to_state_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this tokenizer."""
        return {
            "version": self._STATE_VERSION,
            "vocab": [
                [token_id, list(token_bytes)]
                for token_id, token_bytes in sorted(self.params.vocab.items())
            ],
            "merges": [
                [list(left), list(right)]
                for left, right in self.params.merges
            ],
            "special_tokens": list(self._special_tokens),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "BPETokenizer":
        """Build a tokenizer from the output of `to_state_dict`."""
        version = state.get("version")
        if version != cls._STATE_VERSION:
            raise ValueError(f"Unsupported BPETokenizer state version: {version}")

        vocab_data = state["vocab"]
        merges_data = state["merges"]
        special_tokens_data = state.get("special_tokens", [])

        vocab = {
            int(token_id): bytes(token_bytes)
            for token_id, token_bytes in vocab_data
        }
        merges = [
            (bytes(left), bytes(right))
            for left, right in merges_data
        ]
        special_tokens = [str(token) for token in special_tokens_data]

        return cls(BPETokenizerParams(vocab=vocab, merges=merges, special_tokens=special_tokens))

    def save(self, path: str | os.PathLike) -> None:
        """Store the tokenizer state as JSON at `path`."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_state_dict(), f, separators=(",", ":"))

    @classmethod
    def load(cls, path: str | os.PathLike) -> "BPETokenizer":
        """Load a tokenizer state previously written by `save`."""
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        return cls.from_state_dict(state)

    def to_file(self, path: str | os.PathLike) -> None:
        """Alias for `save`."""
        self.save(path)

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "BPETokenizer":
        """Alias for `load`."""
        return cls.load(path)

    def _streaming_suffix_length(self, string: str) -> int:
        """Return the length of the suffix that may merge with the next chunk."""
        if not string:
            return 0

        suffix_length = 0

        # Whitespace handling in the GPT-2 pretokenization regex depends on
        # the following character, and a final space can become the leading
        # space of the next pretoken.
        split_start = len(string)
        while split_start > 0 and string[split_start - 1].isspace():
            split_start -= 1

        if split_start < len(string):
            suffix_length = max(suffix_length, len(string) - split_start)
        else:
            # A trailing non-whitespace run can be continued by the next
            # iterable chunk. Keep one leading space too, since GPT-2 pretokens
            # include an optional literal space before words/numbers/punctuation.
            while split_start > 0 and not string[split_start - 1].isspace():
                split_start -= 1
            if split_start > 0 and string[split_start - 1] == " ":
                split_start -= 1
            suffix_length = max(suffix_length, len(string) - split_start)

        boundary_tokens = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", *self._special_tokens]
        for token in boundary_tokens:
            for prefix_length in range(1, len(token)):
                if string.endswith(token[:prefix_length]):
                    suffix_length = max(suffix_length, prefix_length)

        return suffix_length

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode strings from `iterable` lazily, yielding token IDs one at a time."""
        buffer = ""
        for chunk in iterable:
            if not chunk:
                continue
            buffer += chunk
            suffix_length = self._streaming_suffix_length(buffer)
            encode_end = len(buffer) - suffix_length
            if encode_end <= 0:
                continue

            yield from self.encode(buffer[:encode_end])
            buffer = buffer[encode_end:]

        if buffer:
            yield from self.encode(buffer)

    def _get_best_merge_pair(self, indices: tuple[int, ...]) -> tuple[int, int] | None:
        best_pair: tuple[int, int] | None = None
        best_rank: int | None = None
        for pair in zip(indices, indices[1:]):
            rank = self._merge_pair_to_rank.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_pair = pair
                best_rank = rank
        return best_pair

    @lru_cache(maxsize=100_000)
    def _encode_pretoken_cached(self, string: str) -> tuple[int, ...]:
        indices = tuple(self._bytes_to_indices[bytes([byte])] for byte in string.encode("utf-8"))
        while len(indices) >= 2:
            best_pair = self._get_best_merge_pair(indices)
            if best_pair is None:
                break
            indices = merge_tuple(indices, best_pair, self._merge_pair_to_new_id[best_pair])
        return indices

    def _encode_pretoken(self, string: str) -> list[int]:
        return list(self._encode_pretoken_cached(string))

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

    # Compact word store: each unique pretokenized word once with frequency.
    words_by_id: dict[int, tuple[int, ...]] = {}
    word_freq_by_id: dict[int, int] = {}
    # Per-word pair multiplicities, e.g. (a, b) might occur multiple times in a word.
    word_pair_counts_by_id: dict[int, dict[tuple[int, int], int]] = {}

    pair_counts: dict[tuple[int, int], int] = collections.defaultdict(int)
    pair_to_word_ids: dict[tuple[int, int], set[int]] = collections.defaultdict(set)

    for word_id, (word, freq) in enumerate(word_counts.items()):
        words_by_id[word_id] = word
        word_freq_by_id[word_id] = freq

        local_pair_counts: dict[tuple[int, int], int] = collections.defaultdict(int)
        for pair in zip(word, word[1:]):
            local_pair_counts[pair] += 1
        word_pair_counts_by_id[word_id] = dict(local_pair_counts)

        for pair, occurrences in local_pair_counts.items():
            pair_counts[pair] += occurrences * freq
            pair_to_word_ids[pair].add(word_id)

    pair_heap: list[_PairHeapEntry] = []
    for pair, count in pair_counts.items():
        if count > 0:
            pair_heap.append(_PairHeapEntry(count, pair, vocab[pair[0]], vocab[pair[1]]))
    heapq.heapify(pair_heap)

    num_merges = vocab_size - len(special_tokens) - 256
    if num_merges > 0:
        for _ in range(num_merges):
            best_pair: tuple[int, int] | None = None
            while pair_heap:
                candidate = heapq.heappop(pair_heap)
                current_count = pair_counts.get(candidate.pair, 0)
                # Skip stale heap entries.
                if current_count <= 0 or current_count != candidate.count:
                    continue
                best_pair = candidate.pair
                break

            if best_pair is None:
                break

            left_id, right_id = best_pair
            new_id = len(vocab)
            vocab[new_id] = vocab[left_id] + vocab[right_id]
            merges.append((vocab[left_id], vocab[right_id]))

            affected_word_ids = list(pair_to_word_ids.get(best_pair, set()))
            for word_id in affected_word_ids:
                freq = word_freq_by_id[word_id]
                old_word_pair_counts = word_pair_counts_by_id[word_id]

                # Remove old pair contributions for this word.
                for pair, occurrences in old_word_pair_counts.items():
                    updated_total = pair_counts[pair] - (occurrences * freq)
                    if updated_total > 0:
                        pair_counts[pair] = updated_total
                        heapq.heappush(
                            pair_heap,
                            _PairHeapEntry(updated_total, pair, vocab[pair[0]], vocab[pair[1]]),
                        )
                    else:
                        pair_counts.pop(pair, None)

                    pair_word_ids = pair_to_word_ids.get(pair)
                    if pair_word_ids is not None:
                        pair_word_ids.discard(word_id)
                        if not pair_word_ids:
                            pair_to_word_ids.pop(pair, None)

                # Merge selected pair in this word.
                merged_word = merge_tuple(words_by_id[word_id], best_pair, new_id)
                words_by_id[word_id] = merged_word

                # Add new pair contributions for this word.
                new_word_pair_counts: dict[tuple[int, int], int] = collections.defaultdict(int)
                for pair in zip(merged_word, merged_word[1:]):
                    new_word_pair_counts[pair] += 1
                word_pair_counts_by_id[word_id] = dict(new_word_pair_counts)

                for pair, occurrences in new_word_pair_counts.items():
                    pair_counts[pair] = pair_counts.get(pair, 0) + (occurrences * freq)
                    pair_to_word_ids[pair].add(word_id)
                    heapq.heappush(
                        pair_heap,
                        _PairHeapEntry(pair_counts[pair], pair, vocab[pair[0]], vocab[pair[1]]),
                    )

    vocab_values = set(vocab.values())
    for special_token in special_tokens:
        special_token_bytes_value = special_token.encode("utf-8")
        if special_token_bytes_value not in vocab_values:
            vocab[len(vocab)] = special_token_bytes_value
            vocab_values.add(special_token_bytes_value)
    end_time = time.time()
    print(f"Finished in {end_time - start_time} seconds.")
    return vocab, merges
