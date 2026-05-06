from cs336_basics.bpe_tokenizer import train_tokenizer, BPETokenizer, BPETokenizerParams


def main() -> None:
    tiny_path = "/Users/uelipeter/Documents/Development/private/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|> "]
    vocab, merges = train_tokenizer(tiny_path, 10000, special_tokens)
    tokenizer = BPETokenizer(BPETokenizerParams(vocab=vocab, merges=merges, special_tokens=special_tokens))
    tokenizer.save("/Users/uelipeter/Documents/Development/private/assignment1-basics/data/TinyStoriesV2-GPT4-train-tokenizer.json")


if __name__ == "__main__":
    main()
