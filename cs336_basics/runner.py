from cs336_basics.bpe_tokenizer import train_tokenizer

def main() -> None:
    tiny_path = "/Users/uelipeter/Documents/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    train_tokenizer(tiny_path, 10000, ["<|endoftext|> "])


if __name__ == "__main__":
    main()
