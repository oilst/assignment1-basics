from cs336_basics.bpe_tokenizer import train_tokenizer

def main() -> None:
    tiny_path = "/Users/uelipeter/Documents/Development/private/assignment1-basics/data/owt_train.txt"
    train_tokenizer(tiny_path, 10000, ["<|endoftext|> "])


if __name__ == "__main__":
    main()
