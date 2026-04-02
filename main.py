from config import load_config
from dataset.text_dataset import load_text
from dataset.tokenizer import CharTokenizer


def main():
    config = load_config("configs/tiny.yaml")

    train_text = load_text(config.data.train_path)
    val_text = load_text(config.data.val_path)

    tokenizer = CharTokenizer(train_text)

    encoded = tokenizer.encode(train_text[:20])
    decoded = tokenizer.decode(encoded)

    print("=== Config ===")
    print(config)

    print("\n=== Train Text Preview ===")
    print(train_text[:100])

    print("\n=== Val Text Preview ===")
    print(val_text[:100])

    print("\n=== Tokenizer Info ===")
    print("vocab_size:", tokenizer.vocab_size)
    print("encoded:", encoded)
    print("decoded:", decoded)


if __name__ == "__main__":
    main()