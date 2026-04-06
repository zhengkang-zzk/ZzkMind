import torch
from torch.utils.data import DataLoader
from config import load_config
from dataset.text_dataset import load_text
from dataset.tokenizer import CharTokenizer
from dataset.loader import LMDataset
from model.ZzkModel import ZzkModel


def main():
    config = load_config("configs/tiny.yaml")

    text = load_text(config.data.train_path)
    text = text * 10
    tokenizer = CharTokenizer(text)
    encoded = tokenizer.encode(text)

    config.model.vocab_size = tokenizer.vocab_size

    dataset = LMDataset(
        tokens=encoded,
        seq_len=config.model.max_position_embeddings
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    model = ZzkModel(config.model)
    model.eval()

    batch_x, batch_y = next(iter(loader))

    print("batch_x shape:", batch_x.shape)
    print("batch_y shape:", batch_y.shape)

    with torch.no_grad():
        logits = model(batch_x)

    print("logits shape:", logits.shape)
    print("expected shape:", (batch_x.size(0), batch_x.size(1), config.model.vocab_size))


if __name__ == "__main__":
    main()