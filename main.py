import torch
import torch.nn.functional as F
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
        batch_size=config.train.batch_size,
        shuffle=True
    )

    model = ZzkModel(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
    model.train()

    for step, (batch_x, batch_y) in enumerate(loader):
        logits = model(batch_x) # logits (batch_size, seq_len, vocab_size)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch_y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step={step}, loss={loss.item():.4f}")
        if step >= 20:
            break

if __name__ == "__main__":
    main()