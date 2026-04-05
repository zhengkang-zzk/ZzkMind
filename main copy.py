import torch
from torch.utils.data import DataLoader
from config import load_config
from dataset.text_dataset import load_text
from dataset.tokenizer import CharTokenizer
from dataset.loader import LMDataset
from model.ZzkModel import ZzkModel



def main():
    config = load_config("configs/tiny.yaml")

    train_text = load_text(config.data.train_path)
    train_text = train_text * 10

    tokenizer = CharTokenizer(train_text)


    encoded = tokenizer.encode(train_text)

    train_dataset = LMDataset(
        tokens=encoded,
        seq_len=config.model.max_position_embeddings
    )

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size)
    model = ZzkModel(config.model)

    print("===开始第一次前向传播===")
    for batch_x, batch_y in train_loader:
        print(batch_x.shape)
        logits = model(batch_x)
        
        print(f"输出 Logits 的形状: {logits.shape}") 
        # 预期: (batch_size, seq_len, vocab_size)
        print(batch_y.shape)
        break
    


if __name__ == "__main__":
    main()