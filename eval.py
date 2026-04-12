import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import load_config
from dataset.text_dataset import load_text
from dataset.tokenizer import HFLocalTokenizer
from dataset.loader import LMDataset
from model.ZzkModel import ZzkModel


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)  # (B, T, vocab_size)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch_y.view(-1)
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    greedy: bool = False,
    top_k: int | None = 5,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    logits = logits / temperature

    if greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)   # (1, k), (1, k)

        probs = torch.softmax(topk_vals, dim=-1)
        sampled_pos = torch.multinomial(probs, num_samples=1)   # (1, 1)

        next_token = torch.gather(topk_idx, dim=-1, index=sampled_pos)
        return next_token

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    max_position_embeddings: int,
    temperature: float = 1.0,
    greedy: bool = False,
    top_k: int | None = 5,
):
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_position_embeddings:]

            logits = model(idx_cond)      # (1, T, vocab_size)
            logits = logits[:, -1, :]     # (1, vocab_size)

            next_token = sample_next_token(
                logits=logits,
                temperature=temperature,
                greedy=greedy,
                top_k=top_k,
            )

            idx = torch.cat([idx, next_token], dim=1)

    return idx


def main():
    config = load_config("configs/tiny.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt_path = "checkpoints/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    print("loaded checkpoint:", ckpt_path)

    # 1. 恢复 tokenizer
    tokenizer_dir = ckpt["tokenizer_dir"]
    tokenizer = HFLocalTokenizer(tokenizer_dir)

    # 2. 对齐配置
    config.model.vocab_size = tokenizer.vocab_size

    if hasattr(config.model, "dropout"):
        config.model.dropout = 0.0

    # 3. 重建验证集（和 train.py 保持一致）
    text = load_text(config.data.train_path)
    text = text * 20
    encoded = tokenizer.encode(text)

    seq_len = config.model.max_position_embeddings
    split_ratio = getattr(config.train, "train_val_split", 0.9)
    split_idx = int(len(encoded) * split_ratio)

    val_tokens = encoded[split_idx:]
    val_dataset = LMDataset(tokens=val_tokens, seq_len=seq_len)

    if len(val_dataset) == 0:
        raise ValueError("val_dataset 为空，请检查 seq_len、数据长度或 split_ratio。")

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
    )

    # 4. 恢复模型
    model = ZzkModel(config.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 5. 计算 val loss / ppl
    val_loss, val_ppl = evaluate(model, val_loader, device)

    print(f"val_loss={val_loss:.6f}")
    print(f"val_ppl={val_ppl:.6f}")

    # 6. 固定几个 prompt 做生成抽查
    prompts = [
        "今天天气",
        "这个项目",
        "我们开始",
    ]

    print("\n=== Generation Samples ===")
    for prompt in prompts:
        try:
            encoded_prompt = tokenizer.encode(prompt)
        except Exception as e:
            print(f"\nprompt={repr(prompt)}")
            print("编码失败：", e)
            continue

        idx = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

        out = generate(
            model=model,
            idx=idx,
            max_new_tokens=80,
            max_position_embeddings=config.model.max_position_embeddings,
            temperature=0.8,
            greedy=False,
            top_k=5,
        )

        out_ids = out[0].tolist()
        out_text = tokenizer.decode(out_ids)

        print(f"\nprompt={repr(prompt)}")
        print(out_text)


if __name__ == "__main__":
    main()