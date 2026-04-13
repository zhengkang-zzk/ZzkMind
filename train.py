# 兼容旧命令：如果用户仍然运行 train.py，就直接转到新的预训练入口。
# 下面的旧训练代码不会执行；后续确认没有占用后可以删除这个文件。
from train_pretrain import main as _pretrain_main


if __name__ == "__main__":
    _pretrain_main()
    raise SystemExit

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


def main():
    config = load_config("configs/tiny.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 1. 读取训练文本
    text = load_text(config.data.train_path)
    text = text * 20

    # 2. tokenizer
    tokenizer_dir = r"E:/ZzkMind/dataset/xlm-roberta-base"
    tokenizer = HFLocalTokenizer(tokenizer_dir)
    encoded = tokenizer.encode(text)

    # 3. 对齐词表大小
    config.model.vocab_size = tokenizer.vocab_size

    # 4. 序列长度
    seq_len = config.model.max_position_embeddings

    if len(encoded) <= seq_len + 1:
        raise ValueError(
            f"文本太短：len(encoded)={len(encoded)}, seq_len={seq_len}，"
            "无法同时构造训练集和验证集。"
        )

    # 5. train / val 切分
    split_ratio = getattr(config.train, "train_val_split", 0.9)
    split_idx = int(len(encoded) * split_ratio)

    train_tokens = encoded[:split_idx]
    val_tokens = encoded[split_idx:]

    print("text length:", len(text))
    print("encoded length:", len(encoded))
    print("vocab_size:", tokenizer.vocab_size)
    print("seq_len:", seq_len)
    print("train token length:", len(train_tokens))
    print("val token length:", len(val_tokens))

    # 6. 构造 dataset
    train_dataset = LMDataset(tokens=train_tokens, seq_len=seq_len)
    val_dataset = LMDataset(tokens=val_tokens, seq_len=seq_len)

    print("train dataset size:", len(train_dataset))
    print("val dataset size:", len(val_dataset))

    if len(train_dataset) == 0:
        raise ValueError("train_dataset 为空，请减小 seq_len 或增大训练文本。")

    if len(val_dataset) == 0:
        raise ValueError(
            "val_dataset 为空，请减小 seq_len，或增加训练文本，或提高 train_val_split 前的文本长度。"
        )

    # 7. 构造 dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
    )

    print("train steps per epoch:", len(train_loader))
    print("val steps:", len(val_loader))

    # 8. 初始化模型与优化器
    model = ZzkModel(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)

    num_epochs = getattr(config.train, "num_epochs", 5)
    log_interval = getattr(config.train, "log_interval", 10)
    grad_clip = getattr(config.train, "grad_clip", None)
    moe_aux_loss_weight = getattr(config.train, "moe_aux_loss_weight", 0.0001)
    print(
        "moe:",
        getattr(config.model, "use_moe", False),
        "| num_experts:",
        getattr(config.model, "moe_num_experts", 0),
        "| top_k:",
        getattr(config.model, "moe_top_k", 0),
        "| aux_weight:",
        moe_aux_loss_weight,
    )

    save_dir = Path(getattr(config.train, "save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")



    # 9. 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        aux_running_loss = 0.0

        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # aux_loss 是 MoE 路由的负载均衡约束；dense FFN 时它恒为 0。
            logits, aux_loss = model(batch_x, return_aux_loss=True)  # (B, T, vocab_size)

            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch_y.view(-1)
            )
            # 总 loss = 语言模型 loss + 小权重的 MoE 辅助 loss，避免 router 只偏爱少数专家。
            loss = lm_loss + moe_aux_loss_weight * aux_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_running_loss += lm_loss.item()
            aux_running_loss += aux_loss.item()
            global_step += 1


        # 10. 统计 train loss
        avg_train_loss = train_running_loss / len(train_loader)
        avg_aux_loss = aux_running_loss / len(train_loader)
        train_ppl = math.exp(avg_train_loss) if avg_train_loss < 20 else float("inf")

        # 11. 计算 val loss
        avg_val_loss, val_ppl = evaluate(model, val_loader, device)



        print(
            f"[epoch {epoch}] "
            f"train_loss={avg_train_loss:.6f}, train_ppl={train_ppl:.6f} | "
            f"val_loss={avg_val_loss:.6f}, val_ppl={val_ppl:.6f} | "
            f"moe_aux_loss={avg_aux_loss:.6f}"
        )

        # 12. 保存 last checkpoint
        last_ckpt = save_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": tokenizer.vocab_size,
                "tokenizer_dir": tokenizer_dir,
                "seq_len": seq_len,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            last_ckpt,
        )
        print(f"last checkpoint saved to: {last_ckpt}")

        # 13. 保存 best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt = save_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab_size": tokenizer.vocab_size,
                    "tokenizer_dir": tokenizer_dir,
                    "seq_len": seq_len,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                best_ckpt,
            )
            print(f"best checkpoint updated: {best_ckpt}")

    tokenizer_name = "xlm-roberta-base-local"

    rope_scaling = getattr(config.model, "rope_scaling", None)
    if rope_scaling is None:
        position_type = "rope"
    else:
        position_type = "rope+yarn"
    
    print(
        f"[summary] tokenizer={tokenizer_name}, position={position_type}, "
        f"seq_len={seq_len}, hidden_size={config.model.hidden_size}, "
        f"layers={config.model.num_hidden_layers}, "
        f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_ppl={val_ppl:.4f}"
    )


if __name__ == "__main__":
    main()
