import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import load_config
from dataset.text_dataset import load_text
from dataset.tokenizer import HFLocalTokenizer
from dataset.loader import LMDataset
from model.ZzkModel import ZzkModel


def load_pretrain_text(data_config):
    path = Path(data_config.train_path)
    if path.suffix.lower() != ".jsonl":
        # 小文本仍然支持直接读取；repeat 只建议用于 tiny overfit，不建议大数据集重复。
        text = load_text(path)
        return text * max(1, getattr(data_config, "repeat", 1))

    # JSONL 数据集按行读取，只取配置指定字段，避免一次性把原始大文件全部展开。
    text_field = getattr(data_config, "jsonl_text_field", "text")
    max_records = getattr(data_config, "max_records", None)
    max_chars = getattr(data_config, "max_chars", None)

    chunks = []
    total_chars = 0
    total_records = 0
    skipped_records = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            # 用 max_records / max_chars 控制实验规模，先小跑通再逐步放大。
            if max_records is not None and total_records >= max_records:
                break
            if max_chars is not None and total_chars >= max_chars:
                break

            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped_records += 1
                continue

            value = item.get(text_field, "")
            if not isinstance(value, str) or not value.strip():
                skipped_records += 1
                continue

            if max_chars is not None:
                # 最后一条样本可能被截断到刚好不超过 max_chars。
                remain = max_chars - total_chars
                if remain <= 0:
                    break
                value = value[:remain]

            chunks.append(value)
            total_chars += len(value)
            total_records += 1

    if not chunks:
        raise ValueError(f"No usable text loaded from {path}")

    print(
        "jsonl loaded:",
        f"records={total_records}",
        f"chars={total_chars}",
        f"skipped={skipped_records}",
        f"field={text_field}",
    )

    text = "\n".join(chunks)
    return text * max(1, getattr(data_config, "repeat", 1))


def evaluate(model, loader, device):
    # 训练脚本里的验证默认跑完整 val_loader；eval.py 里另有 max_batches 快速评估。
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch_y.view(-1),
            )
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


def save_checkpoint(
    path,
    model,
    optimizer,
    tokenizer,
    tokenizer_dir,
    seq_len,
    epoch,
    global_step,
    train_loss=None,
    val_loss=None,
    best_val_loss=None,
):
    # checkpoint 保存模型、优化器和训练进度，支持中断后继续训练。
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_dir": tokenizer_dir,
            "seq_len": seq_len,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def format_duration(seconds):
    # 将 ETA 秒数转成更适合训练日志阅读的格式。
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def main():
    config = load_config("configs/tiny.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 先按配置读取 JSONL / TXT，再统一 tokenizer 成 token 序列。
    text = load_pretrain_text(config.data)

    tokenizer_dir = r"E:/ZzkMind/dataset/xlm-roberta-base"
    tokenizer = HFLocalTokenizer(tokenizer_dir)
    encoded = tokenizer.encode(text)

    config.model.vocab_size = tokenizer.vocab_size
    seq_len = config.model.max_position_embeddings

    if len(encoded) <= seq_len + 1:
        raise ValueError(
            f"text too short: len(encoded)={len(encoded)}, seq_len={seq_len}"
        )

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

    # stride 控制样本窗口移动步长；设为 seq_len 时基本是不重叠 block。
    dataset_stride = getattr(config.data, "stride", None) or seq_len
    train_dataset = LMDataset(tokens=train_tokens, seq_len=seq_len, stride=dataset_stride)
    val_dataset = LMDataset(tokens=val_tokens, seq_len=seq_len, stride=dataset_stride)

    if len(train_dataset) == 0:
        raise ValueError("train_dataset is empty; reduce seq_len or use more text.")
    if len(val_dataset) == 0:
        raise ValueError("val_dataset is empty; adjust split_ratio or use more text.")

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
    print("dataset stride:", dataset_stride)
    print("tokens per train step:", config.train.batch_size * seq_len)
    print("approx train tokens per epoch:", len(train_loader) * config.train.batch_size * seq_len)

    model = ZzkModel(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)

    num_epochs = getattr(config.train, "num_epochs", 5)
    log_interval = max(1, getattr(config.train, "log_interval", 10))
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
    avg_train_loss = None
    avg_val_loss = None
    start_epoch = 0

    resume_from = getattr(config.train, "resume_from", None)
    if resume_from:
        # resume_from 指向 interrupted.pt / last.pt / best.pt 时，恢复模型和优化器状态。
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", best_val_loss))
        avg_train_loss = ckpt.get("train_loss")
        avg_val_loss = ckpt.get("val_loss")
        print(
            f"resumed from {resume_path}: "
            f"start_epoch={start_epoch}, global_step={global_step}"
        )
        if start_epoch >= num_epochs:
            raise ValueError(
                f"checkpoint epoch is {start_epoch - 1}, but num_epochs is {num_epochs}. "
                "Increase train.num_epochs if you want to continue training."
            )

    # 训练过程中用已完成 step 的平均耗时估算 epoch_eta 和 total_eta。
    train_start_time = time.time()
    total_train_steps = (num_epochs - start_epoch) * len(train_loader)

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_running_loss = 0.0
            aux_running_loss = 0.0

            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits, aux_loss = model(batch_x, return_aux_loss=True)
                lm_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch_y.view(-1),
                )
                loss = lm_loss + moe_aux_loss_weight * aux_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

                train_running_loss += lm_loss.item()
                aux_running_loss += aux_loss.item()
                global_step += 1

                if (
                    step == 0
                    or (step + 1) % log_interval == 0
                    or (step + 1) == len(train_loader)
                ):
                    # ETA 是运行时动态估计，通常比启动前的静态估计更可靠。
                    done_steps = (epoch - start_epoch) * len(train_loader) + step + 1
                    elapsed = time.time() - train_start_time
                    sec_per_step = elapsed / max(1, done_steps)
                    epoch_eta = sec_per_step * (len(train_loader) - step - 1)
                    total_eta = sec_per_step * (total_train_steps - done_steps)
                    print(
                        f"[epoch {epoch} step {step + 1}/{len(train_loader)}] "
                        f"lm_loss={train_running_loss / (step + 1):.6f}, "
                        f"moe_aux_loss={aux_running_loss / (step + 1):.6f}, "
                        f"sec/step={sec_per_step:.3f}, "
                        f"epoch_eta={format_duration(epoch_eta)}, "
                        f"total_eta={format_duration(total_eta)}"
                    )

            avg_train_loss = train_running_loss / len(train_loader)
            avg_aux_loss = aux_running_loss / len(train_loader)
            train_ppl = math.exp(avg_train_loss) if avg_train_loss < 20 else float("inf")

            avg_val_loss, val_ppl = evaluate(model, val_loader, device)

            print(
                f"[epoch {epoch}] "
                f"train_loss={avg_train_loss:.6f}, train_ppl={train_ppl:.6f} | "
                f"val_loss={avg_val_loss:.6f}, val_ppl={val_ppl:.6f} | "
                f"moe_aux_loss={avg_aux_loss:.6f}"
            )

            last_ckpt = save_dir / "last.pt"
            save_checkpoint(
                last_ckpt,
                model,
                optimizer,
                tokenizer,
                tokenizer_dir,
                seq_len,
                epoch,
                global_step,
                avg_train_loss,
                avg_val_loss,
                best_val_loss,
            )
            print(f"last checkpoint saved to: {last_ckpt}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_ckpt = save_dir / "best.pt"
                save_checkpoint(
                    best_ckpt,
                    model,
                    optimizer,
                    tokenizer,
                    tokenizer_dir,
                    seq_len,
                    epoch,
                    global_step,
                    avg_train_loss,
                    avg_val_loss,
                    best_val_loss,
                )
                print(f"best checkpoint updated: {best_ckpt}")

    except KeyboardInterrupt:
        # 终端 Ctrl+C 时保存 interrupted.pt，避免半路停训丢掉当前进度。
        interrupted_ckpt = save_dir / "interrupted.pt"
        save_checkpoint(
            interrupted_ckpt,
            model,
            optimizer,
            tokenizer,
            tokenizer_dir,
            seq_len,
            epoch if "epoch" in locals() else -1,
            global_step,
            avg_train_loss,
            avg_val_loss,
            best_val_loss,
        )
        print(f"\ntraining interrupted; checkpoint saved to: {interrupted_ckpt}")
        return

    position_type = "rope+yarn" if getattr(config.model, "rope_scaling", None) else "rope"
    print(
        f"[summary] tokenizer=xlm-roberta-base-local, position={position_type}, "
        f"seq_len={seq_len}, hidden_size={config.model.hidden_size}, "
        f"layers={config.model.num_hidden_layers}, "
        f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_ppl={val_ppl:.4f}"
    )


if __name__ == "__main__":
    main()
