import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import load_config
from dataset.tokenizer import HFLocalTokenizer
from dataset.loader import LMDataset
from model.ZzkModel import ZzkModel
from train_pretrain import load_pretrain_text


CHECKPOINT_PATH = "checkpoints/best.pt"
MAX_NEW_TOKENS = 40
TEMPERATURE = 0.8
TOP_K = 5
GREEDY = False

DEFAULT_PROMPTS = [
    "请用一句话介绍这个项目。",
    "写一首关于秋天的短诗。",
    "解释一下什么是语言模型。",
]


def choose_eval_mode() -> str:
    # 运行脚本后只做一次模式选择，比命令行参数更适合手动检查模型。
    while True:
        print("\n请选择评测模式：")
        print("1. 自动评测：计算 val loss / ppl，并运行默认 prompts")
        print("2. 人工评测：手动输入 prompt；直接回车退出")
        choice = input("请输入 1 或 2：").strip()

        if choice in {"1", "2"}:
            return choice

        print("输入无效，请输入 1 或 2。")


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(loader):
            # 大数据集完整验证会很慢，max_batches 用来快速估计 loss / ppl。
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)  # (B, T, vocab_size)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch_y.view(-1)
            )

            total_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        raise ValueError("no evaluation batches were processed")

    avg_loss = total_loss / total_batches
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


@torch.no_grad()
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

    # 重要：你当前实现的 KV cache 是“累计缓存”版本，
    # 不是滑动窗口版本，所以总长度不能超过 max_position_embeddings
    if idx.size(1) >= max_position_embeddings:
        return idx

    # 先把 prompt 整段送进去，建立初始 cache
    logits, past_key_values = model(idx, past_key_values=None, use_cache=True)
    logits = logits[:, -1, :]   # 只取最后一个位置预测下一个 token

    for _ in range(max_new_tokens):
        next_token = sample_next_token(
            logits=logits,
            temperature=temperature,
            greedy=greedy,
            top_k=top_k,
        )  # shape: (B, 1)

        idx = torch.cat([idx, next_token], dim=1)

        # 达到最大上下文长度就停止
        if idx.size(1) >= max_position_embeddings:
            break

        # 后续只喂一个 token，同时传入 cache
        logits, past_key_values = model(
            next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = logits[:, -1, :]

    return idx


def build_val_loader(config, tokenizer):
    # 复用 train_pretrain.py 的数据读取逻辑，避免 JSONL 在 eval 中被当成普通文本整文件读取。
    text = load_pretrain_text(config.data)
    encoded = tokenizer.encode(text)

    seq_len = config.model.max_position_embeddings
    split_ratio = getattr(config.train, "train_val_split", 0.9)
    split_idx = int(len(encoded) * split_ratio)

    val_tokens = encoded[split_idx:]
    # eval 和训练使用相同 stride，避免验证阶段退回 stride=1 的高重叠滑窗。
    dataset_stride = getattr(config.data, "stride", None) or seq_len
    val_dataset = LMDataset(tokens=val_tokens, seq_len=seq_len, stride=dataset_stride)

    if len(val_dataset) == 0:
        raise ValueError("val_dataset is empty; check seq_len, data length or split_ratio.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
    )

    print("encoded length:", len(encoded))
    print("val token length:", len(val_tokens))
    print("val dataset size:", len(val_dataset))
    print("val steps:", len(val_loader))
    print("dataset stride:", dataset_stride)
    return val_loader


def run_generation(
    model,
    tokenizer,
    prompt,
    device,
    config,
):
    # 自动评测和人工评测都走这个公共入口，保证生成参数一致。
    encoded_prompt = tokenizer.encode(prompt)
    idx = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=MAX_NEW_TOKENS,
        max_position_embeddings=config.model.max_position_embeddings,
        temperature=TEMPERATURE,
        greedy=GREEDY,
        top_k=TOP_K,
    )
    return tokenizer.decode(out[0].tolist())


def load_model_and_tokenizer(config, device):
    # 从固定 checkpoint 恢复 tokenizer 和模型，保持 eval 入口足够简单。
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    print("loaded checkpoint:", CHECKPOINT_PATH)

    tokenizer_dir = ckpt["tokenizer_dir"]
    tokenizer = HFLocalTokenizer(tokenizer_dir)

    # checkpoint 的 embedding / lm_head 大小必须和 tokenizer vocab 对齐。
    config.model.vocab_size = tokenizer.vocab_size
    config.model.dropout = 0.0

    model = ZzkModel(config.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, tokenizer


def run_auto_eval(model, tokenizer, config, device):
    # 自动评测包含客观指标和固定 prompt 生成样例。
    val_loader = build_val_loader(config, tokenizer)
    eval_max_batches = getattr(config.train, "eval_max_batches", None)
    print("eval max batches:", eval_max_batches)

    val_loss, val_ppl = evaluate(
        model,
        val_loader,
        device,
        max_batches=eval_max_batches,
    )
    print(f"val_loss={val_loss:.6f}")
    print(f"val_ppl={val_ppl:.6f}")

    print("\n=== 自动生成样例 ===")
    for prompt in DEFAULT_PROMPTS:
        out_text = run_generation(model, tokenizer, prompt, device, config)
        print(f"\nprompt={repr(prompt)}")
        print(out_text)


def run_manual_eval(model, tokenizer, config, device):
    # 人工评测适合训练过程中快速肉眼检查模型输出；空 prompt 直接退出。
    print("\n=== 人工评测 ===")
    print("输入 prompt 后回车生成；直接回车退出。")

    while True:
        prompt = input("\nprompt> ").strip()
        if not prompt:
            print("退出人工评测。")
            break

        out_text = run_generation(model, tokenizer, prompt, device, config)
        print(out_text)


def main():
    config = load_config("configs/tiny.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model, tokenizer = load_model_and_tokenizer(config, device)
    mode = choose_eval_mode()

    if mode == "1":
        run_auto_eval(model, tokenizer, config, device)
    else:
        run_manual_eval(model, tokenizer, config, device)


if __name__ == "__main__":
    main()
