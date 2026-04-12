import torch

from config import load_config
from dataset.tokenizer import HFLocalTokenizer
from model.ZzkModel import ZzkModel


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    greedy: bool = False,
    top_k: int | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    logits = logits / temperature

    if greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)

        probs = torch.softmax(topk_vals, dim=-1)
        sampled_pos = torch.multinomial(probs, num_samples=1)

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

            logits = model(idx_cond)
            logits = logits[:, -1, :]

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

    tokenizer_dir = ckpt["tokenizer_dir"]
    tokenizer = HFLocalTokenizer(tokenizer_dir)

    config.model.vocab_size = tokenizer.vocab_size

    if hasattr(config.model, "dropout"):
        config.model.dropout = 0.0

    model = ZzkModel(config.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prompt = "今天天气"
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor([encoded], dtype=torch.long, device=device)

    print("prompt:", repr(prompt))
    print("input idx shape:", idx.shape)

    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=100,
        max_position_embeddings=config.model.max_position_embeddings,
        temperature=0.8,
        greedy=False,
        top_k=5,
    )

    out_ids = out[0].tolist()
    out_text = tokenizer.decode(out_ids)

    print("generated ids:", out_ids)
    print("generated text:")
    print(out_text)


if __name__ == "__main__":
    main()