from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    max_position_embeddings: int
    dropout: float

    rope_base: float 
    rope_scaling: Optional[dict]

    use_moe: bool = False
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_expert_capacity: int | None = None
    moe_capacity_factor: float = 1.25
    moe_use_shared_expert: bool = True


@dataclass
class TrainConfig:
    batch_size: int
    lr: float
    device: str
    num_epochs: int = 5
    log_interval: int = 10
    grad_clip: float | None = None
    save_dir: str = "checkpoints"
    train_val_split: float = 0.9
    # MoE 开启时加入总 loss 的辅助损失权重；dense FFN 下 aux_loss 为 0。
    moe_aux_loss_weight: float = 0.0001
    # 为空表示从头训练；填 checkpoints/interrupted.pt 等路径表示续训。
    resume_from: str | None = None
    # eval.py 快速评估最多跑多少个 batch；为空表示完整验证。
    eval_max_batches: int | None = None


@dataclass
class DataConfig:
    train_path: str
    val_path: str
    # JSONL 数据中用于预训练文本的字段名。
    jsonl_text_field: str = "text"
    # 小文本 overfit 时可重复数据；真实预训练建议保持 1。
    repeat: int = 1
    # 限制 JSONL 读取条数，方便先做小规模实验。
    max_records: int | None = None
    # 限制 JSONL 读取字符数，避免一开始吞入过大的本地数据集。
    max_chars: int | None = None
    # LM 样本窗口步长；通常设为 seq_len，避免 stride=1 的高度重叠滑窗。
    stride: int | None = None


@dataclass
class AppConfig:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_cfg = ModelConfig(**raw["model"])
    train_cfg = TrainConfig(**raw["train"])
    data_cfg = DataConfig(**raw["data"])

    return AppConfig(
        model=model_cfg,
        train=train_cfg,
        data=data_cfg,
    )
