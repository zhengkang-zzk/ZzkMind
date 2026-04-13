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
    moe_aux_loss_weight: float = 0.01


@dataclass
class DataConfig:
    train_path: str
    val_path: str


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
