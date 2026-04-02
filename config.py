from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    max_position_embeddings: int
    dropout: float


@dataclass
class TrainConfig:
    batch_size: int
    lr: float
    max_steps: int
    device: str


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