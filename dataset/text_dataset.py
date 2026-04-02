from pathlib import Path


def load_text(path: str | Path) -> str:
    path = Path(path)
    return path.read_text(encoding="utf-8")