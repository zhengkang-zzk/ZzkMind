from pathlib import Path

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
    
    def encode(self, text:str) -> list[int]:
        return [self.stoi[ch] for ch in text]
    
    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
    
def build_tokenizer_from_file(path:str | Path) -> CharTokenizer:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return CharTokenizer(text)