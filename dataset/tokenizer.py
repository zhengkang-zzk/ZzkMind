from pathlib import Path
from transformers import XLMRobertaTokenizerFast


class HFLocalTokenizer:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir).expanduser().resolve()

        if not model_dir.is_dir():
            raise FileNotFoundError(f"本地 tokenizer 目录不存在: {model_dir}")

        spm_file = model_dir / "sentencepiece.bpe.model"
        tokenizer_json = model_dir / "tokenizer.json"

        if not spm_file.exists():
            raise FileNotFoundError(f"缺少文件: {spm_file}")

        if not tokenizer_json.exists():
            raise FileNotFoundError(f"缺少文件: {tokenizer_json}")

        self.model_dir = model_dir

        # 直接按本地文件构造，绕开 AutoTokenizer 的 repo_id 识别
        self.tk = XLMRobertaTokenizerFast(
            vocab_file=str(spm_file),
            tokenizer_file=str(tokenizer_json),
        )

    @property
    def vocab_size(self) -> int:
        return self.tk.vocab_size

    def encode(self, text: str) -> list[int]:
        return self.tk.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )