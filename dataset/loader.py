import torch
from torch.utils.data import Dataset

class LMDataset(Dataset):
    def __init__(self, tokens: list[int], seq_len: int):
        super().__init__()
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        # 减去 seq_len 确保取最后一个样本时不会越界
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        # 截取长度为 seq_len 的输入序列
        x = self.tokens[idx : idx + self.seq_len]
        # 截取向后偏移 1 位的标签序列
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
