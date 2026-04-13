import torch
from torch.utils.data import Dataset

class LMDataset(Dataset):
    def __init__(self, tokens: list[int], seq_len: int, stride: int = 1):
        super().__init__()
        self.tokens = tokens
        self.seq_len = seq_len
        # stride 控制相邻样本窗口的距离；设为 seq_len 可避免大量重叠样本。
        self.stride = max(1, stride)

    def __len__(self):
        # 每个输入窗口后面还要多留 1 个 token 作为 next-token 标签。
        max_start = len(self.tokens) - self.seq_len - 1
        if max_start < 0:
            return 0
        return max_start // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        # x 是当前窗口，y 是整体右移 1 位的 next-token 标签。
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
