import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        torch.manual_seed(0)
        tokens = raw_dataset.split()
        X, Y = [], []
        for i in range(batch_size):
            start = torch.randint(len(tokens) - context_length, (1,))[0]
            X += [tokens[start : start + context_length]]
            Y += [tokens[start + 1: start + 1 + context_length]]
        return X, Y
