import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(
        self, 
        data: TensorType[int], 
        context_length: int, 
        batch_size: int
    ) -> Tuple[TensorType[int], TensorType[int]]:

        torch.manual_seed(0)
        X = [[] for _ in range(batch_size)]
        Y = list(X)
        for i in range(batch_size):
            start = torch.randint(data.size()[0] - context_length, (1,))[0]
            X[i] = data[start : start + context_length]
            Y[i] = data[start + 1 : start + 1 + context_length]
        print(X)
        print(Y)
        return (torch.stack(X), torch.stack(Y))
