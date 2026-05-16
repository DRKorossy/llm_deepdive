import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.attention_dim = attention_dim
        self.W_k = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        Q = self.W_q(embedded)
        K = self.W_k(embedded)
        V = self.W_v(embedded)

        attention_scores = Q @ K.mT / self.attention_dim ** 0.5
        mask = torch.tril(torch.ones_like(attention_scores))
        masked_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        softmaxed = nn.functional.softmax(masked_scores, dim=2)
        
        return torch.round(softmaxed @ V, decimals=4)



