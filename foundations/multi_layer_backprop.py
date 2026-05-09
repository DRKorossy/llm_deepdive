import numpy as np
from typing import List


class Solution:
    def round_clean(self, arr, decimals=0):
        arr = np.round(arr, decimals)
        arr[np.isclose(arr, 0.0)] = 0.0
        return arr.tolist()


    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float],
    ) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        # forward pass
        z1 = np.array(x, dtype=np.float64) @ np.array(W1, dtype=np.float64).transpose() + np.array(b1, dtype=np.float64)
        a1 = np.maximum(0, z1)
        z2 = np.array(a1, dtype=np.float64) @ np.array(W2, dtype=np.float64).transpose() + np.array(b2, dtype=np.float64)
        loss = np.mean((z2 - y_true) ** 2)

        # backward pass
        dLdz2 = 2 * (z2 - y_true) / z2.size
        dLdW2 =  np.outer(dLdz2, a1)
        dLdb2 = dLdz2
        dLda1 = dLdz2 @ W2
        dLdz1 = dLda1 * (z1 > 0)
        dLdW1 = np.outer(dLdz1, x)
        dLdb1 = dLdz1

        return {
            "loss": round(loss, 4),
            "dW1": self.round_clean(dLdW1, 4),
            "db1": self.round_clean(dLdb1, 4),
            "dW2": self.round_clean(dLdW2, 4),
            "db2": self.round_clean(dLdb2, 4)
        }
