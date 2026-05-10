import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        mu = np.mean(x)
        sigma = np.std(x)
        eps = 1e-5

        x_hat = (x - mu)/np.sqrt((sigma**2 + eps)) * gamma + beta
        return np.round(x_hat, 5)
    
