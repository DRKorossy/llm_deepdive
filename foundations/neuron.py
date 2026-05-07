import numpy as np
from numpy.typing import NDArray


class Solution:
    def relu(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(np.zeros_like(x), x)
    
    def sigmoid(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1/(1+np.exp(-x))

    
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        # return round(your_answer, 5)
        activation_map = {"relu": self.relu, "sigmoid": self.sigmoid}
        activation_function = activation_map[activation]
        pre_activation = np.dot(x, w) + b
        forward_value = activation_function(pre_activation)
        return round(forward_value, 5)
        
