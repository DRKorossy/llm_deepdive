import numpy as np
from numpy.typing import NDArray

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        y_pred += 1e-7
        _bce = -np.mean(np.multiply(y_true, np.log(y_pred)) + np.multiply((1 - y_true), np.log(1 - y_pred)))
        return round(_bce, 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        y_pred += 1e-7
        _cce = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return round(_cce, 4)
        
