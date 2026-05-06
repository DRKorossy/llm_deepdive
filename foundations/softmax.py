import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        _softmax = np.divide(np.exp(z - np.max(z)), np.sum(np.exp(z - np.max(z))))
        return np.round(_softmax, 4)
