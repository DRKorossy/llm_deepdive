import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(
        self, 
        model_prediction: NDArray[np.float64], 
        ground_truth: NDArray[np.float64], 
        N: int, 
        X: NDArray[np.float64], 
        desired_weight: int
    ) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(
        self, X: NDArray[np.float64], 
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        W = initial_weights
        for _ in range(num_iterations):
            prediction = self.get_model_prediction(X, W)
            gradients = np.zeros_like(W)

            for j in range(len(W)):
                gradients[j] = self.get_derivative(
                    prediction,
                    Y,
                    len(X),
                    X,
                    j
                )
            W -= self.learning_rate * gradients
        return np.round(W, 5)
