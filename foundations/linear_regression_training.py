import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # model_prediction
        # ground_truth
        # N: number of data points
        # X: input array
        # desired_weight: index of the weight to update

        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
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
        N,M = X.shape
        weights = initial_weights
        for i in range(num_iterations):
            prediction = self.get_model_prediction(X, weights)
            derivatives = np.zeros(M)
            for j in range(M):
                derivatives[j] = self.get_derivative(prediction, Y, N, X, j)
            weights = weights - (self.learning_rate * derivatives)
        return np.round(weights, 5)
            
        
