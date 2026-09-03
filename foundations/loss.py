import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        epsilon = 1e-7
        n = len(y_true)
        loss = -(np.sum(np.dot(y_true, np.log(y_pred+epsilon))) + np.sum(np.dot(1-y_true, np.log(1-y_pred+epsilon))))/n
        return round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        epsilon = 1e-7
        y_pred = y_pred + epsilon
        n = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred))/n
        return round(loss, 4)
