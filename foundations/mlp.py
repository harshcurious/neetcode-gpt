import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def _relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        zero = np.zeros_like(z)
        return np.maximum(zero, z)

    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        l = len(weights)
        h = x
        for i in range(l):
            if i == l-1:
                h = h @ weights[i] + biases[i]
            else:
                h = self._relu(h @ weights[i] + biases[i])
            print(h)

        return np.round(h, 5)

