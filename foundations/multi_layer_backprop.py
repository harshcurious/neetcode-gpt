import numpy as np
from typing import List


class Solution:
    def _relu(self, a: NDArray(np.float64)):
        zero = np.zeros_like(a)
        return np.maximum(zero, a)
    def _relu_derivative(self, a):
        return np.where(a > 0, 1, 0)

    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        # Converting to numpy
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        n = len(y_true)

        # forward pass
        z1 = x @ W1.T + b1
        y1 = self._relu(z1)
        y_hat = y1 @ W2.T + b2
        # y_hat = self._relu(z_2)

        # Loss
        loss = np.mean((y_hat - y_true)**2)

        # Gradients
        dy_hat = 2 * np.mean(y_hat - y_true, keepdims=1)
        # dz2 = dy_hat * self._relu_derivative(z2)
        dW2 = (dy_hat.T) * y1.reshape(1,-1)
        db2 = dy_hat

        dy1 = dy_hat.T * W2
        dz1 = dy1 * self._relu_derivative(z1)

        dW1 = dz1.T * x.reshape(1,-1)
        db1 = dz1.flatten()

        return {
            "loss": np.round(loss,4),
            "dW1": np.round(dW1, 4),
            "db1": np.round(db1, 4),
            "dW2": np.round(dW2, 4),
            "db2": np.round(db2, 4),
        }



