import numpy as np
from .layers.base import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Forward pass for ReLU"""
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout):
        """Backward pass for ReLU"""
        x = self.cache
        dx = dout * (x > 0)
        return dx


class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Forward pass for Softmax"""
        # Numerically stable softmax
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.cache = exps / np.sum(exps, axis=1, keepdims=True)
        return self.cache

    def backward(self, dout):
        """Backward pass for Softmax"""
        s = self.cache
        # Jacobian matrix of softmax
        jacobian = np.einsum("ij,jk->ijk", s, np.eye(s.shape[1])) - np.einsum(
            "ij,ik->ijk", s, s
        )
        return np.einsum("ijk,ik->ij", jacobian, dout)
