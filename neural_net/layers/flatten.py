import numpy as np
from .base import Layer


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Forward pass - flatten input"""
        self.cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        """Backward pass - reshape to original"""
        original_shape = self.cache
        return dout.reshape(original_shape)
