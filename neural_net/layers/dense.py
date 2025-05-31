import numpy as np
from .base import Layer


class Dense(Layer):
    def __init__(self, units, input_dim=None):
        """
        Fully connected (dense) layer

        Args:
            units: Number of neurons in the layer
            input_dim: Dimension of input (optional, can be inferred)
        """
        super().__init__()
        self.units = units
        self.input_dim = input_dim

        # Initialize weights and bias
        self.params = {
            "W": None,  # Weight matrix
            "b": None,  # Bias vector
        }

        # Gradients
        self.grads = {"dW": None, "db": None}

    def initialize_params(self, input_dim):
        """Initialize parameters with He initialization"""
        scale = np.sqrt(2 / input_dim)
        self.params["W"] = np.random.randn(input_dim, self.units) * scale
        self.params["b"] = np.zeros((1, self.units))

    def forward(self, x):
        """
        Forward pass for dense layer: y = xW + b

        Args:
            x: Input data of shape (batch_size, input_dim)

        Returns:
            Output of shape (batch_size, units)
        """
        # Initialize parameters if first pass
        if self.params["W"] is None:
            self.initialize_params(x.shape[1])

        # Save input for backward pass
        self.cache = x

        # Compute forward pass
        return np.dot(x, self.params["W"]) + self.params["b"]

    def backward(self, dout):
        """
        Backward pass for dense layer

        Args:
            dout: Upstream derivative of shape (batch_size, units)

        Returns:
            dx: Gradient with respect to input x
        """
        x = self.cache

        # Compute gradients
        self.grads["dW"] = np.dot(x.T, dout)
        self.grads["db"] = np.sum(dout, axis=0, keepdims=True)

        # Compute gradient with respect to input
        dx = np.dot(dout, self.params["W"].T)

        # Ensure gradient shapes match parameter shapes
        if self.grads["dW"].shape != self.params["W"].shape:
            self.grads["dW"] = self.grads["dW"].reshape(self.params["W"].shape)
        if self.grads["db"].shape != self.params["b"].shape:
            self.grads["db"] = self.grads["db"].reshape(self.params["b"].shape)

        return dx

    def update_params(self, optimizer):
        """Update parameters using the optimizer"""
        self.params["W"] = optimizer.update(self.params["W"], self.grads["dW"])
        self.params["b"] = optimizer.update(self.params["b"], self.grads["db"])
