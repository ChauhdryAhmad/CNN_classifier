import numpy as np
from .base import Layer
from neural_net.utils import im2col, col2im
from ..gpu_util import get_xp


class Conv2D(Layer):
    def __init__(self, num_filters, kernel_size, stride=1, padding=0):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding
        self.input_channels = None
        self.params = {"W": None, "b": None}

    def forward(self, x):
        """Forward pass using im2col optimization"""
        xp = get_xp(x)
        batch_size, h, w, c = x.shape

        # Initialize weights if first pass
        if self.params["W"] is None:
            self.input_channels = c
            scale = xp.sqrt(2 / (self.kernel_size[0] * self.kernel_size[1] * c))
            self.params["W"] = (
                xp.random.randn(
                    self.num_filters,
                    self.kernel_size[0],
                    self.kernel_size[1],
                    self.input_channels,
                )
                * scale
            )
            self.params["b"] = xp.zeros((1, 1, 1, self.num_filters))

        # Apply padding if needed
        if self.padding > 0:
            x_padded = xp.pad(
                x,
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                ),
                mode="constant",
            )
        else:
            x_padded = x

        # Calculate output dimensions
        out_h = (h + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size[1]) // self.stride + 1

        # Convert images to column matrix
        x_col = im2col(
            x_padded, self.kernel_size, self.stride, 0
        )  # shape: (batch*out_h*out_w, k_h*k_w*c)

        # Reshape weights for matrix multiplication
        W_col = (
            self.params["W"].reshape(self.num_filters, -1).T
        )  # shape: (k_h*k_w*c, num_filters)

        # Perform convolution via matrix multiplication
        out = x_col @ W_col  # shape: (batch*out_h*out_w, num_filters)

        # Add bias and reshape
        out = out.reshape(batch_size, out_h, out_w, self.num_filters) + self.params["b"]

        # Store for backward pass
        self.cache = (x, x_col, W_col)
        return out

    def backward(self, dout):
        """Backward pass using col2im"""
        xp = get_xp(dout)
        x, x_col, W_col = self.cache
        batch_size, h, w, c = x.shape
        _, out_h, out_w, num_filters = dout.shape

        # Gradient of bias
        self.grads["db"] = xp.sum(dout, axis=(0, 1, 2), keepdims=True)

        # Reshape dout for matrix multiplication
        dout_reshaped = dout.reshape(
            -1, num_filters
        )  # shape: (batch*out_h*out_w, num_filters)

        # Gradient of weights
        dW = dout_reshaped.T @ x_col  # shape: (num_filters, k_h*k_w*c)
        self.grads["dW"] = dW.reshape(self.params["W"].shape)

        # Gradient of input
        dx_col = dout_reshaped @ W_col.T  # shape: (batch*out_h*out_w, k_h*k_w*c)
        dx = col2im(dx_col, x.shape, self.kernel_size, self.stride, self.padding)

        # Ensure gradient shapes match parameter shapes
        if self.grads["dW"].shape != self.params["W"].shape:
            self.grads["dW"] = self.grads["dW"].reshape(self.params["W"].shape)
        if self.grads["db"].shape != self.params["b"].shape:
            self.grads["db"] = self.grads["db"].reshape(self.params["b"].shape)

        return dx
