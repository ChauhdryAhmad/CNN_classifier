import numpy as np
from .base import Layer


class MaxPool2D(Layer):
    def __init__(self, pool_size=(2, 2), stride=2):
        super().__init__()
        self.pool_size = (
            pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        )
        self.stride = stride

    def forward(self, x):
        """Forward pass for max pooling"""
        batch_size, in_height, in_width, in_channels = x.shape
        out_height = (in_height - self.pool_size[0]) // self.stride + 1
        out_width = (in_width - self.pool_size[1]) // self.stride + 1

        output = np.zeros((batch_size, out_height, out_width, in_channels))

        for b in range(batch_size):
            for c in range(in_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size[0]
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size[1]

                        region = x[b, h_start:h_end, w_start:w_end, c]
                        output[b, h, w, c] = np.max(region)

        self.cache = (x, output.shape)
        return output

    def backward(self, dout):
        """Backward pass for max pooling"""
        x, output_shape = self.cache
        batch_size, out_height, out_width, out_channels = output_shape
        dx = np.zeros_like(x)

        for b in range(batch_size):
            for c in range(out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size[0]
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size[1]

                        region = x[b, h_start:h_end, w_start:w_end, c]
                        max_val = np.max(region)

                        # Create mask of where the max values are
                        mask = region == max_val
                        dx[b, h_start:h_end, w_start:w_end, c] += (
                            mask * dout[b, h, w, c]
                        )

        return dx
