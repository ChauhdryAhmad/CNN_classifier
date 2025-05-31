import numpy as np
from ..gpu_util import to_gpu, to_cpu, get_xp


# _{id(self.params[param_name])}
class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.cache = {}
        self.layer_id = id(self)  # Unique identifier for each layer instance

    def get_param_id(self, param_name):
        """Generate unique parameter identifier"""
        return f"{self.layer_id}_{param_name}"

    def to_gpu(self):
        """Move all parameters to GPU"""
        for name in self.params:
            self.params[name] = to_gpu(self.params[name])
        for name in self.grads:
            self.grads[name] = to_gpu(self.grads[name])
        return self

    def to_cpu(self):
        """Move all parameters to CPU"""
        for name in self.params:
            self.params[name] = to_cpu(self.params[name])
        for name in self.grads:
            self.grads[name] = to_cpu(self.grads[name])
        return self

    def forward(self, x):
        """Forward pass - to be implemented by each layer"""
        raise NotImplementedError

    def backward(self, dout):
        """Backward pass - to be implemented by each layer"""
        raise NotImplementedError

    def update_params(self, optimizer):
        """Update layer parameters using the optimizer"""
        for param_name in self.params:
            self.params[param_name] = optimizer.update(
                self.params[param_name], self.grads[f"d{param_name}"]
            )
