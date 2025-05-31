import numpy as np
from .layers import Conv2D, MaxPool2D, Flatten, Dense
from .activations import ReLU, Softmax


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.loss = None
        self.optimizer = None

    def to_gpu(self):
        """Move entire network to GPU"""
        for layer in self.layers:
            if hasattr(layer, "to_gpu"):
                layer.to_gpu()
        return self

    def to_cpu(self):
        """Move entire network to CPU"""
        for layer in self.layers:
            if hasattr(layer, "to_cpu"):
                layer.to_cpu()
        return self

    def forward(self, x):
        """Forward pass through all layers"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        """Backward pass through all layers"""
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def compile(self, loss, optimizer):
        """Configure loss and optimizer"""
        self.loss = loss
        self.optimizer = optimizer

    def train_step(self, x, y):
        """Single training step"""
        # Forward pass
        output = self.forward(x)
        loss = self.loss.forward(output, y)

        # Backward pass
        dout = self.loss.backward()
        self.backward(dout)

        # Update parameters
        for layer in self.layers:
            if hasattr(layer, "update_params"):
                for param_name in layer.params:
                    grad = layer.grads[f"d{param_name}"]
                    param = layer.params[param_name]

                    # print(
                    #     f"Updating {param_name} with shape {param.shape} and grad shape {grad.shape}"
                    # )
                    # Ensure gradient matches parameter shape
                    if grad.shape != param.shape:
                        grad = grad.reshape(param.shape)

                    param_id = layer.get_param_id(param_name)

                    # Update parameter
                    layer.params[param_name] = self.optimizer.update(
                        param, param_id, grad
                    )

        return loss

    def predict(self, x):
        """Make predictions"""
        return self.forward(x)

    def save_weights(self, path):
        """Save model weights to file"""
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "params"):
                weights[f"layer_{i}"] = {
                    param: val.tolist() for param, val in layer.params.items()
                }
        import json

        with open(path, "w") as f:
            json.dump(weights, f)

    def load_weights(self, path):
        """Load model weights from file"""
        import json

        with open(path, "r") as f:
            weights = json.load(f)

        for i, layer in enumerate(self.layers):
            if f"layer_{i}" in weights:
                for param, val in weights[f"layer_{i}"].items():
                    layer.params[param] = np.array(val)
