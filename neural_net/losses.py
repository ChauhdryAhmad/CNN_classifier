import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.cache = {}

    def forward(self, y_pred, y_true):
        """Forward pass for cross-entropy loss"""
        m = y_pred.shape[0]
        # Clip values to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # One-hot encode if necessary
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]

        loss = -np.sum(y_true * np.log(y_pred)) / m
        self.cache = {"y_pred": y_pred, "y_true": y_true}
        return loss

    def backward(self):
        """Backward pass for cross-entropy loss"""
        y_pred = self.cache["y_pred"]
        y_true = self.cache["y_true"]
        m = y_pred.shape[0]
        return (y_pred - y_true) / m
