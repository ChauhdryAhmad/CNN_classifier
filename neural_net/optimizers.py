import numpy as np


class SGD:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def update(self, param, grad):
        """Update parameters with momentum"""
        param_id = id(param)
        if param_id not in self.velocities:
            self.velocities[param_id] = np.zeros_like(grad)

        # Update velocity
        self.velocities[param_id] = (
            self.momentum * self.velocities[param_id] - self.lr * grad
        )
        return param + self.velocities[param_id]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, param, param_id, grad):
        """Adam parameter update"""
        if param_id not in self.m:
            # print(f"Initializing m and v for param_id: {param_id}")
            self.m[param_id] = np.zeros_like(param)
            self.v[param_id] = np.zeros_like(param)

        self.t += 1
        self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad**2)

        # print(
        #     f"Shape of m: {self.m[param_id].shape}, Shape of v: {self.v[param_id].shape}"
        # )
        # print(f"Shape of param: {param.shape}, Shape of grad: {grad.shape}")

        # Bias correction
        m_hat = self.m[param_id] / (1 - self.beta1**self.t)
        v_hat = self.v[param_id] / (1 - self.beta2**self.t)

        # Update parameters
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        # print(f"\033[92mShape of update: {update.shape}\033[92m")

        if update.shape != param.shape:
            update = update.reshape(param.shape)

        return param - update
