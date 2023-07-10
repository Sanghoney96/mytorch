import math
import numpy as np
from mytorch import Parameter


class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target  # Model or Layer
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        # preprocess the parameters using the function f if needed
        for f in self.hooks:
            f(params)

        # update parameters
        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data = param.data - self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}  # dictionary to store velocity vectors

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = Parameter(np.zeros_like(param.data))

        v = self.vs[v_key]
        v = self.momentum * v - self.lr * param.grad.data
        param.data += v


class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}
        self.iter = 0

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    def update_one(self, param):
        key = id(param)
        if key not in self.ms:
            self.ms[key] = Parameter(np.zeros_like(param.data))
            self.vs[key] = Parameter(np.zeros_like(param.data))

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad

        m_hat = m / (1 - beta1**self.t)
        v_hat = v / (1 - beta2**self.t)

        param.data -= self.alpha * m_hat / (np.sqrt(v_hat) + eps)
