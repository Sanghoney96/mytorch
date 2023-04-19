import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def __call__(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def __call__(self, params, grads):
        self.v = {}
        for key, param in params.items():
            self.v[key] = np.zeros((param.shape[0], param.shape[1])).astype(np.float32)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] -= self.v[key]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epoch = 0

    def __call__(self, params, grads):
        self.m, self.v = {}, {}

        for key, param in params.items():
            self.m[key] = np.zeros((param.shape[0], param.shape[1])).astype(np.float32)
            self.v[key] = np.zeros((param.shape[0], param.shape[1])).astype(np.float32)

        self.epoch += 1

        for key in params.keys():
            self.m[key] = (
                self.beta1 * self.m[key] + (1.0 - self.beta1) * grads[key]
            ) / (1.0 - self.beta1**self.epoch)
            self.v[key] = (
                self.beta2 * self.v[key] + (1.0 - self.beta2) * grads[key]
            ) / (1.0 - self.beta2**self.epoch)

            params[key] -= (self.lr / (np.sqrt(self.v[key]) + 1e-7)) * self.m[key]
