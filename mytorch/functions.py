import numpy as np
from mytorch.base import Function


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, dy):
        y = self.outputs[0]()
        dx = dy * (1 - y**2)
        return dx


def tanh(x):
    return Tanh()(x)
