import numpy as np
from mytorch import Function, as_variable
from mytorch import utils

"""
## tensor operation
"""


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, dy):
        return reshape(dy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, dy):
        if self.axes is None:
            return transpose(dy)

        inv_axes = tuple(np.array(self.axes).argsort())

        return transpose(dy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, dy):
        dx = sum_to(dy, self.x_shape)
        return dx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, dy):
        dx = broadcast_to(dy, self.x_shape)
        return dx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, dy):
        dy = utils.reshape_sum_backward(dy, self.x_shape, self.axis, self.keepdims)
        dx = broadcast_to(dy, self.x_shape)

        return dx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, dy):
        x, W = self.inputs
        dx = matmul(dy, W.T)
        dW = matmul(x.T, dy)
        return dx, dW


def matmul(x, W):
    return MatMul()(x, W)


"""
## Activation functions
"""


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, dy):
        x, W, b = self.inputs

        if b.data is None:
            db = None
        else:
            db = sum_to(dy, b.shape)

        dx = matmul(dy, W.T)
        dW = matmul(x.T, dy)
        return dx, dW, db


def linear(x, W, b=None):
    return Linear()(x, W, b)


class ReLU(Function):
    def forward(self, x):
        self.mask = np.where(x > 0, 1.0, 0.0)
        y = np.maximum(x, 0.0)
        return y

    def backward(self, dy):
        dx = dy * self.mask
        return dx


def relu(x):
    return ReLU()(x)


class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, dy):
        y = self.outputs[0]()
        dx = dy * y * (1 - y)
        return dx


def sigmoid(x):
    return Sigmoid()(x)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = np.exp(x)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, dy):
        y = self.outputs[0]()
        dx = y * dy
        sumdx = dx.sum(axis=self.axis, keepdims=True)
        dx -= y * sumdx
        return dx


def softmax(x, axis=1):
    return Softmax(axis)(x)


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


"""
## Loss functions
"""


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        error = x0 - x1
        y = (error**2).sum() / len(error)
        return y

    def backward(self, dy):
        x0, x1 = self.inputs
        error = x0 - x1
        dx0 = dy * error * (2.0 / len(error))
        dx1 = -dx0
        return dx0, dx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)
