import numpy as np
import weakref

"""
This file includes Config / Variable / Function class 
that support automatic differentiation.
"""


class Variable:
    def __init__(self, data):
        """
        Restore input data as ndarray.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("{} type cannot be used.".format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        """
        Refer function as creator of output variable.
        """
        self.creator = func

    def backward(self):
        """
        Backpropagate from the output(dy/dy=1) to the input.
        """
        if self.grad is None:
            self.grad = np.ones(self.data.shape)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            dys = [output.grad for output in f.outputs]
            dxs = f.backward(*dys)
            if not isinstance(dxs, tuple):
                dxs = (dxs,)

            for x, dx in zip(f.inputs, dxs):
                if x.grad is None:
                    x.grad = dx
                else:
                    x.grad = x.grad + dx

                if x.creator is not None:
                    funcs.append(x.creator)

    def cleargrad(self):
        self.grad = None


def as_ndarray(x):
    """
    Convert scalar(float64 or 32) to ndarray.
    """
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    """
    Function class for implementing functions.
    """

    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_ndarray(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs  # Save inputs/outputs for backprop
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, dys):
        raise NotImplementedError()


"""
## functions
"""


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y  # return tuple

    def backward(self, dy):
        return dy, dy


def add(x0, x1):
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, dy):
        x = self.inputs[0].data
        dx = 2 * x * dy
        return dx


def square(x):
    return Square()(x)


class Exponential(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, dy):
        x = self.inputs[0].data
        dx = dy * np.exp(x)
        return dx


def exp(x):
    return Exponential()(x)


"""
## Test : variable overwritting
"""

x = Variable(np.array(2.0))
y = add(x, x)
y.backward()
print(x.grad)  # 2.0

x.cleargrad()  # 미분값 초기화
y = add(add(x, x), x)
y.backward()
print(x.grad)  # 3.0
