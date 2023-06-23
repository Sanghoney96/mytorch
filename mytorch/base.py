import numpy as np
import weakref

"""
This file includes Config / Variable / Function class 
that support automatic differentiation.
"""


class Variable:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("{} cannot be supported.".format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones(self.data.shape)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


def as_ndarray(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_ndarray(y))
        output.set_creator(self)
        self.output = output
        self.input = input  # Save input for backprop
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, dy):
        raise NotImplementedError()


"""
## functions
"""


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, dy):
        x = self.input.data
        dx = dy * 2 * x
        return dx


def square(x):
    return Square()(x)


class Exponential(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, dy):
        x = self.input.data
        dx = dy * np.exp(x)
        return dx


def exp(x):
    return Exponential()(x)


"""
## Test : simple forwardprop / backprop
"""

x = Variable(np.array(0.5))
w1 = square(x)
w2 = exp(w1)
y = square(w2)

y.backward()
print(x.grad)
