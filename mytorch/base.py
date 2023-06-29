import numpy as np
import weakref
from contextlib import contextmanager

"""
This file includes Config / Variable / Function class 
that support automatic differentiation.
"""


class Config:
    enable_backward = True


@contextmanager
def using_config(flag, bool):
    """
    Turn off the backpropagation mode when inference is ongoing.
    """
    old_bool = getattr(Config, flag)
    setattr(Config, flag, bool)
    try:
        yield
    finally:
        setattr(Config, flag, old_bool)


def no_backward():
    return using_config("enable_backward", False)


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
        self.generation = 0

    def set_creator(self, func):
        """
        Refer function as creator of output variable.
        """
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        """
        Backpropagate from the output(dy/dy=1) to the input.
        """
        if self.grad is None:
            self.grad = np.ones(self.data.shape)

        funcs = []
        seen_set = set()

        # add functions in funcs and sort based on the generation of functions.
        def add_func(f):
            if f not in seen_set:
                seen_set.add(f)
                funcs.append(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            dys = [output().grad for output in f.outputs]
            dxs = f.backward(*dys)
            if not isinstance(dxs, tuple):
                dxs = (dxs,)

            for x, dx in zip(f.inputs, dxs):
                if x.grad is None:
                    x.grad = dx
                else:
                    x.grad = x.grad + dx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

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

        if Config.enable_backward:
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs  # Save inputs/outputs for backprop
            self.outputs = [
                weakref.ref(output) for output in outputs
            ]  # doesn't count reference of output

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
## Test : retain_grad
"""

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))

t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad, t.grad)
print(x0.grad, x1.grad)
