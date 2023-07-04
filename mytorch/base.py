import numpy as np
import mytorch
import weakref
from contextlib import contextmanager

"""
This file includes Config / Variable / Function class 
that support automatic differentiation.
"""


"""
## Variable class
"""


class Variable:
    def __init__(self, data, name=None):
        """
        Restore input data as ndarray.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("{} type cannot be used.".format(type(data)))

        self.name = name
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    __array_priority__ = 1972

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        """Print variable that has a shape of ndarray"""
        if self.data is None:
            return "Variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

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
            self.grad = Variable(np.ones(self.data.shape))

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

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return mytorch.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], tuple) or axes[0] is None:
                axes = axes[0]
        return mytorch.functions.transpose(self, axes)

    @property
    def T(self):
        return mytorch.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return mytorch.functions.sum(self, axis, keepdims)


"""
## Config class and utility functions 
"""


class Config:
    enable_backward = True


@contextmanager
def using_config(flag, bool):
    old_bool = getattr(Config, flag)
    setattr(Config, flag, bool)
    try:
        yield
    finally:
        setattr(Config, flag, old_bool)


def no_backward():
    """
    Turn off the backpropagation mode when inference is ongoing.
    """
    return using_config("enable_backward", False)


def as_variable(x):
    if isinstance(x, Variable):
        return x
    return Variable(x)


def as_ndarray(x):
    """
    Convert scalar(float64 or 32) to ndarray.
    """
    if np.isscalar(x):
        return np.array(x)
    return x


"""
## Function class
"""


class Function:
    """
    Function class for implementing functions.
    """

    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

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
## operators
"""


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, dy):
        return dy, dy


def add(x0, x1):
    x1 = as_ndarray(x1)
    return Add()(x0, x1)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, dy):
        return dy, -dy


def sub(x0, x1):
    x1 = as_ndarray(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_ndarray(x1)
    return Sub()(x1, x0)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, dy):
        x0, x1 = self.inputs
        return dy * x1, dy * x0


def mul(x0, x1):
    return Mul()(x0, x1)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, dy):
        x0, x1 = self.inputs
        dx0 = dy / x1
        dx1 = dy * (-x0 / x1**2)
        return dx0, dx1


def div(x0, x1):
    x1 = as_ndarray(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_ndarray(x1)
    return Div()(x1, x0)


class Neg(Function):
    def forward(self, x):
        y = -x
        return y

    def backward(self, dy):
        dx = -dy
        return dx


def neg(x):
    return Neg()(x)


class Pow(Function):
    def __init__(self, a):
        self.a = a

    def forward(self, x):
        y = x**self.a
        return y

    def backward(self, dy):
        x = self.inputs
        a = self.a
        dx = a * x ** (a - 1) * dy
        return dx


def pow(x, a):
    return Pow(a)(x)


def setup_operator():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__neg__ = neg
    Variable.__pow__ = pow
