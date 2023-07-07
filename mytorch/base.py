import weakref
from contextlib import contextmanager
import numpy as np
import mytorch


"""
This file includes Config / Variable / Function classes:
that support automatic differentiation.
"""

"""
## Config class
"""


class Config:
    enable_backprop = True


@contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_backward():
    """Turn off backpropagation for test."""
    return using_config("enable_backprop", False)


"""
## Variable class
"""


class Variable:
    def __init__(self, data, name=None):
        """
        Restore data and gradient as numpy.ndarray.
        """
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.name = name
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    __array_priority__ = 200

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

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

    def set_creator(self, func):
        """Refer function as creator of output variable."""
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        """Backpropagation from the output to the input."""
        if self.grad is None:
            self.grad = Variable(np.ones(self.data.shape))

        funcs = []
        seen_set = set()

        def add_func(f):
            """Add function to funcs list and sort by generation."""
            if f not in seen_set:
                seen_set.add(f)
                funcs.append(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            dys = [output().grad for output in f.outputs]  # output is weakref
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
## Function class and utility functions
"""


def as_variable(x):
    if isinstance(x, Variable):
        return x
    return Variable(x)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    """Function class for implementing functions."""

    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs  # save inputs for backpropagation
            self.outputs = [
                weakref.ref(output) for output in outputs
            ]  # doesn't hold output in memory

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
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, dy):
        dx0, dx1 = dy, dy
        if self.x0_shape != self.x1_shape:
            dx0 = mytorch.functions.sum_to(dx0, self.x0_shape)
            dx1 = mytorch.functions.sum_to(dx1, self.x1_shape)
        return dx0, dx1


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, dy):
        dx0, dx1 = dy, -dy
        if self.x0_shape != self.x1_shape:  # for broadcast
            dx0 = mytorch.functions.sum_to(dx0, self.x0_shape)
            dx1 = mytorch.functions.sum_to(dx1, self.x1_shape)
        return dx0, dx1


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return sub(x1, x0)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, dy):
        x0, x1 = self.inputs
        dx0 = dy * x1
        dx1 = dy * x0
        if x0.shape != x1.shape:
            dx0 = mytorch.functions.sum_to(dx0, x0.shape)
            dx1 = mytorch.functions.sum_to(dx1, x1.shape)
        return dx0, dx1


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, dy):
        x0, x1 = self.inputs
        dx0 = dy / x1
        dx1 = dy * (-x0 / x1**2)
        if x0.shape != x1.shape:  # for broadcast
            dx0 = mytorch.functions.sum_to(dx0, x0.shape)
            dx1 = mytorch.functions.sum_to(dx1, x1.shape)
        return dx0, dx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return div(x1, x0)


class Neg(Function):
    def forward(self, x):
        y = -x
        return y

    def backward(self, dy):
        dx = -dy
        return -dx


def neg(x):
    return Neg()(x)


class Pow(Function):
    def __init__(self, a):
        self.a = a

    def forward(self, x):
        y = x**self.a
        return y

    def backward(self, dy):
        (x,) = self.inputs
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
