import numpy as np

"""
This file includes Config / Variable / Function class 
that support automatic differentiation.
"""


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
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


class Exponential(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, dy):
        x = self.input.data
        dx = dy * np.exp(x)
        return dx


"""
## test simple forwardprop / backprop
"""

f1 = Square()
f2 = Exponential()
f3 = Square()

x = Variable(np.array(0.5))
w1 = f1(x)
w2 = f2(w1)
y = f3(w2)

y.grad = np.array(1.0)
w2.grad = f3.backward(y.grad)
w1.grad = f2.backward(w2.grad)
x.grad = f1.backward(w1.grad)

print(x.grad)
