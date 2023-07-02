if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from mytorch import Variable

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
a = Variable(np.array([[10, 20, 30], [40, 50, 60]]))

y = x * a

y.backward(retain_grad=True)
print(y.grad)
print(x.grad)
print(a.grad)
