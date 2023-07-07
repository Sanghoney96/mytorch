if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from mytorch import Variable
from mytorch.functions import reshape, matmul

x0 = Variable(np.random.randn(2, 3))
x1 = Variable(np.random.randn(3, 4))

y = matmul(x0, x1)

y.backward()

print(x0.grad.shape)
print(x1.grad.shape)
