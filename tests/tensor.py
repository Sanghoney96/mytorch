if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from mytorch import Variable
from mytorch.functions import reshape

x = Variable(np.random.rand(2, 3, 4))

x_T = x.transpose(0, 2, 1)

x_T.backward()


print(x_T.shape)
print(x.grad)
