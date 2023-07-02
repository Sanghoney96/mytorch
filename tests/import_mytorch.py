if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from mytorch import Variable
import mytorch.functions as F

"""
## Test : functions
"""

x = Variable(np.array(2.0))
y1 = F.tanh(x)
y1.backward()

dx = x.grad

print(dx)
