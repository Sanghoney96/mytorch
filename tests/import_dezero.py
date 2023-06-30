if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from mytorch import Variable

"""
## Test : operator
"""

x = Variable(np.array(2.0))
y1 = 1.0 / x
print(y1)
