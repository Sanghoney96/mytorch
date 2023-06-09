"""
This file is activated when import mytorch.
You can import class like this.
"from mytorch import Variable"
"""

from mytorch.base import Variable
from mytorch.base import Function
from mytorch.base import Parameter
from mytorch.base import using_config
from mytorch.base import no_backward
from mytorch.base import as_array
from mytorch.base import as_variable
from mytorch.base import setup_operator
from mytorch.base import Config

from mytorch.layers import Layer
from mytorch.models import Model

import mytorch.functions
import mytorch.layers
import mytorch.utils

setup_operator()
