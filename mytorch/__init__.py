"""
This file is activated when import mytorch.
You can import class like this.
"from mytorch import Variable"
"""

from mytorch.base import Variable
from mytorch.base import Function
from mytorch.base import using_config
from mytorch.base import no_backward
from mytorch.base import as_array
from mytorch.base import as_variable
from mytorch.base import setup_operator

setup_operator()
