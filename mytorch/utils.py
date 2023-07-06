import numpy as np
from mytorch import Variable, as_variable


def reshape_sum_backward(dy, x_shape, axis, keepdims):
    """
    Reshape gradient appropriately for backward method of mytorch.functions.sum.
    It returns gradient variable which is reshaped appropriately.
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        dy_shape = list(dy.shape)
        for a in sorted(actual_axis):
            dy_shape.insert(a, 1)
    else:
        dy_shape = dy.shape

    dy = dy.reshape(dy_shape)  # reshape
    return dy


def sum_to(x, shape):
    """
    Sum elements along axes to output an array of a given shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y
