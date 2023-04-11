import numpy as np

"""
Contains the methods to set the initial random weights of FC layers.
"""


def zeros(in_features, out_features):
    """
    Creates initial weight matrix(out_features, in_features)
    and initial bias vector(out_features, 1).
    It generates weight matrixes and bias vectors initialized to 0.
    """
    init_W = np.zeros((out_features, in_features))
    init_b = np.zeros((out_features, 1))
    return init_W, init_b


def normal(in_features, out_features):
    """
    Creates initial weight matrix(out_features, in_features)
    and initial bias vector(out_features, 1).
    It generates weight matrixes with a normal distribution.
    """
    init_W = np.random.randn(out_features, in_features)
    init_b = np.zeros((out_features, 1))
    return init_W, init_b


def he_normal(in_features, out_features):
    """
    Creates initial weight matrix(out_features, in_features)
    and initial bias vector(out_features, 1).
    It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in)
    where fan_in is the number of input units in the weight matrix.
    """
    init_W = np.random.randn(out_features, in_features) * np.sqrt(2 / in_features)
    init_b = np.zeros((out_features, 1))
    return init_W, init_b


def xavier_normal(in_features, out_features):
    """
    Creates initial weight matrix(out_features, in_features)
    and initial bias vector(out_features, 1).
    It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out))
    where fan_in/out is the number of input/output units in the weight matrix.
    """
    init_W = np.random.randn(out_features, in_features) * np.sqrt(
        2 / (in_features + out_features)
    )
    init_b = np.zeros((out_features, 1))
    return init_W, init_b


def init(in_features, out_features, init_method="zeros"):
    """
    Determines initialization method and return initialized weight and bias.
    """
    if init_method == "zeros":
        W, b = zeros(in_features, out_features)
    elif init_method == "normal":
        W, b = normal(in_features, out_features)
    elif init_method == "he":
        W, b = he_normal(in_features, out_features)
    elif init_method == "xavier":
        W, b = xavier_normal(in_features, out_features)

    return W, b
