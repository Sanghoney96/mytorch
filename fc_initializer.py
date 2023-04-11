import numpy as np


def zeros(in_features, out_features):
    init_W = np.zeros((out_features, in_features))
    init_b = np.zeros((out_features, 1))
    return init_W, init_b


def normal(in_features, out_features):
    init_W = np.random.randn(out_features, in_features)
    init_b = np.zeros((out_features, 1))
    return init_W, init_b


def he_normal(in_features, out_features):
    init_W = np.random.randn(out_features, in_features) * np.sqrt(2 / in_features)
    init_b = np.zeros((out_features, 1))
    return init_W, init_b


def xavier_normal(in_features, out_features):
    init_W = np.random.randn(out_features, in_features) * np.sqrt(
        2 / (in_features + out_features)
    )
    init_b = np.zeros((out_features, 1))
    return init_W, init_b


def init(in_features, out_features, init_method="zeros"):
    if init_method == "zeros":
        W, b = zeros(in_features, out_features)
    elif init_method == "normal":
        W, b = normal(in_features, out_features)
    elif init_method == "he":
        W, b = he_normal(in_features, out_features)
    elif init_method == "xavier":
        W, b = xavier_normal(in_features, out_features)

    return W, b
