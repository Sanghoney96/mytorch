import numpy as np
from fc_initializer import init


class Flatten:
    """
    Flattens the 3D input for converting to 1D vector.
    Then stack into the variable 'vector' as a shape of column vector.
    Does not affect the batch size.
    """

    def __call__(self, matrices):
        self.matrix_shape = matrices.shape
        vector = np.array([[]])

        for i in range(matrices.shape[0]):
            reshaped_matrix = np.reshape(
                matrices[i],
                (1, self.matrix_shape[1] * self.matrix_shape[2] * self.matrix_shape[3]),
            )
            vector = np.column_stack((vector, reshaped_matrix))
        return vector


class Linear:
    """
    Applies a linear transformation to the incoming data for feeding activation function.
    It conducts backpropagation computation as well.
    """

    def __init__(self, in_features, out_features, init_method="normal"):
        self.in_features = in_features
        self.out_features = out_features
        self.W, self.b = init(self.in_features, self.out_features, init_method="normal")

    def __call__(self, A_prev):
        Z = np.dot(self.W, A_prev) + self.b
        self.A_prev = A_prev
        return Z

    def backward(self, dout):
        pass


class Dropout:
    """
    During training, randomly zeroes some of the elements of the input tensor
    with probability p using samples from a Bernoulli distribution.
    Each channel will be zeroed out independently on every forward call.
    Furthermore, the outputs are scaled by a factor of 1/(1-p) during training.
    This means that during evaluation the module simply computes an identity function.
    """

    def __init__(self, ratio=0.5):
        self.ratio = ratio
