import numpy as np


class sigmoid:
    """
    Sigmoid activation layer.
    """

    def __call__(self, Z):
        Z = Z.astype(np.float32)
        A = 1 / (1 + np.exp(-Z))
        self.A = A
        return A

    def backprop(self, dA):
        dZ = dA * self.A * (1 - self.A)
        return dZ


class softmax:
    """
    Softmax activation layer.
    You should use this activation 'only' for output layer!!
    """

    def __call__(self, Z):
        Z = Z.astype(np.float32)
        Al = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        self.Al = Al
        return Al

    def backprop(self):
        batch_size = self.Al.shape[1]
        dZ = (self.Al - self.y) / batch_size
        return dZ


class relu:
    """
    ReLU activation layer.
    """

    def __call__(self, Z):
        """
        Make the mask that memorizes the position of activated values.
        Then put on input Z to compute activated output A.
        """
        Z = Z.astype(np.float32)
        self.mask = (Z > 0).astype(np.int)
        A = Z * self.mask
        self.A = A
        return A

    def backprop(self, dA):
        """
        Put mask on input dA to compute activated output dZ.
        """
        dZ = dA * self.mask
        return dZ
