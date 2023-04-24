import numpy as np

"""
Includes loss functions.
"""


class MSE:
    """
    Mean Squared Error loss function.
    L2 norm of the error between label and prediction.
    """

    def __call__(self, Al, Y):
        batch_size = Al.shape[1]
        loss = 0.5 * np.sum((Al - Y) ** 2) / batch_size
        return loss

    def backprop(self, Al, Y):
        batch_size = Al.shape[1]
        dAl = (Al - Y) / batch_size
        return dAl


class CategorialCrossentropy:
    """
    Cross Entropy loss function for softmax output layer.
    We assume that each label and prediction of data is a (one-hot) vector.
    So the shape of Al and Y is (num_input_neurons, num_data).
    """

    def __call__(self, Al, Y):
        batch_size = Al.shape[1]
        loss = -np.sum(Y * np.log(Al)) / batch_size
        self.y = Y
        return loss


class BinaryCrossentropy:
    """
    Cross Entropy loss function for sigmoid output layer.
    We assume that each label and prediction of data is a scalar, not a (one-hot) vector.
    So the shape of al and y is (1, num_data).
    """

    def __call__(self, al, y):
        batch_size = al.shape[1]
        loss = -np.sum(y * np.log(al) + (1 - y) * np.log(1 - al)) / batch_size
        return loss

    def backprop(self, al, y):
        batch_size = al.shape[1]
        dal = (-y / al + (1 - y) / (1 - al)) / batch_size
        return dal
