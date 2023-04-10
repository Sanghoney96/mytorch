import numpy as np

class Flatten:
    def __call__(self, matrices):
        self.matrix_shape = matrices.shape
        vector = np.array([[]])

        for i in range(matrices.shape[0]):
            reshaped_matrix = np.reshape(
                matrices[i], (1, self.matrix_shape[1]*self.matrix_shape[2]*self.matrix_shape[3]))
            vector = np.column_stack((vector, reshaped_matrix))
        return vector

class Linear:

class Dropout: