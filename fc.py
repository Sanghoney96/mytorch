import numpy as np

class Flatten:
    def tensor2vector(self, matrices):
        vector = np.array([[]])
        self.matrix_shape = matrices.shape

        for i in range(matrices.shape[0]):
            reshaped_matrix = np.reshape(
                matrices[i], (1, self.matrix_shape[1] * self.matrix_shape[2] * self.matrix_shape[3]))
            vector = np.vstack((vector, reshaped_matrix))
        return vector

class Linear:

class Dropout:
    