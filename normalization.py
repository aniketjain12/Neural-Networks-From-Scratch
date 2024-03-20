import numpy as np
import nnfs

nnfs.init()

layer_output = [[4.8, 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_output)
#print(np.sum(layer_output, axis=1, keepdims=True))#row
#print(np.sum(layer_output, axis=0))#column
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)


