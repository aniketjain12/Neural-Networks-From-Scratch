import math
import numpy as np
layer_output = [4.8, 1.21, 2.385]
exp_values = np.exp(layer_output)

norm_values = exp_values / np.sum(exp_values)
print(norm_values)
print(sum(norm_values))

'''
layer_output = [4.8, 1.21, 2.385]
E = math.e
exp_values = []
for output in layer_output:
    exp_values.append(E**output) # softmax = e^x(output)
print(exp_values)

norm_base = sum(exp_values)#normalization
norm_values = []
for values in exp_values:
    norm_values.append(values / norm_base) 

print(norm_values)
print(sum(norm_values))
'''