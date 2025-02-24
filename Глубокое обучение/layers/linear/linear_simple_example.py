from tensor import *

inputs = [[2, 1], [1, 3]]  # in_features = 2
weights = [[0.1, 0.2], [0.3, 0.4]]  # out_features = 2
bias = [[0.1, 0.1]]

z = matrix_by_matrix(inputs, weights)  # out_features = 2
print(z)

expanded_bias = expand_item(item=bias[0], dim=0, copies_num=2)
print(expanded_bias)

z = add_matrixes(z, expanded_bias)
print(z)

h = sigmoid(z)
print(h)