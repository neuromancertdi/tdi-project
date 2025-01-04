from random import randint
from math import e

def sigmoid(z):
    sigmoid_out = 1 / (1 + e ** (-z))
    return sigmoid_out

def sigmoid_der(sigmoid_out):  # derivative of sigmoid operation
    return sigmoid_out * (1 - sigmoid_out)


def init_weights_matrix(inp_size, out_size):
    # first axis - next layer neurons, second - current layer neurons
    weights_matrix = [[] for _ in range(out_size)]

    for j in range(out_size):  # number of output neuron
        for i in range(inp_size + 1):  # # number of input neuron, +1 - bias
            weights_matrix[j].append((randint(0, 200) / 100 - 1) / randint(1, 100))

    return weights_matrix


def matrix_by_vector(weights_matrix, input_vector, bias=True):
    z = [0 for _ in range(len(weights_matrix))]

    for j in range(len(weights_matrix)):  # number of output neuron
        for i in range(len(input_vector)):  # number of input neuron
            z[j] += weights_matrix[j][i] * input_vector[i]

        if bias: z[j] += weights_matrix[j][-1]

    return z


def vector_activation(z):
    output = [sigmoid(z[i]) for i in range(len(z))]
    return output


def output_derivative(output_vector, target_vertor):
    out_der = [0 for _ in range(len(output_vector))]

    for i in range(len(output_vector)):
        out_der[i] = (output_vector[i] - target_vertor[i]) * sigmoid_der(output_vector[i]) * 2

    return out_der


def hidden_derivative(current_layer, weights_matrix, next_layer_der):
    current_layer_der = [0 for _ in range(len(current_layer))]

    for i in range(len(current_layer)):
        next_layer_der_sum = 0

        for j in range(len(next_layer_der)):
            next_layer_der_sum += weights_matrix[j][i] * next_layer_der[j]

        current_layer_der[i] = sigmoid_der(current_layer[i]) * next_layer_der_sum

    return current_layer_der


def update_weights_matrix(input_vector, weights_matrix, out_der, lr, bias=True):
    for j in range(len(out_der)):  # number of output neuron
        for i in range(len(input_vector)):  # number of input neuron
            weights_matrix[j][i] -= lr * input_vector[i] * out_der[j]

        if bias: weights_matrix[j][-1] -= lr * out_der[j]