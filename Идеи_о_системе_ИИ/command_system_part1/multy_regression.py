from random import randint
from math import e

def mse(outputs, targets):
    error = 0

    for i, output in enumerate(outputs):
        error += (output - targets[i]) ** 2

    return error / len(outputs)


def batch_mse(inputs, targets):
    error = 0

    for inp, tar in zip(inputs, targets):
        error += mse(inp, tar)

    return error / len(inputs)


def transpose_matrix(matrix):
    new_matrix = [[0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new_matrix[j][i] = matrix[i][j]

    return new_matrix


def logistic_function(z):
    return 1 / (1 + e ** (-z))



class LogisticRegression:
    def __init__(self, features_num):
        # +1 for bias, bias is last weight
        self.weights = [randint(-100, 100) / 100 for _ in range(features_num + 1)]


    def forward(self, input_features):
        output = 0

        for i, feature in enumerate(input_features):
            output += self.weights[i] * feature

        return logistic_function(output + self.weights[-1])


    def train(self, inp, output, target, samples_num, lr):
        for j in range(len(self.weights) - 1):
            self.weights[j] += lr * (1 / samples_num) * (target - output) * inp[j]

        self.weights[-1] += lr * (1 / samples_num) * (target - output)


    def fit(self, inputs, targets, epochs=100, lr=0.1):
        for epoch in range(epochs):
            outputs = []

            for i, inp in enumerate(inputs):
                output = self.forward(inp)
                outputs.append(output)

                self.train(inp, output, targets[i], len(inputs), lr)



class MultyRegression:
    def __init__(self, in_features, out_features):
        self.logr_list = [LogisticRegression(in_features) for _ in range(out_features)]


    def forward(self, input_features):
        return [logr.forward(input_features) for logr in self.logr_list]


    def fit(self, inputs, targets, epochs=100, lr=0.1):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                outputs = self.forward(inputs[i])

                for j, logr in enumerate(self.logr_list):
                    logr.train(inputs[i], outputs[j], targets[j][i], len(inputs), lr)




if __name__ == '__main__':
    def main():
        inputs = [[0, 1], [0, 0], [0, 0], [0, 0], [1, 0], [1, 1]]  # input data
        targets = [[1, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0]]  # target data
        targets = transpose_matrix(targets)
        # print(targets)
        labels = [1, 0, 0, 0, 1, 1]

        model = MultyRegression(2, 2)
        model.fit(inputs, targets, epochs=100, lr=0.8)

        for i, inp in enumerate(inputs):
            print(model.forward(inp), targets[0][i], targets[1][i])


    main()