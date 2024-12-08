from random import randint
from math import e, log

def logistic_function(z):
    return 1 / (1 + e ** z)

def logistic_error(outputs, targets):
    error = 0

    for i, point in enumerate(inputs):
        error -= targets[i] * log(outputs[i], e) - (1 - targets[i]) * log(1 - outputs[i], e)

    return error


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
            self.weights[j] -= lr * (1 / samples_num) * (target - output) * inp[j]

        self.weights[-1] -= lr * (1 / samples_num) * (target - output)


    def forward_list(self, inputs):
        outputs = []

        for inp in inputs:
            output = self.forward(inp)
            outputs.append(output)

        return outputs


    def fit(self, inputs, targets, epochs=100, lr=0.1):
        for epoch in range(epochs):
            outputs = []

            for i, inp in enumerate(inputs):
                output = self.forward(inp)
                outputs.append(output)

                self.train(inp, output, targets[i], len(inputs), lr)

            print(f"epoch: {epoch}, error: {logistic_error(outputs, targets)}")



def accuracy(outputs, targets):
    true_outputs = 0

    for i, output in enumerate(outputs):
        if round(output) == targets[i]:
            true_outputs += 1

    return true_outputs, len(inputs)


if __name__ == '__main__':
    x1 = [0.38, 1.42, 0.55, 1.34, 1.76, 1.62, 0.83, 0.84, 1.77, 1.06]
    y1 = [1.79, 0.54, 0.34, 0.678, 1.64, 0.92, 1.49, 0.3, 0.7, 0.99]
    x2 = [3.9, 6.14, 6.1, 2.11, 3.23, 1.62, 1.88]
    y2 = [4.93, 4.95, 0.97, 0.77, 0.43, 4.61, 0.25]

    inputs = [(x1[i], y1[i]) for i in range(len(x1))]
    targets = [0 for i in range(len(x1))]
    inputs += [(x2[i], y2[i]) for i in range(len(x2))]
    targets += [1 for i in range(len(x2))]

    logr_model = LogisticRegression(features_num=2)
    logr_model.fit(inputs, targets, epochs=100, lr=0.1)

    outputs = logr_model.forward_list(inputs)

    print(logr_model.weights)
    print("accuracy:", accuracy(outputs, targets))