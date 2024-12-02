from random import randint

def mse(outputs, targets):
    error = 0

    for i, output in enumerate(outputs):
        error += (output - targets[i]) ** 2

    return error / len(outputs)


class LinearRegression:
    def __init__(self, features_num):
        # +1 for bias, bias is last weight
        self.weights = [randint(-100, 100) / 100 for _ in range(features_num + 1)]


    def forward(self, input_features):
        output = 0

        for i, feature in enumerate(input_features):
            output += self.weights[i] * feature

        output += self.weights[-1]

        return output


    def train(self, inp, output, target, samples_num, lr):
        for j in range(len(self.weights) - 1):
            self.weights[j] -= lr * (2 / samples_num) * (output - target) * inp[j]

        self.weights[-1] -= lr * (2 / samples_num) * (output - target)


    def fit(self, inputs, targets, epochs=100, lr=0.1):
        for epoch in range(epochs):
            outputs = []

            for i, inp in enumerate(inputs):
                output = self.forward(inp)
                outputs.append(output)

                self.train(inp, output, targets[i], len(inputs), lr)

            print(f"epoch: {epoch}, error: {mse(outputs, targets)}")


if __name__ == '__main__':
    inputs = [[0.12], [1.4], [1.9], [3.5], [4.44]]
    targets = [0.7708, -0.004, 0.051, -0.245, -1.0704]

    lr_model = LinearRegression(features_num=1)
    lr_model.fit(inputs, targets, epochs=100, lr=0.1)

    print(lr_model.weights)