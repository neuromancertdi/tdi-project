import torch
from torch import tensor  # это код из фреймворка torch
from tensor import rand  # это пользовательский файл

class Layer:
    def __init__(self):
        self.parameters = []

    def forward(self, inputs):
        return None

    def get_parameters(self):
        return self.parameters



class Linear(Layer):
    def __init__(self, in_features, out_features):
        Layer.__init__(self)

        self.weights = tensor(rand((in_features, out_features)), dtype=torch.float64, requires_grad=True)
        self.bias = tensor(rand((1, out_features)), dtype=torch.float64, requires_grad=True)
        self.parameters.append(self.weights)
        self.parameters.append(self.bias)


    def forward(self, inp):
        out = inp.matmul(self.weights)
        out += self.bias.expand(inp.shape[0], -1)

        return out



class Sequential(Layer):
    def __init__(self, layers: list):
        Layer.__init__(self)
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)


    def forward(self, inp):
        for layer in self.layers:
            inp = layer.forward(inp)

        return inp


    def get_parameters(self):
        parameters = []

        for layer in self.layers:
            parameters += layer.get_parameters()

        return parameters



class Sigmoid(Layer):
    def __init__(self):
        Layer.__init__(self)

    def forward(self, inputs):
        return inputs.sigmoid()



if __name__ == '__main__':
    def main():
        inputs = tensor([[1, 0], [0, 0], [0, 1], [0, 0], [1, 1], [0, 0]], dtype=torch.float64)
        targets = tensor([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float64)

        model = Sequential([Linear(2, 3), Sigmoid(), Linear(3, 2), Sigmoid()])

        for epoch in range(100):
            out = model.forward(inputs)
            error = ((out - targets) * (out - targets)).sum()
            print(f"epoch: {epoch}: {error}")
            error.backward()

            lr = 20

            for param in model.get_parameters():
                param.data -= lr * param.grad.data
                param.grad.zero_()  # занулить градиент

        print(model.forward(inputs))

        for param in model.get_parameters():
            print(param.data)


    main()
