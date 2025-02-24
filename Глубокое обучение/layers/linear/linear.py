from tensor import *

class Layer:
    def __init__(self):
        self.parameters = []

    def get_parameters(self):
        return self.parameters



class Linear(Layer):
    def __init__(self, in_features, out_features):
        Layer.__init__(self)

        self.weights = Tensor(rand((in_features, out_features)))
        self.bias = Tensor(rand((1, out_features)))
        self.parameters.append(self.weights)
        self.parameters.append(self.bias)


    def forward(self, inp):
        out = inp.dot(self.weights)
        out += self.bias.expand(0, inp.shape()[0])

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



class GradientDescent:
    def __init__(self, parameters, lr=0.1):
        self.parameters = parameters
        self.lr = lr


    def zero(self):  # занулить градиент
        for param in self.parameters:
            # param.grad.data = zeros(shape=param.grad.shape())
            param.grad = None


    def step(self, zero=True):
        for param in self.parameters:
            param.data = sub_matrixes(param.data, num_mul_matrix(self.lr, param.grad.data))

        if zero:
            self.zero()



class Sigmoid(Layer):
    def __init__(self):
        Layer.__init__(self)

    def forward(self, inputs):
        return inputs.sigmoid()



class MSELoss(Layer):
    def __init__(self):
        Layer.__init__(self)

    def forward(self, output, target):
        features_num = output.shape()[0] * output.shape()[1]
        return ((output - target) * (output - target)).sum(dim=0).sum(dim=1) / features_num
        # return ((output - target) * (output - target)).sum(dim=0).sum(dim=1)



if __name__ == '__main__':
    def main():
        # задача "логическое или", на выходе 2 класса
        inputs = Tensor([[1, 0], [0, 0], [0, 1], [0, 0], [1, 1], [0, 0]])
        targets = Tensor([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])

        model = Sequential([Linear(2, 3), Sigmoid(), Linear(3, 2), Sigmoid()])
        optim = GradientDescent(parameters=model.get_parameters(), lr=20)
        loss = MSELoss()

        for epoch in range(200):
            outputs = model.forward(inputs)
            error = loss.forward(outputs, targets)
            print(f"epoch: {epoch}: {error}")
            error.backward()
            optim.step()

        print("\noutputs:", model.forward(inputs))

        print()
        for i, param in enumerate(model.get_parameters()):
            print(f"param {i}:", param.data)


    main()