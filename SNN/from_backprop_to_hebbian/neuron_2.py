from random import randint
from math import e
from neuron_1 import Neuron

def sigmoid(z):
    return 1 / (1 + e ** (-z))


def sigmoid_der(sigmoid_out):  # derivative of sigmoid
    return sigmoid_out * (1 - sigmoid_out)


def init_synapse():
    return (randint(-100, 100) / 100) / randint(1, 100)



class Neuron1(Neuron):
    def __init__(self, is_hidden):
        Neuron.__init__(self, is_hidden)


    def connect_pre_neurons(self, pre_neurons):
        for neuron in pre_neurons:
            self.append_input_neuron(neuron)



class Network:
    def __init__(self, structure):  # structure = [inp_size, hid1_size, ..., out_size]
        self.layers = []

        for i in range(len(structure) - 1):
            self.layers.append([Neuron1(is_hidden=True) for _ in range(structure[i])])

        self.layers.append([Neuron1(is_hidden=False) for _ in range(structure[-1])])

        for i, layer in enumerate(self.layers[1:]):
            for neuron in layer:  # индекс текущего слоя == i + 1
                # соединить каждый нейрон текущего слоя со всеми нейронами предыдущего
                neuron.connect_pre_neurons(self.layers[i])


    def set_input(self, inp):  # inp - one dimentional array
        for i, inp_i in enumerate(inp):
            self.layers[0][i].state = inp_i


    def forward(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.forward()


    def predict(self, x):
        self.set_input(x)
        self.forward()

        return [round(neuron.state, 4) for neuron in self.layers[-1]]


    def backward(self, target):  # target - one dimentional array
        for i, out_neuron in enumerate(self.layers[-1]):
            out_neuron.train(target[i])

        for layer in self.layers[1:-1][::-1]:  # slice and reverse
            for hidden_neuron in layer:
                hidden_neuron.train()


    def update_weights(self, lr):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.update_wetghts(lr)


    def fit(self, inputs, targets, lr=0.1):  # inputs, targets - two dimentional arrays
        for i, inp in enumerate(inputs):
            self.set_input(inp)
            self.forward()
            self.backward(targets[i])
            self.update_weights(lr)



if __name__ == '__main__':
    def main():
        from datasets import mnist_100

        inputs, targets, labels = mnist_100(samples_n=3)

        model = Network(structure=[784, 100, 10])

        # for i, inp in enumerate(inputs):
        #     print(labels[i], model.predict(inp))

        for i in range(10):
            print("epoch:", i)
            model.fit(inputs, targets, lr=1)

        print()
        for i, inp in enumerate(inputs):
            print(labels[i], model.predict(inp))


    main()