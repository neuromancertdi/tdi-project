from random import randint
from math import e
from datasets import binary_or

def sigmoid(z):
    return 1 / (1 + e ** (-z))


def sigmoid_der(sigmoid_out):  # derivative of sigmoid
    return sigmoid_out * (1 - sigmoid_out)


def init_weight():
    return (randint(-100, 100) / 100) / randint(1, 100)


NEURON_ID = []

def get_neuron_id(a=0, b=1000000):
    neuron_id = randint(a, b)

    while neuron_id in NEURON_ID:
        neuron_id = randint(a, b)

    NEURON_ID.append(neuron_id)

    return neuron_id



class Neuron:
    def __init__(self, is_hidden):
        self.weights = dict()
        self.state = 0
        self.derivative = 0

        self.inp_neurons = []
        self.out_neurons = []

        self.is_hidden = is_hidden
        self.id = get_neuron_id()


    def append_input_neuron(self, input_neuron):
        self.inp_neurons.append(input_neuron)
        self.weights[input_neuron.id] = init_weight()
        input_neuron.out_neurons.append(self)


    def forward(self):
        z = 0

        for inp_neuron in self.inp_neurons:
            z += self.weights[inp_neuron.id] * inp_neuron.state

        self.state = sigmoid(z)


    def train(self, target=None):
        if self.is_hidden:
            self.derivative = 0

            for out_neuron in self.out_neurons:
                self.derivative += out_neuron.weights[self.id] * out_neuron.derivative

            self.derivative = self.derivative * sigmoid_der(self.state)

        else:
            self.derivative = (self.state - target) * sigmoid_der(self.state)

    def update_wetghts(self, lr=0.1):
        for inp_neuron in self.inp_neurons:
            self.weights[inp_neuron.id] -= lr * inp_neuron.state * self.derivative



if __name__ == '__main__':
    x, t, labels = binary_or()

    model = [
        Neuron(False),  # x1
        Neuron(False),  # x2
        Neuron(True),  # h1
        Neuron(True),  # h2
        Neuron(False),  # o1
        Neuron(False),  # o2
    ]

    model[2].append_input_neuron(model[0])
    model[2].append_input_neuron(model[1])

    model[3].append_input_neuron(model[0])
    model[3].append_input_neuron(model[1])

    model[4].append_input_neuron(model[2])
    model[4].append_input_neuron(model[3])

    model[5].append_input_neuron(model[2])
    model[5].append_input_neuron(model[3])


    for _ in range(100):
        for i, inp in enumerate(x):
            model[0].state = inp[0]
            model[1].state = inp[1]

            for neuron in model[2:]:
                neuron.forward()

            model[-2].train(t[i][0])
            model[-1].train(t[i][1])

            model[-3].train()
            model[-4].train()

            for neuron in model[2:]:
                neuron.update_wetghts(lr=1.5)


    for i, inp in enumerate(x):
        model[0].state = inp[0]
        model[1].state = inp[1]

        for neuron in model[2:]:
            neuron.forward()

        for j, neuron in enumerate(model[4:]):
            print(f"target: {t[i][j]}, output: {neuron.state}")