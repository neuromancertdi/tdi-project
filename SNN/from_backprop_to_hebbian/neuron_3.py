from random import randint
import numpy as np

from datasets import mnist_100
from neuron_1 import get_neuron_id

def init_synapse():
    return randint(-10, 10) / 100


def hebbian_learning(spike_pre, spike_post, w, lr):
    if spike_pre and spike_post:
        w += lr

    elif spike_pre and not spike_post:
        w -= lr

    return np.clip(w, -1, 1)  # ограничение значения



class Neuron:
    def __init__(self):
        self.state = 0
        self.id = get_neuron_id()

        self.synapses = dict()
        self.inp_neurons = []
        self.out_neurons = []


    def connect_neuron(self, input_neuron):
        self.inp_neurons.append(input_neuron)
        self.synapses[input_neuron.id] = init_synapse()
        input_neuron.out_neurons.append(self)


    def connect_pre_neurons(self, pre_neurons):
        for neuron in pre_neurons:
            self.connect_neuron(neuron)


    def integrate(self):
        self.state = 0

        for inp_neuron in self.inp_neurons:
            self.state += self.synapses[inp_neuron.id] * inp_neuron.state


    def fire(self):
        if self.state > 1:
            self.state = 1

        else:
            self.state = 0


    def train(self, inp, target, lr):  # target - a number
        # если нейрон не активировался и его целевое значение - 1, то нужно увеличить веса
        # если нейрон активировался и его целевое значение - 0, то нужно уменьшить веса
        if self.state != target:
            for i, inp_i in enumerate(inp):
                self.synapses[self.inp_neurons[i].id] = hebbian_learning(
                    spike_pre=inp_i,
                    spike_post=target,
                    w=self.synapses[self.inp_neurons[i].id],
                    lr=lr
                )



# для получения активированных нейронов в слое
def get_activated_neurons(neurons: list[Neuron]):
    activated = []

    for neuron in neurons:
        if neuron.state == 1:
            activated.append(1)

        else:
            activated.append(0)

    return activated


# случайно задает определенный процент активных нейронов в слое
def select_active_neurons(neurons: list[Neuron], percent=0.1):
    activated = []
    neurons_num = int(len(neurons) * percent)
    selected_ids = []

    while len(selected_ids) < neurons_num:
        rand_neuron_idx = randint(0, len(neurons) - 1)
        neuron_id = neurons[rand_neuron_idx].id

        if neuron_id not in selected_ids:
            selected_ids.append(neuron_id)

    for neuron in neurons:
        if neuron.id in selected_ids:
            activated.append(1)

        else:
            activated.append(0)

    return activated



class Network:
    def __init__(self, structure):  # structure = [inp_size, hid1_size, ..., out_size]
        self.layers = []

        for i in range(len(structure)):
            self.layers.append([Neuron() for _ in range(structure[i])])

        for i, layer in enumerate(self.layers[1:]):
            for neuron in layer:  # this layer idx == i + 1
                # connect each neuron of current layer to before layer neurons
                neuron.connect_pre_neurons(self.layers[i])


    def _set_input_(self, inp):  # inp - one dimentional array
        for i in range(len(inp)):
            self.layers[0][i].state = inp[i]


    def _forward_(self):
        for i, layer in enumerate(self.layers[1:]):
            for neuron in layer:
                neuron.integrate()
                neuron.fire()


    def predict(self, x):
        self._set_input_(x)
        self._forward_()

        return [neuron.state for neuron in self.layers[-1]]


    def update_weights(self, inp, target_to_last_layer, activated_neurons, lr):
        target_to_first_layer = activated_neurons[1]

        for i, neuron in enumerate(self.layers[1]):
            neuron.train(inp, target_to_first_layer[i], lr)

        inp_to_last_layer = activated_neurons[-2]

        for i, neuron in enumerate(self.layers[-1]):
            neuron.train(inp_to_last_layer, target_to_last_layer[i], lr)

        for l, layer in enumerate(self.layers[2:-1]):
            inp_to_cur_layer = activated_neurons[l + 1]
            target_to_cur_layre = activated_neurons[l + 2]

            for i, neuron in enumerate(layer):
                neuron.train(inp_to_cur_layer, target_to_cur_layre[i], lr)


    def make_targets(self):
        activated_neurons = [[]]

        for i, layer in enumerate(self.layers[1:]):
            activated = select_active_neurons(layer, percent=0.2)
            activated_neurons.append(activated)

        return activated_neurons


    def train(self, inp, target_to_last_layer, activated_neurons, lr=0.01):
        # inputs, targets - one dimentional arrays

        self._set_input_(inp)
        self._forward_()
        self.update_weights(inp, target_to_last_layer, activated_neurons, lr)



if __name__ == '__main__':
    def main():
        inputs, targets, labels = mnist_100(samples_n=10)

        model = Network(structure=[784, 100, 50, 10])

        generated_targets = []

        for _ in range(len(inputs)):
            generated_targets.append(model.make_targets())


        # оптимизация по набору данных - работает
        for _ in range(20):
            for i, inp in enumerate(inputs):
                model.train(inp, targets[i], generated_targets[i], lr=0.01)


        # запоминание по одному примеру - не работает
        # for i, inp in enumerate(inputs):
        #     for _ in range(20):
        #         model.train(inp, targets[i], generated_targets[i], lr=0.01)


        print("\n========================================")
        for k, inp_k in enumerate(inputs):
            print(labels[k], model.predict(inp_k))


        # посмотреть состояния нейронов
        # for k, inp_k in enumerate(inputs):
        #     model.predict(inp_k)
        #
        #     print("-------------------")
        #     for i, layer in enumerate(model.layers[1:]):
        #         print([neuron.state for neuron in layer])


        # посмотреть значения весов
        # for layer in model.layers:
        #     print("+++++++++++++")
        #     for neuron in layer[:10]:
        #         weights = list(neuron.synapses.values())[:10]
        #         print(weights)


    main()