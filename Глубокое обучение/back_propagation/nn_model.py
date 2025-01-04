from nn_module import *

class Network:
    def __init__(self, nn_structure):  # [inp_size, hid1_size, hid2_size, ... , out_size]
        self.weights = []

        for layer_n in range(len(nn_structure) - 1):
            w = init_weights_matrix(
                inp_size=nn_structure[layer_n],
                out_size=nn_structure[layer_n + 1]
            )
            self.weights.append(w)

        self.layers_z = [[] for _ in range(len(self.weights))]
        self.layers = [[] for _ in range(len(self.weights))]


    def predict(self, x):
        self._forward(x)
        return [round(self.layers[-1][i], 2) for i in range(len(self.layers[-1]))]


    def _forward(self, x):
        self.layers_z[0] = matrix_by_vector(self.weights[0], x)
        self.layers[0] = vector_activation(self.layers_z[0])

        for i in range(1, len(self.weights)):
            self.layers_z[i] = matrix_by_vector(self.weights[i], self.layers[i - 1])
            self.layers[i] = vector_activation(self.layers_z[i])


    def train(self, x, t, lr=0.5):  # lr - learning rate
        self._forward(x)

        layers_der = [[] for _ in range(len(self.layers))]  # derivatives

        layers_der[-1] = output_derivative(self.layers[-1], t)
        for i in range(1, len(self.layers)):
            idx = -(i + 1)
            layers_der[idx] = hidden_derivative(self.layers[idx], self.weights[idx + 1], layers_der[idx + 1])

        update_weights_matrix(x, self.weights[0], layers_der[0], lr)
        for i in range(1, len(self.weights)):
            update_weights_matrix(self.layers[i - 1], self.weights[i], layers_der[i], lr)


if __name__ == '__main__':
    from binary_or import mse

    def test(model, x, t):
        for i, x_i in enumerate(x):
            out = model.predict(x_i)
            print(f"output: {out}, error: {round(mse(model.layers[-1], t[i]), 4)}")


    def batch_mse(model, x, t):
        res = 0
        for i, inp in enumerate(x):
            model.predict(inp)
            res += mse(t[i], model.layers[-1])
        return res / len(x)


    def main():
        from datasets import mnist_100

        model = Network(nn_structure=[784, 100, 10])
        x, t, labels = mnist_100(samples_n=3)

        epochs_n = 20

        print("\nlabels =", labels)
        test(model, x, t)
        print()

        for epoch in range(1, epochs_n + 1):
            for i, inp in enumerate(x):
                model.train(inp, t[i], lr=0.1)

            print(f"epoch: {epoch}, error: {round(batch_mse(model, x, t), 4)}")

            test(model, x, t)
            print()

        print("\nlabels =", labels)
        test(model, x, t)


    main()