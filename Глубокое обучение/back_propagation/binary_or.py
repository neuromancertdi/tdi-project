from nn_module import *

def forward(inp, w1, w2):  # forward propagation
    z = matrix_by_vector(w1, inp)
    hidden_vector = vector_activation(z)

    output_vector = matrix_by_vector(w2, hidden_vector)
    output_vector = vector_activation(output_vector)

    return hidden_vector, output_vector


def mse(output_vector, target_vector):
    res = 0

    for i in range(len(output_vector)):
        res += (output_vector[i] - target_vector[i]) ** 2

    return res


def test(inputs, target_vector, w1, w2):
    for i, inp in enumerate(inputs):
        _, output = forward(inp, w1, w2)
        print(f"output: {[round(o, 2) for o in output]},"
              f"error: {round(mse(output, target_vector[i]), 3)}")


if __name__ == '__main__':
    def main():
        from datasets import binary_or
        inputs, targets, labels = binary_or()
        inputs = inputs[:1]
        targets = targets[:1]
        labels = labels[:1]

        w1 = [[0.2, 0.3], [0.4, 0.5]]  # weights for h1 and h2
        w2 = [[0.9, 0.8], [0.7, 0.6]]  # weights for o1 and o2

        lr = 1  # learning rate

        test(inputs, targets, w1, w2)

        for epoch in range(1):
            for i, inp in enumerate(inputs):
                hidden_vector, output_vector = forward(inp, w1, w2)

                print(hidden_vector, output_vector)

                out_der = output_derivative(output_vector, targets[i])
                hid_der = hidden_derivative(hidden_vector, w2, out_der)

                update_weights_matrix(inp, w1, hid_der, lr, bias=False)
                update_weights_matrix(hidden_vector, w2, out_der, lr, bias=False)

        print()
        for i, inp in enumerate(inputs):
            hidden_vector, output_vector = forward(inp, w1, w2)

            print(hidden_vector, output_vector)
        test(inputs, targets, w1, w2)


    main()