from sklearn import datasets
from linear import *
import numpy as np

digits = datasets.load_digits()

inputs = list(digits.data[:20] / 16)
labels = list(digits.target[:20])

test_inputs = list(digits.data[20:30] / 16)
test_labels = list(digits.target[20:30])

targets = zeros(shape=(20, 10))

for i in range(20):
    digit = labels[i]
    targets[i][digit] = 1

for i in range(len(inputs)):
    inputs[i] = list(inputs[i])

for i in range(len(test_inputs)):
    test_inputs[i] = list(test_inputs[i])

inputs = Tensor(inputs)
test_inputs = Tensor(test_inputs)
targets = Tensor(targets)

model = Sequential([Linear(64, 32), Sigmoid(), Linear(32, 10), Sigmoid()])
optim = GradientDescent(parameters=model.get_parameters(), lr=2)
loss = MSELoss()

for epoch in range(200):
    outputs = model.forward(inputs)
    error = loss.forward(outputs, targets)
    print(f"epoch: {epoch}: {error}")
    error.backward()
    optim.step()


outputs = model.forward(test_inputs)
outputs = np.array(outputs.data)

for i in range(10):
    output_label = np.argmax(outputs[i])
    print(f"output_label: {output_label}, true_label: {test_labels[i]}")