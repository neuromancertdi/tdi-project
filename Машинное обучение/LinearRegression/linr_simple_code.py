from random import randint

n_samples = 5  # target: k = -0.41, b = 0.81
x_values = [0.12, 1.4, 1.9, 3.5, 4.44]  # inputs
y_values = [0.7708, -0.004, 0.051, -0.245, -1.0704]  # targets
lr = 0.1
b, k = randint(-100, 100) / 100, randint(-100, 100) / 100  # bias, w

def f(x):
    return k * x + b

def mse():
    errors = []

    for i in range(n_samples):
        target = y_values[i]
        output = f(x_values[i])

        errors.append((target - output) ** 2)

    return sum(errors) / n_samples


for epoch in range(100):
    for i in range(n_samples):
        target = y_values[i]
        output = f(x_values[i])

        b -= lr * (2 / n_samples) * (output - target)
        k -= lr * (2 / n_samples) * (output - target) * x_values[i]

    print(f"epoch: {epoch}, error: {mse()}")

print(k, b)
