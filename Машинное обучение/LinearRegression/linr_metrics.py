from linr_improve_code import LinearRegression

def mse(outputs, targets):
    error = 0

    for i, output in enumerate(outputs):
        error += (output - targets[i]) ** 2

    return error / len(outputs)


def r2_score(outputs, targets):
    mean_target = sum(targets) / len(targets)

    ess = sum([(outputs[i] - targets[i]) ** 2 for i in range(len(outputs))])
    tss = sum([(targets[i] - mean_target) ** 2 for i in range(len(outputs))])

    return 1 - ess / tss


inputs = [[0.12], [1.4], [1.9], [3.5], [4.44]]
targets = [0.7708, -0.004, 0.051, -0.245, -1.0704]

lr_model = LinearRegression(1)
lr_model.fit(inputs, targets, epochs=100, lr=0.1)

outputs = [lr_model.forward(inp) for inp in inputs]

print(f"mse: {mse(outputs, targets)}")
print(f"r2: {r2_score(outputs, targets)}")
