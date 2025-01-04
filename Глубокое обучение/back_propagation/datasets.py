

def binary_or():
    x = [[0, 1], [0, 0], [0, 0], [0, 0], [1, 0], [1, 1]]  # input data
    t = [[1, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0]]  # target data
    labels = [1, 0, 0, 0, 1, 1]
    return x, t, labels


def mnist_100(samples_n=5):
    import matplotlib.pyplot as plt
    # max samples == 100
    x = []  # input data

    for i in range(samples_n):  # loading images
        img = plt.imread(f"digits/images/{i}.jpg")
        img = img[:, :, 0].reshape(784) / 255  # change shape from 28x28x3 to 1x784
        x.append(list(img))

    t = []  # target data
    labels = []
    with open("digits/labels.txt", 'r') as file:  # loading labels
        for i in range(samples_n):
            t.append([0 for _ in range(10)])
            digit_label = int(file.readline())
            labels.append(digit_label)
            t[i][digit_label] = 1

    return x, t, labels