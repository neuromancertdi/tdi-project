from random import randint
from matplotlib import pyplot as plt

def linear_data_points(n_samples, noise=1):
    k = randint(-100, 100) / 100
    b = randint(0, 100) / 100
    print(k, b)

    def f(input_value):
        offset_x = randint(-100 * noise, noise * 100) / 100
        offset_y = randint(-100 * noise, noise * 100) / 100
        x = input_value + offset_x
        y = k * x + b + offset_y
        return x, y

    points = [f(i) for i in range(n_samples)]
    x_list = [points[i][0] for i in range(n_samples)]
    y_list = [points[i][1] for i in range(n_samples)]

    return x_list, y_list


x_list, y_list = linear_data_points(n_samples=5, noise=0.5)

print(x_list, y_list)

plt.scatter(x=x_list, y=y_list)
plt.show()
