from random import randint
from matplotlib import pyplot as plt

def data_points(n_samples, class_point, noise=1):
    def offset_point():
        offset_x = randint(-100 * noise, noise * 100) / 100
        offset_y = randint(-100 * noise, noise * 100) / 100
        x = class_point[0] + offset_x
        y = class_point[1] + offset_y
        return x, y

    points = [offset_point() for _ in range(n_samples)]
    x_list = [points[i][0] for i in range(n_samples)]
    y_list = [points[i][1] for i in range(n_samples)]

    return x_list, y_list


x1_list, y1_list = data_points(n_samples=10, class_point=(1, 1), noise=0.9)
x2_list, y2_list = data_points(n_samples=7, class_point=(4, 2.5), noise=2.5)

print(x1_list, y1_list)
print(x2_list, y2_list)

plt.scatter(x=x1_list, y=y1_list, color='red')
plt.scatter(x=x2_list, y=y2_list, color='green')
plt.show()