

target_labels = [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
points = [(1.84, 3.56), (1.01, -0.52), (1.29, 3.45), (1.93, 4.15),
          (2.84, 3.33), (1.42, 4.64), (0.87, 4.71), (2.21, 1.28),
          (4.33, -0.56), (-1.58, 4.96), (2.1, 0.71), (1.17, -1.08),
          (3.59, 2.37), (1.12, 5.76), (1.67, 0.6), (3.29, 2.1),
          (1.71, 1.05), (0.35, 2.85), (2.47, 4.1), (1.74, 4.43)]

centers = [(1, 5), (2.5, 1)]  # центры кластеров, они выбраны вручную по графику


def euclidean_distances_to_centers(point):
    dists = []

    for center in centers:
        distance = 0

        for i in range(2):
            distance += (point[i] - center[i]) ** 2

        dists.append(distance)

    return dists


def get_min_item_idx(distances):  # найти минимальный элемент
    min_dist, min_item_idx = 1000000, 0

    for idx, dist in enumerate(distances):
        if dist < min_dist:
            min_dist = dist
            min_item_idx = idx

    return min_item_idx


clusters = [[], []]

def update_centers():
    for cluster_idx, cluster in enumerate(clusters):
        cluster_mean_value = [0, 0]

        for point in cluster:
            for i, feature in enumerate(point):
                cluster_mean_value[i] += feature

        for i in range(2):
            cluster_mean_value[i] = cluster_mean_value[i] / len(cluster)

        centers[cluster_idx] = cluster_mean_value


labels = []

for _ in range(100):
    clusters = [[], []]
    labels = []

    for point in points:
        dists = euclidean_distances_to_centers(point)
        min_item_idx = get_min_item_idx(dists)  # получить индекс класрера
        clusters[min_item_idx].append(point)  # добавляем точку в ближайший кластер
        labels.append(min_item_idx)

    update_centers()

print(labels)


import numpy as np
from matplotlib import pyplot as plt

data = np.array(points)
labels = np.array(labels)

x1_list = data[labels==0, 0]
x2_list = data[labels==1, 0]
y1_list = data[labels==0, 1]
y2_list = data[labels==1, 1]

plt.scatter(x=x1_list, y=y1_list, color='red')
plt.scatter(x=x2_list, y=y2_list, color='green')
plt.show()