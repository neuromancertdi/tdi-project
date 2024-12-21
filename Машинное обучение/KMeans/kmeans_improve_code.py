import numpy as np
from random import randint

def euclidean(point1, point2):
    return sum([(point1[i] - point2[i]) ** 2 for i in range(len(point1))])


def euclidean_distances(cur_point, points):
    dists = []

    for point in points:
        dists.append(euclidean(cur_point, point))

    return dists


# найти индекс минимального элемента
def get_min_item_idx(distances):
    min_item_idx = 0

    for idx, dist in enumerate(distances):
        if dist < distances[min_item_idx]:
            min_item_idx = idx

    return min_item_idx



class KMeans:
    def __init__(self, features_num):
        self.features_num = features_num
        self.centers = []
        self.clusters = [[] for _ in range(features_num)]

    def set_centers(self, centers):
        self.centers = centers


    def init_centers(self, points, k):
        min_value = np.min(points, axis=0) * 100
        max_value = np.max(points, axis=0) * 100
        self.centers = []

        for _ in range(k):
            x = randint(min_value[0], max_value[0]) / 100
            y = randint(min_value[1], max_value[1]) / 100
            self.centers.append([x, y])


    def _update_centers_(self):
        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_mean_value = [0, 0]

            for point in cluster:
                for i, feature in enumerate(point):
                    cluster_mean_value[i] += feature

            for i in range(2):
                cluster_mean_value[i] = cluster_mean_value[i] / len(cluster)

            self.centers[cluster_idx] = cluster_mean_value


    def fit(self, points, epochs):
        labels = []

        for _ in range(epochs):
            self.clusters = [[] for _ in range(self.features_num)]
            labels = []

            for point in points:
                dists = euclidean_distances(point, self.centers)
                min_item_idx = get_min_item_idx(dists)  # получить индекс класрера
                self.clusters[min_item_idx].append(point)  # добавляем точку в ближайший кластер
                labels.append(min_item_idx)

            self._update_centers_()

        return self.clusters, labels



if __name__ == '__main__':
    target_labels = [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
    points = [(1.84, 3.56), (1.01, -0.52), (1.29, 3.45), (1.93, 4.15),
              (2.84, 3.33), (1.42, 4.64), (0.87, 4.71), (2.21, 1.28),
              (4.33, -0.56), (-1.58, 4.96), (2.1, 0.71), (1.17, -1.08),
              (3.59, 2.37), (1.12, 5.76), (1.67, 0.6), (3.29, 2.1),
              (1.71, 1.05), (0.35, 2.85), (2.47, 4.1), (1.74, 4.43)]

    centers = [(1, 5), (2.5, 1)]  # центры кластеров, задаем вручную по графику

    kmeans = KMeans(features_num=2)
    kmeans.set_centers(centers)
    clusters, labels = kmeans.fit(points, 10)

    print(labels)
