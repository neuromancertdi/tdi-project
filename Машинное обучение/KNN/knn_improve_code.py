

class KNN:
    def __init__(self, points, labels):
        self.points, self.labels = points[:], labels[:]


    def _euclidean_dinstance_(self, new_point):
        distances = []

        for point in self.points:
            point_distance = 0

            for i in range(2):
                point_distance += (new_point[i] - point[i]) ** 2

            distances.append(point_distance)

        return distances


    def _get_min_item_idx_(self, distances):  # найти индекс минимального элемента
        min_dist, min_item_idx = 1000000, 0

        for idx, dist in enumerate(distances):
            if dist < min_dist:
                min_dist = dist
                min_item_idx = idx

        return min_item_idx


    def predict(self, new_point, k):
        distances = self._euclidean_dinstance_(new_point)
        labels = self.labels[:]  # скорировать список меток
        label_score = {label: 0 for label in labels}

        for _ in range(k):
            min_item_idx = self._get_min_item_idx_(distances)
            distances.pop(min_item_idx)
            label = labels.pop(min_item_idx)
            label_score[label] += 1

        max_score, max_label = 0, 'c1'

        for label, score in label_score.items():
            if score > max_score:
                max_score = score
                max_label = label

        return max_label


if __name__ == '__main__':
    x1 = [1.34, 1.16, 1.54, 0.46, 1.8, 1.86, 0.7, 0.52, 1.24, 1.32]
    y1 = [1.8, 1.33, 1.37, 1.71, 0.68, 0.29, 1.3, 0.25, 1.07, 0.25]
    x2 = [5.0, 5.88, 3.69, 2.61, 4.66, 2.09, 4.82]
    y2 = [0.14, 0.61, 4.37, 1.74, 3.3, 2.98, 4.25]

    points = [(x1[i], y1[i]) for i in range(len(x1))]
    points += [(x2[i], y2[i]) for i in range(len(x2))]
    labels = ['c1' for i in range(len(x1))]
    labels += ['c2' for i in range(len(x2))]

    knn = KNN(points, labels)
    print(knn.predict(new_point=(4, 2), k=10))
    print(knn.predict(new_point=(0.6, 0.5), k=10))