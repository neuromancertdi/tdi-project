from math import sqrt, e, pi
from nbc_multinomial import argmax, NBC

def product(items):
    res = 1

    for item in items:
        res *= item

    return res



class GaussianNBC(NBC):
    def __init__(self):
        NBC.__init__(self)

        self.mean = dict()
        self.std = dict()


    def _mean_(self, feature_items):
        return sum([item for item in feature_items]) / len(feature_items)


    def _std_(self, feature_items):  # standart deviation
        mean = self._mean_(feature_items)
        squared_sum = sum([(item - mean) ** 2 for item in feature_items])

        return sqrt(squared_sum / len(feature_items))


    def _feature_items_per_label_(self, feature_idx, label, inputs, labels):
        feature_items = list()

        for i, inp in enumerate(inputs):
            if labels[i] == label:
                feature_items.append(inp[feature_idx])

        return feature_items


    def _features_items_(self, inputs, labels):
        features_num = len(inputs[0])
        features_items = {label: dict() for label in self.unique_labels}

        for label in self.unique_labels:
            features_items[label] = {i: list() for i in range(features_num)}

            for i in range(features_num):
                features_items[label][i] = self._feature_items_per_label_(i, label, inputs, labels)

        return features_items


    # calc mean and std for each feature for each label
    def _get_features_parameters_(self, inputs, labels):
        features_items = self._features_items_(inputs, labels)
        mean = {label: dict() for label in self.unique_labels}
        std = {label: dict() for label in self.unique_labels}

        for label, feature_items in features_items.items():
            mean[label] = {i: 0 for i in range(len(inputs[0]))}
            std[label] = {i: 0 for i in range(len(inputs[0]))}

            for feature_idx, items in feature_items.items():
                mean[label][feature_idx] = self._mean_(feature_items=items)
                std[label][feature_idx] = self._std_(feature_items=items)

        return mean, std


    def fit(self, inputs, labels):
        self.unique_labels = self._unique_labels_(labels)
        self.labels_probabilities = self._labels_probabilities_(labels)
        self.mean, self.std = self._get_features_parameters_(inputs, labels)


    def _gaus_scaling_(self, inp, label):
        scaled_inp = list()

        for idx, value in enumerate(inp):
            numerator = e ** (-(value - self.mean[label][idx]) ** 2 / (2 * self.std[label][idx] ** 2))
            denominator = sqrt(2 * pi * self.std[label][idx] ** 2)
            scaled_inp.append(numerator / denominator)

        return scaled_inp


    def _predict_label_(self, inp, label):  # calculate posterior probability for each class
        return self.labels_probabilities[label] * product(self._gaus_scaling_(inp, label))


    def predict(self, inp):
        probabilities = [self._predict_label_(inp, label) for label in self.unique_labels]
        max_item_idx = argmax(probabilities)

        return self.unique_labels[max_item_idx]  # return class with the highest posterior



if __name__ == '__main__':
    def main():
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_blobs

        x_dataset, y_dataset = make_blobs(n_samples=150, n_features=4, centers=3, random_state=123)
        x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=123)

        gauss_nbc = GaussianNBC()
        gauss_nbc.fit(inputs=x_train, labels=y_train)

        for i in range(len(x_test)):
            print(y_test[i], gauss_nbc.predict(x_test[i]))


    main()