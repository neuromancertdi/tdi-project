

# найти индекс максимального элемента
def argmax(items):
    max_item_idx = 0

    for idx, item in enumerate(items):
        if item > items[max_item_idx]:
            max_item_idx = idx

    return max_item_idx



class NBC:
    def __init__(self):
        self.labels_probabilities = dict()
        self.unique_labels = list()


    def _unique_labels_(self, labels):
        return list(set(labels))


    def _labels_probabilities_(self, labels):
        label_score = {label: 0 for label in self.unique_labels}

        for label in labels:
            label_score[label] += 1

        return {label: label_score[label] / len(labels) for label in self.unique_labels}



class MultinomialNBC(NBC):
    def __init__(self):
        NBC.__init__(self)

        self.unique_features = list()
        self.features_probabilities = dict()
        self.alpha = 1


    def _unique_features_(self, inputs):
        features = list()

        for inp in inputs:
            for feature in inp:
                if feature not in features:
                    features.append(feature)

        return features


    def _feature_count_per_label_(self, feature, label, inputs, labels):
        feature_count = 0

        for i, inp in enumerate(inputs):
            if labels[i] == label:

                for inp_feature in inp:
                    if inp_feature == feature:
                        feature_count += 1

        return feature_count


    # calculate conditional probability for each feature
    def _features_probabilities_(self, inputs, labels):
        probabilities = {label: dict() for label in self.unique_labels}

        for label in self.unique_labels:
            probabilities[label] = {feature: 0 for feature in self.unique_features}
            features_count = 0

            for feature in self.unique_features:
                feature_count = self._feature_count_per_label_(feature, label, inputs, labels)
                probabilities[label][feature] = feature_count
                features_count += feature_count

            for feature in self.unique_features:
                numerator = probabilities[label][feature] + self.alpha
                denominator = features_count + self.alpha * len(self.unique_features)
                probabilities[label][feature] = numerator / denominator

        return probabilities


    def fit(self, inputs, labels, alpha=1):
        self.alpha = alpha
        self.unique_features = self._unique_features_(inputs)
        self.unique_labels = self._unique_labels_(labels)
        self.features_probabilities = self._features_probabilities_(inputs, labels)
        self.labels_probabilities = self._labels_probabilities_(labels)


    def _predict_label_(self, label, inp):  # calculate posterior probability for each class
        inp_probability = 1

        for feature in inp:
            if feature in self.unique_features:
                inp_probability *= self.features_probabilities[label][feature]

        return inp_probability * self.labels_probabilities[label]


    def predict(self, inp):
        probabilities = [self._predict_label_(label, inp) for label in self.unique_labels]
        max_item_idx = argmax(probabilities)

        return self.unique_labels[max_item_idx]  # return class with the highest posterior



if __name__ == '__main__':
    def main():
        from nbc_data import positive, negative, neutral

        all_messages = positive[:-1] + negative[:-1] + neutral[:-1]
        test_messages = [positive[-1], negative[-1], neutral[-1]]

        translate_table = str.maketrans({char: '' for char in "!,.?"})

        inputs = [inp.translate(translate_table).split() for inp in all_messages]
        test_inputs = [inp.translate(translate_table).split() for inp in test_messages]

        labels = [0 for _ in positive[:-1]]
        labels += [1 for _ in negative[:-1]]
        labels += [2 for _ in neutral[:-1]]

        test_labels = [0, 1, 2]

        mult_nbc = MultinomialNBC()
        mult_nbc.fit(inputs, labels)

        for i, msg in enumerate(test_inputs):
            print(test_labels[i], mult_nbc.predict(msg))

        print("================================")
        for i, msg in enumerate(inputs):
            print(labels[i], mult_nbc.predict(msg))


    main()