from nbc_multinomial import MultinomialNBC

class BernoullyNBC(MultinomialNBC):
    def __init__(self):
        MultinomialNBC.__init__(self)


    def _feature_count_per_label_(self, feature, label, inputs, labels):
        feature_count, label_count = 0, 0

        for i, inp in enumerate(inputs):
            if labels[i] == label:
                label_count += 1

                for inp_feature in inp:
                    if inp_feature == feature:
                        feature_count += 1
                        break  # до первого появления в примере

        return feature_count, label_count


    def _features_probabilities_(self, inputs, labels):  # calculate conditional probability for each feature
        probabilities = {label: dict() for label in self.unique_labels}

        for label in self.unique_labels:
            probabilities[label] = {feature: 0 for feature in self.unique_features}

            for feature in self.unique_features:
                feature_count, label_count = self._feature_count_per_label_(feature, label, inputs, labels)
                probabilities[label][feature] = feature_count / label_count

        return probabilities


    def _unique_features_per_input_(self, inp):
        features = list()

        for feature in inp:
            if feature not in features:
                features.append(feature)

        return features


    # calculate posterior probability for each class
    def _predict_label_(self, label, inp):
        inp_probability = 1
        unique_inp_features = self._unique_features_per_input_(inp)

        for feature in inp:
            if feature in self.unique_features:
                if feature in unique_inp_features:
                    inp_probability *= self.features_probabilities[label][feature]

                else:
                    inp_probability *= 1 - self.features_probabilities[label][feature]

        return inp_probability * self.labels_probabilities[label]



if __name__ == '__main__':
    def main():
        from nbc_data import positive, negative

        train_pos, train_neg = positive[:-2], negative[:-2]
        test_pos, test_neg = positive[-2:], negative[-2:]

        translate_table = str.maketrans({char: '' for char in "!,.?"})

        inputs = [inp.translate(translate_table).split() for inp in train_pos + train_neg]
        test_inputs = [inp.translate(translate_table).split() for inp in test_pos + test_neg]

        labels = [0 for _ in train_pos]
        labels += [1 for _ in train_neg]
        test_labels = [0 for _ in test_pos]
        test_labels += [1 for _ in test_neg]

        bernoully_nbc = BernoullyNBC()
        bernoully_nbc.fit(inputs, labels)

        for i, msg in enumerate(inputs):
            print(labels[i], bernoully_nbc.predict(msg))

        print("==========================")

        for i, msg in enumerate(test_inputs):
            print(test_labels[i], bernoully_nbc.predict(msg))


    main()