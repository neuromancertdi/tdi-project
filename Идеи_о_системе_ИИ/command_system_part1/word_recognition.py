from multy_regression import MultyRegression, transpose_matrix
from tokenizer import Tokenizer

class WordModel:
    def __init__(self, letters, unique_words):
        self.letter_tokenizer = Tokenizer(letters)
        self.words_tokenizer = Tokenizer(unique_words)
        self.model = MultyRegression(len(letters), len(unique_words))


    def fit(self, inputs, targets, epochs=100, lr=0.1):
        self.model.fit(inputs, targets, epochs, lr)


    def get_input(self, word):
        return self.letter_tokenizer.sequence_to_vector(word)

    def get_target(self, word):
        return self.words_tokenizer.token_to_ohe(word)


    def forward(self, inp):
        return self.model.forward(inp)


    def predict(self, word):
        inp = self.get_input(word)
        out = self.forward(inp)

        word = self.words_tokenizer.vector_to_token(out)
        return word



def make_words_data(word_model, input_words, target_words):
    inputs = [word_model.get_input(word) for word in input_words]
    targets = [word_model.get_target(word) for word in target_words]
    targets = transpose_matrix(targets)

    return inputs, targets


def print_output(model, items_list, inputs, targets):
    print("=========================")
    targets = transpose_matrix(targets)

    for i, inp in enumerate(inputs):
        print(items_list[i], model.forward(inp), targets[i])



letters = "абвгдеёжзийклмнопрстуфхцчщшъыьэюя"

words_list = ["привет", "прив", "привеет", "пока", "покаа", "как", "каак", "дела", "сделай"]
target_words = ["привет", "привет", "привет", "пока", "пока", "как", "как", "дела", "сделай"]



if __name__ == '__main__':
    def main():
        test_words = ["првт", "пка", "поа"]
        test_target_words = ["привет", "пока", "пока"]

        unique_words = list(set(target_words))

        word_model = WordModel(letters, unique_words)

        inputs, targets = make_words_data(word_model, words_list, target_words)
        # print(targets)
        # print(inputs)

        test_inputs, test_targets = make_words_data(word_model, test_words, test_target_words)
        # print(test_targets)
        # print(test_inputs)

        word_model.fit(inputs, targets, epochs=100, lr=0.8)

        print_output(word_model, words_list, inputs, targets)
        print_output(word_model, test_words, test_inputs, test_targets)

        print(word_model.predict("прввт"))


    main()