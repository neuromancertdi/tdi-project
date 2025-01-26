from word_recognition import *
from multy_regression import transpose_matrix, MultyRegression
from tokenizer import Tokenizer

class CommandModel:
    def __init__(self, unique_commands_names, words_recognizer):
        self.words_recognizer = words_recognizer
        self.words_tokenizer = words_recognizer.words_tokenizer

        unique_words_num = words_recognizer.words_tokenizer.tokens_num

        self.command_name_tokenizer = Tokenizer(unique_commands_names)
        self.model = MultyRegression(unique_words_num, len(unique_commands_names))


    def fit(self, inputs, targets, epochs=100, lr=0.1):
        self.model.fit(inputs, targets, epochs, lr)


    def get_input(self, input_command):
        command_sequence = [self.words_recognizer.predict(word) for word in input_command.split()]
        return self.words_tokenizer.sequence_to_vector(command_sequence)


    def get_target(self, command_name):
        return self.command_name_tokenizer.token_to_ohe(command_name)


    def forward(self, inp):
        return self.model.forward(inp)


    def predict(self, input_command):
        inp = self.get_input(input_command)
        out = self.forward(inp)

        command_name = self.command_name_tokenizer.vector_to_token(out)
        return command_name



def make_commands_data(command_model, input_commands, target_commands):
    inputs = [command_model.get_input(input_command) for input_command in input_commands]
    targets = [command_model.get_target(command_name) for command_name in target_commands]
    targets = transpose_matrix(targets)

    return inputs, targets


commands = {
    "привет": "привет",
    "как дела": "как дела",
    "пока": "отключение",
    "отключить": "отключение",
    "выйти": "отключение",
    "изменить переменную": "изменить переменную"
}

words_list += ["отключить", "выйти", "изменить", "переменную"]
target_words += ["отключить", "выйти", "изменить", "переменная"]


def get_models():
    unique_words = list(set(target_words))
    unique_commands_names = list(set(list(commands.values())))

    input_commands = list(commands.keys())
    target_commands = list(commands.values())

    word_model = WordModel(letters, unique_words)
    word_inputs, word_targets = make_words_data(word_model, words_list, target_words)
    word_model.fit(word_inputs, word_targets)

    command_model = CommandModel(unique_commands_names, word_model)
    command_inputs, command_targets = make_commands_data(command_model, input_commands, target_commands)
    command_model.fit(command_inputs, command_targets, epochs=100, lr=0.9)

    return word_model, command_model



if __name__ == '__main__':
    def main():
        word_model, command_model = get_models()

        input_commands = list(commands.keys())
        target_commands = list(commands.values())

        command_inputs, command_targets = make_commands_data(command_model, input_commands, target_commands)

        print_output(command_model, input_commands, command_inputs, command_targets)

        input_command = "какк дделла"
        print(command_model.predict(input_command))


    main()