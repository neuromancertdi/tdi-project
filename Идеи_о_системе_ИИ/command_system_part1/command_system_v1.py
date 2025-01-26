from command_recognition import *

word_model, command_model = get_models()

variables = {"is_run": True}

def bye():
    variables["is_run"] = False

actions = {"отключение": bye}

while variables["is_run"]:
    input_command = input("enter a command: ")

    command_name = command_model.predict(input_command)

    print(command_name)

    if command_name in actions:
        actions[command_name]()



