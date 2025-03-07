import json


def load_config(filename="config.json"):
    with open(filename, 'r') as file:
        return json.load(file)
