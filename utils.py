from types import SimpleNamespace
import json


def load_config(filename="config.json"):
    with open(filename, "r") as file:
        config_dict = json.load(file)

    # Recursively convert dictionary to an object
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d

    return dict_to_namespace(config_dict)
