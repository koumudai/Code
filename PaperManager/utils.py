import json


def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)