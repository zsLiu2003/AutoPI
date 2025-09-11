import yaml
import json
def load_config(config_path='config.yaml') -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_api_config(config_path='api_config.json'):

    with open(config_path, 'r') as file:
        config = json.load(file)
    return config