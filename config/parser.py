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

def get_system_prompt(config_path='system_prompt.json', agent_name="cursor") -> str:
    config = load_api_config(config_path)
    for message in config.get('messages', []):
        if message.get('role') == 'system':
            return message.get('content', '')
    return ""  # Return empty string if no system prompt is found