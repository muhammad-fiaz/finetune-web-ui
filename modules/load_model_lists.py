import json
import os
from modules.config import ConfigManager, config_manager
from modules.logly import logly

def load_models_from_json(json_file=None):
    if json_file is None:
        # Get the default path from the config.toml file
        json_file = config_manager.load_config()["paths"]["models_json_path"]

    try:
        with open(json_file, 'r') as file:
            models = json.load(file)
        return models
    except FileNotFoundError:
        logly.error(f"Error: The file {json_file} was not found.")
        return {}
    except json.JSONDecodeError:
        logly.error(f"Error: Failed to decode JSON from {json_file}.")
        return {}

def get_model_data(models, model_name):
    return models.get(model_name, "Model not found")
