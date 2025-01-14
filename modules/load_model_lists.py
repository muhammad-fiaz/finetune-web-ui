import json
import os
from modules.config import MODELS_JSON_PATH  # If needed, still use this for config-based path

def load_models_from_json(json_file=None):
    """
    Load the models from the specified JSON file. By default, looks for `models.json`
    in the current working directory.

    Args:
    - json_file (str): The path to the JSON file containing the models. If None,
                        defaults to `models.json` in the current working directory.

    Returns:
    - dict: The loaded models from the JSON file.
    """
    if json_file is None:
        # Use the current working directory and look for models.json
        json_file = os.path.join(os.getcwd(), MODELS_JSON_PATH)

    try:
        with open(json_file, 'r') as file:
            models = json.load(file)
        return models
    except FileNotFoundError:
        print(f"Error: The file {json_file} was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_file}.")
        return {}


def get_model_data(models, model_name):
    """
    Get the model data from the loaded models.

    Args:
    - models (dict): The loaded models.
    - model_name (str): The model name (key).

    Returns:
    - str: The model's data if found, or a 'not found' message.
    """
    return models.get(model_name, "Model not found")
