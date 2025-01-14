import json
import os
from modules.config import DATASETS_JSON_PATH

def load_datasets_from_json(json_file=None):

    if json_file is None:
        # Use the current working directory and look for datasets.json
        json_file = os.path.join(os.getcwd(),DATASETS_JSON_PATH)

    try:
        with open(json_file, 'r') as file:
            datasets = json.load(file)
        return datasets
    except FileNotFoundError:
        print(f"Error: The file {json_file} was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_file}.")
        return {}


def get_dataset_data(datasets, dataset_name):
    """
    Get the dataset data from the loaded datasets.

    Args:
    - datasets (dict): The loaded datasets.
    - dataset_name (str): The dataset name (key).

    Returns:
    - str: The dataset's data if found, or a 'not found' message.
    """
    return datasets.get(dataset_name, "dataset not found")
