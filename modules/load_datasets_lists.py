import json
import os
from modules.config import ConfigManager, config_manager
from modules.logly import logly


def load_datasets_from_json(json_file=None):
    if json_file is None:
        # Get the default path from the config.toml file
        json_file = config_manager.load_config()["paths"]["datasets_json_path"]

    try:
        with open(json_file, "r") as file:
            datasets = json.load(file)
        return datasets
    except FileNotFoundError:
        logly.error(f"Error: The file {json_file} was not found.")
        return {}
    except json.JSONDecodeError:
        logly.error(f"Error: Failed to decode JSON from {json_file}.")
        return {}


def get_dataset_data(datasets, dataset_name):
    return datasets.get(dataset_name, "dataset not found")
