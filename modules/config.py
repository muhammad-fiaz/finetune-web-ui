import toml
import os
from modules.logly import logly


class ConfigManager:
    CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config.toml")

    default_config = {
        "project_host": {
            "host": "127.0.0.1",
            "port": 7860,
            "debug": False,
            "ssr_mode": False,
        },
        "paths": {
            "models_json_path": "configs/models.json",
            "datasets_json_path": "configs/datasets.json",
        },
        "project_name": {
            "name": "FineTuneWebUI",
            "version": "0.0.0",
        },
        "settings": {
            "logging_enabled": True,
        },
        "huggingface": {
            "api_token": "",  # Default token is empty
        },
    }

    def __init__(self):
        self.create_config()

    def create_config(self):
        """Create a new config.toml file with default settings if it doesn't exist."""
        if not os.path.exists(self.CONFIG_FILE_PATH):
            with open(self.CONFIG_FILE_PATH, "w") as config_file:
                toml.dump(self.default_config, config_file)
            logly.info(f"Config file created at {self.CONFIG_FILE_PATH}")

    def load_config(self):
        """Load the config from the config.toml file."""
        if os.path.exists(self.CONFIG_FILE_PATH):
            with open(self.CONFIG_FILE_PATH, "r") as config_file:
                config = toml.load(config_file)
            return config
        else:
            logly.warn(f"Config file {self.CONFIG_FILE_PATH} not found. Creating one.")
            self.create_config()
            return self.default_config

    def update_config(self, section, key, value):
        """Update a specific setting in the config.toml file within a section."""
        config = self.load_config()
        if section in config:
            config[section][key] = value
            with open(self.CONFIG_FILE_PATH, "w") as config_file:
                toml.dump(config, config_file)
            logly.info(f"Updated {section} -> {key} to {value} in {self.CONFIG_FILE_PATH}")
        else:
            logly.error(f"Section {section} not found in config.")

    def get_config_value(self, section, key):
        """Get a value from a specific section in the config.toml file."""
        config = self.load_config()
        return config.get(section, {}).get(key)


# Initialize config manager
config_manager = ConfigManager()
