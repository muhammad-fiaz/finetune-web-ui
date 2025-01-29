
from modules.config import ConfigManager
from modules.logly import logly


class Settings:
    def __init__(self):
        self.config_manager = ConfigManager()  # Initialize ConfigManager

    def is_logging_enabled(self):
        """
        Check if logging is enabled.
        Returns:
            bool: True if logging is enabled, False otherwise.
        """
        logging_enabled = self.config_manager.get_config_value("settings", "logging_enabled")
        logly.info(f"Logging enabled status: {logging_enabled}")
        return logging_enabled

    def set_logging_enabled(self, enabled):
        """
        Set the logging enabled status.
        Args:
            enabled (bool): True to enable logging, False to disable it.
        """
        self.config_manager.update_config("settings", "logging_enabled", enabled)
        status = "enabled" if enabled else "disabled"
        logly.info(f"Logging has been {status} successfully.")

    def toggle_logging(self):
        """
        Toggle the logging enabled status.
        """
        current_status = self.is_logging_enabled()
        self.set_logging_enabled(not current_status)
        new_status = "enabled" if not current_status else "disabled"
        logly.info(f"Logging toggled to {new_status}.")


