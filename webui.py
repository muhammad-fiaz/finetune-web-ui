import gradio as gr
from modules.async_worker import AsyncWorker
from modules.download import main
from modules.load_datasets_lists import load_datasets_from_json
from modules.load_model_lists import load_models_from_json
from modules.logly import logly


import os
import toml
from modules.logly import logly
from modules.config import ConfigManager


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


class AdvancedOptionsUI:
    def __init__(self):
        self.block = None
        self.options = None

    def create_ui(self):
        """Create the advanced options UI."""
        with gr.Group(visible=False) as advanced_block:
            with gr.Row():
                learning_rate = gr.Slider(label="Learning Rate", minimum=0.0001, maximum=0.1, step=0.0001, value=0.001)
                batch_size = gr.Number(label="Batch Size", value=32)
                epochs = gr.Number(label="Epochs", value=10)
                gradient_accumulation_steps = gr.Number(label="Gradient Accumulation Steps", value=4)
                warmup_steps = gr.Number(label="Warmup Steps", value=5)
                max_steps = gr.Number(label="Max Steps", value=60)
                lora_r = gr.Number(label="LoRA r", value=16)
                lora_alpha = gr.Number(label="LoRA Alpha", value=16)
                lora_dropout = gr.Number(label="LoRA Dropout", value=0.0)
                random_state = gr.Number(label="Random State", value=3407)
            self.block = advanced_block
            self.options = {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "random_state": random_state,
            }
            return advanced_block, self.options


# Background Handler Class
class FineTuneHandler:
    def __init__(self):
        self.models = load_models_from_json()
        self.datasets = load_datasets_from_json()
        self.handler = AsyncWorker()

    def reload_datasets(self):
        """Reload datasets from JSON."""
        logly.info("Reloading datasets from JSON")
        self.datasets = load_datasets_from_json()
        logly.info("Reloaded datasets from JSON successfully")
        return gr.update(choices=list(self.datasets.keys()), value=list(self.datasets.keys())[0])

    def reload_models(self):
        """Reload models from JSON."""
        logly.info("Reloading models from JSON")
        self.models = load_models_from_json()
        logly.info("Reloaded models from JSON successfully")
        return gr.update(choices=list(self.models.keys()), value=list(self.models.keys())[0])

    def handle_download(self, dataset_url, model_name, token):
        """Handle the download process."""
        logly.info("Initiating download process")
        logly.info(f"Dataset URL: {dataset_url}, Model Name: {model_name}, Token Provided: {'Yes' if token else 'No'}")
        try:
            dataset_result, model_result, token = main(dataset_url, model_name, token)
            logly.info("Download process completed successfully")
            return dataset_result, model_result, token
        except Exception as e:
            logly.error(f"Error during download: {e}")
            raise

    def start_finetuning(self, dataset_name, model_name, learning_rate, batch_size, epochs, gradient_accumulation_steps,
                         warmup_steps, max_steps, lora_r, lora_alpha, lora_dropout, random_state):
        """Handle the fine-tuning process."""
        logly.info(f"Fine-Tuning Background Process Started!")

        advanced_options = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "random_state": random_state,
        }

        finetune_process = self.handler.unsloth_trainer(dataset_name, model_name, advanced_options)
        return finetune_process


# UI Class
class FineTuneUI:
    def __init__(self, handler):
        self.handler = handler
        self.settings = Settings()

    def create_ui(self):
        logging_enabled = self.settings.is_logging_enabled()

        """Create and return the Gradio UI."""
        with gr.Blocks(css="footer {visibility: hidden;}",title="Finetune WebUI") as demo:
            gr.Markdown("""
                <h1 style="text-align: center; font-size: 36px; font-weight: bold;">Fine-Tuning Model via Web UI</h1>
            """)

            with gr.Tabs():
                # Tab 1: Fine-Tune
                with gr.Tab("Fine-Tune"):
                    with gr.Row(equal_height=True, elem_id="main-row"):
                        # Left Column
                        with gr.Column(elem_id="left-column"):
                            gr.Markdown("Select the Dataset you want to fine-tune the model with.")
                            dataset_name = gr.Dropdown(
                                choices=list(self.handler.datasets.keys()),
                                label="Dataset Name",
                                value=list(self.handler.datasets.keys())[0],
                                interactive=True
                            )
                            refresh_datasets_button = gr.Button("Refresh Datasets", elem_id="refresh-datasets-button")
                            refresh_datasets_button.click(self.handler.reload_datasets, outputs=dataset_name)

                        # Right Column
                        with gr.Column(elem_id="right-column"):
                            gr.Markdown("Select the model you want to fine-tune.")
                            model_name = gr.Dropdown(
                                choices=list(self.handler.models.keys()),
                                label="Model Name",
                                value=list(self.handler.models.keys())[0],
                                interactive=True
                            )
                            refresh_models_button = gr.Button("Refresh Models", elem_id="refresh-models-button")
                            refresh_models_button.click(self.handler.reload_models, outputs=model_name)

                    finetune_progressbar = gr.Textbox(label="Progress", interactive=False)

                    advanced_options_ui = AdvancedOptionsUI()
                    advanced_block, advanced_options = advanced_options_ui.create_ui()

                    # Link the checkbox to show/hide advanced options
                    advanced_options_checkbox = gr.Checkbox(label="Show Advanced Options", value=False, container=False)
                    advanced_options_checkbox.change(
                        lambda show: gr.update(visible=show),
                        inputs=[advanced_options_checkbox],
                        outputs=[advanced_block]
                    )

                    finetune_button = gr.Button("Fine-Tune", elem_id="fine-tune-button")

                    # Trigger fine-tuning process
                    finetune_button.click(
                        self.handler.start_finetuning,
                        inputs=[dataset_name, model_name] + list(advanced_options.values()),
                        outputs=[finetune_progressbar]
                    )


                # Tab 2: Download
                with gr.Tab("Download"):
                    with gr.Row(equal_height=True, elem_id="download-row"):
                        with gr.Column(elem_id="left-column"):
                            gr.Markdown("Enter the dataset URL to download.")
                            download_dataset = gr.Textbox(
                                label="Dataset URL",
                                placeholder="Enter the dataset URL (e.g., mlabonne/FineTome-100k)"
                            )
                        with gr.Column(elem_id="right-column"):
                            gr.Markdown("Enter the model name to download.")
                            download_model = gr.Textbox(
                                label="Model Name",
                                placeholder="Enter the model name (e.g., mlabonne/FineTome-100k)"
                            )
                    with gr.Row(equal_height=True, elem_id="download-action-row"):
                        download_progress = gr.Textbox(
                            label="Download Progress",
                            placeholder="Progress will be displayed here...",
                            interactive=False
                        )
                    download_button = gr.Button("Download", elem_id="download-button")


                # Tab 3: Settings
                with gr.Tab("Settings"):
                        gr.Markdown("<h2 style='text-align: center;'>Fine-tuning settings for the model</h2>")
                        with gr.Row(equal_height=True):
                            api_token = gr.Textbox(label="API Token", placeholder="Enter your Hugging Face API Token")
                        with gr.Row(equal_height=True):
                            enable_logging_checkbox = gr.Checkbox(
                                label="Enable Logging",
                                value=logging_enabled,
                                interactive=True
                            )

                        # Save the updated value of logging_enabled
                        enable_logging_checkbox.change(
                            lambda value: self.settings.set_logging_enabled(value),
                            inputs=[enable_logging_checkbox],
                            outputs=[]
                        )

                download_button.click(
                    self.handler.handle_download,
                    inputs=[download_dataset, download_model, api_token],
                    outputs=[download_progress]
                )

        logly.info("UI started successfully.")
        return demo

