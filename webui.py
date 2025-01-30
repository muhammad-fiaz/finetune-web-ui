import gradio as gr

from modules.logly import logly
from ui.components.AdvancedOptionsUI import AdvancedOptionsUI
from ui.components.Export import Export
from ui.components.Settings import Settings


# UI Class
class FineTuneUI:
    def __init__(self, handler):
        self.handler = handler
        self.settings = Settings()

    def create_ui(self):
        logging_enabled = self.settings.is_logging_enabled()

        """Create and return the Gradio UI."""
        with gr.Blocks(
            css="footer {visibility: hidden;}", title="Finetune WebUI"
        ) as demo:
            gr.Markdown("""
                <h1 style="text-align: center; font-size: 36px; font-weight: bold;">Fine-Tuning Model via Web UI</h1>
            """)

            with gr.Tabs():
                # Tab 1: Fine-Tune
                with gr.Tab("Fine-Tune"):
                    with gr.Row(equal_height=True, elem_id="main-row"):
                        # Left Column
                        with gr.Column(elem_id="left-column"):
                            dataset_name = gr.Dropdown(
                                choices=list(self.handler.datasets.keys()),
                                label="Dataset Name",
                                value=list(self.handler.datasets.keys())[0],
                                interactive=True,
                                info="Select the dataset you want to fine-tune the model with.",
                            )
                            refresh_datasets_button = gr.Button(
                                "Refresh Datasets", elem_id="refresh-datasets-button"
                            )
                            refresh_datasets_button.click(
                                self.handler.reload_datasets, outputs=dataset_name
                            )

                        # Right Column
                        with gr.Column(elem_id="right-column"):
                            model_name = gr.Dropdown(
                                choices=list(self.handler.models.keys()),
                                label="Model Name",
                                value=list(self.handler.models.keys())[0],
                                interactive=True,
                                info="Select the model you want to fine-tune.",
                            )
                            refresh_models_button = gr.Button(
                                "Refresh Models", elem_id="refresh-models-button"
                            )
                            refresh_models_button.click(
                                self.handler.reload_models, outputs=model_name
                            )

                    finetune_progressbar = gr.Textbox(
                        label="Progress",
                        interactive=False,
                        info="Displays the progress of the fine-tuning process.",
                    )
                    advanced_options_checkbox = gr.Checkbox(
                        label="Show Advanced Options",
                        value=False,
                        container=False,
                        info="Toggle to show or hide advanced fine-tuning options.",
                    )
                    finetune_button = gr.Button("Fine-Tune", elem_id="fine-tune-button")

                    advanced_options_ui = AdvancedOptionsUI()
                    advanced_block, advanced_options = advanced_options_ui.create_ui()

                    # Link the checkbox to show/hide advanced options
                    advanced_options_checkbox.change(
                        lambda show: gr.update(visible=show),
                        inputs=[advanced_options_checkbox],
                        outputs=[advanced_block],
                    )

                    # Trigger fine-tuning process
                    finetune_button.click(
                        self.handler.start_finetuning,
                        inputs=[dataset_name, model_name]
                        + list(advanced_options.values()),
                        outputs=[finetune_progressbar],
                    )

                # Tab 2: Download
                with gr.Tab("Download"):
                    with gr.Row(equal_height=True, elem_id="download-row"):
                        with gr.Column(elem_id="left-column"):
                            download_dataset = gr.Textbox(
                                label="Dataset URL",
                                placeholder="Enter the dataset URL (e.g., mlabonne/FineTome-100k)",
                                info="Provide the URL of the dataset you want to download.",
                            )
                        with gr.Column(elem_id="right-column"):
                            download_model = gr.Textbox(
                                label="Model Name",
                                placeholder="Enter the model name (e.g., mlabonne/FineTome-100k)",
                                info="Provide the name of the model you want to download.",
                            )
                    with gr.Row(equal_height=True, elem_id="download-action-row"):
                        download_progress = gr.Textbox(
                            label="Download Progress",
                            placeholder="Progress will be displayed here...",
                            interactive=False,
                            info="Displays the progress of the download process.",
                        )
                    download_button = gr.Button("Download", elem_id="download-button")
                # Tab 3: Export
                with gr.Tab("Export"):
                    with gr.Row(equal_height=True):
                        export_model_ui = Export()
                        export_model_ui.create_ui()

                # Tab 4: Settings
                with gr.Tab("Settings"):
                    with gr.Row(equal_height=True):
                        api_token = gr.Textbox(
                            label="API Token",
                            placeholder="Enter your Hugging Face API Token",
                            info="Enter your Hugging Face API token for authentication.",
                        )
                    with gr.Row(equal_height=True):
                        enable_logging_checkbox = gr.Checkbox(
                            label="Enable Logging",
                            value=logging_enabled,
                            interactive=True,
                            info="Toggle to enable or disable logging of operations.",
                        )

                    # Save the updated value of logging_enabled
                    enable_logging_checkbox.change(
                        lambda value: self.settings.set_logging_enabled(value),
                        inputs=[enable_logging_checkbox],
                        outputs=[],
                    )

                download_button.click(
                    self.handler.handle_download,
                    inputs=[download_dataset, download_model, api_token],
                    outputs=[download_progress],
                )

        logly.info("UI started successfully.")
        return demo
