import gradio as gr
from modules.download import main
from modules.load_datasets_lists import load_datasets_from_json
from modules.load_model_lists import load_models_from_json
from modules.logly import logly


# Background Handler Class
class FineTuneHandler:
    def __init__(self):
        self.models = load_models_from_json()
        self.datasets = load_datasets_from_json()

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

# UI Class
class FineTuneUI:
    def __init__(self, handler):
        self.handler = handler

    def create_ui(self):
        """Create and return the Gradio UI."""
        with gr.Blocks() as demo:
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
                        gr.Checkbox(label="Enable Logging", value=True)
                        gr.Checkbox(label="Save Logs", value=True)


                download_button.click(
                    self.handler.handle_download,
                    inputs=[download_dataset, download_model, api_token],
                    outputs=[download_progress]
                )
        logly.info("UI created successfully.")
        return demo

