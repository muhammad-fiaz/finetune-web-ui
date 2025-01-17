import gradio as gr

from modules.async_worker import AsyncWorker
from modules.download import main
from modules.load_datasets_lists import load_datasets_from_json
from modules.load_model_lists import load_models_from_json
from modules.logly import logly



class AdvancedOptionsUI:
    def __init__(self):
        self.block = None

    def create_ui(self):
        """Create the advanced options UI."""
        with gr.Group(visible=False) as advanced_block:
            with gr.Row():
                gr.Slider(label="Learning Rate", minimum=0.0001, maximum=0.1, step=0.0001, value=0.001)
                gr.Number(label="Batch Size", value=32)
                gr.Number(label="Epochs", value=10)
        self.block = advanced_block
        return advanced_block

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


    def start_finetuning(self, dataset_name, model_name):
        """Handle the fine-tuning process."""
        self.handler = AsyncWorker()
        logly.info(f"Fine-Tuning Background Process Started!")
        finetune_process= self.handler.unsloth_trainer(dataset_name, model_name)
        return finetune_process


# UI Class
class FineTuneUI:
    def __init__(self, handler):
        self.handler = handler

    def create_ui(self):
        """Create and return the Gradio UI."""
        with gr.Blocks(css="footer {visibility: hidden;}") as demo:
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
                    with gr.Row(equal_height=True, elem_id="fine-tune-action-row"):
                        with gr.Column( elem_id="fine-tune-action-row"):
                          drap_and_drop_datasets = gr.File(label="Upload Dataset")
                          file_type=gr.Radio(["csv","json","txt"],label="File Type")
                        with gr.Column( elem_id="fine-tune-action-row"):
                            drap_and_drop_model = gr.File(label="Upload Model")
                            file_type=gr.Radio(["zip"],label="File Type")
                    with gr.Row(equal_height=True, elem_id="fine-tune-action-row"):
                        finetune_progressbar= gr.Textbox(label="Progress",  interactive=False)

                    with gr.Row(equal_height=True, elem_id="fine-tune-action-row"):
                          finetune_button = gr.Button("Fine-Tune", elem_id="fine-tune-button")
                    advanced_options = gr.Checkbox(label="Show Advanced Options", value=False, container=False)

                    # Include Advanced Options UI
                    advanced_block = AdvancedOptionsUI().create_ui()

                    # Link the checkbox to show/hide advanced options
                    advanced_options.change(
                        lambda show: gr.update(visible=show),
                        inputs=[advanced_options],
                        outputs=[advanced_block]
                    )
                    # Trigger fine-tuning process
                    finetune_button.click(
                        self.handler.start_finetuning,
                        inputs=[dataset_name, model_name],
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
                        gr.Checkbox(label="Enable Logging", value=True)
                        gr.Checkbox(label="Save Logs", value=True)


                download_button.click(
                    self.handler.handle_download,
                    inputs=[download_dataset, download_model, api_token],
                    outputs=[download_progress]
                )
        logly.info("UI started successfully.")
        return demo

