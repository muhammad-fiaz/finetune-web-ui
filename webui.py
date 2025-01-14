import gradio as gr
from modules.download import main
from modules.load_datasets_lists import load_datasets_from_json
from modules.load_model_lists import load_models_from_json

# Load models and datasets from JSON initially
models = load_models_from_json()
datasets = load_datasets_from_json()

# Function to reload datasets
def reload_datasets():
    datasets = load_datasets_from_json()  # Reload the datasets
    return gr.update(choices=list(datasets.keys()), value=list(datasets.keys())[0])  # Update dropdown

# Function to reload models
def reload_models():
    models = load_models_from_json()  # Reload the models
    return gr.update(choices=list(models.keys()), value=list(models.keys())[0])  # Update dropdown

def handle_download(dataset_url, model_name, token):
    # The 'main' function will handle the download logic and return the results
    dataset_result, model_result = main(dataset_url, model_name, token)
    return dataset_result, model_result


# Gradio UI setup
with gr.Blocks() as demo:
    # Title with larger and centered styling
    gr.Markdown("""
        <h1 style="text-align: center; font-size: 36px; font-weight: bold;">Fine-Tuning Model via Web UI</h1>
    """)

    # Add tabs for navigation
    with gr.Tabs():
        # Tab 1: Fine-Tune
        with gr.Tab("Fine-Tune"):
            with gr.Row(equal_height=True, elem_id="main-row"):
                # Left Column (Fine-Tune)
                with gr.Column(elem_id="left-column"):
                    gr.Markdown("Select the Dataset you want to fine-tune the model with.")
                    dataset_name = gr.Dropdown(
                        choices=list(datasets.keys()),  # Load dataset names directly from the JSON keys
                        label="Dataset Name",
                        value=list(datasets.keys())[0],  # Default to the first dataset in the list
                        interactive=True
                    )
                    refresh_datasets_button = gr.Button("Refresh Datasets", elem_id="refresh-datasets-button")
                    refresh_datasets_button.click(reload_datasets,
                                                  outputs=dataset_name)  # Bind the button click to reload datasets

                # Right Column (Fine-Tune)
                with gr.Column(elem_id="right-column"):
                    gr.Markdown("Select the model you want to fine-tune.")
                    model_name = gr.Dropdown(
                        choices=list(models.keys()),  # Load model names directly from the JSON keys
                        label="Model Name",
                        value=list(models.keys())[0],  # Default to the first model in the list
                        interactive=True
                    )
                    refresh_models_button = gr.Button("Refresh Models", elem_id="refresh-models-button")
                    refresh_models_button.click(reload_models,
                                                outputs=model_name)  # Bind the button click to reload models

        # Tab 2: Download
        with gr.Tab("Download"):
            with gr.Row(equal_height=True, elem_id="download-row"):
                # Left Column (Download)
                with gr.Column(elem_id="left-column"):
                    gr.Markdown("Enter the dataset URL and model name to download (optional).")
                    download_dataset = gr.Textbox(
                        label="Dataset URL",
                        placeholder="Enter the dataset URL to download (e.g., mlabonne/FineTome-100k)"
                    )

                # Right Column (Download)
                with gr.Column(elem_id="right-column"):
                    gr.Markdown("Enter the model name to download (optional).")
                    download_model = gr.Textbox(
                        label="Model Name",
                        placeholder="Enter model name to download (e.g., mlabonne/FineTome-100k)"
                    )

            with gr.Row(equal_height=True, elem_id="download-action-row"):
                download_button = gr.Button("Download", elem_id="download-button")

        # Tab 3: Settings
        with gr.Tab("Settings"):
            # Center the subheading for fine-tuning settings
            gr.Markdown("<h2 style='text-align: center;'>Fine-tuning settings for the model</h2>")

            with gr.Row(equal_height=True, elem_id="api-token-row"):
                api_token = gr.Textbox(label="API Token", placeholder="Enter your Hugging Face API Token")

            with gr.Row(equal_height=True, elem_id="logging-settings-row"):
                enable_logging = gr.Checkbox(label="Enable Logging", value=True)
                save_logs = gr.Checkbox(label="Save Logs", value=True)

    # Define the logic for the "Download" button
    download_button.click(
        handle_download,  # Call the handle_download function
        inputs=[download_dataset, download_model, api_token],  # Pass the inputs
        outputs=[gr.Textbox(), gr.Textbox()]  # Output fields for results
    )

    demo.launch()
