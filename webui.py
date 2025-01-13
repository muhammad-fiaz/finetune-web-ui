import gradio as gr
from modules.train import train_model


def start_training(dataset_url, model_name="meta-llama/Llama-3.2-1B"):
    """Function to trigger the training from the UI."""
    # Here you can call the training function, passing dataset_url and model_name
    train_model(dataset_url, model_name)

    return f"Training started with dataset: {dataset_url} and model: {model_name}"


# Gradio UI setup
with gr.Blocks() as demo:
    gr.Markdown("### Fine-Tuning Model via Web UI")

    dataset_url = gr.Textbox(label="Dataset URL", placeholder="Enter URL of your dataset")
    model_name = gr.Textbox(label="Model Name", value="meta-llama/Llama-3.2-1B", placeholder="Enter the model name")

    start_button = gr.Button("Start Training")

    output = gr.Textbox(label="Training Status")

    # When the button is clicked, trigger the start_training function
    start_button.click(
        fn=start_training,
        inputs=[dataset_url, model_name],
        outputs=output
    )


