import os
from datasets import load_dataset
from modules.logly import logly


# Function to download the model
def download_model(model_name, token=None):
    try:
        # Load model from Hugging Face using optional token for authentication
        from unsloth import FastLanguageModel
        FastLanguageModel.from_pretrained(model_name, token=token)
        logly.info(f"Model '{model_name}' downloaded successfully!")
        return f"Model '{model_name}' downloaded successfully!"

    except Exception as e:
        logly.error(f"Model '{model_name}' download failed: {e}")
        return f"Error downloading model: {e}. Please check the Hugging Face API token if required."


# Function to download the dataset
def download_dataset(dataset_url, token=None):
    try:
        # Load dataset from Hugging Face Hub
        load_dataset(dataset_url, token=token)
        logly.info(f"Dataset '{dataset_url}' downloaded successfully!")
        # The dataset is automatically cached in the default Hugging Face dataset cache directory
        return f"Dataset '{dataset_url}' downloaded successfully!"

    except Exception as e:
        logly.error(f"Dataset '{dataset_url}' download failed: {e}")
        return f"Error downloading dataset: {e}"


# Main function to handle model and dataset downloads
def main(download_dataset_url=None, download_model_name=None, huggingface_token=None):
    logly.info("Starting main download process.")
    logly.info(f"Input parameters: Dataset URL={download_dataset_url}, Model Name={download_model_name}, Token Provided={'Yes' if huggingface_token else 'No'}")

    dataset_response = ""
    model_response = ""
    token = huggingface_token

    # Download dataset if URL is provided
    if download_dataset_url:
        logly.info(f"Attempting to download dataset: {download_dataset_url}")
        dataset_response = download_dataset(download_dataset_url, token=huggingface_token)
        logly.info(f"Dataset download response: {dataset_response}")

    # Download model if model name is provided
    if download_model_name:
        logly.info(f"Attempting to download model: {download_model_name}")
        model_response = download_model(download_model_name, token=huggingface_token)
        logly.info(f"Model download response: {model_response}")

    # Return responses as a tuple of three values
    return dataset_response, model_response, token
