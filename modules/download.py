import os
from datasets import load_dataset
from unsloth import FastLanguageModel


# Function to download the model
def download_model(model_name, token=None):
    try:
        print(f"Downloading model: {model_name}")

        # Load model from Hugging Face using optional token for authentication
        model, tokenizer = FastLanguageModel.from_pretrained(model_name, token=token)

        print(f"Model '{model_name}' downloaded successfully!")
        return model, tokenizer

    except Exception as e:
        print(f"Error downloading model: {e}")
        return None, f"Error downloading model: {e}. Please check the Hugging Face API token if required."


# Function to download the dataset
def download_dataset(dataset_url):
    try:
        print(f"Downloading dataset: {dataset_url}")

        # Load dataset from Hugging Face Hub
        dataset = load_dataset(dataset_url)

        # The dataset is automatically cached in the default Hugging Face dataset cache directory
        print(f"Dataset '{dataset_url}' downloaded successfully!")
        return dataset

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return f"Error downloading dataset: {e}"


# Main function to handle model and dataset downloads
def main(download_dataset_url=None, download_model_name=None, huggingface_token=None):
    dataset_response = "No dataset URL provided, skipping dataset download."
    model_response = "No model name provided, skipping model download."

    # Download dataset if URL is provided
    if download_dataset_url:
        dataset_response = download_dataset(download_dataset_url)

    # Download model if model name is provided
    if download_model_name:
        model, model_response = download_model(download_model_name, token=huggingface_token)

    return dataset_response, model_response
