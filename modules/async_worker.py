import time
from modules.logly import logly

class AsyncWorker:
    def __init__(self):
        pass
    def unsloth_trainer(self, dataset_name, model_name):
        """Simulate the fine-tuning process and update progress."""
        logly.info(f"Starting fine-tuning with dataset: {dataset_name} and model: {model_name}")


        # Simulate a completed fine-tuning
        logly.info(f"Fine-tuning completed for dataset: {dataset_name} and model: {model_name}")
        return "Fine-tuning completed successfully!"
