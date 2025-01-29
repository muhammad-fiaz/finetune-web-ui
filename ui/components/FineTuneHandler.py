from modules.async_worker import AsyncWorker
from modules.download import main
from modules.load_datasets_lists import load_datasets_from_json
from modules.load_model_lists import load_models_from_json
from modules.logly import logly
import gradio as gr


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

    def start_finetuning(self, dataset_name, model_name, max_seq_length, load_in_4bit, learning_rate, batch_size,
                         epochs, gradient_accumulation_steps, warmup_steps, max_steps, lora_r, lora_alpha,
                         lora_dropout, random_state, loftq_config, use_rslora, target_modules, logging_steps,
                         trainer_max_seq_length, weight_decay, optim, lr_scheduler_type, output_dir, dataset_split, dataset_num_proc, dataset_packing, train_report_to,
                         set_chat_template, instruction_part, response_part, use_gradient_checkpointing, map_eos_token, mapping_template):
        """Handle the fine-tuning process."""
        logly.info(f"Fine-Tuning Background Process Started!")

        advanced_options = {
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
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
            "loftq_config": loftq_config,
            "use_rslora": use_rslora,
            "target_modules": target_modules,
            "logging_steps": logging_steps,
            "trainer_max_seq_length": trainer_max_seq_length,
            "weight_decay": weight_decay,
            "optim": optim,
            "lr_scheduler_type": lr_scheduler_type,
            "output_dir": output_dir,
            "dataset_split": dataset_split,
            "dataset_num_proc": dataset_num_proc,
            "dataset_packing": dataset_packing,
            "train_report_to": train_report_to,
            "set_chat_template": set_chat_template,
            "instruction_part": instruction_part,
            "response_part": response_part,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "map_eos_token": map_eos_token,
            "mapping_template": mapping_template,

        }

        finetune_process = self.handler.unsloth_trainer(dataset_name, model_name, advanced_options)
        return finetune_process
