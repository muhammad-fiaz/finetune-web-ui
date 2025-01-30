import torch
from unsloth import is_bfloat16_supported
from modules.finetune.unsloth import UnslothTrainer


class AsyncWorker:
    def __init__(self):
        # Clear GPU memory at the start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared GPU cache at start of application.")
        self.trainer = UnslothTrainer()

    def unsloth_trainer(self, dataset_name, model_name, advanced_options):
        # Load model with parameters
        self.trainer.load_model(
            model_name=model_name,
            max_seq_length=advanced_options["max_seq_length"],
            dtype=None,
            load_in_4bit=advanced_options["load_in_4bit"],
            token=None,
        )

        # Apply PEFT with parameters
        self.trainer.apply_peft(
            r=advanced_options["lora_r"],
            target_modules=advanced_options["target_modules"],
            lora_alpha=advanced_options["lora_alpha"],
            lora_dropout=advanced_options["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=advanced_options["random_state"],
            use_rslora=advanced_options["use_rslora"],
            loftq_config=advanced_options["loftq_config"],
        )

        # Set chat template
        self.trainer.set_chat_template(
            chat_template=advanced_options["set_chat_template"],
            mapping={
                "role": "role",
                "content": "content",
                "user": "user",
                "assistant": "assistant",
            },
            system_message=None,
            map_eos_token=advanced_options["map_eos_token"],
        )

        # Load dataset with parameters
        self.trainer.load_dataset(
            dataset_name=dataset_name,
            split=advanced_options["dataset_split"],
            dataset_num_proc=advanced_options["dataset_num_proc"],
            packing=advanced_options["dataset_packing"],
        )

        # Setup trainer with parameters
        self.trainer.setup_trainer(
            max_seq_length=advanced_options["trainer_max_seq_length"],
            per_device_train_batch_size=advanced_options["batch_size"],
            gradient_accumulation_steps=advanced_options["gradient_accumulation_steps"],
            warmup_steps=advanced_options["warmup_steps"],
            max_steps=advanced_options["max_steps"],
            learning_rate=advanced_options["learning_rate"],
            logging_steps=advanced_options["logging_steps"],
            optim=advanced_options["optim"],
            weight_decay=advanced_options["weight_decay"],
            lr_scheduler_type=advanced_options["lr_scheduler_type"],
            seed=advanced_options["random_state"],
            output_dir=advanced_options["output_dir"],
            report_to=advanced_options["train_report_to"],
        )

        # Train on responses only
        self.trainer.train_on_responses_only(
            instruction_part=advanced_options["instruction_part"],
            response_part=advanced_options["response_part"],
        )

        # Show memory stats before training
        self.trainer.show_memory_stats()

        # Train the model
        trainer_stats = self.trainer.train()

        # Show final memory and time stats after training
        self.trainer.show_final_memory_and_time_stats(trainer_stats.metrics)

        # Inference example
        messages = [
            {
                "role": "user",
                "content": "Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8,",
            }
        ]
        print(self.trainer.inference(messages))
        # Inference stream example
        self.trainer.inference_stream(messages)
        # Save model with parameters
        self.trainer.save_model(save_path="lora_model")

        return "Fine-tuning completed successfully!"

    def export_model(self, model_name, advanced_options):
        return "Model exported successfully!"
