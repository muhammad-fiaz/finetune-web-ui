import torch
from unsloth import is_bfloat16_supported
from modules.finetune.unsloth import UnslothTrainer

class AsyncWorker:
    def __init__(self):
        self.trainer = UnslothTrainer()

    def unsloth_trainer(self, dataset_name, model_name, advanced_options):
        # Load model with parameters
        self.trainer.load_model(
            model_name=model_name,
            max_seq_length=2048,
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            load_in_4bit=True,
            token=None
        )

        # Apply PEFT with parameters
        self.trainer.apply_peft(
            r=advanced_options["lora_r"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=advanced_options["lora_alpha"],
            lora_dropout=advanced_options["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=advanced_options["random_state"],
            use_rslora=False,
            loftq_config=None
        )

        # Set chat template
        self.trainer.set_chat_template(chat_template="llama-3.1")

        # Load dataset with parameters
        self.trainer.load_dataset(
            dataset_name=dataset_name,
            split="train",
            dataset_num_proc=1,
            packing=False
        )

        # Setup trainer with parameters
        self.trainer.setup_trainer(
            max_seq_length=2048,
            per_device_train_batch_size=advanced_options["batch_size"],
            gradient_accumulation_steps=advanced_options["gradient_accumulation_steps"],
            warmup_steps=advanced_options["warmup_steps"],
            max_steps=advanced_options["max_steps"],
            learning_rate=advanced_options["learning_rate"],
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=advanced_options["random_state"],
            output_dir="outputs",
            report_to="none"
        )

        # Train on responses only
        self.trainer.train_on_responses_only(
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        # Show memory stats before training
        self.trainer.show_memory_stats()

        # Train the model
        trainer_stats = self.trainer.train()

        # Show final memory and time stats after training
        self.trainer.show_final_memory_and_time_stats(trainer_stats.metrics)

        # Inference example
        messages = [{"role": "user", "content": "Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8,"}]
        print(self.trainer.inference(messages))
        # Inference stream example
        self.trainer.inference_stream(messages)
        # Save model with parameters
        self.trainer.save_model(save_path="lora_model")

        return "Fine-tuning completed successfully!"